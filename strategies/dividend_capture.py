"""股息捕获策略 — 基于Alpaca公司行动数据的股息事件驱动策略"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

from qf.strategy import BaseStrategy
from qf.data_alpaca import AlpacaDataLoader


# ════════════════════════════════════════════════════════════════════
#  1. 股息捕获策略 (短期事件驱动)
# ════════════════════════════════════════════════════════════════════

class DividendCaptureStrategy(BaseStrategy):
    """
    股息捕获策略 — 在除息日前买入，除息日后卖出，赚取股息收入。

    逻辑:
        1. 从Alpaca获取即将到来的股息公告
        2. 除息日前2-5天买入（捕获股息资格）
        3. 除息日当天或之后卖出
        4. 过滤条件: 年化股息收益率>1%, 价格>$20(市值代理)
        5. 按股息收益率加权持仓（收益率越高仓位越大）
        6. 风险控制: 前20天跌幅超5%的票排除（避免"股息陷阱"）
    """

    name = "Dividend Capture"
    description = "除息日前买入捕获股息，除息日后卖出"

    # 策略参数
    buy_days_before_ex = 3       # 除息日前几天买入（范围2-5天）
    sell_days_after_ex = 1       # 除息日后几天卖出
    min_annualized_yield = 0.01  # 最低年化股息收益率 1%
    min_price = 20.0             # 最低股价（市值>$1B代理）
    max_drawdown_20d = 0.05      # 前20天最大跌幅阈值
    lookback_days = 60           # 向前扫描股息公告的天数

    def __init__(self, loader=None, **kwargs):
        """
        初始化股息捕获策略。

        Parameters:
            loader: AlpacaDataLoader实例，为None时自动创建
            **kwargs: 覆盖默认参数
        """
        self.loader = loader or AlpacaDataLoader()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _fetch_upcoming_dividends(self, as_of=None):
        """
        获取即将到来的股息数据并做基础清洗。

        Returns:
            DataFrame: 含 target_symbol, cash, ex_date 等字段的股息公告
        """
        as_of = as_of or date.today()
        since = as_of - timedelta(days=self.lookback_days)
        until = as_of + timedelta(days=self.lookback_days)

        div_df = self.loader.get_corporate_actions('dividend', since=since, until=until)
        if div_df.empty:
            return div_df

        # 确保日期格式
        div_df['ex_date'] = pd.to_datetime(div_df['ex_date'])
        div_df['cash'] = pd.to_numeric(div_df['cash'], errors='coerce').fillna(0)

        # 只保留有效记录
        div_df = div_df[
            (div_df['cash'] > 0) &
            (div_df['target_symbol'].str.len() > 0) &
            (div_df['ex_date'].notna())
        ].copy()

        return div_df

    def _get_price_data(self, symbols, end_date, lookback=30):
        """
        获取历史价格数据用于过滤和评分。

        Returns:
            DataFrame: MultiIndex (symbol, timestamp) 的日K线
        """
        start = (pd.Timestamp(end_date) - timedelta(days=lookback)).strftime('%Y-%m-%d')
        end = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        # 分批获取，Alpaca对单次请求符号数有限制
        batch_size = 50
        frames = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            bars = self.loader.get_daily_bars(batch, start, end)
            if bars is not None and not bars.empty:
                frames.append(bars)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    def _compute_scores(self, div_df, as_of=None):
        """
        计算每只股票的综合得分。

        评分公式: score = dividend_yield * (1 - recent_drawdown)
            - dividend_yield: 年化股息收益率
            - recent_drawdown: 前20天最大回撤（惩罚下跌股票）

        Returns:
            DataFrame: 含 symbol, score, annualized_yield, drawdown_20d 等
        """
        as_of = as_of or date.today()
        if div_df.empty:
            return pd.DataFrame()

        symbols = div_df['target_symbol'].unique().tolist()
        price_data = self._get_price_data(symbols, as_of, lookback=30)

        if price_data.empty:
            return pd.DataFrame()

        scores = []
        for _, row in div_df.iterrows():
            sym = row['target_symbol']
            cash = float(row['cash'])
            ex_dt = pd.Timestamp(row['ex_date'])

            # 获取该股票的价格序列
            try:
                if isinstance(price_data.index, pd.MultiIndex):
                    sym_prices = price_data.loc[sym]['close'] if sym in price_data.index.get_level_values(0) else None
                else:
                    sym_prices = price_data[price_data['symbol'] == sym]['close'] if 'symbol' in price_data.columns else None
            except (KeyError, TypeError):
                continue

            if sym_prices is None or len(sym_prices) < 5:
                continue

            last_price = float(sym_prices.iloc[-1])

            # 过滤: 价格>$20 (市值代理)
            if last_price < self.min_price:
                continue

            # 年化股息收益率 (假设季度分红则*4)
            quarterly_yield = cash / last_price
            annualized_yield = quarterly_yield * 4

            if annualized_yield < self.min_annualized_yield:
                continue

            # 计算近20天回撤
            recent = sym_prices.tail(20)
            if len(recent) >= 2:
                peak = recent.max()
                drawdown = (peak - recent.iloc[-1]) / peak if peak > 0 else 0
            else:
                drawdown = 0

            # 风险过滤: 排除"股息陷阱"
            if drawdown > self.max_drawdown_20d:
                continue

            # 综合得分: 收益率 * (1 - 回撤)
            score = annualized_yield * (1.0 - drawdown)

            scores.append({
                'symbol': sym,
                'ex_date': ex_dt,
                'cash_dividend': cash,
                'last_price': last_price,
                'annualized_yield': annualized_yield,
                'drawdown_20d': drawdown,
                'score': score,
            })

        return pd.DataFrame(scores)

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """
        生成信号DataFrame (date x symbol)。

        基于即将到来的股息事件生成交易信号:
            - 除息日前2-5天: 正信号（买入）
            - 除息日当天/之后: 零信号（卖出）
            - 信号强度按评分加权

        Parameters:
            data: 数据字典，可含 'as_of_date' 键指定日期

        Returns:
            DataFrame: index=日期, columns=股票代码, values=信号强度
        """
        as_of = data.get('as_of_date', date.today())
        if isinstance(as_of, str):
            as_of = pd.Timestamp(as_of).date()

        # 获取股息数据
        div_df = self._fetch_upcoming_dividends(as_of=as_of)
        if div_df.empty:
            print("未找到即将到来的股息公告")
            return pd.DataFrame()

        # 筛选: 除息日在买入窗口内
        as_of_ts = pd.Timestamp(as_of)
        buy_start = as_of_ts + timedelta(days=2)
        buy_end = as_of_ts + timedelta(days=self.buy_days_before_ex + 5)
        upcoming = div_df[
            (div_df['ex_date'] >= buy_start) &
            (div_df['ex_date'] <= buy_end)
        ].copy()

        if upcoming.empty:
            print(f"当前窗口内无符合条件的股息事件 ({buy_start.date()} - {buy_end.date()})")
            return pd.DataFrame()

        # 计算得分
        scored = self._compute_scores(upcoming, as_of=as_of)
        if scored.empty:
            return pd.DataFrame()

        # 构建信号矩阵: 在买入到卖出窗口内填充信号
        dates = pd.bdate_range(
            start=as_of_ts,
            end=as_of_ts + timedelta(days=self.buy_days_before_ex + self.sell_days_after_ex + 10),
        )
        symbols = scored['symbol'].unique().tolist()
        signal = pd.DataFrame(0.0, index=dates, columns=symbols)

        for _, row in scored.iterrows():
            sym = row['symbol']
            ex_dt = pd.Timestamp(row['ex_date'])
            entry_date = ex_dt - pd.offsets.BDay(self.buy_days_before_ex)
            exit_date = ex_dt + pd.offsets.BDay(self.sell_days_after_ex)

            # 买入期间信号为正（按score加权）
            mask_buy = (signal.index >= entry_date) & (signal.index < ex_dt)
            signal.loc[mask_buy, sym] = row['score']

            # 卖出期间信号为负
            mask_sell = (signal.index >= ex_dt) & (signal.index <= exit_date)
            signal.loc[mask_sell, sym] = -row['score']

        # 去除全零行
        signal = signal.loc[(signal != 0).any(axis=1)]
        return signal

    def get_current_targets(self, as_of=None):
        """
        获取当前推荐买入/卖出的标的（便捷方法）。

        Returns:
            dict: {'buy': [...], 'sell': [...]}，每个含 symbol, score, ex_date 等
        """
        as_of = as_of or date.today()
        div_df = self._fetch_upcoming_dividends(as_of=as_of)
        if div_df.empty:
            return {'buy': [], 'sell': []}

        as_of_ts = pd.Timestamp(as_of)
        scored = self._compute_scores(div_df, as_of=as_of)
        if scored.empty:
            return {'buy': [], 'sell': []}

        buys = []
        sells = []
        for _, row in scored.iterrows():
            ex_dt = pd.Timestamp(row['ex_date'])
            days_to_ex = (ex_dt - as_of_ts).days

            entry = row.to_dict()
            entry['days_to_ex'] = days_to_ex

            if 2 <= days_to_ex <= self.buy_days_before_ex + 2:
                buys.append(entry)
            elif days_to_ex <= 0 and days_to_ex >= -self.sell_days_after_ex:
                sells.append(entry)

        buys.sort(key=lambda x: x['score'], reverse=True)
        sells.sort(key=lambda x: x['score'], reverse=True)
        return {'buy': buys, 'sell': sells}


# ════════════════════════════════════════════════════════════════════
#  2. 股息收益率长期价值策略
# ════════════════════════════════════════════════════════════════════

class DividendYieldStrategy(BaseStrategy):
    """
    股息收益率价值策略 — 基于12个月滚动股息收益率的多空因子。

    逻辑:
        - 收集过去12个月所有股息支付
        - 计算年化股息收益率 = 12个月累计股息 / 当前股价
        - 按收益率排名: 做多前20%（高收益率），做空后20%（低/无收益率）
        - 排除"股息陷阱": 收益率>10%的异常值视为数据错误或公司困境

    适用于长期价值组合，月度换仓。
    """

    name = "Dividend Yield Value"
    description = "12个月滚动股息收益率多空因子"

    long_pct = 0.20    # 做多前20%
    short_pct = 0.20   # 做空后20%
    max_yield = 0.10   # 收益率上限（排除异常）
    min_price = 20.0   # 最低价格门槛

    def __init__(self, loader=None, **kwargs):
        """
        初始化股息收益率策略。

        Parameters:
            loader: AlpacaDataLoader实例
        """
        self.loader = loader or AlpacaDataLoader()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _compute_trailing_yield(self, as_of=None):
        """
        计算所有股票的12个月滚动股息收益率。

        Returns:
            DataFrame: symbol, ttm_dividends, price, ttm_yield
        """
        as_of = as_of or date.today()
        since = as_of - timedelta(days=365)

        # 获取过去12个月的股息数据
        div_df = self.loader.get_corporate_actions('dividend', since=since, until=as_of)
        if div_df.empty:
            return pd.DataFrame()

        div_df['cash'] = pd.to_numeric(div_df['cash'], errors='coerce').fillna(0)
        div_df = div_df[div_df['cash'] > 0].copy()

        # 按股票汇总12个月累计股息
        ttm_divs = (
            div_df
            .groupby('target_symbol')['cash']
            .sum()
            .reset_index()
            .rename(columns={'target_symbol': 'symbol', 'cash': 'ttm_dividends'})
        )

        if ttm_divs.empty:
            return pd.DataFrame()

        # 获取当前价格
        symbols = ttm_divs['symbol'].tolist()
        try:
            snapshots = self.loader.get_snapshots(symbols)
            prices = {}
            for sym, snap in snapshots.items():
                if isinstance(snap, dict):
                    prices[sym] = snap.get('last', snap.get('close', 0))
                else:
                    prices[sym] = getattr(snap, 'latest_trade', {})
                    if hasattr(prices[sym], 'price'):
                        prices[sym] = prices[sym].price
                    else:
                        prices[sym] = 0
        except Exception:
            # 回退到日K线获取最近价格
            bars = self.loader.get_daily_bars(
                symbols[:50],
                (pd.Timestamp(as_of) - timedelta(days=5)).strftime('%Y-%m-%d'),
                pd.Timestamp(as_of).strftime('%Y-%m-%d'),
            )
            prices = {}
            if bars is not None and not bars.empty:
                if isinstance(bars.index, pd.MultiIndex):
                    for sym in symbols:
                        try:
                            prices[sym] = float(bars.loc[sym]['close'].iloc[-1])
                        except (KeyError, IndexError):
                            pass

        ttm_divs['price'] = ttm_divs['symbol'].map(prices)
        ttm_divs = ttm_divs.dropna(subset=['price'])
        ttm_divs = ttm_divs[ttm_divs['price'] >= self.min_price]

        # 计算收益率
        ttm_divs['ttm_yield'] = ttm_divs['ttm_dividends'] / ttm_divs['price']

        # 排除异常高收益率（股息陷阱/数据错误）
        ttm_divs = ttm_divs[ttm_divs['ttm_yield'] <= self.max_yield]

        return ttm_divs.sort_values('ttm_yield', ascending=False).reset_index(drop=True)

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """
        生成基于股息收益率排名的多空信号。

        做多前20%高收益率股票，做空后20%低/无收益率股票。
        信号值 = 标准化后的收益率排名分数。

        Parameters:
            data: 数据字典，可含 'as_of_date', 'all_symbols' 键

        Returns:
            DataFrame: index=日期, columns=股票代码, values=信号(-1到+1)
        """
        as_of = data.get('as_of_date', date.today())
        if isinstance(as_of, str):
            as_of = pd.Timestamp(as_of).date()

        yield_df = self._compute_trailing_yield(as_of=as_of)
        if yield_df.empty:
            return pd.DataFrame()

        n = len(yield_df)
        top_n = max(1, int(n * self.long_pct))
        bottom_n = max(1, int(n * self.short_pct))

        # 排名信号: 高收益率 = 正信号, 低收益率 = 负信号
        yield_df['rank'] = yield_df['ttm_yield'].rank(ascending=True, method='average')
        yield_df['signal'] = 0.0

        # 做多: 前quintile
        long_mask = yield_df['rank'] > (n - top_n)
        yield_df.loc[long_mask, 'signal'] = yield_df.loc[long_mask, 'ttm_yield']

        # 做空: 后quintile
        short_mask = yield_df['rank'] <= bottom_n
        yield_df.loc[short_mask, 'signal'] = -1.0

        # 如果提供all_symbols，对无股息的股票也做空
        all_symbols = data.get('all_symbols', [])
        if all_symbols:
            div_payers = set(yield_df['symbol'].tolist())
            non_payers = [s for s in all_symbols if s not in div_payers]
            if non_payers:
                non_payer_df = pd.DataFrame({
                    'symbol': non_payers,
                    'signal': -0.5,  # 无股息的票给予较弱做空信号
                })
                yield_df = pd.concat([yield_df[['symbol', 'signal']], non_payer_df], ignore_index=True)

        # 构建信号矩阵（单行 — 横截面信号）
        as_of_ts = pd.Timestamp(as_of)
        signal = pd.DataFrame(
            {row['symbol']: row['signal'] for _, row in yield_df.iterrows()},
            index=[as_of_ts],
        )

        return signal


# ════════════════════════════════════════════════════════════════════
#  3. 回测工具函数
# ════════════════════════════════════════════════════════════════════

def run_dividend_backtest(
    start_date='2024-01-01',
    end_date=None,
    initial_capital=100_000,
    loader=None,
    strategy_params=None,
):
    """
    股息捕获策略历史回测。

    逻辑:
        1. 按月滚动扫描历史股息公告
        2. 对每个除息事件模拟: 除息前买入 → 收取股息 → 除息后卖出
        3. 汇总盈亏、胜率、年化收益率

    Parameters:
        start_date: 回测开始日期
        end_date: 回测结束日期（默认今天）
        initial_capital: 初始资金
        loader: AlpacaDataLoader实例
        strategy_params: 策略参数字典

    Returns:
        dict: {
            'trades': DataFrame（每笔交易明细）,
            'summary': dict（汇总统计）,
            'equity_curve': Series（资金曲线）
        }
    """
    loader = loader or AlpacaDataLoader()
    end_date = end_date or date.today().isoformat()

    params = strategy_params or {}
    strategy = DividendCaptureStrategy(loader=loader, **params)

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    trades = []
    capital = initial_capital
    equity_history = []

    # 按月滚动
    current = start
    while current < end:
        month_end = min(current + timedelta(days=30), end)

        # 获取这个月的股息公告
        div_df = loader.get_corporate_actions('dividend', since=current, until=month_end)
        if div_df.empty:
            current = month_end
            continue

        div_df['ex_date'] = pd.to_datetime(div_df['ex_date'])
        div_df['cash'] = pd.to_numeric(div_df['cash'], errors='coerce').fillna(0)
        div_df = div_df[(div_df['cash'] > 0) & (div_df['ex_date'].notna())].copy()

        # 计算得分，过滤
        scored = strategy._compute_scores(div_df, as_of=current)
        if scored.empty:
            current = month_end
            continue

        # 模拟每笔交易
        for _, row in scored.iterrows():
            sym = row['symbol']
            ex_dt = pd.Timestamp(row['ex_date'])
            entry_dt = ex_dt - pd.offsets.BDay(strategy.buy_days_before_ex)
            exit_dt = ex_dt + pd.offsets.BDay(strategy.sell_days_after_ex)

            # 按score分配仓位（归一化）
            total_score = scored['score'].sum()
            weight = row['score'] / total_score if total_score > 0 else 1.0 / len(scored)
            position_value = capital * 0.9 * weight  # 90%资金投入
            shares = int(position_value / row['last_price'])
            if shares < 1:
                continue

            # 获取买入卖出价格
            try:
                trade_bars = loader.get_daily_bars(
                    [sym],
                    entry_dt.strftime('%Y-%m-%d'),
                    (exit_dt + timedelta(days=5)).strftime('%Y-%m-%d'),
                )
            except Exception:
                continue

            if trade_bars is None or trade_bars.empty:
                continue

            try:
                if isinstance(trade_bars.index, pd.MultiIndex):
                    sym_bars = trade_bars.loc[sym]
                else:
                    sym_bars = trade_bars
                closes = sym_bars['close']
            except (KeyError, TypeError):
                continue

            if len(closes) < 2:
                continue

            buy_price = float(closes.iloc[0])
            sell_price = float(closes.iloc[-1])
            dividend_income = row['cash_dividend'] * shares

            # 价格损益 + 股息收入
            price_pnl = (sell_price - buy_price) * shares
            total_pnl = price_pnl + dividend_income

            trades.append({
                'symbol': sym,
                'entry_date': entry_dt,
                'exit_date': exit_dt,
                'ex_date': ex_dt,
                'shares': shares,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'dividend_per_share': row['cash_dividend'],
                'dividend_income': dividend_income,
                'price_pnl': price_pnl,
                'total_pnl': total_pnl,
                'annualized_yield': row['annualized_yield'],
                'score': row['score'],
            })

            capital += total_pnl

        equity_history.append({
            'date': pd.Timestamp(month_end),
            'equity': capital,
        })
        current = month_end

    # 汇总
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_history)

    if trades_df.empty:
        summary = {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_pnl_per_trade': 0,
            'total_dividend_income': 0,
            'final_capital': capital,
            'total_return': 0,
        }
    else:
        wins = (trades_df['total_pnl'] > 0).sum()
        total_div = trades_df['dividend_income'].sum()
        total_pnl = trades_df['total_pnl'].sum()
        days = (end - start).days
        ann_factor = 365.0 / max(days, 1)

        summary = {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'win_rate': wins / len(trades_df),
            'avg_pnl_per_trade': total_pnl / len(trades_df),
            'total_dividend_income': total_div,
            'total_price_pnl': trades_df['price_pnl'].sum(),
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital,
            'annualized_return': ((capital / initial_capital) ** ann_factor) - 1,
            'best_trade': trades_df['total_pnl'].max(),
            'worst_trade': trades_df['total_pnl'].min(),
            'avg_holding_days': (
                (trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean()
            ),
        }

    equity_curve = (
        equity_df.set_index('date')['equity']
        if not equity_df.empty
        else pd.Series(dtype=float)
    )

    return {
        'trades': trades_df,
        'summary': summary,
        'equity_curve': equity_curve,
    }


# ════════════════════════════════════════════════════════════════════
#  便捷入口
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("股息捕获策略 — 当前推荐标的")
    print("=" * 60)

    strategy = DividendCaptureStrategy()
    targets = strategy.get_current_targets()

    if targets['buy']:
        print("\n【买入推荐】")
        for t in targets['buy']:
            print(f"  {t['symbol']:6s}  收益率={t['annualized_yield']:.2%}"
                  f"  得分={t['score']:.4f}  除息日={t['ex_date']}"
                  f"  ({t['days_to_ex']}天后)")
    else:
        print("\n当前无买入推荐")

    if targets['sell']:
        print("\n【卖出推荐】")
        for t in targets['sell']:
            print(f"  {t['symbol']:6s}  除息日已过 ({abs(t['days_to_ex'])}天)")
    else:
        print("\n当前无卖出推荐")

    print("\n" + "=" * 60)
    print("运行回测...")
    print("=" * 60)
    result = run_dividend_backtest(start_date='2024-06-01')
    s = result['summary']
    print(f"  总交易次数: {s['total_trades']}")
    print(f"  总收益:     ${s['total_pnl']:,.2f}")
    print(f"  胜率:       {s['win_rate']:.1%}")
    print(f"  股息收入:   ${s.get('total_dividend_income', 0):,.2f}")
    print(f"  总回报率:   {s['total_return']:.2%}")
