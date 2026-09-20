"""A-share intraday momentum strategy — first-30-minute momentum stock selection + volume/price confirmation + intraday stop loss

Notes on A-share market rules:
- T+1 settlement: shares bought today cannot be sold the same day, only from the next session
  This strategy simulates a T+0 environment (e.g. securities lending / convertible bonds / ETFs);
  live A-share trading requires closing out on the following day instead
- Price limits: +/-10% on the main board, +/-20% on ChiNext/STAR (this strategy defaults to 10%)
- Opening call auction: 9:15-9:25, the open is set at 9:25, continuous trading starts at 9:30
- Closing call auction: 14:57-15:00, orders cannot be cancelled
- Minimum trade size: 100 shares (1 lot); STAR/ChiNext allow 1 share
"""

import numpy as np
import pandas as pd
from datetime import time as dtime


# ═══════════════════════════════════════════════════════════════════
#  A股日内动量策略
# ═══════════════════════════════════════════════════════════════════

class ChinaIntradayMomentum:
    """
    Intraday momentum strategy (A-shares)

    Core logic:
    1. In the first 30 minutes after the open (9:30 ~ 10:00), compute each stock's gain and pick the N strongest
    2. Volume confirmation: first-30-minute volume > 1.5x the 5-day average first-30-minute volume
    3. Price limit filter: exclude stocks already limit-up/limit-down (cannot be bought/sold)
    4. Moving average filter: only trade stocks above their 20-day moving average (trend confirmation)
    5. Equal weighting, at most 10 names, each capped at 10%
    6. 3% intraday drawdown stop loss
    7. Close all positions 5 minutes before the close (14:55)
    """

    name = "A股日内动量"
    description = "开盘30分钟动量选股 + 量价确认 + 日内止损"

    def __init__(
        self,
        max_positions: int = 10,          # 最大持仓数
        max_single_weight: float = 0.10,  # 单票最大权重
        momentum_top_n: int = 10,         # 选最强N只
        volume_multiplier: float = 1.5,   # 成交量倍数阈值
        volume_lookback: int = 5,         # 成交量均值回看天数
        ma_period: int = 20,              # 均线周期
        stop_loss_pct: float = 0.03,      # 日内止损比例
        limit_up_pct: float = 0.098,      # 涨停判定阈值（留2bp缓冲）
        limit_down_pct: float = -0.098,   # 跌停判定阈值
        entry_time: str = "10:00",        # 入场时间（开盘30分钟后）
        exit_time: str = "14:55",         # 平仓时间（收盘前5分钟）
    ):
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.momentum_top_n = momentum_top_n
        self.volume_multiplier = volume_multiplier
        self.volume_lookback = volume_lookback
        self.ma_period = ma_period
        self.stop_loss_pct = stop_loss_pct
        self.limit_up_pct = limit_up_pct
        self.limit_down_pct = limit_down_pct
        self.entry_time = pd.Timestamp(entry_time).time()
        self.exit_time = pd.Timestamp(exit_time).time()

        # 回测结果
        self.daily_returns = []
        self.trade_log = []

    # ─────────────────────────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────────────────────────

    def _calc_morning_momentum(self, minute_data: pd.DataFrame, date: str) -> pd.Series:
        """计算开盘到10:00的涨幅

        Parameters
        ----------
        minute_data : DataFrame
            分钟级行情，columns: [datetime, code, open, high, low, close, volume]
        date : str
            日期 'YYYY-MM-DD'

        Returns
        -------
        Series
            个股30分钟涨幅 (code -> float)
        """
        day_data = minute_data[minute_data['datetime'].dt.date == pd.Timestamp(date).date()]
        if day_data.empty:
            return pd.Series(dtype=float)

        # 9:30 的开盘价
        open_bar = day_data[day_data['datetime'].dt.time == dtime(9, 30)]
        open_prices = open_bar.set_index('code')['open']

        # 10:00 的收盘价（即前30分钟结束时的价格）
        bar_10 = day_data[day_data['datetime'].dt.time == self.entry_time]
        close_10 = bar_10.set_index('code')['close']

        # 计算涨幅
        common = open_prices.index.intersection(close_10.index)
        momentum = (close_10[common] - open_prices[common]) / open_prices[common]
        return momentum.dropna()

    def _calc_morning_volume(self, minute_data: pd.DataFrame, date: str) -> pd.Series:
        """计算当日前30分钟总成交量

        Parameters
        ----------
        minute_data : DataFrame
            分钟级行情
        date : str
            日期

        Returns
        -------
        Series
            个股前30分钟总成交量 (code -> float)
        """
        day_data = minute_data[minute_data['datetime'].dt.date == pd.Timestamp(date).date()]
        # 9:30 ~ 10:00 的数据
        morning = day_data[
            (day_data['datetime'].dt.time >= dtime(9, 30)) &
            (day_data['datetime'].dt.time <= self.entry_time)
        ]
        return morning.groupby('code')['volume'].sum()

    def _calc_avg_morning_volume(
        self, minute_data: pd.DataFrame, date: str, lookback: int = 5
    ) -> pd.Series:
        """计算过去N日平均前30分钟成交量

        Parameters
        ----------
        minute_data : DataFrame
            分钟级行情
        date : str
            日期
        lookback : int
            回看天数

        Returns
        -------
        Series
            个股平均前30分钟成交量 (code -> float)
        """
        all_dates = sorted(minute_data['datetime'].dt.date.unique())
        target = pd.Timestamp(date).date()
        past_dates = [d for d in all_dates if d < target][-lookback:]

        if not past_dates:
            return pd.Series(dtype=float)

        vols = []
        for d in past_dates:
            v = self._calc_morning_volume(minute_data, str(d))
            vols.append(v)

        vol_df = pd.DataFrame(vols)
        return vol_df.mean()

    def _is_limit_up_down(self, pct_change: float) -> bool:
        """判断是否涨停或跌停

        A股涨跌停规则:
        - 主板: ±10%
        - 创业板(300xxx)/科创板(688xxx): ±20%（简化版统一用10%）
        - ST股: ±5%（本策略不交易ST股）
        """
        return pct_change >= self.limit_up_pct or pct_change <= self.limit_down_pct

    def _calc_ma20(self, daily_data: pd.DataFrame, date: str) -> pd.Series:
        """计算20日均线

        Parameters
        ----------
        daily_data : DataFrame
            日线行情，columns: [date, code, close]
        date : str
            日期

        Returns
        -------
        Series
            个股20日均线值 (code -> float)
        """
        hist = daily_data[daily_data['date'] <= date]
        # 取最近 ma_period 个交易日
        recent_dates = sorted(hist['date'].unique())[-self.ma_period:]
        recent = hist[hist['date'].isin(recent_dates)]
        return recent.groupby('code')['close'].mean()

    # ─────────────────────────────────────────────────────────────
    #  信号生成 & 选股
    # ─────────────────────────────────────────────────────────────

    def select_stocks(
        self,
        minute_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        date: str,
    ) -> list[str]:
        """Single-day stock selection logic

        Parameters
        ----------
        minute_data : DataFrame
            Minute-level market data
        daily_data : DataFrame
            Daily market data (including history)
        date : str
            Trading date

        Returns
        -------
        list[str]
            Tickers of the selected stocks
        """
        # 1) 开盘30分钟动量
        momentum = self._calc_morning_momentum(minute_data, date)
        if momentum.empty:
            return []

        # 2) 成交量确认: 当日前30分钟量 > 5日均量 × 1.5
        today_vol = self._calc_morning_volume(minute_data, date)
        avg_vol = self._calc_avg_morning_volume(
            minute_data, date, self.volume_lookback
        )
        common_vol = today_vol.index.intersection(avg_vol.index)
        vol_confirm = today_vol[common_vol] > avg_vol[common_vol] * self.volume_multiplier
        vol_pass = vol_confirm[vol_confirm].index

        # 3) 涨跌停过滤
        not_limit = momentum[
            momentum.apply(lambda x: not self._is_limit_up_down(x))
        ].index

        # 4) 均线过滤: 当前价 > 20日均线
        ma20 = self._calc_ma20(daily_data, date)
        # 用开盘30分钟后的价格和MA20比较
        day_data = minute_data[minute_data['datetime'].dt.date == pd.Timestamp(date).date()]
        bar_10 = day_data[day_data['datetime'].dt.time == self.entry_time]
        current_price = bar_10.set_index('code')['close']
        common_ma = current_price.index.intersection(ma20.index)
        above_ma = current_price[common_ma][current_price[common_ma] > ma20[common_ma]].index

        # 交集: 所有条件都满足
        candidates = (
            momentum.index
            .intersection(vol_pass)
            .intersection(not_limit)
            .intersection(above_ma)
        )

        if len(candidates) == 0:
            return []

        # 按动量排序，选最强的 N 只
        selected = momentum[candidates].sort_values(ascending=False)
        selected = selected.head(self.momentum_top_n)

        # 只选正动量
        selected = selected[selected > 0]

        return selected.index.tolist()

    # ─────────────────────────────────────────────────────────────
    #  回测引擎
    # ─────────────────────────────────────────────────────────────

    def backtest_intraday(
        self,
        minute_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        """Simplified intraday backtest

        Parameters
        ----------
        minute_data : DataFrame
            Minute-level market data, columns: [datetime, code, open, high, low, close, volume]
            datetime is a pd.Timestamp carrying both date and time
        daily_data : DataFrame
            Daily market data, columns: [date, code, open, high, low, close, volume]
        start : str
            Backtest start date 'YYYY-MM-DD'
        end : str
            Backtest end date 'YYYY-MM-DD'

        Returns
        -------
        Series
            Series of daily returns
        """
        self.daily_returns = []
        self.trade_log = []

        # 获取回测区间的交易日
        all_dates = sorted(minute_data['datetime'].dt.date.unique())
        trade_dates = [
            d for d in all_dates
            if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date()
        ]

        print(f"回测区间: {start} ~ {end}, 共 {len(trade_dates)} 个交易日")

        for i, date in enumerate(trade_dates):
            date_str = str(date)
            day_ret = self._simulate_one_day(minute_data, daily_data, date_str)
            self.daily_returns.append({'date': date, 'return': day_ret})

            if (i + 1) % 50 == 0:
                print(f"  已回测 {i + 1}/{len(trade_dates)} 天...")

        ret_series = pd.Series(
            [r['return'] for r in self.daily_returns],
            index=[r['date'] for r in self.daily_returns],
            name='daily_return',
        )
        print(f"回测完成，总交易日 {len(ret_series)} 天")
        return ret_series

    def _simulate_one_day(
        self,
        minute_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        date: str,
    ) -> float:
        """模拟单日交易

        流程:
        1. 10:00 选股并买入
        2. 盘中逐分钟检查止损（回撤 > 3% 则平仓该票）
        3. 14:55 全部平仓

        Parameters
        ----------
        minute_data : DataFrame
            分钟级行情
        daily_data : DataFrame
            日线行情
        date : str
            交易日期

        Returns
        -------
        float
            当日组合收益率
        """
        # 选股
        stocks = self.select_stocks(minute_data, daily_data, date)
        if not stocks:
            return 0.0

        # 等权分配，单票不超过 max_single_weight
        n = min(len(stocks), self.max_positions)
        stocks = stocks[:n]
        weight = min(1.0 / n, self.max_single_weight)

        # 获取入场价（10:00 的收盘价）
        day_data = minute_data[minute_data['datetime'].dt.date == pd.Timestamp(date).date()]
        entry_bar = day_data[day_data['datetime'].dt.time == self.entry_time]
        entry_prices = entry_bar.set_index('code')['close']

        # 逐分钟模拟: 10:00 之后到 14:55
        afternoon_data = day_data[
            (day_data['datetime'].dt.time > self.entry_time) &
            (day_data['datetime'].dt.time <= self.exit_time)
        ]

        # 逐股票追踪收益
        holdings = {}  # code -> {'entry_price': float, 'stopped': bool}
        for code in stocks:
            if code in entry_prices.index:
                holdings[code] = {
                    'entry_price': entry_prices[code],
                    'stopped': False,
                    'exit_price': entry_prices[code],  # 默认值
                }

        if not holdings:
            return 0.0

        # 获取每分钟行情用于止损检查
        minutes = sorted(afternoon_data['datetime'].unique())
        for ts in minutes:
            bar = afternoon_data[afternoon_data['datetime'] == ts].set_index('code')
            for code, info in holdings.items():
                if info['stopped'] or code not in bar.index:
                    continue
                current = bar.loc[code, 'low']  # 用最低价检查止损更保守
                drawdown = (current - info['entry_price']) / info['entry_price']
                if drawdown <= -self.stop_loss_pct:
                    # 触发止损，用止损价作为退出价
                    info['stopped'] = True
                    info['exit_price'] = info['entry_price'] * (1 - self.stop_loss_pct)
                else:
                    # 更新退出价为当前收盘价（平仓时使用最后的价格）
                    info['exit_price'] = bar.loc[code, 'close']

        # 计算组合收益
        total_return = 0.0
        for code, info in holdings.items():
            stock_ret = (info['exit_price'] - info['entry_price']) / info['entry_price']
            total_return += stock_ret * weight
            self.trade_log.append({
                'date': date,
                'code': code,
                'entry_price': info['entry_price'],
                'exit_price': info['exit_price'],
                'return': stock_ret,
                'stopped': info['stopped'],
            })

        return total_return

    # ─────────────────────────────────────────────────────────────
    #  回测报告
    # ─────────────────────────────────────────────────────────────

    def generate_report(self) -> dict:
        """Print backtest statistics

        Returns
        -------
        dict
            Metrics keyed by the literal Chinese strings used in the code:
            '回测天数' (backtest days), '有交易天数' (days with trades),
            '胜率' (win rate), '日均收益' (average daily return),
            '年化收益' (annualized return), '年化波动率' (annualized vol),
            '最大回撤' (max drawdown), 'Sharpe比率' (Sharpe ratio),
            '累计收益' (cumulative return), '总交易笔数' (total trades).
        """
        if not self.daily_returns:
            print("尚未运行回测，请先调用 backtest_intraday()")
            return {}

        rets = pd.Series(
            [r['return'] for r in self.daily_returns],
            index=[r['date'] for r in self.daily_returns],
        )

        # 基础统计
        total_days = len(rets)
        trading_days = (rets != 0).sum()  # 有持仓的天数
        win_days = (rets > 0).sum()
        loss_days = (rets < 0).sum()

        # 胜率（仅计算有交易的天数）
        win_rate = win_days / trading_days if trading_days > 0 else 0

        # 日均收益
        avg_daily_ret = rets.mean()

        # 累计收益
        cum_ret = (1 + rets).cumprod()
        total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0

        # 最大回撤
        peak = cum_ret.expanding().max()
        drawdown = (cum_ret - peak) / peak
        max_drawdown = drawdown.min()

        # 年化收益（按242个交易日）
        annual_ret = (1 + avg_daily_ret) ** 242 - 1

        # 年化波动率
        annual_vol = rets.std() * np.sqrt(242)

        # Sharpe比率（无风险利率按年化2.5%，即日化约0.01%）
        rf_daily = 0.025 / 242
        sharpe = (avg_daily_ret - rf_daily) / rets.std() if rets.std() > 0 else 0

        # 止损统计
        trades = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        stop_count = trades['stopped'].sum() if not trades.empty else 0
        total_trades = len(trades) if not trades.empty else 0

        # 盈亏比
        if not trades.empty:
            wins = trades[trades['return'] > 0]['return']
            losses = trades[trades['return'] < 0]['return']
            profit_factor = (
                wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
            )
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
        else:
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        report = {
            '回测天数': total_days,
            '有交易天数': int(trading_days),
            '胜率': f"{win_rate:.2%}",
            '日均收益': f"{avg_daily_ret:.4%}",
            '年化收益': f"{annual_ret:.2%}",
            '年化波动率': f"{annual_vol:.2%}",
            '最大回撤': f"{max_drawdown:.2%}",
            'Sharpe比率': f"{sharpe:.2f}",
            '累计收益': f"{total_ret:.2%}",
            '总交易笔数': total_trades,
            '止损次数': int(stop_count),
            '止损比例': f"{stop_count / total_trades:.2%}" if total_trades > 0 else "N/A",
            '盈亏比': f"{profit_factor:.2f}",
            '平均盈利': f"{avg_win:.4%}",
            '平均亏损': f"{avg_loss:.4%}",
        }

        print("\n" + "=" * 50)
        print("  A股日内动量策略 — 回测报告")
        print("=" * 50)
        for k, v in report.items():
            print(f"  {k:　<10s}: {v}")
        print("=" * 50)

        return report


# ═══════════════════════════════════════════════════════════════════
#  模拟数据生成（用于演示）
# ═══════════════════════════════════════════════════════════════════

def _generate_mock_minute_data(
    codes: list[str],
    dates: list[str],
    freq_minutes: int = 1,
) -> pd.DataFrame:
    """生成模拟分钟级行情数据

    Parameters
    ----------
    codes : list[str]
        股票代码列表
    dates : list[str]
        交易日列表
    freq_minutes : int
        分钟频率（1=逐分钟）

    Returns
    -------
    DataFrame
        columns: [datetime, code, open, high, low, close, volume]
    """
    np.random.seed(42)
    rows = []

    # A股交易时间: 9:30-11:30, 13:00-15:00
    morning_times = pd.date_range("09:30", "11:30", freq=f"{freq_minutes}min").time
    afternoon_times = pd.date_range("13:00", "15:00", freq=f"{freq_minutes}min").time
    all_times = list(morning_times) + list(afternoon_times)

    for date_str in dates:
        date = pd.Timestamp(date_str)
        for code in codes:
            # 随机基础价格 10~100
            base_price = np.random.uniform(10, 100)
            # 日内随机游走，带轻微动量
            drift = np.random.normal(0.0002, 0.001)  # 微小日内趋势
            price = base_price

            for t in all_times:
                dt = pd.Timestamp.combine(date.date(), t)
                ret = np.random.normal(drift, 0.003)  # 分钟级波动
                price *= (1 + ret)
                high = price * (1 + abs(np.random.normal(0, 0.001)))
                low = price * (1 - abs(np.random.normal(0, 0.001)))
                vol = int(np.random.exponential(50000))

                rows.append({
                    'datetime': dt,
                    'code': code,
                    'open': round(price * (1 + np.random.normal(0, 0.0005)), 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(price, 2),
                    'volume': vol,
                })

    return pd.DataFrame(rows)


def _generate_mock_daily_data(
    codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """生成模拟日线数据

    Parameters
    ----------
    codes : list[str]
        股票代码列表
    start, end : str
        日期范围

    Returns
    -------
    DataFrame
        columns: [date, code, open, high, low, close, volume]
    """
    np.random.seed(123)
    dates = pd.bdate_range(start, end)  # 工作日
    rows = []

    for code in codes:
        price = np.random.uniform(10, 100)
        for d in dates:
            ret = np.random.normal(0.0005, 0.02)
            price *= (1 + ret)
            rows.append({
                'date': d.strftime('%Y-%m-%d'),
                'code': code,
                'open': round(price * 0.999, 2),
                'high': round(price * 1.015, 2),
                'low': round(price * 0.985, 2),
                'close': round(price, 2),
                'volume': int(np.random.exponential(1_000_000)),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  主入口 — 模拟数据演示
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  A股日内动量策略 — 模拟数据演示")
    print("=" * 60)

    # 模拟20只股票、10个交易日
    codes = [f"{600000 + i}" for i in range(20)]
    trade_dates = pd.bdate_range("2025-03-10", "2025-03-21").strftime("%Y-%m-%d").tolist()

    print(f"\n生成模拟数据: {len(codes)} 只股票, {len(trade_dates)} 个交易日...")

    # 生成日线数据（需要更早的历史用于计算MA20）
    daily_data = _generate_mock_daily_data(codes, "2025-01-01", "2025-03-21")
    print(f"  日线数据: {len(daily_data)} 条")

    # 生成分钟数据
    minute_data = _generate_mock_minute_data(codes, trade_dates, freq_minutes=5)
    print(f"  分钟数据: {len(minute_data)} 条")

    # 初始化策略
    strategy = ChinaIntradayMomentum(
        max_positions=5,
        momentum_top_n=5,
        stop_loss_pct=0.03,
    )

    # 运行回测
    print("\n开始回测...")
    returns = strategy.backtest_intraday(
        minute_data=minute_data,
        daily_data=daily_data,
        start="2025-03-10",
        end="2025-03-21",
    )

    # 输出报告
    report = strategy.generate_report()

    # 打印交易日志摘要
    if strategy.trade_log:
        trades_df = pd.DataFrame(strategy.trade_log)
        print(f"\n交易日志摘要 (共 {len(trades_df)} 笔):")
        print(trades_df.groupby('date').agg(
            持仓数=('code', 'count'),
            平均收益=('return', 'mean'),
            止损数=('stopped', 'sum'),
        ).to_string())
