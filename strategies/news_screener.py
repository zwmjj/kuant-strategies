"""News + screener composite strategies — a strategy family combining Alpaca news sentiment and screener data

Strategies included:
    1. NewsMomentumStrategy      — news sentiment trend following
    2. NewsContrarianStrategy     — news overreaction contrarian
    3. AttentionMomentumStrategy  — attention momentum
    4. MarketMoversStrategy       — market mover continuation
    5. CompositeAlpacaStrategy    — Alpaca full-data flagship
"""

import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy
from qf.signals_news import NewsSentimentGenerator
from qf.signals_screener import ScreenerSignalGenerator, _cross_sectional_rank


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _momentum(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """计算滚动动量 (收益率), 返回截面排名 [-1, 1]"""
    raw = prices.pct_change(window)
    ranked = raw.rank(axis=1, pct=True) * 2 - 1
    return ranked


def _safe_align(*frames: pd.DataFrame) -> list:
    """对齐多个DataFrame的索引和列"""
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)
    return [f.loc[idx, cols] for f in frames]


def run_signal_backtest(
    signal_series: pd.DataFrame,
    returns_df: pd.DataFrame,
    long_n: int = 20,
    short_n: int = 20,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Quick signal backtest — go long/short on the signal and compute returns

    Parameters
    ----------
    signal_series : pd.DataFrame
        Signal matrix (date x symbol); higher values are more bullish
    returns_df : pd.DataFrame
        Daily return matrix (date x symbol), aligned with the signal
    long_n : int
        Number of stocks held long, default 20
    short_n : int
        Number of stocks held short, default 20
    cost_bps : float
        One-way trading cost in basis points, default 5bps

    Returns
    -------
    pd.DataFrame
        Daily backtest results with columns long_ret, short_ret, ls_ret, cum_ret, turnover
    """
    signal_series, returns_df = _safe_align(signal_series, returns_df)

    # 信号滞后一期（t日信号 => t+1日持仓）
    signal_lag = signal_series.shift(1)
    signal_lag = signal_lag.iloc[1:]
    returns_aligned = returns_df.iloc[1:]

    results = []
    prev_long = set()
    prev_short = set()

    for date in signal_lag.index:
        sig_row = signal_lag.loc[date].dropna()
        ret_row = returns_aligned.loc[date]

        if len(sig_row) < long_n + short_n:
            continue

        # 选股
        sorted_syms = sig_row.sort_values(ascending=False)
        long_syms = set(sorted_syms.index[:long_n])
        short_syms = set(sorted_syms.index[-short_n:])

        # 等权收益
        long_ret = ret_row.reindex(long_syms).mean() if long_syms else 0.0
        short_ret = ret_row.reindex(short_syms).mean() if short_syms else 0.0
        ls_ret = long_ret - short_ret

        # 换手率
        long_to = len(long_syms - prev_long) / max(long_n, 1)
        short_to = len(short_syms - prev_short) / max(short_n, 1)
        turnover = (long_to + short_to) / 2

        # 扣除交易成本
        cost = turnover * cost_bps / 10000 * 2  # 双边
        ls_ret_net = ls_ret - cost

        results.append({
            'date': date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': ls_ret,
            'ls_ret_net': ls_ret_net,
            'turnover': turnover,
        })

        prev_long = long_syms
        prev_short = short_syms

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index('date')
    df['cum_ret'] = (1 + df['ls_ret_net']).cumprod() - 1

    # 汇总统计
    n_days = len(df)
    ann = 252
    mean_ret = df['ls_ret_net'].mean()
    std_ret = df['ls_ret_net'].std()
    sharpe = mean_ret / std_ret * np.sqrt(ann) if std_ret > 0 else 0.0
    max_dd = (df['cum_ret'] - df['cum_ret'].cummax()).min()

    df.attrs['summary'] = {
        'n_days': n_days,
        'ann_return': mean_ret * ann,
        'ann_vol': std_ret * np.sqrt(ann),
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_turnover': df['turnover'].mean(),
    }

    return df


# ═══════════════════════════════════════════════════════════════════════
# 策略 1: 新闻情绪顺势策略
# ═══════════════════════════════════════════════════════════════════════

class NewsMomentumStrategy(BaseStrategy):
    """News sentiment + price momentum trend-following strategy

    Core logic: positive news sentiment + positive price momentum = strong buy signal.
    News count acts as an amplifier — the more attention, the stronger the signal.

    Signal formula:
        signal = 0.5 * news_sentiment + 0.3 * momentum_5d + 0.2 * news_volume

    Best suited to: the start of a trend, where a news catalyst and price confirmation reinforce each other.
    """
    name = "新闻顺势动量"
    description = "正面情绪+价格动量共振，新闻量放大信号"

    def __init__(
        self,
        w_sentiment: float = 0.5,
        w_momentum: float = 0.3,
        w_volume: float = 0.2,
        mom_window: int = 5,
    ):
        self.w_sentiment = w_sentiment
        self.w_momentum = w_momentum
        self.w_volume = w_volume
        self.mom_window = mom_window
        self._news_gen = NewsSentimentGenerator()

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the news trend-following momentum signal

        Parameters
        ----------
        data : dict
            Must contain:
            - 'prices': pd.DataFrame (date x symbol), closing prices
            - 'news_by_symbol': dict[str, list], daily news data
              or 'news_by_date': dict[date, dict[str, list]], historical daily news

        Returns
        -------
        pd.DataFrame : (date x symbol) signal matrix
        """
        prices = data['prices']
        mom = _momentum(prices, self.mom_window)

        # ── 构建情绪和新闻量截面 ──
        if 'news_by_date' in data:
            # 历史模式: 每日有独立的新闻数据
            sentiment_frames = {}
            volume_frames = {}
            for date, news_by_sym in data['news_by_date'].items():
                sent = self._news_gen.build_sentiment_signal(news_by_sym)
                vol = self._news_gen.news_volume_signal(news_by_sym)
                sentiment_frames[date] = sent
                volume_frames[date] = vol

            sentiment_df = pd.DataFrame(sentiment_frames).T
            news_vol_df = pd.DataFrame(volume_frames).T
        elif 'news_by_symbol' in data:
            # 实时模式: 只有当天的新闻
            sent = self._news_gen.build_sentiment_signal(data['news_by_symbol'])
            vol = self._news_gen.news_volume_signal(data['news_by_symbol'])
            # 扩展为单行DataFrame
            last_date = prices.index[-1]
            sentiment_df = pd.DataFrame({last_date: sent}).T
            news_vol_df = pd.DataFrame({last_date: vol}).T
        else:
            # 无新闻数据时退化为纯动量
            return mom

        # ── 对齐并合成 ──
        aligned = _safe_align(
            mom.reindex(sentiment_df.index).reindex(columns=sentiment_df.columns, fill_value=0),
            sentiment_df,
            news_vol_df,
        )
        mom_a, sent_a, vol_a = aligned

        signal = (
            self.w_sentiment * sent_a.fillna(0)
            + self.w_momentum * mom_a.fillna(0)
            + self.w_volume * vol_a.fillna(0)
        )

        # 截面排名归一化
        signal = signal.rank(axis=1, pct=True) * 2 - 1
        return signal


# ═══════════════════════════════════════════════════════════════════════
# 策略 2: 新闻过度反应逆势策略
# ═══════════════════════════════════════════════════════════════════════

class NewsContrarianStrategy(BaseStrategy):
    """News overreaction contrarian strategy

    Core logic: a fundamentally sound stock hit by extremely negative news = market overreaction = buying opportunity.
    Filter: only trigger a buy when the stock's 20-day momentum is positive (fundamentals still look intact).

    Signal formula:
        signal = -sentiment * (1 if prior_momentum_20d > 0 else 0)

    Best suited to: catching the bounce after panic selling. Works best on blue chips that were sold off unfairly.
    """
    name = "新闻逆势（过度反应）"
    description = "极端负面情绪+基本面良好=超卖反弹"

    long_pct: float = 1.0
    short_pct: float = 0.0
    short_n: int = 1  # 纯多头策略

    def __init__(self, prior_mom_window: int = 20):
        self.prior_mom_window = prior_mom_window
        self._news_gen = NewsSentimentGenerator()

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the news contrarian signal

        Parameters
        ----------
        data : dict
            Must contain:
            - 'prices': pd.DataFrame (date x symbol)
            - 'news_by_symbol' or 'news_by_date'

        Returns
        -------
        pd.DataFrame : (date x symbol) signal matrix; only stocks meeting the filter get a positive signal
        """
        prices = data['prices']

        # 20日动量（原始收益率，用于判断方向）
        prior_mom_raw = prices.pct_change(self.prior_mom_window)
        prior_mom_positive = (prior_mom_raw > 0).astype(float)

        # ── 构建情绪截面 ──
        if 'news_by_date' in data:
            sentiment_frames = {}
            for date, news_by_sym in data['news_by_date'].items():
                sent = self._news_gen.build_sentiment_signal(news_by_sym)
                sentiment_frames[date] = sent
            sentiment_df = pd.DataFrame(sentiment_frames).T
        elif 'news_by_symbol' in data:
            sent = self._news_gen.build_sentiment_signal(data['news_by_symbol'])
            last_date = prices.index[-1]
            sentiment_df = pd.DataFrame({last_date: sent}).T
        else:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # 对齐
        common_idx = sentiment_df.index.intersection(prior_mom_positive.index)
        common_cols = sentiment_df.columns.intersection(prior_mom_positive.columns)

        if common_idx.empty or common_cols.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        sent_a = sentiment_df.loc[common_idx, common_cols]
        mom_filter = prior_mom_positive.loc[common_idx, common_cols]

        # 信号: 情绪越负 => 信号越正（取反），但只在prior_mom > 0时激活
        signal = -sent_a * mom_filter

        # 截面排名
        signal = signal.rank(axis=1, pct=True) * 2 - 1
        return signal.reindex(index=prices.index, columns=prices.columns, fill_value=0)


# ═══════════════════════════════════════════════════════════════════════
# 策略 3: 关注度动量策略
# ═══════════════════════════════════════════════════════════════════════

class AttentionMomentumStrategy(BaseStrategy):
    """Attention + momentum dual-confirmation strategy

    Core logic: stocks on the Alpaca most_actives list exhibit short-term momentum (attention-driven returns).
    Confirmed against 5-day price momentum to filter out pure attention noise.

    Signal formula:
        signal = 0.6 * attention + 0.4 * momentum_5d

    Best suited to: short-term trend following on hot stocks. Fits 1-5 day holding periods.
    """
    name = "关注度动量"
    description = "screener关注度+价格动量双确认"

    def __init__(
        self,
        w_attention: float = 0.6,
        w_momentum: float = 0.4,
        mom_window: int = 5,
    ):
        self.w_attention = w_attention
        self.w_momentum = w_momentum
        self.mom_window = mom_window

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the attention momentum signal

        Parameters
        ----------
        data : dict
            Must contain:
            - 'prices': pd.DataFrame (date x symbol)
            - 'most_actives': list[dict] (current day)
              or 'most_actives_by_date': dict[date, list[dict]] (history)

        Returns
        -------
        pd.DataFrame : (date x symbol) signal matrix
        """
        prices = data['prices']
        universe = list(prices.columns)
        mom = _momentum(prices, self.mom_window)

        if 'most_actives_by_date' in data:
            # 历史模式
            attention_frames = {}
            for date, actives in data['most_actives_by_date'].items():
                att = ScreenerSignalGenerator.attention_signal(actives, universe)
                attention_frames[date] = att
            attention_df = pd.DataFrame(attention_frames).T
        elif 'most_actives' in data:
            # 实时模式
            att = ScreenerSignalGenerator.attention_signal(
                data['most_actives'], universe
            )
            last_date = prices.index[-1]
            attention_df = pd.DataFrame({last_date: att}).T
        else:
            return mom

        # 对齐
        common_idx = attention_df.index.intersection(mom.index)
        common_cols = attention_df.columns.intersection(mom.columns)

        if common_idx.empty:
            return mom

        att_a = attention_df.loc[common_idx, common_cols].fillna(0)
        mom_a = mom.loc[common_idx, common_cols].fillna(0)

        signal = self.w_attention * att_a + self.w_momentum * mom_a

        signal = signal.rank(axis=1, pct=True) * 2 - 1
        return signal.reindex(index=prices.index, columns=prices.columns, fill_value=0)


# ═══════════════════════════════════════════════════════════════════════
# 策略 4: 市场异动延续策略
# ═══════════════════════════════════════════════════════════════════════

class MarketMoversStrategy(BaseStrategy):
    """Market mover continuation strategy

    Core logic:
    - Today's top gainers + positive news = continued gains tomorrow (trend continuation)
    - Today's top losers + no negative news = oversold bounce tomorrow (mispricing correction)

    Two sub-signals, equally weighted:
    - gainer_continuation = gainer_rank * max(sentiment, 0)
    - loser_bounce = loser_rank * max(-sentiment + neutral_bonus, 0)

    Best suited to: daily trading, capturing next-day continuation/reversal after a large move.
    """
    name = "市场异动延续"
    description = "涨幅榜+正面新闻延续，跌幅榜+无利空反弹"

    def __init__(self, sentiment_threshold: float = -0.2):
        self.sentiment_threshold = sentiment_threshold
        self._news_gen = NewsSentimentGenerator()

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the market mover continuation signal

        Parameters
        ----------
        data : dict
            Must contain:
            - 'prices': pd.DataFrame (date x symbol)
            - 'gainers': list[dict] (current day's top gainers)
            - 'losers': list[dict] (current day's top losers)
            - 'news_by_symbol': dict[str, list] or 'news_by_date'

        Returns
        -------
        pd.DataFrame : (date x symbol) signal matrix
        """
        prices = data['prices']
        universe = list(prices.columns)

        # ── 涨跌幅信号 ──
        gainers = data.get('gainers', [])
        losers = data.get('losers', [])

        gainer_signal = pd.Series(0.0, index=universe)
        for item in (gainers or []):
            sym = item.get('symbol', '')
            pct = item.get('percent_change', 0.0)
            if sym in gainer_signal.index:
                gainer_signal[sym] = abs(pct)

        loser_signal = pd.Series(0.0, index=universe)
        for item in (losers or []):
            sym = item.get('symbol', '')
            pct = item.get('percent_change', 0.0)
            if sym in loser_signal.index:
                loser_signal[sym] = abs(pct)  # 取绝对值，跌幅越大信号越强

        # ── 情绪信号 ──
        if 'news_by_symbol' in data:
            news_by_sym = data['news_by_symbol']
        elif 'news_by_date' in data:
            # 取最新日期的新闻
            dates = sorted(data['news_by_date'].keys())
            news_by_sym = data['news_by_date'][dates[-1]] if dates else {}
        else:
            news_by_sym = {}

        sentiment = {}
        for sym in universe:
            if sym in news_by_sym and news_by_sym[sym]:
                sentiment[sym] = self._news_gen.aggregate_sentiment(
                    news_by_sym[sym], sym
                )
            else:
                sentiment[sym] = 0.0
        sentiment_s = pd.Series(sentiment, dtype=float)

        # ── 子信号1: 涨幅榜 + 正面新闻 = 延续 ──
        # 正面情绪放大gainer信号
        positive_sent = sentiment_s.clip(lower=0)
        continuation = gainer_signal * (0.5 + 0.5 * positive_sent)

        # ── 子信号2: 跌幅榜 + 非负面新闻 = 反弹 ──
        # 情绪不是很负面时（无利空）才给反弹信号
        no_bad_news = (sentiment_s > self.sentiment_threshold).astype(float)
        bounce = loser_signal * no_bad_news

        # ── 合成 ──
        combined = continuation + bounce

        # 构造单行DataFrame
        last_date = prices.index[-1]
        signal_df = pd.DataFrame(
            {last_date: combined},
        ).T
        signal_df = signal_df.reindex(columns=universe, fill_value=0)

        # 截面排名
        signal_df = signal_df.rank(axis=1, pct=True) * 2 - 1
        return signal_df.reindex(
            index=prices.index, columns=prices.columns, fill_value=0
        )


# ═══════════════════════════════════════════════════════════════════════
# 策略 5: Alpaca全数据旗舰策略
# ═══════════════════════════════════════════════════════════════════════

class CompositeAlpacaStrategy(BaseStrategy):
    """Alpaca full-data flagship strategy — the "kitchen sink" strategy combining every Alpaca data source

    Weight allocation:
        20% news sentiment  — NewsSentimentGenerator
        20% VWAP momentum   — cross-sectional rank of (price - vwap) / vwap
        15% volume surge    — cross-sectional rank of volume / avg_volume_20d
        15% IV rank         — implied volatility percentile (short high IV, long low IV)
        15% attention       — screener most_actives attention signal
        15% dividend yield  — long high yield, cross-sectional rank

    This is the flagship strategy designed for Alpaca paper trading, drawing on every data
    dimension Alpaca provides. Turnover control is fairly tight (turnover_penalty=0.30),
    suiting medium-frequency holding periods (5-20 days).
    """
    name = "Alpaca旗舰复合"
    description = "六维因子复合: 情绪+VWAP+量+IV+关注度+股息"

    turnover_penalty: float = 0.30

    # 默认权重
    DEFAULT_WEIGHTS = {
        'sentiment': 0.20,
        'vwap_mom': 0.20,
        'volume_surge': 0.15,
        'iv_rank': 0.15,
        'attention': 0.15,
        'div_yield': 0.15,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._news_gen = NewsSentimentGenerator()

    def _build_sentiment(self, data: dict, universe: list) -> pd.Series:
        """构建情绪信号"""
        if 'news_by_symbol' in data:
            return self._news_gen.build_sentiment_signal(data['news_by_symbol'])
        return pd.Series(0.0, index=universe)

    @staticmethod
    def _build_vwap_momentum(data: dict, universe: list) -> pd.Series:
        """构建VWAP动量信号: (close - vwap) / vwap

        价格在VWAP之上 = 买方主导，正信号
        """
        if 'vwap' not in data or 'prices' not in data:
            return pd.Series(0.0, index=universe)

        prices = data['prices']
        vwap = data['vwap']

        if isinstance(prices, pd.DataFrame):
            close = prices.iloc[-1]
        else:
            close = prices

        if isinstance(vwap, pd.DataFrame):
            vwap_last = vwap.iloc[-1]
        else:
            vwap_last = vwap

        common = close.index.intersection(vwap_last.index)
        if common.empty:
            return pd.Series(0.0, index=universe)

        raw = (close[common] - vwap_last[common]) / vwap_last[common].replace(0, np.nan)
        return _cross_sectional_rank(raw).reindex(universe, fill_value=0)

    @staticmethod
    def _build_volume_surge(data: dict, universe: list) -> pd.Series:
        """构建成交量飙升信号: volume_today / avg_volume_20d"""
        if 'volume' not in data:
            return pd.Series(0.0, index=universe)

        volume = data['volume']
        if isinstance(volume, pd.DataFrame) and len(volume) >= 20:
            avg_vol = volume.iloc[-20:].mean()
            today_vol = volume.iloc[-1]
            raw = today_vol / avg_vol.replace(0, np.nan)
            return _cross_sectional_rank(raw).reindex(universe, fill_value=0)

        return pd.Series(0.0, index=universe)

    @staticmethod
    def _build_iv_rank(data: dict, universe: list) -> pd.Series:
        """构建IV排名信号: 低IV多，高IV空（波动率风险溢价）"""
        if 'iv_rank' not in data:
            return pd.Series(0.0, index=universe)

        iv = data['iv_rank']
        if isinstance(iv, pd.Series):
            # 取反: 低IV = 正信号（便宜的期权保护，安全边际高）
            return _cross_sectional_rank(-iv).reindex(universe, fill_value=0)
        return pd.Series(0.0, index=universe)

    @staticmethod
    def _build_attention(data: dict, universe: list) -> pd.Series:
        """构建关注度信号"""
        if 'most_actives' not in data:
            return pd.Series(0.0, index=universe)
        return ScreenerSignalGenerator.attention_signal(
            data['most_actives'], universe
        )

    @staticmethod
    def _build_div_yield(data: dict, universe: list) -> pd.Series:
        """构建股息率信号: 高股息多"""
        if 'div_yield' not in data:
            return pd.Series(0.0, index=universe)

        dy = data['div_yield']
        if isinstance(dy, pd.Series):
            return _cross_sectional_rank(dy).reindex(universe, fill_value=0)
        return pd.Series(0.0, index=universe)

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the Alpaca flagship composite signal

        Parameters
        ----------
        data : dict
            May contain the following keys (missing dimensions are given zero weight):
            - 'prices': pd.DataFrame (date x symbol), required
            - 'volume': pd.DataFrame (date x symbol), trading volume
            - 'vwap': pd.DataFrame (date x symbol), VWAP
            - 'news_by_symbol': dict[str, list], news
            - 'most_actives': list[dict], screener data
            - 'iv_rank': pd.Series (symbol -> iv_percentile)
            - 'div_yield': pd.Series (symbol -> dividend_yield)

        Returns
        -------
        pd.DataFrame : (date x symbol) signal matrix
        """
        prices = data['prices']
        universe = list(prices.columns)
        w = self.weights

        # 构建各维度信号
        sub_signals = {
            'sentiment': self._build_sentiment(data, universe),
            'vwap_mom': self._build_vwap_momentum(data, universe),
            'volume_surge': self._build_volume_surge(data, universe),
            'iv_rank': self._build_iv_rank(data, universe),
            'attention': self._build_attention(data, universe),
            'div_yield': self._build_div_yield(data, universe),
        }

        # 检测哪些维度有有效数据（非全零）
        active_weights = {}
        for key, sig in sub_signals.items():
            if sig.abs().sum() > 0:
                active_weights[key] = w.get(key, 0.0)

        # 重新归一化权重（缺失维度的权重按比例分配给有效维度）
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            # 所有维度都无数据，退化为纯动量
            mom = _momentum(prices, 5)
            return mom

        # 加权合成
        composite = pd.Series(0.0, index=universe)
        for key, weight in active_weights.items():
            sig = sub_signals[key].reindex(universe, fill_value=0)
            composite += weight * sig

        # 截面排名
        composite = _cross_sectional_rank(composite)

        # 构造单行DataFrame（实时模式）
        last_date = prices.index[-1]
        signal_df = pd.DataFrame({last_date: composite}).T
        signal_df = signal_df.reindex(columns=universe, fill_value=0)

        return signal_df.reindex(
            index=prices.index, columns=prices.columns, fill_value=0
        )
