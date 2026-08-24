"""Statistical arbitrage suite — 10 complete stat-arb strategies + vectorized backtester

Contents:
1. PairsTrading           — Engle-Granger cointegration pairs trading
2. SectorNeutralMomentum  — sector-neutral momentum (pure stock-picking alpha)
3. ResidualReversion      — Fama-French residual mean reversion
4. ETFArbitrage           — ETF vs. constituent-basket arbitrage
5. RelativeValueVolatility— intra-sector volatility relative value
6. CointegrationPortfolio — multi-asset cointegration portfolio (PCA method)
7. MeanReversionBasket    — cross-sectional mean-reversion basket
8. LeadLagArbitrage       — lead-lag arbitrage (Lo & MacKinlay 1990)
9. DispersionTrading      — adaptive dispersion trading
10. OrderFlowImbalance    — order flow imbalance (volume-price proxy)

Every strategy implements the generate_signal(data) interface, where data = {ticker: DataFrame(OHLCV)}.
At the end, run_all_stat_arb_backtests(start, end) downloads the data and runs every backtest in one call.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════
#  常量 & 工具函数
# ═══════════════════════════════════════════════════════════════════════

# 50只美国大盘股（涵盖11个行业）
UNIVERSE_50 = [
    # 科技
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC",
    # 金融
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
    # 医疗
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO",
    # 消费
    "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT",
    # 工业
    "CAT", "BA", "HON", "UPS", "GE",
    # 能源
    "XOM", "CVX", "COP",
    # 公用事业
    "NEE", "DUK",
    # 材料
    "LIN", "APD",
    # 通信
    "DIS", "NFLX", "CMCSA",
    # 地产
    "PLD", "AMT",
    # 必需消费
    "PG", "KO", "WMT",
]

# 行业ETF映射
SECTOR_ETFS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC"],
    "XLF": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO"],
    "XLY": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT"],
    "XLI": ["CAT", "BA", "HON", "UPS", "GE"],
    "XLE": ["XOM", "CVX", "COP"],
    "XLU": ["NEE", "DUK"],
    "XLB": ["LIN", "APD"],
    "XLC": ["DIS", "NFLX", "CMCSA"],
    "XLRE": ["PLD", "AMT"],
    "XLP": ["PG", "KO", "WMT"],
}

# 反向映射：股票 -> 行业ETF
STOCK_TO_SECTOR: dict[str, str] = {}
for _etf, _members in SECTOR_ETFS.items():
    for _s in _members:
        STOCK_TO_SECTOR[_s] = _etf

# 宏观ETF
MACRO_ETFS = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "VXX"]

ALL_TICKERS = list(set(UNIVERSE_50 + list(SECTOR_ETFS.keys()) + MACRO_ETFS))


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """滚动z-score"""
    m = series.rolling(window, min_periods=max(20, window // 2)).mean()
    s = series.rolling(window, min_periods=max(20, window // 2)).std()
    return (series - m) / s.replace(0, np.nan)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """滚动OLS beta = cov(y,x)/var(x)"""
    cov = y.rolling(window, min_periods=30).cov(x)
    var = x.rolling(window, min_periods=30).var()
    return cov / var.replace(0, np.nan)


def _rolling_corr(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """滚动相关系数"""
    return y.rolling(window, min_periods=30).corr(x)


def _build_close_df(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从 data dict 构建收盘价 DataFrame (date x ticker)"""
    frames = {}
    for ticker, df in data.items():
        if "Close" in df.columns:
            frames[ticker] = df["Close"]
        elif "Adj Close" in df.columns:
            frames[ticker] = df["Adj Close"]
    return pd.DataFrame(frames).sort_index()


def _build_ohlcv_field(data: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """从 data dict 构建任意字段 DataFrame (date x ticker)"""
    frames = {}
    for ticker, df in data.items():
        if field in df.columns:
            frames[ticker] = df[field]
    return pd.DataFrame(frames).sort_index()


def _simple_backtest(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    long_top: int = 5,
    short_bottom: int = 5,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """简易截面多空回测

    Parameters
    ----------
    signal : DataFrame (date x ticker)，值越大越看多
    returns : DataFrame (date x ticker)，日收益率
    long_top : 做多排名前N
    short_bottom : 做空排名后N
    cost_bps : 单边交易成本(基点)

    Returns
    -------
    DataFrame 含 long_ret, short_ret, ls_ret, cum_ret 列
    """
    sig = signal.dropna(how="all")
    ret = returns.reindex(sig.index, columns=sig.columns)

    # 截面排名
    ranks = sig.rank(axis=1, ascending=True, pct=True)

    n_stocks = sig.count(axis=1)
    long_thresh = 1.0 - long_top / n_stocks
    short_thresh = short_bottom / n_stocks

    # 对齐 Series 与 DataFrame 比较 (axis=0 广播)
    long_mask = ranks.ge(long_thresh, axis=0)
    short_mask = ranks.le(short_thresh, axis=0)

    long_w = long_mask.div(long_mask.sum(axis=1), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1), axis=0).fillna(0)

    # 日度收益
    long_ret = (long_w.shift(1) * ret).sum(axis=1)
    short_ret = (short_w.shift(1) * ret).sum(axis=1)
    ls_ret = long_ret - short_ret

    # 换手成本
    turnover = long_w.diff().abs().sum(axis=1) + short_w.diff().abs().sum(axis=1)
    cost = turnover * cost_bps / 10000

    ls_ret_net = ls_ret - cost

    result = pd.DataFrame({
        "long_ret": long_ret,
        "short_ret": short_ret,
        "ls_ret": ls_ret,
        "ls_ret_net": ls_ret_net,
        "cum_ret": (1 + ls_ret_net).cumprod(),
        "turnover": turnover,
    })
    return result


def _pairs_backtest(
    spread_z: pd.Series,
    ret_a: pd.Series,
    ret_b: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """配对交易回测：做多价差回归

    当 z > entry_z: 做空A做多B (价差偏高，期望回落)
    当 z < -entry_z: 做多A做空B
    当 |z| < exit_z 或 |z| > stop_z: 平仓
    """
    position = pd.Series(0.0, index=spread_z.index)
    pos = 0.0

    for i in range(1, len(spread_z)):
        z = spread_z.iloc[i]
        if np.isnan(z):
            pos = 0.0
        elif abs(z) > stop_z:
            pos = 0.0          # 止损
        elif abs(z) < exit_z:
            pos = 0.0          # 止盈
        elif z > entry_z and pos == 0:
            pos = -1.0         # 做空价差
        elif z < -entry_z and pos == 0:
            pos = 1.0          # 做多价差
        position.iloc[i] = pos

    spread_ret = ret_a - ret_b
    strat_ret = position.shift(1) * spread_ret

    # 换手成本
    trades = position.diff().abs()
    cost = trades * 2 * cost_bps / 10000  # 两腿各一次

    net_ret = strat_ret - cost
    cum = (1 + net_ret).cumprod()

    return pd.DataFrame({
        "position": position,
        "spread_ret": spread_ret,
        "strat_ret": strat_ret,
        "net_ret": net_ret,
        "cum_ret": cum,
    })


def _print_stats(name: str, rets: pd.Series) -> dict[str, float]:
    """打印并返回策略统计"""
    rets = rets.dropna()
    if len(rets) < 30:
        print(f"  [{name}] 数据不足，跳过")
        return {}
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (1 + rets).cumprod().div((1 + rets).cumprod().cummax()).min() - 1
    win_rate = (rets > 0).mean()
    stats = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }
    print(f"  [{name}] 年化={ann_ret:.2%} 波动={ann_vol:.2%} "
          f"Sharpe={sharpe:.2f} 最大回撤={max_dd:.2%} 胜率={win_rate:.2%}")
    return stats


# ═══════════════════════════════════════════════════════════════════════
#  基类
# ═══════════════════════════════════════════════════════════════════════

class StatArbStrategy(ABC):
    """Base class for statistical arbitrage strategies"""
    name: str = "BaseStatArb"

    @abstractmethod
    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate the signal matrix (date x ticker); larger values are more bullish"""
        ...

    def backtest(self, data: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Default backtest: cross-sectional long/short"""
        close = _build_close_df(data)
        returns = close.pct_change()
        signal = self.generate_signal(data)
        return _simple_backtest(signal, returns, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  1. PairsTrading — 协整配对交易
# ═══════════════════════════════════════════════════════════════════════

class PairsTrading(StatArbStrategy):
    """Cointegration pairs trading strategy

    Method:
    1. Over the formation period, compute the correlation of every stock pair as a cointegration proxy
    2. Keep the top_pairs most correlated pairs
    3. Compute the z-score of the log spread
    4. Enter when |z| > entry_z, close when |z| < exit_z, stop loss when |z| > stop_z
    5. Roll the window forward, holding up to max_active pairs at once

    Reference: Gatev, Goetzmann & Rouwenhorst (2006) "Pairs Trading"
    """
    name = "PairsTrading"

    def __init__(
        self,
        formation_window: int = 252,
        trading_window: int = 60,
        top_pairs: int = 20,
        max_active: int = 5,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
    ):
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.top_pairs = top_pairs
        self.max_active = max_active
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def _find_pairs(self, close: pd.DataFrame) -> list[tuple[str, str, float]]:
        """找到相关性最高的配对（协整代理）

        Returns: [(ticker_a, ticker_b, correlation), ...]
        """
        log_p = np.log(close.dropna(axis=1, how="any"))
        if log_p.shape[1] < 2:
            return []
        tickers = log_p.columns.tolist()
        pairs = []
        for i, j in combinations(range(len(tickers)), 2):
            a, b = tickers[i], tickers[j]
            # 同行业优先：协整更可能成立
            corr = log_p[a].corr(log_p[b])
            if not np.isnan(corr):
                pairs.append((a, b, corr))
        pairs.sort(key=lambda x: -abs(x[2]))
        return pairs[: self.top_pairs]

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate pairs trading signals

        Returns a DataFrame (date x ticker) where positive = long and negative = short.
        Each pair produces its own +1/-1 signal, which is added onto the corresponding stocks.
        """
        close = _build_close_df(data)
        stocks = [t for t in UNIVERSE_50 if t in close.columns]
        close_stk = close[stocks].dropna(axis=1, thresh=int(len(close) * 0.8))

        signal = pd.DataFrame(0.0, index=close.index, columns=close_stk.columns)

        # 用前formation_window天找配对
        if len(close_stk) < self.formation_window + self.trading_window:
            return signal

        formation = close_stk.iloc[: self.formation_window]
        pairs = self._find_pairs(formation)

        if not pairs:
            return signal

        # 在整个样本内计算z-score并产生信号
        for a, b, _ in pairs[: self.max_active]:
            if a not in close_stk.columns or b not in close_stk.columns:
                continue
            spread = np.log(close_stk[a]) - np.log(close_stk[b])
            z = _zscore(spread, window=self.trading_window)

            # 做空价差 (z高) = 做空A做多B
            short_spread = (z > self.entry_z).astype(float)
            # 做多价差 (z低) = 做多A做空B
            long_spread = (z < -self.entry_z).astype(float)
            # 止损/止盈 mask
            flat = (z.abs() < self.exit_z) | (z.abs() > self.stop_z)

            pos = long_spread - short_spread
            # 简化：直接用信号强度，不追踪持仓状态（向量化近似）
            pos[flat] = 0

            signal[a] = signal[a] + pos
            signal[b] = signal[b] - pos

        return signal

    def backtest(self, data: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Backtest specific to pairs trading"""
        close = _build_close_df(data)
        returns = close.pct_change()
        signal = self.generate_signal(data)

        # 按信号方向加权收益
        port_ret = (signal.shift(1).div(signal.shift(1).abs().sum(axis=1), axis=0).fillna(0)
                    * returns.reindex(columns=signal.columns).fillna(0)).sum(axis=1)

        turnover = signal.diff().abs().sum(axis=1) / signal.abs().sum(axis=1).replace(0, 1)
        cost = turnover * 5 / 10000

        net = port_ret - cost
        return pd.DataFrame({
            "strat_ret": port_ret,
            "net_ret": net,
            "cum_ret": (1 + net).cumprod(),
            "turnover": turnover,
        })


# ═══════════════════════════════════════════════════════════════════════
#  2. SectorNeutralMomentum — 行业中性动量
# ═══════════════════════════════════════════════════════════════════════

class SectorNeutralMomentum(StatArbStrategy):
    """Sector-neutral momentum strategy

    Method:
    1. Compute each stock's 20-day momentum
    2. Project the sector ETF momentum onto the stock via beta to get its sector momentum contribution
    3. Stock momentum - sector momentum = pure stock-picking alpha
    4. Go long the 5 highest-alpha names and short the 5 lowest

    Goal: pure stock picking with no sector exposure
    """
    name = "SectorNeutralMomentum"

    def __init__(self, mom_window: int = 20, beta_window: int = 60,
                 long_n: int = 5, short_n: int = 5):
        self.mom_window = mom_window
        self.beta_window = beta_window
        self.long_n = long_n
        self.short_n = short_n

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate sector-neutral momentum signals

        Signal = stock 20-day momentum - beta * sector ETF 20-day momentum
        """
        close = _build_close_df(data)
        returns = close.pct_change()
        stocks = [t for t in UNIVERSE_50 if t in close.columns]

        # 20日动量
        mom = close[stocks].pct_change(self.mom_window)

        # 行业中性化
        signal = pd.DataFrame(index=close.index, columns=stocks, dtype=float)

        for etf, members in SECTOR_ETFS.items():
            if etf not in returns.columns:
                continue
            etf_mom = close[etf].pct_change(self.mom_window) if etf in close.columns else None
            if etf_mom is None:
                continue

            for stock in members:
                if stock not in returns.columns:
                    continue
                # 滚动beta
                beta = _rolling_beta(returns[stock], returns[etf], self.beta_window)
                # 行业中性动量 = 个股动量 - beta * 行业ETF动量
                sector_contribution = beta * etf_mom
                signal[stock] = mom[stock] - sector_contribution

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  3. ResidualReversion — 残差均值回复
# ═══════════════════════════════════════════════════════════════════════

class ResidualReversion(StatArbStrategy):
    """Fama-French residual mean-reversion strategy

    Method:
    1. Run a 60-day rolling OLS regression per stock: stock_ret = alpha + beta * spy_ret + epsilon
    2. Residual = stock_ret - alpha - beta * spy_ret (idiosyncratic stock return)
    3. Accumulate the residual over 10 days
    4. Signal = -cumulative_residual (mean reversion)

    Rationale: once the market factor is stripped out, idiosyncratic returns mean-revert over short horizons
    Reference: Lehmann (1990), Lo & MacKinlay (1990)
    """
    name = "ResidualReversion"

    def __init__(self, reg_window: int = 60, cum_window: int = 10):
        self.reg_window = reg_window
        self.cum_window = cum_window

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate residual reversion signals

        Signal = -cumsum(residual, 10d); a negative cumulative residual -> go long (expecting reversion)
        """
        close = _build_close_df(data)
        returns = close.pct_change()
        stocks = [t for t in UNIVERSE_50 if t in returns.columns]

        if "SPY" not in returns.columns:
            return pd.DataFrame(index=close.index, columns=stocks, dtype=float)

        spy_ret = returns["SPY"]
        signal = pd.DataFrame(index=close.index, columns=stocks, dtype=float)

        for stock in stocks:
            stk_ret = returns[stock]
            beta = _rolling_beta(stk_ret, spy_ret, self.reg_window)
            alpha = (stk_ret.rolling(self.reg_window, min_periods=30).mean()
                     - beta * spy_ret.rolling(self.reg_window, min_periods=30).mean())

            residual = stk_ret - alpha - beta * spy_ret
            cum_resid = residual.rolling(self.cum_window, min_periods=5).sum()
            signal[stock] = -cum_resid  # 均值回复：反向操作

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  4. ETFArbitrage — ETF篮子套利
# ═══════════════════════════════════════════════════════════════════════

class ETFArbitrage(StatArbStrategy):
    """ETF vs. constituent-basket arbitrage

    Method:
    1. Build a synthetic ETF return from the constituents
    2. Compute the spread between the synthetic and the actual ETF
    3. Enter when the spread deviates by more than 1.5 standard deviations
    4. Three trades:
       - AAPL+MSFT+NVDA+GOOG+META vs QQQ
       - JPM+BAC+GS+MS vs XLF
       - XOM+CVX vs XLE

    Rationale: the ETF creation/redemption mechanism forces the spread to converge over time
    """
    name = "ETFArbitrage"

    BASKETS = {
        "QQQ": ["AAPL", "MSFT", "NVDA", "GOOG", "META"],
        "XLF": ["JPM", "BAC", "GS", "MS"],
        "XLE": ["XOM", "CVX"],
    }

    def __init__(self, lookback: int = 60, entry_std: float = 1.5,
                 exit_std: float = 0.3):
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate ETF arbitrage signals

        When the basket is cheap relative to the ETF: long the basket and short the ETF, and vice versa.
        The returned signals carry non-zero values for both the constituents and the ETF.
        """
        close = _build_close_df(data)
        returns = close.pct_change()
        all_tickers = list(set(
            list(self.BASKETS.keys()) +
            [s for v in self.BASKETS.values() for s in v]
        ))
        valid = [t for t in all_tickers if t in close.columns]
        signal = pd.DataFrame(0.0, index=close.index, columns=valid)

        for etf, stocks in self.BASKETS.items():
            avail = [s for s in stocks if s in returns.columns]
            if etf not in returns.columns or len(avail) < 2:
                continue

            # 等权合成ETF收益
            synthetic_ret = returns[avail].mean(axis=1)
            etf_ret = returns[etf]

            # 累计价差
            spread = (synthetic_ret - etf_ret).cumsum()
            z = _zscore(spread, window=self.lookback)

            # 信号：z>0 → 合成ETF跑赢 → 做空篮子做多ETF
            raw_sig = pd.Series(0.0, index=z.index)
            raw_sig[z > self.entry_std] = -1.0
            raw_sig[z < -self.entry_std] = 1.0
            raw_sig[z.abs() < self.exit_std] = 0.0

            # 分配信号
            for s in avail:
                signal[s] = signal[s] + raw_sig / len(avail)
            if etf in signal.columns:
                signal[etf] = signal[etf] - raw_sig

        return signal

    def backtest(self, data: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Backtest specific to ETF arbitrage"""
        close = _build_close_df(data)
        returns = close.pct_change()
        signal = self.generate_signal(data)

        w = signal.shift(1).div(signal.shift(1).abs().sum(axis=1).replace(0, 1), axis=0).fillna(0)
        port_ret = (w * returns.reindex(columns=signal.columns).fillna(0)).sum(axis=1)

        turnover = w.diff().abs().sum(axis=1)
        cost = turnover * 5 / 10000
        net = port_ret - cost

        return pd.DataFrame({
            "strat_ret": port_ret,
            "net_ret": net,
            "cum_ret": (1 + net).cumprod(),
            "turnover": turnover,
        })


# ═══════════════════════════════════════════════════════════════════════
#  5. RelativeValueVolatility — 波动率相对价值
# ═══════════════════════════════════════════════════════════════════════

class RelativeValueVolatility(StatArbStrategy):
    """Intra-sector volatility relative value strategy

    Method:
    1. Compute realized volatility (20-day) for stock pairs within the same sector
    2. Compute the volatility ratio = vol_A / vol_B
    3. When the ratio deviates from its historical mean by more than 1.5 std: long the low-vol name, short the high-vol name
    4. Close when the ratio reverts

    Rationale: stocks in the same sector face similar risks, so their volatilities converge over the long run
    """
    name = "RelativeValueVolatility"

    def __init__(self, vol_window: int = 20, lookback: int = 120,
                 entry_std: float = 1.5):
        self.vol_window = vol_window
        self.lookback = lookback
        self.entry_std = entry_std

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate volatility relative value signals

        Volatility low relative to the sector -> go long; high -> go short
        """
        close = _build_close_df(data)
        returns = close.pct_change()

        stocks = [t for t in UNIVERSE_50 if t in returns.columns]
        rvol = returns[stocks].rolling(self.vol_window, min_periods=10).std() * np.sqrt(252)

        signal = pd.DataFrame(0.0, index=close.index, columns=stocks)

        for etf, members in SECTOR_ETFS.items():
            avail = [s for s in members if s in stocks]
            if len(avail) < 2:
                continue

            # 行业平均波动率
            sector_vol = rvol[avail].mean(axis=1)

            for stock in avail:
                ratio = rvol[stock] / sector_vol.replace(0, np.nan)
                z = _zscore(ratio, window=self.lookback)
                # 波动率偏高 → 做空；偏低 → 做多
                signal[stock] = -z

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  6. CointegrationPortfolio — 协整组合（PCA法）
# ═══════════════════════════════════════════════════════════════════════

class CointegrationPortfolio(StatArbStrategy):
    """Multi-asset cointegration portfolio strategy (simplified Johansen method)

    Method:
    1. Run PCA on the log prices of 3-4 stocks within a sector
    2. PC1 (the first principal component) approximates the cointegrating vector
    3. When the PC1 score deviates from zero by more than 2 std: trade the whole basket
    4. More robust than pairs trading (less prone to breaking down)

    Reference: Alexander & Dimitriu (2005) "Cointegration-Based Trading Strategies"
    """
    name = "CointegrationPortfolio"

    def __init__(self, window: int = 120, entry_std: float = 2.0,
                 exit_std: float = 0.5):
        self.window = window
        self.entry_std = entry_std
        self.exit_std = exit_std

    def _pca_spread(self, log_prices: pd.DataFrame, window: int) -> pd.Series:
        """用滚动PCA的PC1作为协整价差代理

        对对数价格做demeaned PCA，取第一主成分得分
        """
        # 标准化
        centered = log_prices.sub(log_prices.rolling(window, min_periods=60).mean())
        std = log_prices.rolling(window, min_periods=60).std()
        normed = centered / std.replace(0, np.nan)
        normed = normed.dropna()

        if len(normed) < window:
            return pd.Series(dtype=float)

        # 简化PCA：等权组合的残差 ≈ PC1
        # 完整PCA需要逐窗口计算特征向量，此处用市值等权简化
        # 取均值作为共同因子，各股偏离均值的部分作为价差
        common = normed.mean(axis=1)
        # PC1 score ≈ 第一个stock减去共同因子（简化）
        # 更好的做法：取normed的行均值作为spread
        spread = normed.iloc[:, 0] - common
        return spread

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate cointegration portfolio signals

        Run PCA on the constituents of each sector ETF and trade when PC1 deviates
        """
        close = _build_close_df(data)
        stocks = [t for t in UNIVERSE_50 if t in close.columns]
        log_p = np.log(close[stocks].dropna(axis=1, thresh=int(len(close) * 0.8)))

        signal = pd.DataFrame(0.0, index=close.index, columns=stocks)

        for etf, members in SECTOR_ETFS.items():
            avail = [s for s in members if s in log_p.columns]
            if len(avail) < 3:
                continue

            # 对行业成分做PCA
            sector_lp = log_p[avail]
            # 标准化
            centered = sector_lp.sub(sector_lp.rolling(self.window, min_periods=60).mean())
            std = sector_lp.rolling(self.window, min_periods=60).std()
            normed = centered / std.replace(0, np.nan)

            # 简化PCA：每只股票的z相对于行业均值z的偏离
            sector_mean = normed.mean(axis=1)
            for stock in avail:
                deviation = normed[stock] - sector_mean
                z = _zscore(deviation, window=self.window)
                # 偏离过大 → 均值回复
                sig = pd.Series(0.0, index=z.index)
                sig[z > self.entry_std] = -1.0
                sig[z < -self.entry_std] = 1.0
                sig[z.abs() < self.exit_std] = 0.0
                signal[stock] = signal[stock] + sig

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  7. MeanReversionBasket — 截面均值回复
# ═══════════════════════════════════════════════════════════════════════

class MeanReversionBasket(StatArbStrategy):
    """Cross-sectional mean-reversion basket strategy

    Method:
    1. Compute the log price of the equal-weighted basket
    2. Compute each stock's z-score relative to the basket
    3. Long when z < -2, short when z > 2
    4. Pure cross-sectional mean reversion, rebalanced daily

    Rationale: stocks that under- or out-perform the market over short horizons tend to revert
    Reference: Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns"
    """
    name = "MeanReversionBasket"

    def __init__(self, lookback: int = 20, entry_z: float = 2.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate cross-sectional mean-reversion signals

        Signal = -zscore(stock log price - basket log price)
        """
        close = _build_close_df(data)
        stocks = [t for t in UNIVERSE_50 if t in close.columns]
        close_stk = close[stocks].dropna(axis=1, thresh=int(len(close) * 0.8))

        log_p = np.log(close_stk)
        basket = log_p.mean(axis=1)  # 等权篮子

        signal = pd.DataFrame(index=close.index, columns=close_stk.columns, dtype=float)

        for stock in close_stk.columns:
            spread = log_p[stock] - basket
            z = _zscore(spread, window=self.lookback)
            signal[stock] = -z  # 均值回复：z高→做空

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  8. LeadLagArbitrage — 领先-滞后套利
# ═══════════════════════════════════════════════════════════════════════

class LeadLagArbitrage(StatArbStrategy):
    """Lead-lag arbitrage strategy

    Method:
    1. Compute each stock's correlation with SPY lagged by one day: corr(stock_t, SPY_{t-1})
    2. High lagged correlation = slow responder (information travels slowly)
    3. When SPY rises: go long the slow responders (they have not reacted yet)
    4. When SPY falls: short the slow responders

    Reference: Lo & MacKinlay (1990) "When are Contrarian Profits Due to Stock Market Overreaction?"
    """
    name = "LeadLagArbitrage"

    def __init__(self, corr_window: int = 60, top_n: int = 10):
        self.corr_window = corr_window
        self.top_n = top_n

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate lead-lag signals

        Signal = lagged_corr_rank * sign(SPY previous-day return)
        Stocks with high lagged correlation get a stronger signal in the direction SPY moved
        """
        close = _build_close_df(data)
        returns = close.pct_change()
        stocks = [t for t in UNIVERSE_50 if t in returns.columns]

        if "SPY" not in returns.columns:
            return pd.DataFrame(index=close.index, columns=stocks, dtype=float)

        spy_ret = returns["SPY"]
        spy_lag = spy_ret.shift(1)  # 昨日SPY收益

        signal = pd.DataFrame(0.0, index=close.index, columns=stocks)

        for stock in stocks:
            # 滚动滞后相关: corr(stock_t, SPY_{t-1})
            lag_corr = _rolling_corr(returns[stock], spy_lag, self.corr_window)

            # 信号 = 滞后相关 * SPY昨日方向
            # 滞后相关高且SPY昨天涨 → 今天该股也会涨 → 做多
            signal[stock] = lag_corr * np.sign(spy_lag)

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  9. DispersionTrading — 自适应离散度交易
# ═══════════════════════════════════════════════════════════════════════

class DispersionTrading(StatArbStrategy):
    """Adaptive dispersion trading strategy

    Method:
    1. Compute cross-sectional return dispersion = std(cross-sectional returns)
    2. High dispersion -> mean reversion works better (large deviations leave room to revert)
    3. Low dispersion -> momentum works better (trends are more uniform)
    4. Switch dynamically:
       - Dispersion > median: use the 5-day reversal signal
       - Dispersion <= median: use the 20-day momentum signal

    Rationale: the market regime decides which style works better
    Reference: Stivers & Sun (2010) "Cross-Sectional Return Dispersion and Time Variation in Value and Momentum Premiums"
    """
    name = "DispersionTrading"

    def __init__(self, disp_window: int = 20, rev_window: int = 5,
                 mom_window: int = 20):
        self.disp_window = disp_window
        self.rev_window = rev_window
        self.mom_window = mom_window

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate adaptive signals

        Use the reversal signal in high-dispersion periods and the momentum signal in low-dispersion periods
        """
        close = _build_close_df(data)
        returns = close.pct_change()
        stocks = [t for t in UNIVERSE_50 if t in returns.columns]
        ret_stk = returns[stocks]

        # 截面离散度
        dispersion = ret_stk.std(axis=1)
        disp_median = dispersion.rolling(252, min_periods=60).median()

        high_disp = dispersion > disp_median  # True = 高离散度时期

        # 反转信号（5日收益的负值）
        reversal = -ret_stk.rolling(self.rev_window).sum()
        # 动量信号（20日收益）
        momentum = ret_stk.rolling(self.mom_window).sum()

        # 截面标准化
        rev_z = reversal.sub(reversal.mean(axis=1), axis=0).div(
            reversal.std(axis=1).replace(0, np.nan), axis=0)
        mom_z = momentum.sub(momentum.mean(axis=1), axis=0).div(
            momentum.std(axis=1).replace(0, np.nan), axis=0)

        # 自适应切换
        signal = pd.DataFrame(0.0, index=close.index, columns=stocks)
        signal[high_disp] = rev_z[high_disp]
        signal[~high_disp] = mom_z[~high_disp]

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  10. OrderFlowImbalance — 订单流失衡
# ═══════════════════════════════════════════════════════════════════════

class OrderFlowImbalance(StatArbStrategy):
    """Order flow imbalance strategy (volume-price proxy)

    Method:
    1. Buy volume = volume * (close - low) / (high - low)
    2. Sell volume = volume * (high - close) / (high - low)
    3. Net inflow = buy - sell, accumulated over 5 days
    4. Cross-sectionally: long the highest net inflow, short the lowest

    Rationale: an intraday close near the high = strong buying pressure (institutional accumulation)
    Reference: Chordia, Roll & Subrahmanyam (2002) "Order Imbalance, Liquidity, and Market Returns"
    """
    name = "OrderFlowImbalance"

    def __init__(self, cum_window: int = 5):
        self.cum_window = cum_window

    def generate_signal(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate order flow imbalance signals

        Signal = cross-sectional rank of the 5-day cumulative net buying volume
        """
        high = _build_ohlcv_field(data, "High")
        low = _build_ohlcv_field(data, "Low")
        close = _build_ohlcv_field(data, "Close")
        volume = _build_ohlcv_field(data, "Volume")

        stocks = [t for t in UNIVERSE_50
                  if t in close.columns and t in high.columns
                  and t in low.columns and t in volume.columns]

        hl_range = (high[stocks] - low[stocks]).replace(0, np.nan)

        # 买方成交量占比
        buy_pct = (close[stocks] - low[stocks]) / hl_range
        sell_pct = (high[stocks] - close[stocks]) / hl_range

        buy_vol = volume[stocks] * buy_pct
        sell_vol = volume[stocks] * sell_pct

        net_flow = buy_vol - sell_vol

        # 累计5日
        cum_flow = net_flow.rolling(self.cum_window, min_periods=3).sum()

        # 截面标准化
        signal = cum_flow.sub(cum_flow.mean(axis=1), axis=0).div(
            cum_flow.std(axis=1).replace(0, np.nan), axis=0)

        return signal


# ═══════════════════════════════════════════════════════════════════════
#  策略注册表
# ═══════════════════════════════════════════════════════════════════════

ALL_STRATEGIES: list[type[StatArbStrategy]] = [
    PairsTrading,
    SectorNeutralMomentum,
    ResidualReversion,
    ETFArbitrage,
    RelativeValueVolatility,
    CointegrationPortfolio,
    MeanReversionBasket,
    LeadLagArbitrage,
    DispersionTrading,
    OrderFlowImbalance,
]


# ═══════════════════════════════════════════════════════════════════════
#  数据下载 & 全量回测
# ═══════════════════════════════════════════════════════════════════════

def download_data(
    start: str = "2016-01-01",
    end: str = "2024-12-31",
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for every instrument via yfinance

    Parameters
    ----------
    start : str
        Start date, YYYY-MM-DD
    end : str
        End date, YYYY-MM-DD

    Returns
    -------
    dict[str, pd.DataFrame]
        {ticker: DataFrame(Date, Open, High, Low, Close, Volume)}
    """
    import yfinance as yf

    tickers = list(set(UNIVERSE_50 + list(SECTOR_ETFS.keys()) + MACRO_ETFS))
    print(f"下载 {len(tickers)} 个标的 [{start} ~ {end}] ...")

    raw = yf.download(tickers, start=start, end=end,
                      group_by="ticker", auto_adjust=True, threads=True)

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
            df = df.dropna(subset=["Close"])
            if len(df) > 100:
                # 确保列名一致
                df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                data[ticker] = df
        except (KeyError, TypeError):
            print(f"  跳过 {ticker}（数据不足）")

    print(f"成功加载 {len(data)} 个标的，日期范围: "
          f"{min(d.index[0] for d in data.values()):%Y-%m-%d} ~ "
          f"{max(d.index[-1] for d in data.values()):%Y-%m-%d}")
    return data


def run_all_stat_arb_backtests(
    start: str = "2016-01-01",
    end: str = "2024-12-31",
) -> dict[str, dict]:
    """Download the data and backtest all 10 statistical arbitrage strategies

    Parameters
    ----------
    start, end : str
        Backtest date range

    Returns
    -------
    dict[str, dict]
        {strategy name: {"stats": {...}, "equity": Series}}
    """
    data = download_data(start, end)
    close = _build_close_df(data)
    returns = close.pct_change()

    results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("  统计套利策略全量回测")
    print("=" * 70)

    for StrategyCls in ALL_STRATEGIES:
        strat = StrategyCls()
        name = strat.name
        print(f"\n{'─' * 50}")
        print(f"  策略: {name}")
        print(f"{'─' * 50}")

        try:
            # 使用策略自身的backtest方法（部分策略有专用回测）
            if hasattr(strat, "backtest") and type(strat).backtest is not StatArbStrategy.backtest:
                bt = strat.backtest(data)
                rets = bt["net_ret"]
            else:
                signal = strat.generate_signal(data)
                stocks = [c for c in signal.columns if c in UNIVERSE_50]
                if not stocks:
                    print(f"  [{name}] 无有效信号，跳过")
                    continue
                signal = signal[stocks]
                ret_sub = returns.reindex(columns=stocks)
                bt = _simple_backtest(signal, ret_sub, long_top=5, short_bottom=5)
                rets = bt["ls_ret_net"]

            stats = _print_stats(name, rets)
            results[name] = {
                "stats": stats,
                "equity": bt["cum_ret"],
                "backtest": bt,
            }

        except Exception as e:
            print(f"  [{name}] 回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总表
    if results:
        print("\n" + "=" * 70)
        print("  策略汇总")
        print("=" * 70)
        summary = pd.DataFrame({
            name: res["stats"] for name, res in results.items() if res["stats"]
        }).T
        if not summary.empty:
            summary = summary.sort_values("sharpe", ascending=False)
            # 格式化输出
            fmt = summary.copy()
            for col in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
                if col in fmt.columns:
                    fmt[col] = fmt[col].map(lambda x: f"{x:.2%}")
            if "sharpe" in fmt.columns:
                fmt["sharpe"] = fmt["sharpe"].map(lambda x: f"{x:.2f}")
            print(fmt.to_string())

    return results


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_all_stat_arb_backtests("2016-01-01", "2024-12-31")
