"""Daily alpha strategies — daily and weekly rebalanced strategies built on Alpaca daily bars and daily-frequency signals"""
import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy
from qf.signals_daily import DailySignalGenerator, prepare_daily_signals


# ═══════════════════════════════════════════════════════════════════
#  1. VWAP动量策略
# ═══════════════════════════════════════════════════════════════════

class VWAPMomentumStrategy(BaseStrategy):
    """VWAP deviation + short-term momentum — institutional flow meets trend confirmation

    Logic: a close above VWAP indicates institutional buying pressure during the session;
    positive 5-day momentum on top of that indicates a short-term uptrend, and the signal is
    strongest when the two agree.
    signal = 0.6 * vwap_deviation + 0.4 * momentum_5d
    Rebalanced daily.
    """
    name = "VWAP Momentum"
    description = "VWAP偏离(机构资金流)+5日动量共振，日频再平衡"

    w_vwap: float = 0.6
    w_mom: float = 0.4

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the composite VWAP momentum signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume', 'vwap'; values are DataFrames (date x symbol)

        Returns
        -------
        pd.DataFrame
            Composite signal (date x symbol), in [-1, 1]
        """
        sg = DailySignalGenerator
        close = data['close']
        vwap = data['vwap']

        vwap_sig = sg.vwap_deviation(close, vwap)
        mom_sig = sg.momentum_5d(close)

        raw = self.w_vwap * vwap_sig + self.w_mom * mom_sig
        return sg.cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  2. 成交量异动策略
# ═══════════════════════════════════════════════════════════════════

class VolumeSurpriseStrategy(BaseStrategy):
    """Volume anomaly + return direction — spotting institutional accumulation and distribution

    Logic: a sudden volume expansion (> 2x the 20-day average) with a positive return means
    institutional accumulation (buy); heavy volume with a negative return means distribution (sell).
    signal = volume_surge_rank * sign(1-day return)
    Rebalanced daily.
    """
    name = "Volume Surprise"
    description = "量价配合：放量上涨=吸筹买入，放量下跌=派发卖出"

    surge_threshold: float = 2.0
    lookback: int = 20

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the volume anomaly signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            Signal (date x symbol), in [-1, 1]
        """
        sg = DailySignalGenerator
        close = data['close']
        volume = data['volume']

        returns_1d = close.pct_change()
        avg_vol = volume.rolling(self.lookback).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        # 只在成交量超过阈值时给出方向性信号，否则为0
        surge_mask = vol_ratio >= self.surge_threshold
        direction = np.sign(returns_1d)

        # 连续信号：放量程度 * 方向；未放量部分用较弱信号
        raw = vol_ratio * direction
        # 放量部分信号加强
        raw = raw.where(surge_mask, raw * 0.3)

        return sg.cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  3. 微观结构Alpha策略
# ═══════════════════════════════════════════════════════════════════

class MicrostructureAlphaStrategy(BaseStrategy):
    """Microstructure quality factor — Amihud liquidity + institutional footprint + trading cost

    Logic: low Amihud (high liquidity) + large average trade size (institutional participation)
    + narrow spread (low trading cost) = the high-quality trading environment institutions prefer.
    signal = -0.4*amihud + 0.3*trade_intensity + 0.3*(-spread)
    Note: amihud_illiquidity and high_low_spread are already sign-flipped in DailySignalGenerator.
    Rebalanced daily.
    """
    name = "Microstructure Alpha"
    description = "流动性+机构足迹+低交易成本 = 微观结构质量"

    w_amihud: float = 0.4
    w_trade_intensity: float = 0.3
    w_spread: float = 0.3

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the composite microstructure signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume', 'trade_count', 'high', 'low'

        Returns
        -------
        pd.DataFrame
            Signal (date x symbol), in [-1, 1]
        """
        sg = DailySignalGenerator
        close = data['close']
        volume = data['volume']
        trade_count = data['trade_count']
        high = data['high']
        low = data['low']

        returns = close.pct_change()
        dollar_volume = close * volume

        # DailySignalGenerator中amihud_illiquidity已经取负值（低非流动性=高信号）
        amihud_sig = sg.amihud_illiquidity(returns, dollar_volume)
        # trade_intensity: 大单笔成交量=高信号
        ti_sig = sg.trade_intensity(trade_count, volume)
        # high_low_spread已经取负值（低价差=高信号）
        spread_sig = sg.high_low_spread(high, low, close)

        raw = (self.w_amihud * amihud_sig
               + self.w_trade_intensity * ti_sig
               + self.w_spread * spread_sig)
        return sg.cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  4. 隔夜跳空策略
# ═══════════════════════════════════════════════════════════════════

class OvernightGapStrategy(BaseStrategy):
    """Overnight gap + volume confirmation — gap continuation vs gap fill

    Logic: the gap direction plus post-open volume decides continuation or fill.
    Gap up + heavy volume = information driven, the move continues (buy);
    gap up + light volume = sentiment driven and likely to fill (sell).
    Gap downs work the other way round.
    Rebalanced daily.
    """
    name = "Overnight Gap"
    description = "跳空+量确认：放量跳空=延续，缩量跳空=回补"

    vol_lookback: int = 20

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the overnight gap signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'open', 'volume'

        Returns
        -------
        pd.DataFrame
            Signal (date x symbol), in [-1, 1]
        """
        sg = DailySignalGenerator
        close = data['close']
        open_prices = data['open']
        volume = data['volume']

        prev_close = close.shift(1)
        gap = open_prices / prev_close.replace(0, np.nan) - 1

        # 成交量相对20日均量
        avg_vol = volume.rolling(self.vol_lookback).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        # 放量（>1x均量）= 延续 → gap方向不变
        # 缩量（<1x均量）= 回补 → gap方向取反
        # 用连续权重：vol_ratio > 1时正向放大，< 1时反转
        # 信号 = gap * (2 * vol_ratio - 1)，vol_ratio=1时信号为gap本身
        # vol_ratio=0.5时信号为0，vol_ratio=2时信号为3*gap
        vol_confirm = 2 * vol_ratio - 1
        vol_confirm = vol_confirm.clip(-1, 3)  # 限制极端值

        raw = gap * vol_confirm
        return sg.cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  5. 多因子日频策略
# ═══════════════════════════════════════════════════════════════════

class MultiFactorDailyStrategy(BaseStrategy):
    """Equal-weight daily multi-factor blend — four alpha sources at equal allocation

    Logic: combines VWAP momentum, volume anomaly, microstructure and overnight gap — four
    uncorrelated alpha sources — at equal weight (25% each), diversifying away single-factor risk.
    Uses prepare_daily_signals() to obtain the precomputed signals.
    Rebalanced daily.
    """
    name = "Multi-Factor Daily"
    description = "4因子等权: VWAP动量25% + 量异动25% + 微结构25% + 跳空25%"

    w_vwap_mom: float = 0.25
    w_vol_surprise: float = 0.25
    w_microstructure: float = 0.25
    w_overnight: float = 0.25

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the composite daily multi-factor signal

        Parameters
        ----------
        data : dict
            Alpaca daily bar dictionary containing 'close', 'open', 'high', 'low',
            'volume', 'vwap', 'trade_count'

        Returns
        -------
        pd.DataFrame
            Composite signal (date x symbol), in [-1, 1]
        """
        sg = DailySignalGenerator

        # 获取各子策略信号
        vwap_mom = VWAPMomentumStrategy()
        vol_surp = VolumeSurpriseStrategy()
        micro = MicrostructureAlphaStrategy()
        gap = OvernightGapStrategy()

        sig_vwap = vwap_mom.generate_signal(data)
        sig_vol = vol_surp.generate_signal(data)
        sig_micro = micro.generate_signal(data)
        sig_gap = gap.generate_signal(data)

        raw = (self.w_vwap_mom * sig_vwap
               + self.w_vol_surprise * sig_vol
               + self.w_microstructure * sig_micro
               + self.w_overnight * sig_gap)

        return sg.cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  6. 周频再平衡策略
# ═══════════════════════════════════════════════════════════════════

class WeeklyRebalanceStrategy(BaseStrategy):
    """Weekly-rebalanced multi-factor strategy — lower turnover and trading costs

    Logic: the same equal-weight four-factor blend as MultiFactorDaily, but positions are updated
    only every 5 trading days. On the days in between the previous rebalance signal is carried
    forward, cutting turnover by roughly 80%.
    Suited to cost-sensitive accounts or larger books.
    """
    name = "Weekly Rebalance Multi-Factor"
    description = "同MultiFactorDaily但每5日再平衡，降低换手80%"

    rebalance_freq: int = 5  # 每N个交易日再平衡

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the weekly-rebalanced signal

        Parameters
        ----------
        data : dict
            Alpaca daily bar dictionary

        Returns
        -------
        pd.DataFrame
            Signal (date x symbol), refreshed every rebalance_freq days
        """
        # 先获取日频多因子信号
        mf = MultiFactorDailyStrategy()
        daily_signal = mf.generate_signal(data)

        # 只在再平衡日更新，其余日用前值填充
        rebal_mask = pd.Series(False, index=daily_signal.index)
        rebal_mask.iloc[::self.rebalance_freq] = True

        weekly_signal = daily_signal.copy()
        weekly_signal[~rebal_mask.values] = np.nan
        weekly_signal = weekly_signal.ffill()

        return weekly_signal


# ═══════════════════════════════════════════════════════════════════
#  策略注册表
# ═══════════════════════════════════════════════════════════════════

DAILY_STRATEGY_REGISTRY = {
    'vwap_momentum':    VWAPMomentumStrategy,
    'volume_surprise':  VolumeSurpriseStrategy,
    'microstructure':   MicrostructureAlphaStrategy,
    'overnight_gap':    OvernightGapStrategy,
    'multi_factor':     MultiFactorDailyStrategy,
    'weekly_rebalance': WeeklyRebalanceStrategy,
}


def get_daily_strategy(strategy_id: str) -> BaseStrategy:
    """Get a daily strategy instance by ID

    Parameters
    ----------
    strategy_id : str
        Strategy identifier; see DAILY_STRATEGY_REGISTRY

    Returns
    -------
    BaseStrategy
        Strategy instance
    """
    if strategy_id not in DAILY_STRATEGY_REGISTRY:
        avail = list(DAILY_STRATEGY_REGISTRY.keys())
        raise ValueError(f"未知日频策略: {strategy_id}. 可用: {avail}")
    return DAILY_STRATEGY_REGISTRY[strategy_id]()


# ═══════════════════════════════════════════════════════════════════
#  回测引擎
# ═══════════════════════════════════════════════════════════════════

def run_daily_backtest(strategy: BaseStrategy,
                       data_dict: dict,
                       initial_capital: float = 10_000,
                       long_n: int = 20,
                       short_n: int = 0,
                       cost_bps: float = 5.0) -> dict:
    """Simple vectorized daily backtest

    Longs the long_n instruments with the highest signal and optionally shorts the short_n lowest.
    Equal weighting, rebalanced daily on the signal, net of one-way trading costs.

    Parameters
    ----------
    strategy : BaseStrategy
        Strategy instance; must implement generate_signal(data_dict)
    data_dict : dict
        Alpaca daily bar dictionary; must contain 'close' and 'volume'
    initial_capital : float
        Initial capital (default 10000)
    long_n : int
        Number of long positions (default 20)
    short_n : int
        Number of short positions (default 0, i.e. long-only)
    cost_bps : float
        One-way trading cost in basis points (default 5bps)

    Returns
    -------
    dict
        Backtest results:
        - 'sharpe': annualized Sharpe ratio
        - 'total_return': total return
        - 'annual_return': annualized return
        - 'max_drawdown': maximum drawdown
        - 'turnover': average daily turnover
        - 'equity_curve': equity curve (pd.Series)
        - 'daily_returns': daily return series (pd.Series)
    """
    # 生成信号
    signal = strategy.generate_signal(data_dict)

    close = data_dict.get('close')
    if close is None:
        close = data_dict['prices']
    returns = close.pct_change()

    # 对齐信号和收益
    signal, returns = signal.align(returns, join='inner')

    # 每日选股：信号用前一日的（避免前瞻偏差）
    signal_lagged = signal.shift(1)

    # 构建持仓权重矩阵
    weights = pd.DataFrame(0.0, index=signal_lagged.index, columns=signal_lagged.columns)

    for i, date in enumerate(signal_lagged.index):
        row = signal_lagged.iloc[i].dropna()
        if len(row) < long_n + short_n:
            continue

        # 做多：信号最高的N只
        if long_n > 0:
            top = row.nlargest(long_n).index
            weights.loc[date, top] = 1.0 / long_n

        # 做空：信号最低的N只
        if short_n > 0:
            bottom = row.nsmallest(short_n).index
            weights.loc[date, bottom] = -1.0 / short_n

    # 计算组合日收益
    port_returns = (weights * returns).sum(axis=1)

    # 计算换手率并扣除交易成本
    weight_diff = weights.diff().abs().sum(axis=1)
    turnover = weight_diff
    cost_per_day = turnover * cost_bps / 10_000
    port_returns_net = port_returns - cost_per_day

    # 净值曲线
    equity = (1 + port_returns_net).cumprod() * initial_capital

    # 性能指标
    n_days = len(port_returns_net)
    n_years = n_days / 252

    total_return = equity.iloc[-1] / initial_capital - 1 if n_days > 0 else 0.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    daily_std = port_returns_net.std()
    sharpe = (port_returns_net.mean() / max(daily_std, 1e-10)) * np.sqrt(252)

    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()

    avg_turnover = turnover.mean()

    return {
        'sharpe': round(sharpe, 3),
        'total_return': round(total_return, 4),
        'annual_return': round(annual_return, 4),
        'max_drawdown': round(max_drawdown, 4),
        'turnover': round(avg_turnover, 4),
        'n_days': n_days,
        'equity_curve': equity,
        'daily_returns': port_returns_net,
    }


def run_all_daily_backtests(data_dict: dict, **kwargs) -> pd.DataFrame:
    """Backtest every registered daily strategy and return the summary table

    Parameters
    ----------
    data_dict : dict
        Alpaca daily bar dictionary
    **kwargs
        Extra arguments forwarded to run_daily_backtest

    Returns
    -------
    pd.DataFrame
        Strategy x metric summary table
    """
    results = {}
    for sid, cls in DAILY_STRATEGY_REGISTRY.items():
        strat = cls()
        try:
            res = run_daily_backtest(strat, data_dict, **kwargs)
            results[sid] = {k: v for k, v in res.items()
                           if k not in ('equity_curve', 'daily_returns')}
        except Exception as e:
            results[sid] = {'error': str(e)}

    df = pd.DataFrame(results).T
    df.index.name = 'strategy'
    return df
