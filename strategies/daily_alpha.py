"""日频Alpha策略 — 基于Alpaca日线数据+日频信号的每日/每周再平衡策略"""
import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy
from qf.signals_daily import DailySignalGenerator, prepare_daily_signals


# ═══════════════════════════════════════════════════════════════════
#  1. VWAP动量策略
# ═══════════════════════════════════════════════════════════════════

class VWAPMomentumStrategy(BaseStrategy):
    """VWAP偏离+短期动量组合 — 机构资金流向+趋势共振

    逻辑：收盘价高于VWAP说明盘中有机构买压（扫货），
    叠加5日动量为正表示短期趋势向上，两者共振时信号最强。
    信号 = 0.6 * vwap_deviation + 0.4 * momentum_5d
    日频再平衡。
    """
    name = "VWAP Momentum"
    description = "VWAP偏离(机构资金流)+5日动量共振，日频再平衡"

    w_vwap: float = 0.6
    w_mom: float = 0.4

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成VWAP动量复合信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'volume', 'vwap'；值为 DataFrame (date x symbol)

        Returns
        -------
        pd.DataFrame
            复合信号 (date x symbol)，值域 [-1, 1]
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
    """成交量异动+收益方向 — 机构吸筹/派发识别

    逻辑：成交量突然放大（>2x 20日均量）伴随正收益 = 机构吸筹（买入）；
    放量伴随负收益 = 机构派发（卖出）。
    信号 = volume_surge_rank * sign(1日收益率)
    日频再平衡。
    """
    name = "Volume Surprise"
    description = "量价配合：放量上涨=吸筹买入，放量下跌=派发卖出"

    surge_threshold: float = 2.0
    lookback: int = 20

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成成交量异动信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            信号 (date x symbol)，值域 [-1, 1]
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
    """微观结构质量因子 — Amihud流动性+机构足迹+交易成本

    逻辑：低Amihud（高流动性）+ 高单笔成交量（机构参与）+ 窄价差（低交易成本）
    = 机构偏好的高质量交易环境。
    信号 = -0.4*amihud + 0.3*trade_intensity + 0.3*(-spread)
    注意：amihud_illiquidity和high_low_spread在DailySignalGenerator中已取负值。
    日频再平衡。
    """
    name = "Microstructure Alpha"
    description = "流动性+机构足迹+低交易成本 = 微观结构质量"

    w_amihud: float = 0.4
    w_trade_intensity: float = 0.3
    w_spread: float = 0.3

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成微观结构复合信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'volume', 'trade_count', 'high', 'low'

        Returns
        -------
        pd.DataFrame
            信号 (date x symbol)，值域 [-1, 1]
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
    """隔夜跳空+成交量确认 — 跳空延续 vs 跳空回补

    逻辑：隔夜跳空方向 + 开盘后成交量决定延续还是回补。
    跳空上涨 + 放量 = 信息驱动，延续趋势（买入）；
    跳空上涨 + 缩量 = 情绪驱动，大概率回补（卖出）。
    跳空下跌则方向相反。
    日频再平衡。
    """
    name = "Overnight Gap"
    description = "跳空+量确认：放量跳空=延续，缩量跳空=回补"

    vol_lookback: int = 20

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成隔夜跳空信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'open', 'volume'

        Returns
        -------
        pd.DataFrame
            信号 (date x symbol)，值域 [-1, 1]
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
    """多因子日频等权组合 — 四大Alpha源均等配置

    逻辑：将VWAP动量、成交量异动、微观结构、隔夜跳空四个不相关Alpha源
    等权组合（各25%），通过分散化降低单因子风险。
    使用 prepare_daily_signals() 获取预计算信号。
    日频再平衡。
    """
    name = "Multi-Factor Daily"
    description = "4因子等权: VWAP动量25% + 量异动25% + 微结构25% + 跳空25%"

    w_vwap_mom: float = 0.25
    w_vol_surprise: float = 0.25
    w_microstructure: float = 0.25
    w_overnight: float = 0.25

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成多因子日频复合信号

        Parameters
        ----------
        data : dict
            Alpaca日线数据字典，含 'close', 'open', 'high', 'low',
            'volume', 'vwap', 'trade_count'

        Returns
        -------
        pd.DataFrame
            复合信号 (date x symbol)，值域 [-1, 1]
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
    """周频再平衡多因子策略 — 降低换手和交易成本

    逻辑：与MultiFactorDaily相同的四因子等权，但每5个交易日才更新一次持仓。
    中间日沿用上一次再平衡的信号，减少约80%的换手率。
    适合交易成本敏感或资金量较大的账户。
    """
    name = "Weekly Rebalance Multi-Factor"
    description = "同MultiFactorDaily但每5日再平衡，降低换手80%"

    rebalance_freq: int = 5  # 每N个交易日再平衡

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成周频再平衡信号

        Parameters
        ----------
        data : dict
            Alpaca日线数据字典

        Returns
        -------
        pd.DataFrame
            信号 (date x symbol)，每rebalance_freq天更新一次
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
    """按ID获取日频策略实例

    Parameters
    ----------
    strategy_id : str
        策略标识符，见 DAILY_STRATEGY_REGISTRY

    Returns
    -------
    BaseStrategy
        策略实例
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
    """简易向量化日频回测

    做多信号最高的long_n只股票，做空信号最低的short_n只（可选）。
    等权配置，每日按信号再平衡，扣除单边交易成本。

    Parameters
    ----------
    strategy : BaseStrategy
        策略实例，需实现 generate_signal(data_dict)
    data_dict : dict
        Alpaca日线数据字典，必须含 'close' 和 'volume'
    initial_capital : float
        初始资金（默认 10000）
    long_n : int
        做多股票数量（默认 20）
    short_n : int
        做空股票数量（默认 0，纯多头）
    cost_bps : float
        单边交易成本（基点，默认 5bps）

    Returns
    -------
    dict
        回测结果:
        - 'sharpe': 年化夏普比率
        - 'total_return': 总收益率
        - 'annual_return': 年化收益率
        - 'max_drawdown': 最大回撤
        - 'turnover': 日均换手率
        - 'equity_curve': 净值序列 (pd.Series)
        - 'daily_returns': 日收益率序列 (pd.Series)
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
    """对所有注册的日频策略运行回测，返回汇总表

    Parameters
    ----------
    data_dict : dict
        Alpaca日线数据字典
    **kwargs
        传递给 run_daily_backtest 的额外参数

    Returns
    -------
    pd.DataFrame
        策略 x 指标 的汇总表
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
