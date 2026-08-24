"""Commodity ETF trading strategy collection — 8 systematic strategies built on Alpaca-tradable ETFs

Strategies:
    1. GoldMomentumStrategy        — Gold trend momentum (GLD/GDX, 50-day MA + volatility scaling)
    2. OilMeanReversionStrategy    — Crude oil mean reversion (USO/UCO/SCO, Z-score extremes)
    3. GoldOilRatioStrategy        — Gold/oil ratio arbitrage (market-neutral pairs trade)
    4. CommodityCarryStrategy      — Commodity term-structure carry (contango/backwardation)
    5. GoldMinerArbitrage          — Gold miners vs gold arbitrage (GDX/GLD spread reversion)
    6. MacroRegimeStrategy         — Macro regime detection (multi-asset signals set positions)
    7. CommodityOptionsStrategy    — Commodity options strategy (IV rank + event driven)
    8. MultiCommodityMomentum      — Multi-commodity cross-sectional momentum (long strong / short weak)

Plus:
    - run_commodity_backtest()          — single-strategy backtest
    - run_all_commodity_backtests()     — aggregate backtest over all strategies

Available ETFs (Alpaca):
    Crude oil: USO($124), UCO(2x), SCO(-2x), XLE, XOP, BNO, DBO
    Gold: GLD($415), IAU, AAAU, GDX($86), GDXJ, NUGT(2x), DUST(-2x), SLV($63)
    Broad: DBC, GSG, PDBC
    Agriculture: WEAT, CORN, SOYB, DBA
    Natural gas: UNG, BOIL, KOLD
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# 所有策略所需的ETF代码
# =====================================================================
ALL_COMMODITY_ETFS = [
    # 黄金
    'GLD', 'IAU', 'GDX', 'GDXJ', 'NUGT', 'DUST', 'SLV',
    # 原油
    'USO', 'UCO', 'SCO', 'XLE', 'XOP', 'BNO', 'DBO',
    # 综合/农业/天然气
    'DBC', 'WEAT', 'CORN', 'SOYB', 'DBA', 'UNG', 'BOIL', 'KOLD',
    # 基准
    'SPY',
    # 铜矿 (用于多商品动量)
    'COPX',
]


# =====================================================================
# 基类
# =====================================================================

class CommodityStrategyBase(ABC):
    """Base class for commodity ETF strategies"""

    name: str = "未命名商品策略"
    description: str = ""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Parameters:
            data: pd.DataFrame, columns = ETF tickers, index = dates, values = close prices

        Returns:
            pd.DataFrame: columns = ETF tickers, index = dates, values = position weights
                          positive = long, negative = short, 0 = flat
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return the strategy parameters"""
        return {'name': self.name, 'description': self.description}


# =====================================================================
# 辅助函数
# =====================================================================

def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window, min_periods=window).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动Z-score"""
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """已实现波动率 (年化)"""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def _momentum(series: pd.Series, window: int) -> pd.Series:
    """价格动量 (收益率)"""
    return series.pct_change(window)


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """滚动百分位排名 (0-100)"""
    return series.rolling(window, min_periods=window // 2).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )


# =====================================================================
# 策略1: 黄金动量策略
# =====================================================================

class GoldMomentumStrategy(CommodityStrategyBase):
    """Gold trend momentum strategy

    Logic:
        - 60-day momentum confirms the trend direction
        - GLD > 50-day MA -> long GLD (or GDX for leverage)
        - GLD < 50-day MA -> flat, or short via DUST
        - Volatility scaling: position = target volatility / realized volatility

    Historical performance: gold trends strongly; momentum Sharpe around 0.8
    """

    name = "黄金动量"
    description = "Gold momentum: 50d SMA crossover + vol scaling"

    def __init__(self,
                 sma_window: int = 50,
                 mom_window: int = 60,
                 target_vol: float = 0.15,
                 vol_lookback: int = 20,
                 use_miners: bool = False,
                 use_inverse: bool = True):
        """
        参数:
            sma_window: 均线周期 (默认50日)
            mom_window: 动量计算周期 (默认60日)
            target_vol: 目标年化波动率 (默认15%)
            vol_lookback: 波动率计算窗口 (默认20日)
            use_miners: True则用GDX替代GLD获取杠杆
            use_inverse: True则在下跌趋势中通过DUST做空
        """
        self.sma_window = sma_window
        self.mom_window = mom_window
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.use_miners = use_miners
        self.use_inverse = use_inverse

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gold momentum signals"""
        gld = data['GLD']
        sma = _sma(gld, self.sma_window)
        mom = _momentum(gld, self.mom_window)
        rets = gld.pct_change()
        vol = _realized_vol(rets, self.vol_lookback)

        # 波动率缩放因子, 上限2倍
        vol_scale = (self.target_vol / vol).clip(upper=2.0)

        # 趋势判断: GLD > SMA 且动量为正
        trend_up = (gld > sma) & (mom > 0)
        trend_down = (gld < sma) & (mom < 0)

        long_ticker = 'GDX' if self.use_miners else 'GLD'
        short_ticker = 'DUST' if self.use_inverse else None

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 做多信号
        signals.loc[trend_up, long_ticker] = 1.0
        # 做空信号 (通过反向ETF实现, 权重为正代表买入DUST)
        if short_ticker and short_ticker in data.columns:
            signals.loc[trend_down, short_ticker] = 0.5  # DUST是2x杠杆, 减半仓位

        # 波动率缩放
        for col in signals.columns:
            mask = signals[col] != 0
            signals.loc[mask, col] = signals.loc[mask, col] * vol_scale[mask]

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'sma_window': self.sma_window,
            'mom_window': self.mom_window,
            'target_vol': self.target_vol,
            'use_miners': self.use_miners,
            'use_inverse': self.use_inverse,
        }


# =====================================================================
# 策略2: 原油均值回归策略
# =====================================================================

class OilMeanReversionStrategy(CommodityStrategyBase):
    """Crude oil mean reversion strategy

    Logic:
        - Crude oil mean-reverts strongly from extreme levels
        - Compute the Z-score of USO against its 60-day mean
        - Z < -2: buy USO (or UCO for 2x leverage)
        - Z > +2: sell/short USO (or buy SCO for -2x)
        - Hard stop loss: exit when Z reaches +/-3
    """

    name = "原油均值回归"
    description = "Oil mean reversion: Z-score extremes with stop-loss"

    def __init__(self,
                 z_window: int = 60,
                 entry_z: float = 2.0,
                 stop_z: float = 3.0,
                 use_leveraged: bool = False):
        """
        参数:
            z_window: Z-score计算窗口 (默认60日)
            entry_z: 进场阈值 (默认2.0标准差)
            stop_z: 止损阈值 (默认3.0标准差)
            use_leveraged: True则使用UCO/SCO (2x杠杆)
        """
        self.z_window = z_window
        self.entry_z = entry_z
        self.stop_z = stop_z
        self.use_leveraged = use_leveraged

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate crude oil mean reversion signals"""
        uso = data['USO']
        z = _zscore(uso, self.z_window)

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        long_ticker = 'UCO' if self.use_leveraged else 'USO'
        short_ticker = 'SCO' if self.use_leveraged else 'USO'

        # 超卖 → 做多
        oversold = z < -self.entry_z
        stopped_low = z < -self.stop_z
        long_mask = oversold & ~stopped_low
        signals.loc[long_mask, long_ticker] = 1.0

        # 超买 → 做空
        overbought = z > self.entry_z
        stopped_high = z > self.stop_z
        short_mask = overbought & ~stopped_high
        if self.use_leveraged:
            signals.loc[short_mask, short_ticker] = 1.0  # 买入SCO = 做空油
        else:
            signals.loc[short_mask, short_ticker] = -1.0  # 直接做空USO

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'z_window': self.z_window,
            'entry_z': self.entry_z,
            'stop_z': self.stop_z,
            'use_leveraged': self.use_leveraged,
        }


# =====================================================================
# 策略3: 金油比套利策略
# =====================================================================

class GoldOilRatioStrategy(CommodityStrategyBase):
    """Gold/oil ratio pairs trading strategy

    Logic:
        - The gold/oil ratio (GLD/USO) is a classic macro indicator
        - Ratio > mean + 2 sigma -> long crude, short gold (ratio reverts)
        - Ratio < mean - 2 sigma -> long gold, short crude
        - Market-neutral pairs trade that hedges systematic risk
    """

    name = "金油比套利"
    description = "Gold/Oil ratio pairs trade: market-neutral mean reversion"

    def __init__(self,
                 lookback: int = 60,
                 entry_std: float = 2.0,
                 exit_std: float = 0.5):
        """
        参数:
            lookback: 均值/标准差计算窗口 (默认60日)
            entry_std: 进场阈值 (默认2.0σ)
            exit_std: 平仓阈值 (默认0.5σ, 比值回到正常范围)
        """
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gold/oil ratio pairs trading signals"""
        ratio = data['GLD'] / data['USO']
        z = _zscore(ratio, self.lookback)

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 金油比过高 → 做多油, 做空金 (预期比值下降)
        ratio_high = z > self.entry_std
        signals.loc[ratio_high, 'USO'] = 1.0
        signals.loc[ratio_high, 'GLD'] = -1.0

        # 金油比过低 → 做多金, 做空油 (预期比值上升)
        ratio_low = z < -self.entry_std
        signals.loc[ratio_low, 'GLD'] = 1.0
        signals.loc[ratio_low, 'USO'] = -1.0

        # 中间区域 (|z| < exit_std) → 空仓, 已经是0

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'entry_std': self.entry_std,
            'exit_std': self.exit_std,
        }


# =====================================================================
# 策略4: 商品期限结构套利 (Carry)
# =====================================================================

class CommodityCarryStrategy(CommodityStrategyBase):
    """Commodity term-structure carry strategy

    Logic:
        - Contango (futures premium): deferred > front month, negative cost of carry -> short the commodity
        - Backwardation (futures discount): front > deferred month, positive carry -> long the commodity
        - Use the UCO/USO ratio as a contango proxy (UCO decays faster in contango)
        - Alternatively compare USO/BNO (different roll dates)
    """

    name = "商品Carry"
    description = "Commodity carry: contango/backwardation via ETF ratio proxy"

    def __init__(self,
                 lookback: int = 20,
                 entry_z: float = 1.0,
                 carry_proxy: str = 'uco_uso'):
        """
        参数:
            lookback: 比值变化计算窗口
            entry_z: 进场阈值
            carry_proxy: 'uco_uso' 或 'uso_bno'
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.carry_proxy = carry_proxy

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate carry signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if self.carry_proxy == 'uco_uso':
            # UCO/USO比值持续下降 = contango严重 (UCO因每日复合+roll衰减)
            # 比值上升 = backwardation (UCO表现更好)
            if 'UCO' not in data.columns or 'USO' not in data.columns:
                logger.warning("Carry策略需要UCO和USO数据")
                return signals
            ratio = data['UCO'] / data['USO']
        else:
            # USO/BNO: 不同展期日期, 差异反映期限结构
            if 'BNO' not in data.columns or 'USO' not in data.columns:
                logger.warning("Carry策略需要USO和BNO数据")
                return signals
            ratio = data['USO'] / data['BNO']

        # 比值的变化率 (滚动)
        ratio_change = ratio.pct_change(self.lookback)
        z = _zscore(ratio_change, self.lookback * 3)

        # Backwardation信号 (比值上升) → 做多原油
        backwardation = z > self.entry_z
        signals.loc[backwardation, 'USO'] = 1.0

        # Contango信号 (比值下降) → 做空原油 (通过SCO)
        contango = z < -self.entry_z
        if 'SCO' in data.columns:
            signals.loc[contango, 'SCO'] = 1.0  # 买入SCO = 做空油
        else:
            signals.loc[contango, 'USO'] = -1.0

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'carry_proxy': self.carry_proxy,
        }


# =====================================================================
# 策略5: 金矿股vs黄金套利
# =====================================================================

class GoldMinerArbitrage(CommodityStrategyBase):
    """Gold miners vs gold spread arbitrage strategy

    Logic:
        - GDX (gold miners) has a beta of roughly 1.5-2x to gold
        - The GDX/GLD ratio mean-reverts
        - Ratio falling (miners underperform gold) -> buy GDX, sell GLD
        - Ratio rising (miners outperform gold) -> sell GDX, buy GLD
        - Beta adjustment: GLD position = GDX position x beta
    """

    name = "金矿套利"
    description = "Gold miner vs gold spread: mean-reversion of GDX/GLD ratio"

    def __init__(self,
                 lookback: int = 60,
                 entry_z: float = 1.5,
                 exit_z: float = 0.3,
                 beta_window: int = 120,
                 hedge_ratio_cap: float = 2.5):
        """
        参数:
            lookback: 比值Z-score计算窗口
            entry_z: 进场阈值
            exit_z: 平仓阈值
            beta_window: Beta估计窗口
            hedge_ratio_cap: 对冲比率上限
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.beta_window = beta_window
        self.hedge_ratio_cap = hedge_ratio_cap

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gold miner arbitrage signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if 'GDX' not in data.columns or 'GLD' not in data.columns:
            logger.warning("金矿套利策略需要GDX和GLD数据")
            return signals

        ratio = data['GDX'] / data['GLD']
        z = _zscore(ratio, self.lookback)

        # 滚动beta估计 (GDX对GLD的回归系数)
        gdx_ret = data['GDX'].pct_change()
        gld_ret = data['GLD'].pct_change()
        rolling_cov = gdx_ret.rolling(self.beta_window).cov(gld_ret)
        rolling_var = gld_ret.rolling(self.beta_window).var()
        beta = (rolling_cov / rolling_var.replace(0, np.nan)).clip(
            lower=0.5, upper=self.hedge_ratio_cap
        )

        # 矿股跑输 (比值偏低) → 买GDX, 卖GLD
        miners_cheap = z < -self.entry_z
        signals.loc[miners_cheap, 'GDX'] = 1.0
        signals.loc[miners_cheap, 'GLD'] = -1.0 / beta[miners_cheap]

        # 矿股跑赢 (比值偏高) → 卖GDX, 买GLD
        miners_rich = z > self.entry_z
        signals.loc[miners_rich, 'GDX'] = -1.0
        signals.loc[miners_rich, 'GLD'] = 1.0 / beta[miners_rich]

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'beta_window': self.beta_window,
        }


# =====================================================================
# 策略6: 宏观体制识别策略
# =====================================================================

class MacroRegimeStrategy(CommodityStrategyBase):
    """Macro regime detection strategy

    Logic:
        Identify the macro regime from the trend direction of gold, crude oil and SPY:
        - Risk-on:            SPY up, gold flat, oil up -> long oil, overweight equities
        - Risk-off:           SPY down, gold up, oil down -> long gold, hedge equities
        - Inflation:          gold up, oil up, SPY flat -> long commodities
        - Deflation:          everything down -> cash / short
        Rebalanced weekly
    """

    name = "宏观体制"
    description = "Macro regime detection: gold/oil/SPY trends → positioning"

    def __init__(self,
                 trend_window: int = 40,
                 rebalance_freq: int = 5,
                 trend_threshold: float = 0.0):
        """
        参数:
            trend_window: 趋势判断窗口 (默认40个交易日 ≈ 2个月)
            rebalance_freq: 再平衡频率 (默认5日 = 每周)
            trend_threshold: 趋势确认阈值 (收益率, 默认0 = 正负即可)
        """
        self.trend_window = trend_window
        self.rebalance_freq = rebalance_freq
        self.trend_threshold = trend_threshold

    def _classify_regime(self, spy_mom: float, gold_mom: float,
                         oil_mom: float) -> str:
        """根据动量判断宏观体制"""
        t = self.trend_threshold
        spy_up = spy_mom > t
        spy_dn = spy_mom < -t
        gold_up = gold_mom > t
        gold_dn = gold_mom < -t
        oil_up = oil_mom > t
        oil_dn = oil_mom < -t

        if spy_up and oil_up and not gold_up:
            return 'risk_on'
        elif spy_dn and gold_up and oil_dn:
            return 'risk_off'
        elif gold_up and oil_up:
            return 'inflation'
        elif spy_dn and gold_dn and oil_dn:
            return 'deflation'
        else:
            return 'neutral'

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate macro regime signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        required = ['SPY', 'GLD', 'USO']
        if not all(c in data.columns for c in required):
            logger.warning("宏观体制策略需要SPY, GLD, USO数据")
            return signals

        spy_mom = _momentum(data['SPY'], self.trend_window)
        gold_mom = _momentum(data['GLD'], self.trend_window)
        oil_mom = _momentum(data['USO'], self.trend_window)

        # 资产配置方案
        regime_alloc = {
            'risk_on':   {'USO': 0.4, 'XLE': 0.3, 'SPY': 0.3},
            'risk_off':  {'GLD': 0.5, 'SLV': 0.2, 'SPY': -0.3},
            'inflation': {'GLD': 0.3, 'USO': 0.3, 'DBC': 0.2, 'SLV': 0.2},
            'deflation': {'GLD': 0.1, 'USO': -0.2, 'SPY': -0.3},
            'neutral':   {'GLD': 0.2, 'USO': 0.1},
        }

        # 逐日分类 (只在再平衡日更新)
        current_alloc = {}
        for i, dt in enumerate(data.index):
            if i < self.trend_window:
                continue

            # 每 rebalance_freq 天更新一次
            if i % self.rebalance_freq == 0:
                regime = self._classify_regime(
                    spy_mom.iloc[i], gold_mom.iloc[i], oil_mom.iloc[i]
                )
                current_alloc = regime_alloc.get(regime, {})

            for ticker, weight in current_alloc.items():
                if ticker in signals.columns:
                    signals.loc[dt, ticker] = weight

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'trend_window': self.trend_window,
            'rebalance_freq': self.rebalance_freq,
        }


# =====================================================================
# 策略7: 商品期权策略
# =====================================================================

class CommodityOptionsStrategy(CommodityStrategyBase):
    """Commodity options strategy

    Logic:
        - GLD IV rank > 70%: sell covered calls
        - USO Z-score < -1.5: sell puts (cash-secured put, buying the dip)
        - GLD low-volatility regime: iron condor (sell options on both sides)
        - Ahead of an OPEC meeting on USO: buy a strangle (betting on a large move)

    Note: the signals produced here are "ETF-equivalent positions";
          live execution routes orders through the options module in options_strategies.py
    """

    name = "商品期权"
    description = "Commodity options: IV rank sells + event-driven buys"

    def __init__(self,
                 iv_rank_window: int = 252,
                 iv_sell_threshold: float = 70.0,
                 oil_put_z: float = -1.5,
                 low_vol_percentile: float = 30.0):
        """
        参数:
            iv_rank_window: IV rank计算窗口 (默认252日 = 1年)
            iv_sell_threshold: 卖出看涨的IV百分位阈值 (默认70)
            oil_put_z: 卖出原油看跌的Z-score阈值 (默认-1.5)
            low_vol_percentile: 铁鹰进场的低波动百分位 (默认30)
        """
        self.iv_rank_window = iv_rank_window
        self.iv_sell_threshold = iv_sell_threshold
        self.oil_put_z = oil_put_z
        self.low_vol_percentile = low_vol_percentile

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate commodity option-equivalent signals

        Note: realized volatility is used as a proxy for IV (true IV requires options data).
            Meaning of the signal values:
            +0.3 = covered-call-equivalent position (holding the underlying, delta about -0.3)
            +0.5 = short-put-equivalent position (providing liquidity, delta about +0.3)
            Output is still ETF weights; live execution maps them to specific option positions
        """
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # --- GLD Covered Call: 高IV rank时卖call ---
        if 'GLD' in data.columns:
            gld_ret = data['GLD'].pct_change()
            gld_rv = _realized_vol(gld_ret, 20)
            gld_iv_rank = _rolling_percentile(gld_rv, self.iv_rank_window)

            # 高IV → 卖covered call (持有GLD + 卖call ≈ 持有GLD但delta减小)
            high_iv = gld_iv_rank > self.iv_sell_threshold
            signals.loc[high_iv, 'GLD'] = 0.7  # 底层持仓
            # 标记: 这些日期应该同时卖出GLD call

            # 低波动 → 铁鹰 (区间震荡, 不持有方向性仓位)
            low_vol = gld_iv_rank < self.low_vol_percentile
            # 铁鹰等价: 小额正delta (接近中性)
            signals.loc[low_vol, 'GLD'] = 0.1

        # --- USO Put Selling: 超卖时卖put ---
        if 'USO' in data.columns:
            uso_z = _zscore(data['USO'], 60)
            oversold = uso_z < self.oil_put_z
            # 卖put ≈ 小仓位做多 (delta约0.3)
            signals.loc[oversold, 'USO'] = 0.3

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'iv_sell_threshold': self.iv_sell_threshold,
            'oil_put_z': self.oil_put_z,
        }


# =====================================================================
# 策略8: 多商品横截面动量
# =====================================================================

class MultiCommodityMomentum(CommodityStrategyBase):
    """Multi-commodity cross-sectional momentum strategy

    Logic:
        - Cross-sectional momentum: long the N strongest commodities, short the N weakest
        - Universe: GLD, SLV, USO, UNG, WEAT, CORN, SOYB, DBA, COPX
        - Ranked on 20-day momentum, long the top 3, short the bottom 3
        - Equal weighted, rebalanced weekly
        - Academic basis: Erb & Harvey (2006), Asness et al. (2013)
    """

    name = "多商品动量"
    description = "Cross-sectional commodity momentum: long top 3, short bottom 3"

    UNIVERSE = ['GLD', 'SLV', 'USO', 'UNG', 'WEAT', 'CORN', 'SOYB', 'DBA', 'COPX']

    def __init__(self,
                 mom_window: int = 20,
                 long_n: int = 3,
                 short_n: int = 3,
                 rebalance_freq: int = 5):
        """
        参数:
            mom_window: 动量计算窗口 (默认20日)
            long_n: 做多数量 (默认3)
            short_n: 做空数量 (默认3)
            rebalance_freq: 再平衡频率 (默认5日 = 每周)
        """
        self.mom_window = mom_window
        self.long_n = long_n
        self.short_n = short_n
        self.rebalance_freq = rebalance_freq

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional momentum signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 筛选可用资产
        available = [t for t in self.UNIVERSE if t in data.columns]
        if len(available) < self.long_n + self.short_n:
            logger.warning(
                f"可用资产不足: {len(available)}, 需要至少{self.long_n + self.short_n}"
            )
            return signals

        # 计算各资产动量
        mom_df = pd.DataFrame({
            t: _momentum(data[t], self.mom_window) for t in available
        })

        # 逐周排名并分配权重
        weight_per_long = 1.0 / self.long_n
        weight_per_short = -1.0 / self.short_n
        current_weights = {}

        for i, dt in enumerate(data.index):
            if i < self.mom_window:
                continue

            # 每 rebalance_freq 天更新
            if i % self.rebalance_freq == 0:
                row = mom_df.loc[dt].dropna()
                if len(row) < self.long_n + self.short_n:
                    continue
                ranked = row.sort_values(ascending=False)
                current_weights = {}
                # 做多前N
                for t in ranked.index[:self.long_n]:
                    current_weights[t] = weight_per_long
                # 做空后N
                for t in ranked.index[-self.short_n:]:
                    current_weights[t] = weight_per_short

            for ticker, w in current_weights.items():
                if ticker in signals.columns:
                    signals.loc[dt, ticker] = w

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'mom_window': self.mom_window,
            'long_n': self.long_n,
            'short_n': self.short_n,
            'rebalance_freq': self.rebalance_freq,
        }


# =====================================================================
# 回测引擎
# =====================================================================

def run_commodity_backtest(
    strategy: CommodityStrategyBase,
    data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 5.0,
) -> dict:
    """Single-strategy backtest

    Parameters:
        strategy: strategy instance
        data: price data (columns=ETFs, index=dates, values=close prices)
        initial_capital: starting capital
        commission_bps: transaction cost (basis points)

    Returns:
        dict: {
            'name': strategy name,
            'equity_curve': pd.Series (daily equity),
            'returns': pd.Series (daily returns),
            'sharpe': float,
            'cagr': float,
            'max_drawdown': float,
            'win_rate': float,
            'calmar': float,
            'total_trades': int,
        }
    """
    logger.info(f"回测策略: {strategy.name}")

    # 生成信号 (权重矩阵)
    weights = strategy.generate_signal(data)

    # 计算每日收益
    returns = data.pct_change()

    # 加权组合收益 (前一日的权重 × 当日收益)
    shifted_weights = weights.shift(1)  # T日收盘信号 → T+1日持仓
    port_returns = (shifted_weights * returns).sum(axis=1)

    # 交易成本: 权重变化 × commission
    turnover = shifted_weights.diff().abs().sum(axis=1)
    cost = turnover * commission_bps / 10_000
    port_returns = port_returns - cost

    # 去除前面NaN
    port_returns = port_returns.dropna()
    if len(port_returns) == 0:
        return {
            'name': strategy.name,
            'equity_curve': pd.Series(dtype=float),
            'returns': pd.Series(dtype=float),
            'sharpe': 0.0, 'cagr': 0.0, 'max_drawdown': 0.0,
            'win_rate': 0.0, 'calmar': 0.0, 'total_trades': 0,
        }

    # 净值曲线
    equity = (1 + port_returns).cumprod() * initial_capital

    # 绩效指标
    n_years = len(port_returns) / 252
    total_return = equity.iloc[-1] / initial_capital - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (port_returns.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # 胜率
    active_days = port_returns[port_returns != 0]
    win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0.0

    # 交易次数 (权重变化)
    total_trades = int((turnover > 0.01).sum())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'name': strategy.name,
        'equity_curve': equity,
        'returns': port_returns,
        'sharpe': round(sharpe, 3),
        'cagr': round(cagr * 100, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'calmar': round(calmar, 3),
        'total_trades': total_trades,
    }


def run_all_commodity_backtests(
    start: str = '2018-01-01',
    end: str = '2026-03-28',
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """Run backtests for all 8 commodity strategies and print a summary table

    Parameters:
        start: backtest start date
        end: backtest end date
        initial_capital: starting capital

    Returns:
        pd.DataFrame: per-strategy performance summary (Sharpe, CAGR%, MDD%, WR%, Calmar, Trades)
    """
    import yfinance as yf

    # ---- 下载数据 ----
    print(f"下载商品ETF数据: {start} → {end}")
    print(f"  代码: {ALL_COMMODITY_ETFS}")

    raw = yf.download(
        ALL_COMMODITY_ETFS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )

    # yfinance返回MultiIndex columns: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close'].copy()
    else:
        data = raw.copy()

    # 去掉全部为NaN的列 (某些ETF可能无数据)
    data = data.dropna(axis=1, how='all')
    data = data.ffill()  # 前向填充缺失 (节假日等)

    available = list(data.columns)
    print(f"  获取到 {len(available)} 个ETF: {available}")
    print(f"  数据范围: {data.index[0].strftime('%Y-%m-%d')} → "
          f"{data.index[-1].strftime('%Y-%m-%d')}, {len(data)} 个交易日")
    print()

    # ---- 初始化全部策略 ----
    strategies = [
        GoldMomentumStrategy(),
        OilMeanReversionStrategy(),
        GoldOilRatioStrategy(),
        CommodityCarryStrategy(),
        GoldMinerArbitrage(),
        MacroRegimeStrategy(),
        CommodityOptionsStrategy(),
        MultiCommodityMomentum(),
    ]

    # ---- 回测 ----
    results = []
    for strat in strategies:
        try:
            res = run_commodity_backtest(strat, data, initial_capital)
            results.append(res)
            print(f"  ✓ {strat.name:<12s}  "
                  f"Sharpe={res['sharpe']:>6.3f}  "
                  f"CAGR={res['cagr']:>7.2f}%  "
                  f"MDD={res['max_drawdown']:>7.2f}%  "
                  f"WR={res['win_rate']:>5.1f}%")
        except Exception as e:
            logger.error(f"策略 {strat.name} 回测失败: {e}")
            print(f"  ✗ {strat.name:<12s}  ERROR: {e}")
            results.append({
                'name': strat.name, 'sharpe': np.nan, 'cagr': np.nan,
                'max_drawdown': np.nan, 'win_rate': np.nan,
                'calmar': np.nan, 'total_trades': 0,
            })

    # ---- 汇总表 ----
    summary = pd.DataFrame([
        {
            '策略': r['name'],
            'Sharpe': r['sharpe'],
            'CAGR%': r['cagr'],
            'MDD%': r['max_drawdown'],
            'WR%': r['win_rate'],
            'Calmar': r['calmar'],
            'Trades': r['total_trades'],
        }
        for r in results
    ])
    summary = summary.set_index('策略')

    print("\n" + "=" * 72)
    print("大宗商品策略回测汇总")
    print("=" * 72)
    print(summary.to_string())
    print("=" * 72)

    return summary


# =====================================================================
# 直接运行
# =====================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    summary = run_all_commodity_backtests()
