"""FX and futures ETF proxy strategy collection — 8 systematic strategies built on Alpaca/yfinance-tradable ETFs

Strategies:
    1. CurrencyMomentumStrategy    — FX momentum (cross-sectional momentum over 6 currency ETFs)
    2. CurrencyCarryStrategy       — FX carry (long high-yield / short low-yield)
    3. GlobalMacroMomentum         — Global macro trend following (AQR / Man Group style)
    4. RiskParityCrossAsset        — Cross-asset risk parity (Bridgewater All Weather)
    5. YieldCurveStrategy          — Yield curve strategy (TLT/SHY slope signal)
    6. CommodityCrossMomentum      — Commodity cross-sectional momentum (9 commodity ETFs)
    7. FXVolTargetStrategy         — FX volatility targeting (carry in low vol / momentum in high vol)
    8. GlobalValueStrategy         — Global value (P/SMA_200 valuation ranking)

Plus:
    - run_forex_futures_backtest()       — single-strategy backtest
    - run_all_forex_futures_backtests()  — aggregate backtest over all strategies

Available ETF proxies:
    FX: UUP (dollar index), FXE (EUR), FXY (JPY), FXB (GBP), FXA (AUD), FXC (CAD)
    Commodity futures: USO (crude), GLD (gold), SLV (silver), UNG (natural gas), DBC (broad),
              WEAT (wheat), CORN (corn), SOYB (soybeans)
    Bond futures: TLT (20y), IEF (7-10y), SHY (1-3y), TIP (TIPS), HYG (high yield)
    Equity index futures: SPY, QQQ, IWM, EFA (international developed), EEM (emerging), FXI (China), EWJ (Japan)

References:
    - Menkhoff et al. (2012) "Currency Momentum Strategies"
    - Asness et al. (2013) "Value and Momentum Everywhere"
    - AQR "Time Series Momentum" (Moskowitz, Ooi, Pedersen 2012)
    - Bridgewater "All Weather" risk parity framework
"""

import sys
import warnings
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# =====================================================================
# 全部所需ETF代码
# =====================================================================
FOREX_ETFS = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC']
COMMODITY_FUTURES_ETFS = ['USO', 'GLD', 'SLV', 'UNG', 'DBC', 'WEAT', 'CORN', 'SOYB', 'COPX']
BOND_FUTURES_ETFS = ['TLT', 'IEF', 'SHY', 'TIP', 'HYG']
EQUITY_INDEX_ETFS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'FXI', 'EWJ']

ALL_FOREX_FUTURES_ETFS = sorted(set(
    FOREX_ETFS + COMMODITY_FUTURES_ETFS + BOND_FUTURES_ETFS + EQUITY_INDEX_ETFS
))


# =====================================================================
# 辅助函数
# =====================================================================

def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window, min_periods=window).mean()


def _momentum(series: pd.Series, window: int) -> pd.Series:
    """价格动量 (收益率)"""
    return series.pct_change(window)


def _realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """已实现波动率 (年化)"""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def _rank_cross_section(row: pd.Series) -> pd.Series:
    """横截面百分位排名 (0~1)"""
    valid = row.dropna()
    if len(valid) == 0:
        return row * 0.0
    ranked = valid.rank(pct=True)
    result = row.copy() * 0.0
    result[ranked.index] = ranked
    return result


# =====================================================================
# 基类
# =====================================================================

class ForexFuturesStrategyBase(ABC):
    """Base class for FX / futures ETF strategies"""

    name: str = "未命名策略"
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
# 策略1: 外汇动量策略
# =====================================================================

class CurrencyMomentumStrategy(ForexFuturesStrategyBase):
    """FX cross-sectional momentum strategy

    Logic:
        - Compute 20-day momentum for 6 currency ETFs
        - Long the top 2 ranked currencies, short the bottom 2
        - Equal weighting, rebalanced daily
        - FX momentum is well supported in the literature (Menkhoff et al. 2012)

    Rationale:
        Momentum in FX markets stems from central-bank policy inertia,
        persistence of capital flows and investors' gradual reaction to macro data
    """

    name = "外汇动量"
    description = "Currency cross-sectional momentum: long top 2, short bottom 2"

    TICKERS = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC']

    def __init__(self, mom_window: int = 20, n_long: int = 2, n_short: int = 2):
        """
        参数:
            mom_window: 动量计算窗口 (默认20日)
            n_long: 做多数量 (默认2)
            n_short: 做空数量 (默认2)
        """
        self.mom_window = mom_window
        self.n_long = n_long
        self.n_short = n_short

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate FX momentum signals"""
        available = [t for t in self.TICKERS if t in data.columns]
        if len(available) < self.n_long + self.n_short:
            return pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 计算各货币ETF的动量
        mom = pd.DataFrame({t: _momentum(data[t], self.mom_window) for t in available})

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        for i in range(len(mom)):
            row = mom.iloc[i].dropna()
            if len(row) < self.n_long + self.n_short:
                continue
            ranked = row.sort_values()
            # 做空动量最弱的
            shorts = ranked.index[:self.n_short]
            # 做多动量最强的
            longs = ranked.index[-self.n_long:]

            weight_long = 1.0 / self.n_long
            weight_short = -1.0 / self.n_short
            for t in longs:
                signals.iloc[i][t] = weight_long
            for t in shorts:
                signals.iloc[i][t] = weight_short

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'mom_window': self.mom_window,
            'n_long': self.n_long,
            'n_short': self.n_short,
        }


# =====================================================================
# 策略2: 外汇套息策略
# =====================================================================

class CurrencyCarryStrategy(ForexFuturesStrategyBase):
    """FX carry proxy strategy

    Logic:
        - Use 60-day return as a proxy for the interest-rate differential
          (high-yield currency ETFs tend to appreciate -> positive 60-day return)
        - Long the top 2 "high-yield", short the bottom 2 "low-yield"
        - An ETF implementation of the classic carry trade

    Rationale:
        Interest-rate parity is violated systematically over short horizons;
        appreciation of high-yield currencies does not fully offset the rate
        differential, which is the source of carry-trade excess return
    """

    name = "外汇套息"
    description = "Currency carry proxy: 60d return as yield proxy"

    TICKERS = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC']

    def __init__(self, carry_window: int = 60, n_long: int = 2, n_short: int = 2):
        """
        参数:
            carry_window: 利差代理窗口 (默认60日)
            n_long: 做多高息货币数量 (默认2)
            n_short: 做空低息货币数量 (默认2)
        """
        self.carry_window = carry_window
        self.n_long = n_long
        self.n_short = n_short

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate FX carry signals"""
        available = [t for t in self.TICKERS if t in data.columns]
        if len(available) < self.n_long + self.n_short:
            return pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 60日收益作为carry代理
        carry = pd.DataFrame({t: _momentum(data[t], self.carry_window) for t in available})

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        for i in range(len(carry)):
            row = carry.iloc[i].dropna()
            if len(row) < self.n_long + self.n_short:
                continue
            ranked = row.sort_values()
            # 做空低息 (carry最低)
            shorts = ranked.index[:self.n_short]
            # 做多高息 (carry最高)
            longs = ranked.index[-self.n_long:]

            weight_long = 1.0 / self.n_long
            weight_short = -1.0 / self.n_short
            for t in longs:
                signals.iloc[i][t] = weight_long
            for t in shorts:
                signals.iloc[i][t] = weight_short

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'carry_window': self.carry_window,
            'n_long': self.n_long,
            'n_short': self.n_short,
        }


# =====================================================================
# 策略3: 全球宏观趋势跟踪
# =====================================================================

class GlobalMacroMomentum(ForexFuturesStrategyBase):
    """Global macro trend-following strategy

    Logic:
        - Four asset classes:
          Equities (SPY, QQQ, EFA, EEM)
          Bonds (TLT, IEF, TIP)
          Commodities (GLD, USO, DBC)
          Currencies (UUP)
        - Price > 50-day moving average -> long, otherwise short
        - Inverse-volatility weighting (equal risk contribution)
        - Time-series momentum in the style of AQR / Man Group

    Rationale:
        Time-series momentum (TSMOM) is one of the most robust factors;
        cross-asset allocation sharply reduces reliance on any single asset,
        and inverse-volatility weighting delivers risk parity
    """

    name = "全球宏观趋势"
    description = "Global macro trend following: 50d SMA + inverse-vol weighting"

    EQUITY = ['SPY', 'QQQ', 'EFA', 'EEM']
    BONDS = ['TLT', 'IEF', 'TIP']
    COMMODITIES = ['GLD', 'USO', 'DBC']
    CURRENCIES = ['UUP']
    ALL_TICKERS = EQUITY + BONDS + COMMODITIES + CURRENCIES

    def __init__(self, sma_window: int = 50, vol_window: int = 20):
        """
        参数:
            sma_window: 趋势判断均线周期 (默认50日)
            vol_window: 波动率估算窗口 (默认20日)
        """
        self.sma_window = sma_window
        self.vol_window = vol_window

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate global macro trend signals"""
        available = [t for t in self.ALL_TICKERS if t in data.columns]
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if len(available) == 0:
            return signals

        returns = data[available].pct_change()

        for t in available:
            sma = _sma(data[t], self.sma_window)
            vol = _realized_vol(returns[t], self.vol_window)

            # 趋势方向: 价格 > SMA → +1, 否则 -1
            direction = pd.Series(0.0, index=data.index)
            direction[data[t] > sma] = 1.0
            direction[data[t] <= sma] = -1.0

            # 逆波动率加权: 1/vol, 上限归一化
            inv_vol = (1.0 / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)

            signals[t] = direction * inv_vol

        # 归一化: 使总杠杆约为1x
        abs_sum = signals[available].abs().sum(axis=1).replace(0, 1.0)
        for t in available:
            signals[t] = signals[t] / abs_sum

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'sma_window': self.sma_window,
            'vol_window': self.vol_window,
        }


# =====================================================================
# 策略4: 跨资产风险平价
# =====================================================================

class RiskParityCrossAsset(ForexFuturesStrategyBase):
    """Cross-asset risk parity strategy

    Logic:
        - Four asset buckets, each taking 25% of the risk budget:
          Equities: SPY, QQQ
          Bonds: TLT, IEF
          Commodities: GLD, DBC
          Currencies: UUP, FXE
        - Inverse-volatility weighting within each bucket
        - Monthly rebalance
        - Targets 10% annualized portfolio volatility
        - The classic Bridgewater All Weather framework

    Rationale:
        A traditional 60/40 portfolio actually draws over 90% of its risk from equities;
        risk parity makes every asset class contribute the same amount of risk, so some
        asset performs well in each macro environment
    """

    name = "风险平价"
    description = "Risk parity: inverse-vol across 4 asset classes, target 10% vol"

    BUCKETS = {
        'equity': ['SPY', 'QQQ'],
        'bonds': ['TLT', 'IEF'],
        'commodities': ['GLD', 'DBC'],
        'currencies': ['UUP', 'FXE'],
    }
    BUCKET_WEIGHT = 0.25  # 每桶25%风险预算

    def __init__(self, vol_window: int = 60, target_vol: float = 0.10,
                 rebal_freq: int = 21):
        """
        参数:
            vol_window: 波动率估算窗口 (默认60日)
            target_vol: 目标年化组合波动率 (默认10%)
            rebal_freq: 再平衡频率 (默认21个交易日, 约1月)
        """
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.rebal_freq = rebal_freq

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate risk parity signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)
        returns = data.pct_change()

        # 计算各资产波动率
        vols = pd.DataFrame(index=data.index)
        all_tickers = []
        for bucket_name, tickers in self.BUCKETS.items():
            avail = [t for t in tickers if t in data.columns]
            all_tickers.extend(avail)
            for t in avail:
                vols[t] = _realized_vol(returns[t], self.vol_window)

        if len(all_tickers) == 0:
            return signals

        # 月度再平衡: 只在特定日期计算权重
        rebal_dates = data.index[::self.rebal_freq]
        current_weights = pd.Series(0.0, index=data.columns)

        for date in data.index:
            if date in rebal_dates:
                # 重新计算权重
                new_weights = pd.Series(0.0, index=data.columns)

                for bucket_name, tickers in self.BUCKETS.items():
                    avail = [t for t in tickers if t in data.columns]
                    if len(avail) == 0:
                        continue

                    # 桶内逆波动率加权
                    bucket_vols = pd.Series({
                        t: vols[t].loc[date] if date in vols.index and not np.isnan(vols[t].loc[date]) else np.nan
                        for t in avail
                    }).dropna()

                    if len(bucket_vols) == 0 or (bucket_vols <= 0).all():
                        # 等权替代
                        for t in avail:
                            new_weights[t] = self.BUCKET_WEIGHT / len(avail)
                    else:
                        inv_vol = 1.0 / bucket_vols
                        inv_vol_sum = inv_vol.sum()
                        for t in bucket_vols.index:
                            new_weights[t] = self.BUCKET_WEIGHT * (inv_vol[t] / inv_vol_sum)

                current_weights = new_weights

            signals.loc[date] = current_weights

        # 目标波动率缩放
        port_returns = (signals.shift(1) * returns).sum(axis=1)
        realized = port_returns.rolling(self.vol_window, min_periods=20).std() * np.sqrt(252)
        vol_scale = (self.target_vol / realized).clip(0.2, 3.0).fillna(1.0)

        for t in all_tickers:
            signals[t] = signals[t] * vol_scale

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'vol_window': self.vol_window,
            'target_vol': self.target_vol,
            'rebal_freq': self.rebal_freq,
        }


# =====================================================================
# 策略5: 收益率曲线策略
# =====================================================================

class YieldCurveStrategy(ForexFuturesStrategyBase):
    """Yield curve slope strategy

    Logic:
        - The TLT/SHY ratio captures the slope of the yield curve
        - Rising ratio (curve steepening): expansion signal -> risk-on,
          long SPY + QQQ
        - Falling ratio (flattening/inversion): recession signal -> risk-off,
          long TLT + GLD (safe-haven assets)
        - Direction determined by the 20-day rate of change

    Rationale:
        The yield curve is one of the strongest macro leading indicators;
        inversion has preceded every recession of the past ~50 years,
        and it is observable in real time and hard to arbitrage away
    """

    name = "收益率曲线"
    description = "Yield curve slope: TLT/SHY ratio drives risk-on/risk-off"

    def __init__(self, lookback: int = 20, sma_window: int = 50):
        """
        参数:
            lookback: 曲线变化率计算窗口 (默认20日)
            sma_window: 趋势确认均线 (默认50日)
        """
        self.lookback = lookback
        self.sma_window = sma_window

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate yield curve signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if 'TLT' not in data.columns or 'SHY' not in data.columns:
            return signals

        # TLT/SHY比率 (反映曲线斜率)
        ratio = data['TLT'] / data['SHY']
        ratio_mom = _momentum(ratio, self.lookback)
        ratio_sma = _sma(ratio, self.sma_window)

        # 曲线陡峭化: ratio上升且高于均线 → risk-on
        steepening = (ratio_mom > 0) & (ratio > ratio_sma)
        # 曲线平坦化: ratio下降且低于均线 → risk-off
        flattening = (ratio_mom < 0) & (ratio < ratio_sma)

        # Risk-on配置: 股票
        risk_on_tickers = ['SPY', 'QQQ']
        risk_on_avail = [t for t in risk_on_tickers if t in data.columns]

        # Risk-off配置: 债券 + 黄金
        risk_off_tickers = ['TLT', 'GLD']
        risk_off_avail = [t for t in risk_off_tickers if t in data.columns]

        if len(risk_on_avail) > 0:
            w = 1.0 / len(risk_on_avail)
            for t in risk_on_avail:
                signals.loc[steepening, t] = w

        if len(risk_off_avail) > 0:
            w = 1.0 / len(risk_off_avail)
            for t in risk_off_avail:
                signals.loc[flattening, t] = w

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'sma_window': self.sma_window,
        }


# =====================================================================
# 策略6: 商品横截面动量
# =====================================================================

class CommodityCrossMomentum(ForexFuturesStrategyBase):
    """Commodity cross-sectional momentum strategy

    Logic:
        - Compute 20-day momentum for 9 commodity ETFs:
          GLD, SLV, USO, UNG, WEAT, CORN, SOYB, DBC, COPX
        - Long the top 3, short the bottom 3
        - Equal weighting
        - Commodity momentum is a pronounced effect (Asness et al. 2013)

    Rationale:
        Commodity supply-demand fundamentals shift slowly, so trends persist
        through inventory cycles and lagged capacity adjustment; cross-sectional
        momentum is more robust than time-series momentum in commodities
    """

    name = "商品横截面动量"
    description = "Commodity cross-sectional momentum: long top 3, short bottom 3"

    TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'WEAT', 'CORN', 'SOYB', 'DBC', 'COPX']

    def __init__(self, mom_window: int = 20, n_long: int = 3, n_short: int = 3):
        """
        参数:
            mom_window: 动量计算窗口 (默认20日)
            n_long: 做多数量 (默认3)
            n_short: 做空数量 (默认3)
        """
        self.mom_window = mom_window
        self.n_long = n_long
        self.n_short = n_short

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate commodity momentum signals"""
        available = [t for t in self.TICKERS if t in data.columns]
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if len(available) < self.n_long + self.n_short:
            return signals

        mom = pd.DataFrame({t: _momentum(data[t], self.mom_window) for t in available})

        for i in range(len(mom)):
            row = mom.iloc[i].dropna()
            if len(row) < self.n_long + self.n_short:
                continue
            ranked = row.sort_values()
            shorts = ranked.index[:self.n_short]
            longs = ranked.index[-self.n_long:]

            weight_long = 1.0 / self.n_long
            weight_short = -1.0 / self.n_short
            for t in longs:
                signals.iloc[i][t] = weight_long
            for t in shorts:
                signals.iloc[i][t] = weight_short

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'mom_window': self.mom_window,
            'n_long': self.n_long,
            'n_short': self.n_short,
        }


# =====================================================================
# 策略7: 外汇波动率目标策略
# =====================================================================

class FXVolTargetStrategy(ForexFuturesStrategyBase):
    """FX volatility-targeting strategy

    Logic:
        - Trades UUP (dollar index ETF)
        - Volatility regime: 20-day realized volatility vs 60-day average volatility
        - Low-vol regime (20d vol < 60d avg): carry mode
          -> short UUP (equivalent to long high-yield non-USD currencies)
        - High-vol regime (20d vol > 60d avg): momentum mode
          -> trend-follow UUP (long above the 20-day SMA, otherwise short)
        - Volatility-scaled position sizing

    Rationale:
        Carry trades perform best in low volatility (low VIX -> high risk appetite),
        while momentum/trend strategies dominate in high volatility (panic-driven flows);
        switching adaptively avoids the tail risk of running a single strategy
    """

    name = "外汇波动率目标"
    description = "FX vol target: carry in low vol, momentum in high vol"

    def __init__(self, fast_vol: int = 20, slow_vol: int = 60,
                 sma_window: int = 20, target_vol: float = 0.10):
        """
        参数:
            fast_vol: 快速波动率窗口 (默认20日)
            slow_vol: 慢速波动率窗口 (默认60日)
            sma_window: 趋势判断SMA (默认20日)
            target_vol: 目标波动率 (默认10%)
        """
        self.fast_vol = fast_vol
        self.slow_vol = slow_vol
        self.sma_window = sma_window
        self.target_vol = target_vol

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate FX volatility-targeting signals"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if 'UUP' not in data.columns:
            return signals

        uup = data['UUP']
        rets = uup.pct_change()

        # 波动率体制
        fast_v = _realized_vol(rets, self.fast_vol)
        slow_v = _realized_vol(rets, self.slow_vol)
        low_vol = fast_v < slow_v  # 低波动率环境

        # SMA趋势
        sma = _sma(uup, self.sma_window)
        trend_up = uup > sma

        # 仓位方向
        position = pd.Series(0.0, index=data.index)

        # 低波 → 套息: 做空美元 (做空UUP)
        position[low_vol] = -1.0

        # 高波 → 动量: 跟踪UUP趋势
        high_vol = ~low_vol & fast_v.notna() & slow_v.notna()
        position[high_vol & trend_up] = 1.0
        position[high_vol & ~trend_up] = -1.0

        # 波动率缩放
        vol_scale = (self.target_vol / fast_v).clip(0.2, 3.0).fillna(0.0)
        signals['UUP'] = position * vol_scale

        # 对冲腿: 在carry模式下同时做多高息货币ETF
        carry_tickers = ['FXA', 'FXB']  # 澳元英镑通常是高息
        carry_avail = [t for t in carry_tickers if t in data.columns]
        if len(carry_avail) > 0:
            carry_weight = 0.3 / len(carry_avail)  # 辅助仓位
            for t in carry_avail:
                signals.loc[low_vol, t] = carry_weight * vol_scale[low_vol]

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'fast_vol': self.fast_vol,
            'slow_vol': self.slow_vol,
            'sma_window': self.sma_window,
            'target_vol': self.target_vol,
        }


# =====================================================================
# 策略8: 全球价值策略
# =====================================================================

class GlobalValueStrategy(ForexFuturesStrategyBase):
    """Global value strategy

    Logic:
        - Use P/SMA_200 as a valuation proxy:
          price / 200-day moving average -> low = cheap, high = expensive
        - Compares five markets: SPY, EFA, EEM, FXI, EWJ
        - Long the 2 cheapest markets, short the 2 most expensive
        - Monthly rebalance (value signals move slowly)

    Rationale:
        Long-horizon mean reversion is one of the most robust regularities in global equities;
        a low P/SMA_200 means the market trades below its long-run mean,
        and the cross-country value factor has a Sharpe of roughly 0.4-0.6
    """

    name = "全球价值"
    description = "Global value: P/SMA200 ratio, long cheapest 2, short most expensive 2"

    TICKERS = ['SPY', 'EFA', 'EEM', 'FXI', 'EWJ']

    def __init__(self, sma_window: int = 200, n_long: int = 2, n_short: int = 2,
                 rebal_freq: int = 21):
        """
        参数:
            sma_window: 估值基准均线 (默认200日)
            n_long: 做多最便宜市场数量 (默认2)
            n_short: 做空最贵市场数量 (默认2)
            rebal_freq: 再平衡频率 (默认21交易日)
        """
        self.sma_window = sma_window
        self.n_long = n_long
        self.n_short = n_short
        self.rebal_freq = rebal_freq

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate global value signals"""
        available = [t for t in self.TICKERS if t in data.columns]
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if len(available) < self.n_long + self.n_short:
            return signals

        # P/SMA_200 比率
        p_sma = pd.DataFrame({
            t: data[t] / _sma(data[t], self.sma_window)
            for t in available
        })

        # 月度再平衡
        rebal_dates = data.index[::self.rebal_freq]
        current_weights = pd.Series(0.0, index=data.columns)

        for date in data.index:
            if date in rebal_dates:
                row = p_sma.loc[date].dropna()
                if len(row) >= self.n_long + self.n_short:
                    current_weights = pd.Series(0.0, index=data.columns)
                    ranked = row.sort_values()

                    # 低P/SMA = 便宜 → 做多
                    longs = ranked.index[:self.n_long]
                    # 高P/SMA = 贵 → 做空
                    shorts = ranked.index[-self.n_short:]

                    for t in longs:
                        current_weights[t] = 1.0 / self.n_long
                    for t in shorts:
                        current_weights[t] = -1.0 / self.n_short

            signals.loc[date] = current_weights

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'sma_window': self.sma_window,
            'n_long': self.n_long,
            'n_short': self.n_short,
            'rebal_freq': self.rebal_freq,
        }


# =====================================================================
# 回测引擎
# =====================================================================

def run_forex_futures_backtest(
    strategy: ForexFuturesStrategyBase,
    data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 5.0,
) -> dict:
    """Run a backtest for a single FX / futures strategy

    Parameters:
        strategy: strategy instance
        data: price data (columns=ETF tickers, index=dates)
        initial_capital: starting capital (default 100k)
        commission_bps: transaction cost (default 5bps)

    Returns:
        dict: equity curve, return series and performance metrics
    """
    # 生成信号
    weights = strategy.generate_signal(data)

    # 每日收益
    returns = data.pct_change()

    # 加权组合收益 (T日信号 → T+1日持仓)
    shifted_weights = weights.shift(1)
    port_returns = (shifted_weights * returns).sum(axis=1)

    # 交易成本
    turnover = shifted_weights.diff().abs().sum(axis=1)
    cost = turnover * commission_bps / 10_000
    port_returns = port_returns - cost

    # 去除NaN
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

    # 交易次数
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


# =====================================================================
# 全策略汇总回测
# =====================================================================

def run_all_forex_futures_backtests(
    start: str = '2018-01-01',
    end: str = '2026-03-28',
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """Run backtests for all 8 FX / futures strategies and print a summary table

    Parameters:
        start: backtest start date
        end: backtest end date
        initial_capital: starting capital

    Returns:
        pd.DataFrame: per-strategy performance summary (Sharpe, CAGR%, MDD%, WR%, Calmar, Trades)
    """
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    import yfinance as yf

    # ---- 下载数据 ----
    print(f"下载外汇/期货ETF数据: {start} → {end}")
    print(f"  代码: {ALL_FOREX_FUTURES_ETFS}")

    raw = yf.download(
        ALL_FOREX_FUTURES_ETFS,
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

    # 去掉全为NaN的列, 前向填充
    data = data.dropna(axis=1, how='all')
    data = data.ffill()

    available = list(data.columns)
    print(f"  获取到 {len(available)} 个ETF: {available}")
    print(f"  数据范围: {data.index[0].strftime('%Y-%m-%d')} → "
          f"{data.index[-1].strftime('%Y-%m-%d')}, {len(data)} 个交易日")
    print()

    # ---- 初始化全部策略 ----
    strategies = [
        CurrencyMomentumStrategy(),
        CurrencyCarryStrategy(),
        GlobalMacroMomentum(),
        RiskParityCrossAsset(),
        YieldCurveStrategy(),
        CommodityCrossMomentum(),
        FXVolTargetStrategy(),
        GlobalValueStrategy(),
    ]

    # ---- 回测 ----
    results = []
    for strat in strategies:
        try:
            res = run_forex_futures_backtest(strat, data, initial_capital)
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
    print("外汇/期货ETF策略回测汇总")
    print("=" * 72)
    print(summary.to_string())
    print("=" * 72)

    return summary


# =====================================================================
# 直接运行
# =====================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    summary = run_all_forex_futures_backtests()
