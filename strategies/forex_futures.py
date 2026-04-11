"""外汇与期货ETF代理策略集合 — 基于Alpaca/yfinance可交易ETF的8种系统性策略

包含策略:
    1. CurrencyMomentumStrategy    — 外汇动量 (6种货币ETF横截面动量)
    2. CurrencyCarryStrategy       — 外汇套息 (高息做多/低息做空)
    3. GlobalMacroMomentum         — 全球宏观趋势跟踪 (AQR/Man Group风格)
    4. RiskParityCrossAsset        — 跨资产风险平价 (Bridgewater全天候)
    5. YieldCurveStrategy          — 收益率曲线策略 (TLT/SHY斜率信号)
    6. CommodityCrossMomentum      — 商品横截面动量 (9种商品ETF)
    7. FXVolTargetStrategy         — 外汇波动率目标 (低波套息/高波动量)
    8. GlobalValueStrategy         — 全球价值 (P/SMA_200估值排序)

以及:
    - run_forex_futures_backtest()       — 单策略回测
    - run_all_forex_futures_backtests()  — 全策略汇总回测

可用ETF代理:
    外汇: UUP(美元指数), FXE(欧元), FXY(日元), FXB(英镑), FXA(澳元), FXC(加元)
    商品期货: USO(原油), GLD(黄金), SLV(白银), UNG(天然气), DBC(综合),
              WEAT(小麦), CORN(玉米), SOYB(大豆)
    债券期货: TLT(20年), IEF(7-10年), SHY(1-3年), TIP(TIPS), HYG(高收益)
    股指期货: SPY, QQQ, IWM, EFA(国际发达), EEM(新兴), FXI(中国), EWJ(日本)

参考文献:
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
    """外汇/期货ETF策略基类"""

    name: str = "未命名策略"
    description: str = ""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。

        参数:
            data: pd.DataFrame, columns = ETF代码, index = 日期, values = 收盘价

        返回:
            pd.DataFrame: columns = ETF代码, index = 日期, values = 持仓权重
                          正值 = 做多, 负值 = 做空, 0 = 空仓
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """返回策略参数"""
        return {'name': self.name, 'description': self.description}


# =====================================================================
# 策略1: 外汇动量策略
# =====================================================================

class CurrencyMomentumStrategy(ForexFuturesStrategyBase):
    """外汇横截面动量策略

    逻辑:
        - 计算6种货币ETF的20日动量
        - 做多排名前2的货币, 做空排名后2的货币
        - 等权配置, 每日再平衡
        - 外汇动量效应有充分学术支持 (Menkhoff et al. 2012)

    原理:
        外汇市场的动量效应来源于央行政策惯性、
        资本流动持续性和投资者对宏观数据的渐进反应
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
        """生成外汇动量信号"""
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
    """外汇套息代理策略

    逻辑:
        - 用60日收益作为利差代理
          (高息货币ETF趋向升值 → 60日正收益)
        - 做多"高息"前2, 做空"低息"后2
        - 经典carry trade的ETF实现

    原理:
        利率平价在短期内系统性偏离,
        高息货币的升值幅度不足以抵消利差,
        形成carry trade的超额收益来源
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
        """生成外汇套息信号"""
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
    """全球宏观趋势跟踪策略

    逻辑:
        - 四大资产类别:
          股票 (SPY, QQQ, EFA, EEM)
          债券 (TLT, IEF, TIP)
          商品 (GLD, USO, DBC)
          货币 (UUP)
        - 价格 > 50日均线 → 做多, 否则做空
        - 逆波动率加权 (等风险贡献)
        - AQR / Man Group风格的时序动量策略

    原理:
        时序动量 (TSMOM) 是最稳健的因子之一,
        跨资产配置大幅降低单一资产依赖,
        逆波动率加权实现风险平价
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
        """生成全球宏观趋势信号"""
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
    """跨资产风险平价策略

    逻辑:
        - 四大资产桶各占25%风险预算:
          股票: SPY, QQQ
          债券: TLT, IEF
          商品: GLD, DBC
          货币: UUP, FXE
        - 桶内逆波动率加权
        - 月度再平衡
        - 目标10%年化组合波动率
        - 经典Bridgewater全天候框架

    原理:
        传统60/40组合实际上90%以上风险来自股票,
        风险平价让每类资产贡献相同风险,
        在不同宏观环境中都有资产表现良好
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
        """生成风险平价信号"""
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
    """收益率曲线斜率策略

    逻辑:
        - TLT/SHY比率捕捉收益率曲线斜率
        - 比率上升 (曲线陡峭化): 经济扩张信号 → risk-on
          做多SPY + QQQ
        - 比率下降 (曲线平坦化/倒挂): 衰退信号 → risk-off
          做多TLT + GLD (避险资产)
        - 用20日变化率判断方向

    原理:
        收益率曲线是最强的宏观先行指标之一,
        倒挂预测了近50年每次衰退,
        实时可观测且难以被套利消除
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
        """生成收益率曲线信号"""
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
    """商品横截面动量策略

    逻辑:
        - 计算9种商品ETF的20日动量:
          GLD, SLV, USO, UNG, WEAT, CORN, SOYB, DBC, COPX
        - 做多排名前3, 做空排名后3
        - 等权配置
        - 商品动量效应显著 (Asness et al. 2013)

    原理:
        商品供需基本面变化缓慢,
        趋势持续性来自库存周期和产能调整滞后,
        横截面动量比时序动量在商品中更稳健
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
        """生成商品动量信号"""
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
    """外汇波动率目标策略

    逻辑:
        - 交易UUP (美元指数ETF)
        - 波动率体制判断: 20日实现波动率 vs 60日平均波动率
        - 低波环境 (20d vol < 60d avg): 套息模式
          → 做空UUP (等于做多高息非美货币)
        - 高波环境 (20d vol > 60d avg): 动量模式
          → 趋势跟踪UUP方向 (> 20日SMA做多, 否则做空)
        - 波动率缩放仓位

    原理:
        低波动率时carry trade表现最好 (VIX低 → 风险偏好高),
        高波动率时动量/趋势策略占优 (恐慌驱动资金流),
        自适应切换避免单一策略的尾部风险
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
        """生成外汇波动率目标信号"""
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
    """全球价值策略

    逻辑:
        - 用P/SMA_200作为估值代理:
          价格/200日均线 → 低 = 便宜, 高 = 贵
        - 对比: SPY, EFA, EEM, FXI, EWJ 五个市场
        - 做多最便宜的2个市场, 做空最贵的2个市场
        - 月度再平衡 (价值信号慢变)

    原理:
        长期均值回归是全球股市最稳健的规律之一,
        P/SMA_200低意味着市场处于长期均值下方,
        跨国价值因子的夏普比率约0.4-0.6
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
        """生成全球价值信号"""
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
    """运行单个外汇/期货策略回测

    参数:
        strategy: 策略实例
        data: 价格数据 (columns=ETF代码, index=日期)
        initial_capital: 初始资金 (默认10万)
        commission_bps: 交易成本 (默认5bps)

    返回:
        dict: 含净值曲线、收益序列及绩效指标
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
    """运行全部8个外汇/期货策略回测并输出汇总表

    参数:
        start: 回测开始日期
        end: 回测结束日期
        initial_capital: 初始资金

    返回:
        pd.DataFrame: 各策略绩效汇总 (Sharpe, CAGR%, MDD%, WR%, Calmar, Trades)
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
