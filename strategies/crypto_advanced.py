"""高级加密货币交易策略集合 — 基于Alpaca实时数据 + yfinance代理ETF的8种系统性策略

包含策略:
    1. CryptoTrendEnsemble       — 多指标趋势集成 (SMA交叉 + MACD + Donchian突破, 多数投票)
    2. CryptoMeanReversion       — 加密均值回归 (Z-score极端值, BTC基准 + 山寨币偏离)
    3. BTCDominanceStrategy       — BTC统治力轮动 (BTC vs 山寨币相对强弱切换)
    4. CryptoVolatilityHarvest   — 加密波动率收割 (BITO/GBTC隐含波动率溢价)
    5. OnChainProxyStrategy      — 链上数据代理 (COIN/MARA/RIOT作为链上指标替代)
    6. CryptoMomentumCrashFilter — 动量+崩盘过滤器 (20日动量, 单日>15%暴跌则清仓5日)
    7. CrossAssetCryptoRegime    — 跨资产加密体制 (GLD/UUP/TLT宏观信号预测加密方向)
    8. CryptoCalendarStrategy    — 加密日历效应 (周末效应 + 月末再平衡 + 减半周期)

以及:
    - run_crypto_backtest()          — 单策略回测
    - run_all_crypto_backtests()     — 全策略汇总回测

可用数据:
    Alpaca实时加密: BTC/USD, ETH/USD, SOL/USD, DOGE/USD, AVAX/USD, LINK/USD, DOT/USD, ADA/USD
    Alpaca加密订单簿: L2 买卖盘
    yfinance代理ETF: BITO (BTC ETF), GBTC, ETHE, COIN, MARA, RIOT
    yfinance直接: BTC-USD, ETH-USD, SOL-USD, DOGE-USD, AVAX-USD, LINK-USD, DOT-USD, ADA-USD
    宏观资产: GLD, UUP, TLT, SPY, VIX (^VIX)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# 数据代码定义
# =====================================================================

# Alpaca可交易加密货币对
ALPACA_CRYPTO_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD',
    'AVAX/USD', 'LINK/USD', 'DOT/USD', 'ADA/USD',
]

# yfinance加密货币代码 (直接行情)
YF_CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD',
    'AVAX-USD', 'LINK-USD', 'DOT-USD', 'ADA-USD',
]

# yfinance代理ETF/股票
YF_CRYPTO_PROXIES = ['BITO', 'GBTC', 'ETHE', 'COIN', 'MARA', 'RIOT']

# 宏观资产 (跨资产策略)
YF_MACRO_TICKERS = ['GLD', 'UUP', 'TLT', 'SPY']

# 全部需要下载的yfinance代码
ALL_YF_TICKERS = YF_CRYPTO_TICKERS + YF_CRYPTO_PROXIES + YF_MACRO_TICKERS

# 山寨币 (不含BTC)
ALT_TICKERS = [t for t in YF_CRYPTO_TICKERS if t != 'BTC-USD']


# =====================================================================
# 辅助函数
# =====================================================================

def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动Z-score"""
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """已实现波动率 (年化, 加密用365天)"""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(365)


def _momentum(series: pd.Series, window: int) -> pd.Series:
    """价格动量 (收益率)"""
    return series.pct_change(window)


def _macd(series: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD指标

    返回:
        (macd_line, signal_line, histogram)
    """
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _donchian(series: pd.Series, window: int = 20
              ) -> Tuple[pd.Series, pd.Series]:
    """Donchian通道

    返回:
        (upper_band, lower_band)
    """
    upper = series.rolling(window, min_periods=window).max()
    lower = series.rolling(window, min_periods=window).min()
    return upper, lower


def _max_drawdown(equity: pd.Series) -> float:
    """最大回撤 (负值)"""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def _bs_call_price(S: float, K: float, T: float, r: float,
                   sigma: float) -> float:
    """Black-Scholes看涨期权定价 (简化版, 用于估算波动率溢价)"""
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
    except ImportError:
        # 如果scipy不可用, 用简化近似
        def _norm_cdf(x):
            """标准正态CDF的近似"""
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            d = 0.3989422804014327
            p = d * np.exp(-x * x / 2.0) * (
                t * (0.3193815 + t * (-0.3565638 + t * (1.781478
                    + t * (-1.821256 + t * 1.330274))))
            )
            return 1.0 - p if x > 0 else p
        class _Norm:
            cdf = staticmethod(_norm_cdf)
        norm = _Norm()

    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S: float, K: float, T: float, r: float,
                  sigma: float) -> float:
    """Black-Scholes看跌期权定价"""
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
    except ImportError:
        def _norm_cdf(x):
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            d = 0.3989422804014327
            p = d * np.exp(-x * x / 2.0) * (
                t * (0.3193815 + t * (-0.3565638 + t * (1.781478
                    + t * (-1.821256 + t * 1.330274))))
            )
            return 1.0 - p if x > 0 else p
        class _Norm:
            cdf = staticmethod(_norm_cdf)
        norm = _Norm()

    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =====================================================================
# 基类
# =====================================================================

class CryptoStrategyBase(ABC):
    """加密货币策略基类"""

    name: str = "未命名加密策略"
    description: str = ""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。

        参数:
            data: pd.DataFrame, columns = 代码, index = 日期, values = 收盘价

        返回:
            pd.DataFrame: columns = 代码, index = 日期, values = 持仓权重
                          正值 = 做多, 负值 = 做空, 0 = 空仓
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """返回策略参数"""
        return {'name': self.name, 'description': self.description}


# =====================================================================
# 策略1: 多指标趋势集成
# =====================================================================

class CryptoTrendEnsemble(CryptoStrategyBase):
    """多指标趋势集成策略

    逻辑:
        - 对每个加密货币同时计算3种趋势信号:
          1) SMA交叉: 10日/20日/50日 短中长期均线方向
          2) MACD: 快慢线交叉 + 柱状图方向
          3) Donchian突破: 价格突破20日高点/低点
        - 多数投票: 2/3以上看多 → 做多; 2/3以上看空 → 做空
        - 在所有加密货币中选出信号最强的top_n, 等权配置

    特点: 多信号集成减少假突破, 加密趋势性强时效果好
    """

    name = "加密趋势集成"
    description = "Multi-indicator trend ensemble: SMA + MACD + Donchian, majority vote"

    def __init__(self,
                 sma_fast: int = 10,
                 sma_mid: int = 20,
                 sma_slow: int = 50,
                 donchian_window: int = 20,
                 top_n: int = 4):
        """
        参数:
            sma_fast: 快速均线周期
            sma_mid: 中速均线周期
            sma_slow: 慢速均线周期
            donchian_window: Donchian通道周期
            top_n: 选取前N个最强信号的币种
        """
        self.sma_fast = sma_fast
        self.sma_mid = sma_mid
        self.sma_slow = sma_slow
        self.donchian_window = donchian_window
        self.top_n = top_n

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成趋势集成信号"""
        crypto_cols = [c for c in data.columns if c in YF_CRYPTO_TICKERS]
        if not crypto_cols:
            crypto_cols = [c for c in data.columns
                          if c.endswith('-USD') and c != 'SPY']

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)
        vote_scores = pd.DataFrame(0.0, index=data.index, columns=crypto_cols)

        for coin in crypto_cols:
            price = data[coin]
            if price.isna().all():
                continue

            # --- 信号1: SMA交叉 ---
            sma_f = _sma(price, self.sma_fast)
            sma_m = _sma(price, self.sma_mid)
            sma_s = _sma(price, self.sma_slow)
            # 快>中>慢 = 看多; 快<中<慢 = 看空
            sma_bull = ((sma_f > sma_m) & (sma_m > sma_s)).astype(float)
            sma_bear = ((sma_f < sma_m) & (sma_m < sma_s)).astype(float)
            sma_signal = sma_bull - sma_bear  # +1, 0, -1

            # --- 信号2: MACD ---
            macd_line, sig_line, hist = _macd(price)
            macd_signal = pd.Series(0.0, index=data.index)
            macd_signal[hist > 0] = 1.0
            macd_signal[hist < 0] = -1.0

            # --- 信号3: Donchian突破 ---
            upper, lower = _donchian(price, self.donchian_window)
            donchian_signal = pd.Series(0.0, index=data.index)
            donchian_signal[price >= upper] = 1.0
            donchian_signal[price <= lower] = -1.0

            # --- 多数投票 ---
            total_vote = sma_signal + macd_signal + donchian_signal
            vote_scores[coin] = total_vote

        # 选择信号最强的top_n币种
        for date in data.index:
            row = vote_scores.loc[date].dropna()
            if row.empty:
                continue

            # 看多的币: vote >= 2
            bullish = row[row >= 2].nlargest(self.top_n)
            # 看空的币: vote <= -2
            bearish = row[row <= -2].nsmallest(self.top_n)

            n_positions = len(bullish) + len(bearish)
            if n_positions == 0:
                continue

            weight = 1.0 / max(n_positions, 1)
            for coin in bullish.index:
                signals.loc[date, coin] = weight
            for coin in bearish.index:
                signals.loc[date, coin] = -weight

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'sma_fast': self.sma_fast,
            'sma_mid': self.sma_mid,
            'sma_slow': self.sma_slow,
            'donchian_window': self.donchian_window,
            'top_n': self.top_n,
        }


# =====================================================================
# 策略2: 加密均值回归
# =====================================================================

class CryptoMeanReversion(CryptoStrategyBase):
    """加密均值回归策略

    逻辑:
        - 加密货币有显著的3-5日短期反转效应
        - BTC信号: BTC相对20日SMA的Z-score
        - 山寨币信号: 山寨币相对BTC趋势的Z-score偏离
        - 进场: Z < -2 买入, Z > +2 卖出
        - 利用加密市场过度反应的特性赚取反转收益

    风险: 趋势行情中可能亏损, 需严格止损
    """

    name = "加密均值回归"
    description = "Crypto mean reversion: Z-score extremes vs 20d SMA, alt vs BTC"

    def __init__(self,
                 z_window: int = 20,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5,
                 reversion_days: int = 5,
                 btc_weight: float = 0.5,
                 alt_weight: float = 0.5):
        """
        参数:
            z_window: Z-score计算窗口
            entry_z: 进场Z-score阈值
            exit_z: 出场Z-score阈值
            reversion_days: 预期回归天数
            btc_weight: BTC信号权重
            alt_weight: 山寨币信号权重
        """
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.reversion_days = reversion_days
        self.btc_weight = btc_weight
        self.alt_weight = alt_weight

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成均值回归信号"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        btc_col = 'BTC-USD' if 'BTC-USD' in data.columns else None
        if btc_col is None:
            # 尝试其他可能的BTC列名
            for c in data.columns:
                if 'BTC' in c.upper():
                    btc_col = c
                    break
        if btc_col is None:
            logger.warning("未找到BTC数据, 均值回归策略无法运行")
            return signals

        btc = data[btc_col]
        btc_z = _zscore(btc, self.z_window)

        # --- BTC均值回归信号 ---
        btc_signal = pd.Series(0.0, index=data.index)
        btc_signal[btc_z < -self.entry_z] = 1.0    # 超卖买入
        btc_signal[btc_z > self.entry_z] = -1.0     # 超买卖出
        # 回到中性区域平仓
        btc_signal[(btc_z > -self.exit_z) & (btc_z < self.exit_z)
                   & (btc_signal.shift(1) != 0)] = 0.0

        signals[btc_col] = btc_signal * self.btc_weight

        # --- 山寨币相对BTC的均值回归 ---
        btc_return = btc.pct_change()
        alt_cols = [c for c in data.columns
                    if c in ALT_TICKERS or
                    (c.endswith('-USD') and c != btc_col and c not in YF_MACRO_TICKERS)]

        if alt_cols:
            per_alt_weight = self.alt_weight / len(alt_cols)

            for alt_col in alt_cols:
                alt = data[alt_col]
                if alt.isna().all():
                    continue

                # 山寨币相对BTC的超额收益Z-score
                alt_return = alt.pct_change()
                relative_return = alt_return - btc_return
                rel_z = _zscore(relative_return.cumsum(), self.z_window)

                alt_signal = pd.Series(0.0, index=data.index)
                alt_signal[rel_z < -self.entry_z] = 1.0     # 山寨币相对BTC超卖
                alt_signal[rel_z > self.entry_z] = -1.0      # 山寨币相对BTC超买

                signals[alt_col] = alt_signal * per_alt_weight

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'z_window': self.z_window,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'reversion_days': self.reversion_days,
        }


# =====================================================================
# 策略3: BTC统治力轮动
# =====================================================================

class BTCDominanceStrategy(CryptoStrategyBase):
    """BTC统治力轮动策略

    逻辑:
        - BTC统治力 = BTC市值 / 加密总市值
        - 无法直接获取, 用代理: BTC vs 等权山寨币篮子的相对表现
        - BTC跑赢山寨 (统治力上升): 做多BTC, 做空山寨币
        - BTC跑输山寨 (统治力下降): 做多山寨币, 做空BTC
        - 经典的加密货币轮动信号, 山寨季(alt season)时特别有效

    信号构建:
        - 20日滚动BTC超额收益 vs 山寨币等权篮子
        - 趋势方向: BTC超额收益的10日SMA方向
    """

    name = "BTC统治力轮动"
    description = "BTC dominance rotation: BTC vs alt basket relative strength"

    def __init__(self,
                 lookback: int = 20,
                 sma_window: int = 10,
                 top_alts: int = 4,
                 btc_alloc: float = 0.5,
                 alt_alloc: float = 0.5):
        """
        参数:
            lookback: 相对表现计算窗口
            sma_window: 趋势平滑窗口
            top_alts: 山寨币篮子中选取的数量
            btc_alloc: BTC仓位配置比例
            alt_alloc: 山寨币总仓位配置比例
        """
        self.lookback = lookback
        self.sma_window = sma_window
        self.top_alts = top_alts
        self.btc_alloc = btc_alloc
        self.alt_alloc = alt_alloc

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成BTC统治力信号"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        btc_col = 'BTC-USD' if 'BTC-USD' in data.columns else None
        if btc_col is None:
            for c in data.columns:
                if 'BTC' in c.upper() and c.endswith('-USD'):
                    btc_col = c
                    break
        if btc_col is None:
            return signals

        alt_cols = [c for c in data.columns
                    if c in ALT_TICKERS or
                    (c.endswith('-USD') and c != btc_col
                     and c not in YF_MACRO_TICKERS
                     and c not in YF_CRYPTO_PROXIES)]

        if not alt_cols:
            return signals

        btc_ret = data[btc_col].pct_change()
        alt_basket_ret = data[alt_cols].pct_change().mean(axis=1)

        # BTC超额收益 (相对山寨篮子)
        excess = btc_ret - alt_basket_ret
        excess_cum = excess.rolling(self.lookback, min_periods=self.lookback).sum()

        # 平滑趋势
        trend = _sma(excess_cum, self.sma_window)
        trend_direction = trend.diff()  # >0 = BTC趋强, <0 = 山寨趋强

        # 选择趋势方向最强的山寨币 (按动量排名)
        alt_mom = data[alt_cols].pct_change(self.lookback)

        for i, date in enumerate(data.index):
            if pd.isna(trend_direction.iloc[i]):
                continue

            if trend_direction.iloc[i] > 0:
                # BTC统治力上升: 做多BTC, 做空山寨
                signals.loc[date, btc_col] = self.btc_alloc
                # 选最弱的山寨做空
                mom_row = alt_mom.loc[date].dropna()
                if not mom_row.empty:
                    weakest = mom_row.nsmallest(min(self.top_alts, len(mom_row)))
                    per_alt = self.alt_alloc / len(weakest)
                    for alt in weakest.index:
                        signals.loc[date, alt] = -per_alt

            elif trend_direction.iloc[i] < 0:
                # BTC统治力下降 (山寨季): 做多山寨, 做空BTC
                signals.loc[date, btc_col] = -self.btc_alloc
                # 选最强的山寨做多
                mom_row = alt_mom.loc[date].dropna()
                if not mom_row.empty:
                    strongest = mom_row.nlargest(min(self.top_alts, len(mom_row)))
                    per_alt = self.alt_alloc / len(strongest)
                    for alt in strongest.index:
                        signals.loc[date, alt] = per_alt

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'sma_window': self.sma_window,
            'top_alts': self.top_alts,
        }


# =====================================================================
# 策略4: 加密波动率收割
# =====================================================================

class CryptoVolatilityHarvest(CryptoStrategyBase):
    """加密波动率收割策略

    逻辑:
        - 加密货币的隐含波动率(IV)长期高于已实现波动率(RV)
        - 波动率溢价比股票更大, 卖期权策略理论收益更高
        - 通过BITO/GBTC模拟: 估算隐含波动率, 卖出虚值strangle
        - BS模型估算期权价值, 跟踪"波动率溢价"收益

    实现:
        - 用30日RV作为IV下限估计, IV = RV * 1.3 (加密溢价因子)
        - 模拟卖出10% OTM strangle, 月度到期
        - 目标: 捕获30-40%的波动率溢价
        - 风控: 最大持仓10%, 亏损达到2倍权利金时止损

    注意: 这是策略级别的模拟, 非实际期权交易
    """

    name = "加密波动率收割"
    description = "Crypto vol harvest: sell strangles on BITO/GBTC, capture IV-RV spread"

    def __init__(self,
                 vol_window: int = 30,
                 iv_premium_factor: float = 1.3,
                 otm_pct: float = 0.10,
                 max_position: float = 0.10,
                 stop_multiplier: float = 2.0,
                 days_to_expiry: int = 30,
                 risk_free_rate: float = 0.05):
        """
        参数:
            vol_window: 已实现波动率计算窗口
            iv_premium_factor: IV/RV溢价倍数 (加密约1.3)
            otm_pct: OTM幅度 (10% = 虚值10%)
            max_position: 最大仓位限制 (占总组合10%)
            stop_multiplier: 止损倍数 (2倍权利金)
            days_to_expiry: 期权期限 (天)
            risk_free_rate: 无风险利率
        """
        self.vol_window = vol_window
        self.iv_premium_factor = iv_premium_factor
        self.otm_pct = otm_pct
        self.max_position = max_position
        self.stop_multiplier = stop_multiplier
        self.days_to_expiry = days_to_expiry
        self.risk_free_rate = risk_free_rate

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成波动率收割信号

        模拟strangle卖出策略: 每月卖出虚值call+put, 到期日平仓
        信号以等价delta风险表示为现货仓位
        """
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        # 优先使用BITO, 其次GBTC, 最后BTC-USD
        target_col = None
        for candidate in ['BITO', 'GBTC', 'BTC-USD']:
            if candidate in data.columns and not data[candidate].isna().all():
                target_col = candidate
                break

        if target_col is None:
            logger.warning("未找到BITO/GBTC/BTC-USD数据, 波动率收割策略无法运行")
            return signals

        price = data[target_col]
        returns = price.pct_change()
        rv = _realized_vol(returns, self.vol_window)

        # 估算IV = RV * premium_factor
        iv_est = rv * self.iv_premium_factor

        T = self.days_to_expiry / 365.0
        r = self.risk_free_rate

        # 跟踪持仓状态
        in_position = False
        entry_premium = 0.0
        entry_date_idx = 0
        cumulative_pnl = 0.0

        for i in range(self.vol_window + 1, len(data.index)):
            date = data.index[i]
            S = price.iloc[i]
            sigma = iv_est.iloc[i]

            if pd.isna(S) or pd.isna(sigma) or sigma <= 0:
                continue

            if not in_position:
                # 每月开仓: 卖出strangle
                K_call = S * (1 + self.otm_pct)
                K_put = S * (1 - self.otm_pct)

                call_premium = _bs_call_price(S, K_call, T, r, sigma)
                put_premium = _bs_put_price(S, K_put, T, r, sigma)
                entry_premium = call_premium + put_premium

                if entry_premium > 0:
                    in_position = True
                    entry_date_idx = i
                    cumulative_pnl = 0.0
                    # 卖strangle: 等效于short gamma, 微小short delta
                    # 用现货表示: 小额做空对冲
                    signals.loc[date, target_col] = -self.max_position * 0.5

            else:
                # 持仓中: 检查是否到期或止损
                days_held = i - entry_date_idx

                # 简化PnL: 价格变动对strangle的影响
                price_change_pct = abs(price.iloc[i] / price.iloc[entry_date_idx] - 1)
                # 如果价格波动超过OTM幅度, 开始亏损
                if price_change_pct > self.otm_pct:
                    unrealized_loss = (price_change_pct - self.otm_pct) * S
                else:
                    unrealized_loss = 0.0

                # 时间价值衰减收益 (theta)
                time_decay = entry_premium * (days_held / self.days_to_expiry)
                cumulative_pnl = time_decay - unrealized_loss

                # 止损: 亏损 > 2倍权利金
                if unrealized_loss > self.stop_multiplier * entry_premium:
                    signals.loc[date, target_col] = 0.0  # 平仓
                    in_position = False
                    continue

                # 到期平仓
                if days_held >= self.days_to_expiry:
                    signals.loc[date, target_col] = 0.0
                    in_position = False
                    continue

                # 维持仓位
                signals.loc[date, target_col] = -self.max_position * 0.5

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'vol_window': self.vol_window,
            'iv_premium_factor': self.iv_premium_factor,
            'otm_pct': self.otm_pct,
            'max_position': self.max_position,
            'stop_multiplier': self.stop_multiplier,
        }


# =====================================================================
# 策略5: 链上数据代理
# =====================================================================

class OnChainProxyStrategy(CryptoStrategyBase):
    """链上数据代理策略

    逻辑:
        - 无法直接获取链上数据, 但可通过上市公司股票代理:
          1) COIN (Coinbase): 交易所健康 = 交易量/收入代理
             COIN领先BTC → 看多 (交易所看到资金流入)
          2) MARA/RIOT (矿企): 挖矿盈利能力 = 算力代理
             MARA领先BTC → 看多 (矿工在囤积)
          3) BTC/COIN比率: 加密 vs 基础设施的背离
             比率上升 → 加密强于基础设施, 可能回归

    信号:
        - COIN 20日动量 > BTC 20日动量: 看多BTC (+0.4)
        - MARA 20日动量 > BTC 20日动量: 看多BTC (+0.3)
        - BTC/COIN比率Z-score > 1: 做空BTC, 做多COIN (+0.3)
    """

    name = "链上代理"
    description = "On-chain proxy: COIN/MARA/RIOT as on-chain data substitutes"

    def __init__(self,
                 mom_window: int = 20,
                 ratio_z_window: int = 60,
                 ratio_z_threshold: float = 1.0,
                 coin_weight: float = 0.4,
                 miner_weight: float = 0.3,
                 ratio_weight: float = 0.3):
        """
        参数:
            mom_window: 动量计算窗口
            ratio_z_window: BTC/COIN比率Z-score窗口
            ratio_z_threshold: 比率偏离阈值
            coin_weight: COIN信号权重
            miner_weight: 矿企信号权重
            ratio_weight: 比率信号权重
        """
        self.mom_window = mom_window
        self.ratio_z_window = ratio_z_window
        self.ratio_z_threshold = ratio_z_threshold
        self.coin_weight = coin_weight
        self.miner_weight = miner_weight
        self.ratio_weight = ratio_weight

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成链上代理信号"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        btc_col = 'BTC-USD' if 'BTC-USD' in data.columns else None
        if btc_col is None:
            for c in data.columns:
                if 'BTC' in c.upper() and c.endswith('-USD'):
                    btc_col = c
                    break
        if btc_col is None:
            return signals

        btc_mom = _momentum(data[btc_col], self.mom_window)

        # --- 信号1: COIN领先BTC ---
        if 'COIN' in data.columns:
            coin_mom = _momentum(data['COIN'], self.mom_window)
            coin_leads = coin_mom - btc_mom
            # COIN动量 > BTC动量 → 看多BTC
            btc_signal_1 = pd.Series(0.0, index=data.index)
            btc_signal_1[coin_leads > 0] = self.coin_weight
            btc_signal_1[coin_leads < 0] = -self.coin_weight * 0.5  # 不对称: 看空信号弱一点
            signals[btc_col] += btc_signal_1

        # --- 信号2: MARA/RIOT领先BTC ---
        miner_cols = [c for c in ['MARA', 'RIOT'] if c in data.columns]
        if miner_cols:
            miner_avg_mom = data[miner_cols].apply(
                lambda x: _momentum(x, self.mom_window)
            ).mean(axis=1)
            miner_leads = miner_avg_mom - btc_mom

            btc_signal_2 = pd.Series(0.0, index=data.index)
            btc_signal_2[miner_leads > 0] = self.miner_weight
            btc_signal_2[miner_leads < 0] = -self.miner_weight * 0.5
            signals[btc_col] += btc_signal_2

        # --- 信号3: BTC/COIN比率均值回归 ---
        if 'COIN' in data.columns:
            ratio = data[btc_col] / data['COIN']
            ratio_z = _zscore(ratio, self.ratio_z_window)

            # 比率Z > threshold: BTC相对COIN过高, 做空BTC做多COIN
            # 比率Z < -threshold: BTC相对COIN过低, 做多BTC做空COIN
            ratio_signal_btc = pd.Series(0.0, index=data.index)
            ratio_signal_coin = pd.Series(0.0, index=data.index)

            ratio_signal_btc[ratio_z > self.ratio_z_threshold] = -self.ratio_weight
            ratio_signal_coin[ratio_z > self.ratio_z_threshold] = self.ratio_weight

            ratio_signal_btc[ratio_z < -self.ratio_z_threshold] = self.ratio_weight
            ratio_signal_coin[ratio_z < -self.ratio_z_threshold] = -self.ratio_weight

            signals[btc_col] += ratio_signal_btc
            signals['COIN'] += ratio_signal_coin

        # 限制总权重在[-1, 1]
        signals = signals.clip(-1.0, 1.0)

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'mom_window': self.mom_window,
            'ratio_z_window': self.ratio_z_window,
            'ratio_z_threshold': self.ratio_z_threshold,
        }


# =====================================================================
# 策略6: 动量+崩盘过滤器
# =====================================================================

class CryptoMomentumCrashFilter(CryptoStrategyBase):
    """加密动量+崩盘过滤器策略

    逻辑:
        - 加密货币20日动量非常有效, 但遭遇崩盘时会血亏
        - 解决方案: 加入崩盘过滤器
        - 若任一币种单日跌幅 > 15%, 全组合清仓5日
        - 这个简单过滤器能显著提高Sharpe ratio

    信号:
        - 正常情况: 按20日动量排名做多top_n
        - 崩盘触发: 全部清仓, 等待5个交易日
        - 恢复后: 重新按动量排名入场
    """

    name = "动量崩盘过滤"
    description = "Crypto momentum + crash filter: exit all on >15% single-day drop"

    def __init__(self,
                 mom_window: int = 20,
                 crash_threshold: float = -0.15,
                 cooldown_days: int = 5,
                 top_n: int = 4):
        """
        参数:
            mom_window: 动量计算窗口
            crash_threshold: 崩盘阈值 (单日跌幅, 负数)
            cooldown_days: 崩盘后等待天数
            top_n: 做多排名前N的币种
        """
        self.mom_window = mom_window
        self.crash_threshold = crash_threshold
        self.cooldown_days = cooldown_days
        self.top_n = top_n

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成动量+崩盘过滤信号"""
        crypto_cols = [c for c in data.columns if c in YF_CRYPTO_TICKERS]
        if not crypto_cols:
            crypto_cols = [c for c in data.columns
                          if c.endswith('-USD') and c not in YF_MACRO_TICKERS]

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        if not crypto_cols:
            return signals

        returns = data[crypto_cols].pct_change()
        mom = data[crypto_cols].pct_change(self.mom_window)

        # 崩盘检测: 任一币种单日跌幅超过阈值
        crash_detected = (returns < self.crash_threshold).any(axis=1)

        # 计算冷却期: 崩盘后cooldown_days日内不交易
        in_cooldown = pd.Series(False, index=data.index)
        cooldown_remaining = 0

        for i in range(len(data.index)):
            if crash_detected.iloc[i]:
                cooldown_remaining = self.cooldown_days
                in_cooldown.iloc[i] = True
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1
                in_cooldown.iloc[i] = True
            else:
                in_cooldown.iloc[i] = False

        # 生成动量信号 (非冷却期)
        for i, date in enumerate(data.index):
            if in_cooldown.iloc[i]:
                continue  # 冷却期: 全部归零

            mom_row = mom.loc[date].dropna()
            if mom_row.empty or len(mom_row) < 2:
                continue

            # 做多动量最强的top_n
            top = mom_row.nlargest(min(self.top_n, len(mom_row)))
            weight = 1.0 / len(top)
            for coin in top.index:
                if top[coin] > 0:  # 只做多正动量的
                    signals.loc[date, coin] = weight

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'mom_window': self.mom_window,
            'crash_threshold': self.crash_threshold,
            'cooldown_days': self.cooldown_days,
            'top_n': self.top_n,
        }


# =====================================================================
# 策略7: 跨资产加密体制
# =====================================================================

class CrossAssetCryptoRegime(CryptoStrategyBase):
    """跨资产加密体制识别策略

    逻辑:
        - 用传统资产预测加密货币方向:
        - 黄金上涨 + 美元下跌 = 看多加密 (流动性/通胀对冲叙事)
        - 利率上升(TLT下跌) + 美元上涨 = 看空加密 (紧缩)
        - VIX上升 = 短期看空, 但随后转多 (避险需求→BTC避风港)

    信号权重:
        - GLD信号: 40% (黄金与BTC的"数字黄金"叙事高度相关)
        - UUP信号: 30% (美元强弱是全球流动性指标)
        - TLT信号: 30% (利率方向反映货币政策)
    """

    name = "跨资产加密体制"
    description = "Cross-asset crypto regime: GLD/UUP/TLT macro signals predict crypto"

    def __init__(self,
                 lookback: int = 20,
                 gld_weight: float = 0.40,
                 uup_weight: float = 0.30,
                 tlt_weight: float = 0.30):
        """
        参数:
            lookback: 宏观信号计算窗口
            gld_weight: 黄金信号权重
            uup_weight: 美元信号权重
            tlt_weight: 债券信号权重
        """
        self.lookback = lookback
        self.gld_weight = gld_weight
        self.uup_weight = uup_weight
        self.tlt_weight = tlt_weight

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成跨资产体制信号"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        btc_col = 'BTC-USD' if 'BTC-USD' in data.columns else None
        if btc_col is None:
            for c in data.columns:
                if 'BTC' in c.upper() and c.endswith('-USD'):
                    btc_col = c
                    break
        if btc_col is None:
            return signals

        # 计算宏观资产动量
        composite_signal = pd.Series(0.0, index=data.index)

        # --- GLD信号: 黄金上涨 = 看多加密 ---
        if 'GLD' in data.columns:
            gld_mom = _momentum(data['GLD'], self.lookback)
            gld_signal = pd.Series(0.0, index=data.index)
            gld_signal[gld_mom > 0] = 1.0
            gld_signal[gld_mom < 0] = -1.0
            composite_signal += gld_signal * self.gld_weight

        # --- UUP信号: 美元下跌 = 看多加密 (反向) ---
        if 'UUP' in data.columns:
            uup_mom = _momentum(data['UUP'], self.lookback)
            uup_signal = pd.Series(0.0, index=data.index)
            uup_signal[uup_mom > 0] = -1.0   # 美元强 → 看空加密
            uup_signal[uup_mom < 0] = 1.0     # 美元弱 → 看多加密
            composite_signal += uup_signal * self.uup_weight

        # --- TLT信号: 债券上涨(利率下降) = 看多加密 ---
        if 'TLT' in data.columns:
            tlt_mom = _momentum(data['TLT'], self.lookback)
            tlt_signal = pd.Series(0.0, index=data.index)
            tlt_signal[tlt_mom > 0] = 1.0     # 利率下降 → 看多加密
            tlt_signal[tlt_mom < 0] = -1.0     # 利率上升 → 看空加密
            composite_signal += tlt_signal * self.tlt_weight

        # 对BTC和主要山寨币应用信号
        crypto_cols = [c for c in data.columns if c in YF_CRYPTO_TICKERS]
        if not crypto_cols:
            crypto_cols = [c for c in data.columns
                          if c.endswith('-USD') and c not in YF_MACRO_TICKERS]

        if crypto_cols:
            per_coin_weight = 1.0 / len(crypto_cols)
            for coin in crypto_cols:
                signals[coin] = composite_signal * per_coin_weight

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'lookback': self.lookback,
            'gld_weight': self.gld_weight,
            'uup_weight': self.uup_weight,
            'tlt_weight': self.tlt_weight,
        }


# =====================================================================
# 策略8: 加密日历效应
# =====================================================================

class CryptoCalendarStrategy(CryptoStrategyBase):
    """加密日历效应策略

    逻辑:
        - 加密货币有已知的日历效应:
          1) 周末效应: BTC倾向于周日晚间拉升 (散户周末买入)
          2) 月末再平衡: 机构月末卖出压力 (基金再平衡)
          3) 减半周期: 4年周期 (下次约2028年4月)
             代理: 用200周SMA交叉判断长期周期位置

    信号:
        - 周五收盘买入, 周一开盘卖出 (捕获周末效应)
        - 月末最后3日减仓 (避开再平衡卖压)
        - 价格 > 200周SMA: 长期看多 (减半后牛市)
        - 价格 < 200周SMA: 长期看空 (减半前调整)
    """

    name = "加密日历效应"
    description = "Crypto calendar: weekend effect + month-end + halving cycle proxy"

    def __init__(self,
                 use_weekend: bool = True,
                 use_month_end: bool = True,
                 use_halving_cycle: bool = True,
                 sma_weeks: int = 200,
                 month_end_days: int = 3,
                 weekend_weight: float = 0.3,
                 cycle_weight: float = 0.5,
                 base_weight: float = 0.2):
        """
        参数:
            use_weekend: 是否使用周末效应
            use_month_end: 是否使用月末效应
            use_halving_cycle: 是否使用减半周期
            sma_weeks: 周均线周期 (200周)
            month_end_days: 月末减仓天数
            weekend_weight: 周末信号权重
            cycle_weight: 减半周期信号权重
            base_weight: 基础权重
        """
        self.use_weekend = use_weekend
        self.use_month_end = use_month_end
        self.use_halving_cycle = use_halving_cycle
        self.sma_weeks = sma_weeks
        self.month_end_days = month_end_days
        self.weekend_weight = weekend_weight
        self.cycle_weight = cycle_weight
        self.base_weight = base_weight

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成日历效应信号"""
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        btc_col = 'BTC-USD' if 'BTC-USD' in data.columns else None
        if btc_col is None:
            for c in data.columns:
                if 'BTC' in c.upper() and c.endswith('-USD'):
                    btc_col = c
                    break
        if btc_col is None:
            return signals

        crypto_cols = [c for c in data.columns if c in YF_CRYPTO_TICKERS]
        if not crypto_cols:
            crypto_cols = [c for c in data.columns
                          if c.endswith('-USD') and c not in YF_MACRO_TICKERS]

        btc_price = data[btc_col]

        # --- 减半周期: 200周SMA (≈1400个交易日, 用日线近似) ---
        cycle_signal = pd.Series(0.0, index=data.index)
        if self.use_halving_cycle:
            # 200周 ≈ 1400日, 但数据可能不够长, 用min(1400, 可用数据的80%)
            sma_days = min(1400, int(len(data) * 0.8))
            if sma_days >= 200:
                sma_200w = _sma(btc_price, sma_days)
                cycle_signal[btc_price > sma_200w] = 1.0    # 牛市周期
                cycle_signal[btc_price < sma_200w] = -0.5   # 熊市周期 (不对称)

        # --- 周末效应: 周五做多 ---
        weekend_signal = pd.Series(0.0, index=data.index)
        if self.use_weekend:
            # dayofweek: 0=周一, 4=周五
            day_of_week = data.index.dayofweek
            weekend_signal[day_of_week == 4] = 1.0   # 周五买入

        # --- 月末效应: 月末减仓 ---
        month_end_signal = pd.Series(0.0, index=data.index)
        if self.use_month_end:
            # 计算每个日期到月末的天数
            for i, date in enumerate(data.index):
                if hasattr(date, 'month'):
                    # 获取该月最后一天
                    if date.month == 12:
                        last_day = date.replace(year=date.year + 1, month=1, day=1) - timedelta(days=1)
                    else:
                        last_day = date.replace(month=date.month + 1, day=1) - timedelta(days=1)
                    days_to_end = (last_day - date).days
                    if days_to_end <= self.month_end_days:
                        month_end_signal.iloc[i] = -0.5  # 月末减仓

        # --- 合成信号 ---
        composite = (
            cycle_signal * self.cycle_weight
            + weekend_signal * self.weekend_weight
            + month_end_signal * self.base_weight
        )
        # 加上基础多头偏差 (加密长期上涨倾向)
        composite += self.base_weight * 0.5

        composite = composite.clip(-1.0, 1.0)

        if crypto_cols:
            per_coin = 1.0 / len(crypto_cols)
            for coin in crypto_cols:
                signals[coin] = composite * per_coin

        return signals

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            'use_weekend': self.use_weekend,
            'use_month_end': self.use_month_end,
            'use_halving_cycle': self.use_halving_cycle,
            'sma_weeks': self.sma_weeks,
        }


# =====================================================================
# 回测引擎
# =====================================================================

def run_crypto_backtest(
    strategy: CryptoStrategyBase,
    data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 10.0,
) -> dict:
    """单策略回测

    参数:
        strategy: 策略实例
        data: 价格数据 (columns=代码, index=日期, values=收盘价)
        initial_capital: 初始资金
        commission_bps: 交易成本 (基点, 加密默认10bps)

    返回:
        dict: {
            'name': 策略名,
            'equity_curve': pd.Series (日度净值),
            'returns': pd.Series (日度收益率),
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

    # 绩效指标 (加密用365天年化)
    n_years = len(port_returns) / 365
    total_return = equity.iloc[-1] / initial_capital - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = port_returns.std() * np.sqrt(365)
    sharpe = (port_returns.mean() * 365) / ann_vol if ann_vol > 0 else 0.0

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


# =====================================================================
# 全策略回测汇总
# =====================================================================

def run_all_crypto_backtests(
    start: str = '2020-01-01',
    end: str = '2026-03-28',
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """运行全部8个加密策略回测并输出汇总表

    参数:
        start: 回测开始日期
        end: 回测结束日期
        initial_capital: 初始资金

    返回:
        pd.DataFrame: 各策略绩效汇总 (Sharpe, CAGR%, MDD%, WR%, Calmar, Trades)
    """
    import yfinance as yf

    # ---- 下载数据 ----
    print(f"下载加密货币及代理数据: {start} → {end}")
    print(f"  代码: {ALL_YF_TICKERS}")

    raw = yf.download(
        ALL_YF_TICKERS,
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

    # 去掉全部为NaN的列
    data = data.dropna(axis=1, how='all')
    data = data.ffill()  # 前向填充缺失 (周末/节假日等)

    available = list(data.columns)
    print(f"  获取到 {len(available)} 个标的: {available}")
    print(f"  数据范围: {data.index[0].strftime('%Y-%m-%d')} → "
          f"{data.index[-1].strftime('%Y-%m-%d')}, {len(data)} 个交易日")
    print()

    # ---- 初始化全部策略 ----
    strategies = [
        CryptoTrendEnsemble(),
        CryptoMeanReversion(),
        BTCDominanceStrategy(),
        CryptoVolatilityHarvest(),
        OnChainProxyStrategy(),
        CryptoMomentumCrashFilter(),
        CrossAssetCryptoRegime(),
        CryptoCalendarStrategy(),
    ]

    # ---- 回测 ----
    results = []
    for strat in strategies:
        try:
            res = run_crypto_backtest(strat, data, initial_capital)
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
    print("加密货币策略回测汇总")
    print("=" * 72)
    print(summary.to_string())
    print("=" * 72)

    return summary


# =====================================================================
# 直接运行
# =====================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    summary = run_all_crypto_backtests()
