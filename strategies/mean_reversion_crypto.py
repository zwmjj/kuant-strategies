"""加密货币布林带均值回归策略 — 波动率自适应 + RSI 过滤

策略说明:
    基于布林带和RSI的加密货币均值回归策略。核心逻辑:
    1. 布林带信号: 价格触及下轨(超卖)做多, 触及上轨(超买)做空
    2. RSI过滤: RSI<30 确认超卖做多, RSI>70 确认超买做空, 双重确认提高胜率
    3. 波动率自适应: 高波动率环境下自动收窄仓位, 降低风险暴露
    4. 信号强度: 偏离布林带中轨越远, 信号越强; 经波动率调整后输出

支持交易对: BTC/USD, ETH/USD, SOL/USD
可通过 run_crypto_backtest() 进行向量化回测。

作者: KuanQuant
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qf.strategy_crypto import (
    CryptoBaseStrategy,
    run_crypto_backtest,
    _build_close_matrix,
    _build_field_matrix,
)


# ═══════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════

# 默认交易对: 高流动性的三大加密货币
DEFAULT_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数: 技术指标计算
# ═══════════════════════════════════════════════════════════════════════

def _compute_bollinger_bands(close: pd.DataFrame, window: int = 20,
                             num_std: float = 2.0):
    """
    计算布林带。

    参数:
        close: 收盘价矩阵 (date x symbol)
        window: SMA窗口期
        num_std: 标准差倍数

    返回:
        (中轨, 上轨, 下轨) — 三个 DataFrame
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算RSI (Relative Strength Index)。

    使用Wilder平滑法 (指数加权移动平均)。

    参数:
        close: 收盘价矩阵 (date x symbol)
        period: RSI周期

    返回:
        RSI矩阵, 值域 [0, 100]
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder平滑: 等价于 alpha = 1/period 的EMA
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_realized_vol(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算已实现波动率 (滚动标准差, 年化)。

    参数:
        close: 收盘价矩阵 (date x symbol)
        window: 滚动窗口

    返回:
        年化波动率矩阵 (基于365天)
    """
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(window).std() * np.sqrt(365)
    return vol


# ═══════════════════════════════════════════════════════════════════════
#  策略类
# ═══════════════════════════════════════════════════════════════════════

class BollingerMeanReversionCrypto(CryptoBaseStrategy):
    """
    布林带均值回归加密货币策略 (波动率自适应 + RSI双重过滤)。

    信号生成逻辑:
        1. 价格 < 布林带下轨 且 RSI < rsi_oversold → 做多
        2. 价格 > 布林带上轨 且 RSI > rsi_overbought → 做空
        3. 信号强度 = |price - mid| / (upper - lower), 归一化到 [0, 1]
        4. 波动率调整: 仓位 = 基础仓位 * (目标波动率 / 当前波动率)
           高波动时自动缩仓, 低波动时放大仓位 (上限为2x)

    参数:
        bb_window: 布林带SMA窗口, 默认20
        bb_std: 布林带标准差倍数, 默认2.0
        rsi_period: RSI周期, 默认14
        rsi_oversold: RSI超卖阈值, 默认30
        rsi_overbought: RSI超买阈值, 默认70
        vol_window: 波动率计算窗口, 默认20
        target_vol: 目标年化波动率, 默认0.5 (50%)
        max_vol_scale: 波动率缩放上限, 默认2.0
    """

    name = "布林带均值回归 (加密货币)"
    description = "布林带+RSI双重确认, 波动率自适应仓位管理"

    def __init__(
        self,
        symbols: list[str] | None = None,
        capital: float = 10_000,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        vol_window: int = 20,
        target_vol: float = 0.50,
        max_vol_scale: float = 2.0,
    ):
        """
        初始化策略参数。

        参数:
            symbols: 交易对列表, 默认 ['BTC/USD', 'ETH/USD', 'SOL/USD']
            capital: 初始资金
            bb_window: 布林带窗口期
            bb_std: 布林带标准差倍数
            rsi_period: RSI计算周期
            rsi_oversold: RSI超卖阈值 (低于此值确认超卖)
            rsi_overbought: RSI超买阈值 (高于此值确认超买)
            vol_window: 已实现波动率计算窗口
            target_vol: 目标年化波动率 (用于仓位缩放)
            max_vol_scale: 波动率缩放因子的最大值
        """
        super().__init__(symbols or DEFAULT_SYMBOLS.copy(), capital)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_vol_scale = max_vol_scale

    # ── 核心信号生成 ────────────────────────────────────────────────────

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """
        生成均值回归信号矩阵。

        流程:
            1. 计算布林带 (中轨/上轨/下轨)
            2. 计算RSI
            3. 价格触及下轨 + RSI<30 → 做多信号
            4. 价格触及上轨 + RSI>70 → 做空信号
            5. 信号强度按偏离程度归一化
            6. 乘以波动率缩放因子, 高波动缩仓

        参数:
            data_dict: 字典, 键为交易对, 值为含 close 列的 DataFrame

        返回:
            pd.DataFrame — 行=日期, 列=交易对, 值=调整后的信号强度
                正值=做多, 负值=做空, 0=空仓
        """
        close = _build_close_matrix(data_dict, self.symbols)

        # ── 技术指标 ───────────────────────────────────────────────
        mid, upper, lower = _compute_bollinger_bands(
            close, self.bb_window, self.bb_std
        )
        rsi = _compute_rsi(close, self.rsi_period)
        realized_vol = _compute_realized_vol(close, self.vol_window)

        # ── 布林带宽度 (用于归一化信号强度) ──────────────────────────
        bb_width = (upper - lower).replace(0, np.nan)

        # ── 基础信号 ──────────────────────────────────────────────
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # 做多条件: 价格 <= 下轨 且 RSI < 超卖阈值
        long_cond = (close <= lower) & (rsi < self.rsi_oversold)
        # 做空条件: 价格 >= 上轨 且 RSI > 超买阈值
        short_cond = (close >= upper) & (rsi > self.rsi_overbought)

        # 信号强度: 偏离中轨的程度, 归一化到 [0, 1]
        deviation = (close - mid).abs() / bb_width
        # 限制在 [0, 1] 范围, 避免极端值
        strength = deviation.clip(0, 1)

        # 赋值信号: 做多为正, 做空为负
        signals[long_cond] = strength[long_cond]
        signals[short_cond] = -strength[short_cond]

        # ── 波动率自适应仓位调整 ──────────────────────────────────
        # vol_scale = target_vol / realized_vol, 上限为 max_vol_scale
        # 高波动 → vol_scale < 1 → 缩仓
        # 低波动 → vol_scale > 1 → 放大 (但不超过上限)
        vol_scale = self.target_vol / realized_vol.replace(0, np.nan)
        vol_scale = vol_scale.clip(upper=self.max_vol_scale).fillna(1.0)

        # 最终信号 = 基础信号 * 波动率缩放
        signals = signals * vol_scale

        return signals

    # ── 参数报告 ────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """返回策略全部参数, 用于日志和报告。"""
        base = super().get_params()
        base.update({
            'bb_window': self.bb_window,
            'bb_std': self.bb_std,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'vol_window': self.vol_window,
            'target_vol': self.target_vol,
            'max_vol_scale': self.max_vol_scale,
        })
        return base


# ═══════════════════════════════════════════════════════════════════════
#  快捷回测入口
# ═══════════════════════════════════════════════════════════════════════

def backtest_mean_reversion(data_dict: dict, **kwargs) -> dict:
    """
    一键回测布林带均值回归策略。

    参数:
        data_dict: 加密货币数据字典 {symbol: DataFrame}
        **kwargs: 传递给 BollingerMeanReversionCrypto 的参数

    返回:
        dict — 含 total_return, cagr, sharpe, max_drawdown, equity_curve 等

    用法示例:
        from qf.strategy_crypto import prepare_crypto_data
        from strategies.mean_reversion_crypto import backtest_mean_reversion

        data = prepare_crypto_data(alpaca_loader, lookback_days=365)
        result = backtest_mean_reversion(data, bb_window=20, rsi_period=14)
        print(f"夏普比率: {result['sharpe']:.2f}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")
    """
    capital = kwargs.pop('capital', 10_000)
    strategy = BollingerMeanReversionCrypto(capital=capital, **kwargs)
    return run_crypto_backtest(strategy, data_dict, initial_capital=capital)


# ═══════════════════════════════════════════════════════════════════════
#  独立回测 (合成数据, 用于验证逻辑)
# ═══════════════════════════════════════════════════════════════════════

def _demo_backtest():
    """
    使用合成价格数据运行策略回测, 验证信号生成和回测流程。

    生成带均值回归特性的模拟价格序列:
    - BTC: 围绕50000振荡, 高波动
    - ETH: 围绕3000振荡, 中波动
    - SOL: 围绕100振荡, 高波动
    """
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range('2025-01-01', periods=n_days, freq='D')

    # 生成带均值回归特征的价格 (Ornstein-Uhlenbeck过程)
    def _ou_process(mean_price, volatility, mean_reversion_speed=0.05, n=365):
        """Ornstein-Uhlenbeck过程模拟均值回归价格序列。"""
        prices = [mean_price]
        for _ in range(n - 1):
            dp = mean_reversion_speed * (mean_price - prices[-1])
            dp += volatility * mean_price * np.random.randn()
            prices.append(max(prices[-1] + dp, mean_price * 0.1))  # 防止负价格
        return np.array(prices)

    # 合成数据
    configs = {
        'BTC/USD': (50_000, 0.03),   # 均价50000, 日波动3%
        'ETH/USD': (3_000, 0.035),   # 均价3000, 日波动3.5%
        'SOL/USD': (100, 0.05),      # 均价100, 日波动5%
    }

    data_dict = {}
    for symbol, (mean_p, vol) in configs.items():
        prices = _ou_process(mean_p, vol, n=n_days)
        # 构造OHLCV DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            'close': prices,
            'volume': np.random.lognormal(20, 1, n_days),
        }, index=dates)
        data_dict[symbol] = df

    # 运行回测
    strategy = BollingerMeanReversionCrypto(
        bb_window=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        target_vol=0.50,
    )

    print("=" * 60)
    print(f"策略: {strategy.name}")
    print(f"参数: {strategy.get_params()}")
    print("=" * 60)

    result = run_crypto_backtest(strategy, data_dict, initial_capital=10_000)

    print(f"\n回测结果 ({n_days}天):")
    print(f"  总收益率:   {result['total_return']:+.2%}")
    print(f"  年化收益率: {result['cagr']:+.2%}")
    print(f"  夏普比率:   {result['sharpe']:.2f}")
    print(f"  最大回撤:   {result['max_drawdown']:.2%}")
    print(f"  回测天数:   {result['n_days']}")

    # 信号统计
    signals = strategy.generate_signal(data_dict)
    print(f"\n信号统计:")
    for sym in strategy.symbols:
        if sym in signals.columns:
            s = signals[sym]
            n_long = (s > 0).sum()
            n_short = (s < 0).sum()
            n_flat = (s == 0).sum()
            print(f"  {sym}: 做多{n_long}天, 做空{n_short}天, 空仓{n_flat}天")

    return result


if __name__ == '__main__':
    _demo_backtest()
