"""TimesFM策略 — 利用Google TimesFM 2.5基础模型预测价格走势

策略列表:
    1. TimesFMTrendStrategy   — 逐股预测未来收益，截面排名选股
    2. TimesFMQuantileStrategy — 利用分位数预测做风险调整后的选股

原理:
    TimesFM是Google开源的时间序列基础模型（200M参数），
    在大量时序数据上预训练，具备零样本（zero-shot）预测能力。
    本策略利用其预测各股票未来N日价格，据此构建截面信号。
"""

import warnings
import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy

# TimesFM 可选依赖
try:
    import timesfm
    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False
    warnings.warn(
        "timesfm 未安装，TimesFM策略将不可用。"
        "请运行: pip install timesfm[torch]"
    )


def _load_timesfm_model(
    max_context: int = 512,
    max_horizon: int = 128,
):
    """加载并编译TimesFM模型（单例缓存）"""
    if not HAS_TIMESFM:
        raise RuntimeError("timesfm 未安装")

    if not hasattr(_load_timesfm_model, '_model'):
        print("[TimesFM] 首次加载模型，从HuggingFace下载中...")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=False,  # Windows兼容性，避免编译问题
        )
        _load_timesfm_model._model = model
        _load_timesfm_model._compiled_config = None

    model = _load_timesfm_model._model
    config = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,  # 收益率可为负
        fix_quantile_crossing=True,
    )

    # 仅在配置变化时重新编译
    if _load_timesfm_model._compiled_config != (max_context, max_horizon):
        model.compile(config)
        _load_timesfm_model._compiled_config = (max_context, max_horizon)

    return model


def _batch_forecast(
    model,
    series_list: list[np.ndarray],
    horizon: int,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """分批预测，避免内存溢出

    Returns
    -------
    point_forecast : np.ndarray, shape (N, horizon)
    quantile_forecast : np.ndarray, shape (N, horizon, Q)
    """
    all_points = []
    all_quantiles = []

    for i in range(0, len(series_list), batch_size):
        batch = series_list[i:i + batch_size]
        pf, qf = model.forecast(horizon=horizon, inputs=batch)
        all_points.append(pf)
        all_quantiles.append(qf)

    return np.concatenate(all_points, axis=0), np.concatenate(all_quantiles, axis=0)


# ═══════════════════════════════════════════════════════════════════
#  1. TimesFMTrendStrategy — 基础趋势预测
# ═══════════════════════════════════════════════════════════════════

class TimesFMTrendStrategy(BaseStrategy):
    """TimesFM趋势策略 — 利用基础模型预测各股票未来收益，截面排名选股

    工作流程:
        1. 每rebalance_freq天，用过去context_days天的收盘价序列作为输入
        2. TimesFM预测未来forecast_horizon天的价格
        3. 计算预测收益率 = (预测末日价格 / 当前价格) - 1
        4. 截面排名 → 信号值 [-1, 1]
    """
    name = "TimesFM Trend"
    description = "TimesFM基础模型：零样本价格预测 → 截面排名选股"

    context_days: int = 256      # 输入历史天数
    forecast_horizon: int = 10   # 预测未来天数
    rebalance_freq: int = 5      # 每N天重新预测
    min_history: int = 60        # 股票最少需要的历史天数
    batch_size: int = 64         # 推理批次大小

    def generate_signal(self, data: dict) -> pd.DataFrame:
        close = data['close']
        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = _load_timesfm_model(
            max_context=self.context_days,
            max_horizon=self.forecast_horizon,
        )

        # 从有足够历史的日期开始
        start_idx = max(self.min_history, self.context_days)

        for i in range(start_idx, len(dates), self.rebalance_freq):
            today = dates[i]

            # 准备各股票的价格序列
            lookback_start = max(0, i - self.context_days)
            price_window = close.iloc[lookback_start:i]

            # 筛选有效股票（至少min_history天非NaN数据）
            valid_counts = price_window.notna().sum()
            valid_symbols = valid_counts[valid_counts >= self.min_history].index.tolist()

            if len(valid_symbols) < 5:
                continue

            # 构建输入序列列表
            series_list = []
            sym_order = []
            for sym in valid_symbols:
                series = price_window[sym].dropna().values.astype(np.float64)
                if len(series) >= self.min_history:
                    series_list.append(series)
                    sym_order.append(sym)

            if len(series_list) < 5:
                continue

            # TimesFM批量预测
            point_forecast, _ = _batch_forecast(
                model, series_list, self.forecast_horizon, self.batch_size
            )

            # 计算预测收益率：(预测末日价格 / 当前价格) - 1
            current_prices = np.array([
                close.loc[today, sym] for sym in sym_order
            ])

            # point_forecast shape: (N, forecast_horizon)
            # 取预测区间末尾作为目标价格
            predicted_end_prices = point_forecast[:, -1]
            predicted_returns = (predicted_end_prices / current_prices) - 1

            # 处理异常值
            predicted_returns = np.clip(predicted_returns, -0.5, 0.5)

            # 截面排名 → [-1, 1]
            pred_series = pd.Series(predicted_returns, index=sym_order)
            ranked = pred_series.rank(pct=True) * 2 - 1

            # 填充到rebalance_freq天内
            end_idx = min(i + self.rebalance_freq, len(dates))
            for j in range(i, end_idx):
                signal_df.loc[dates[j], ranked.index] = ranked.values

        return signal_df


# ═══════════════════════════════════════════════════════════════════
#  2. TimesFMQuantileStrategy — 分位数风险调整策略
# ═══════════════════════════════════════════════════════════════════

class TimesFMQuantileStrategy(BaseStrategy):
    """TimesFM分位数策略 — 利用概率预测做风险调整后的选股

    相比纯趋势策略，本策略额外考虑预测的不确定性:
        信号 = 预测收益 / 预测波动 (类似夏普比)

    工作流程:
        1. TimesFM预测未来价格的点预测和分位数预测
        2. 用分位数宽度衡量预测不确定性
        3. 信号 = 预测收益 / 不确定性（风险调整）
        4. 截面排名 → 信号值 [-1, 1]
    """
    name = "TimesFM Quantile Risk-Adjusted"
    description = "TimesFM概率预测：收益/不确定性 → 风险调整后截面排名"

    context_days: int = 256
    forecast_horizon: int = 10
    rebalance_freq: int = 5
    min_history: int = 60
    batch_size: int = 64

    def generate_signal(self, data: dict) -> pd.DataFrame:
        close = data['close']
        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = _load_timesfm_model(
            max_context=self.context_days,
            max_horizon=self.forecast_horizon,
        )

        start_idx = max(self.min_history, self.context_days)

        for i in range(start_idx, len(dates), self.rebalance_freq):
            today = dates[i]

            lookback_start = max(0, i - self.context_days)
            price_window = close.iloc[lookback_start:i]

            valid_counts = price_window.notna().sum()
            valid_symbols = valid_counts[valid_counts >= self.min_history].index.tolist()

            if len(valid_symbols) < 5:
                continue

            series_list = []
            sym_order = []
            for sym in valid_symbols:
                series = price_window[sym].dropna().values.astype(np.float64)
                if len(series) >= self.min_history:
                    series_list.append(series)
                    sym_order.append(sym)

            if len(series_list) < 5:
                continue

            point_forecast, quantile_forecast = _batch_forecast(
                model, series_list, self.forecast_horizon, self.batch_size
            )

            current_prices = np.array([
                close.loc[today, sym] for sym in sym_order
            ])

            # 预测收益率
            predicted_returns = (point_forecast[:, -1] / current_prices) - 1

            # 预测不确定性：用分位数范围衡量
            # quantile_forecast shape: (N, horizon, Q)
            # 取最后一天的上下分位数差作为不确定性
            q_last = quantile_forecast[:, -1, :]  # (N, Q)
            # 用IQR近似: 上分位 - 下分位（第一列通常是低分位，最后一列是高分位）
            uncertainty = (q_last[:, -1] - q_last[:, 0]) / current_prices
            uncertainty = np.maximum(uncertainty, 1e-6)  # 避免除零

            # 风险调整信号 = 预测收益 / 不确定性
            risk_adjusted = predicted_returns / uncertainty
            risk_adjusted = np.clip(risk_adjusted, -10, 10)

            pred_series = pd.Series(risk_adjusted, index=sym_order)
            ranked = pred_series.rank(pct=True) * 2 - 1

            end_idx = min(i + self.rebalance_freq, len(dates))
            for j in range(i, end_idx):
                signal_df.loc[dates[j], ranked.index] = ranked.values

        return signal_df
