"""TimesFM 因子策略 — 用基础模型预测生成截面因子信号

因子列表:
    1. timesfm_trend      — 预测收益率截面排名
    2. timesfm_risk_adj   — 预测收益 / 预测不确定性 (夏普型)
    3. timesfm_reversal   — 预测方向 vs 近期动量的分歧信号

策略列表:
    1. TimesFMFactorStrategy     — 三因子等权组合
    2. TimesFMAlphaStrategy      — TimesFM + 传统因子集成
    3. TimesFMPureTrendStrategy  — 纯趋势预测因子

原理:
    框架使用月频数据 (date × permno)。对每只股票，取其月度价格序列
    输入 TimesFM 2.5 预测下一期价格，再截面排名生成因子信号。
    TimesFM 的优势在于零样本捕捉非线性时序模式，与传统动量/反转
    因子低相关，适合因子集成。
"""

import warnings
import numpy as np
import pandas as pd
from qf.strategy import BaseStrategy

try:
    import timesfm
    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False
    warnings.warn("timesfm 未安装，TimesFM因子策略不可用")


# ═══════════════════════════════════════════════════════════════════
#  模型管理
# ═══════════════════════════════════════════════════════════════════

_MODEL_CACHE = {}


def _get_model(context=512, horizon=128):
    """单例加载 TimesFM 模型"""
    if not HAS_TIMESFM:
        raise RuntimeError("timesfm 未安装")

    key = (context, horizon)
    if key not in _MODEL_CACHE:
        print("  [TimesFM] 加载模型...")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=False,
        )
        m.compile(timesfm.ForecastConfig(
            max_context=context,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=False,   # 关闭翻转不变性, 速度翻倍
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=512,       # 大批次充分利用GPU
        ))
        _MODEL_CACHE[key] = m
        print("  [TimesFM] 模型就绪")
    return _MODEL_CACHE[key]


# ═══════════════════════════════════════════════════════════════════
#  因子计算核心
# ═══════════════════════════════════════════════════════════════════

def _cross_sectional_rank(df):
    """截面排名 → [-1, 1]"""
    def _rank_row(row):
        valid = row.dropna()
        if len(valid) < 10:
            return row * np.nan
        return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
    return df.apply(_rank_row, axis=1)


def compute_timesfm_factors(prices, returns, mktcap=None, batch_size=256,
                            recompute_freq=3, max_stocks=500):
    """计算全部 TimesFM 因子

    Parameters
    ----------
    prices : pd.DataFrame
        月频收盘价 (date × permno)
    returns : pd.DataFrame
        月频收益率 (date × permno)
    mktcap : pd.DataFrame, optional
        市值, 用于筛选大盘股以减少计算量
    batch_size : int
        推理批次大小
    recompute_freq : int
        每 N 个月重新预测一次 (中间月份沿用上次信号)
    max_stocks : int
        每期最多预测的股票数 (按市值排序取前N)

    Returns
    -------
    dict[str, pd.DataFrame]
        因子名 → 信号 DataFrame (date × permno, [-1, 1])
    """
    model = _get_model(context=512, horizon=128)
    dates = prices.index
    permnos = prices.columns
    n_dates = len(dates)

    # 预分配
    pred_return = pd.DataFrame(np.nan, index=dates, columns=permnos)
    pred_uncertainty = pd.DataFrame(np.nan, index=dates, columns=permnos)

    MIN_HISTORY = 24

    # 计算需要预测的月份 (每 recompute_freq 个月一次)
    predict_months = list(range(MIN_HISTORY, n_dates, recompute_freq))
    if (n_dates - 1) not in predict_months:
        predict_months.append(n_dates - 1)

    print(f"  [TimesFM] 逐期预测 ({len(predict_months)} 期, 间隔{recompute_freq}月, "
          f"每期≤{max_stocks}只股票)...")

    for pi, i in enumerate(predict_months):
        today = dates[i]

        # 取历史
        hist = prices.iloc[:i]

        # 筛选有效股票
        valid_counts = hist.notna().sum()
        valid_permnos = valid_counts[valid_counts >= MIN_HISTORY].index

        # 市值筛选: 只保留前 max_stocks 只
        if mktcap is not None and len(valid_permnos) > max_stocks:
            mc_today = mktcap.iloc[min(i, len(mktcap) - 1)]
            mc_valid = mc_today.reindex(valid_permnos).dropna()
            if len(mc_valid) > max_stocks:
                valid_permnos = mc_valid.nlargest(max_stocks).index

        if len(valid_permnos) < 20:
            continue

        # 构建输入
        series_list = []
        perm_order = []
        current_prices = []
        for p in valid_permnos:
            s = hist[p].dropna().values.astype(np.float64)
            if len(s) >= MIN_HISTORY and s[-1] > 0:
                series_list.append(s)
                perm_order.append(p)
                current_prices.append(s[-1])

        if len(series_list) < 20:
            continue

        # 分批预测
        all_points = []
        all_quantiles = []
        for b_start in range(0, len(series_list), batch_size):
            batch = series_list[b_start:b_start + batch_size]
            pf, qf = model.forecast(horizon=1, inputs=batch)
            all_points.append(pf)
            all_quantiles.append(qf)

        points = np.concatenate(all_points, axis=0)      # (N, 1)
        quantiles = np.concatenate(all_quantiles, axis=0) # (N, 1, Q)

        cur = np.array(current_prices)
        pred_ret = np.clip((points[:, 0] / cur) - 1, -0.5, 0.5)
        q_spread = np.maximum(
            (quantiles[:, 0, -1] - quantiles[:, 0, 0]) / cur, 1e-6
        )

        # 填充当前月 + 后续 recompute_freq-1 个月 (向量化)
        fill_end = min(i + recompute_freq, n_dates)
        col_idx = [pred_return.columns.get_loc(p) for p in perm_order]
        for fi in range(i, fill_end):
            pred_return.values[fi, col_idx] = pred_ret
            pred_uncertainty.values[fi, col_idx] = q_spread

        pct = 100 * (pi + 1) / len(predict_months)
        print(f"    {today.strftime('%Y-%m')} ({pct:.0f}%) — {len(perm_order)} 只股票")

    # ── 生成三个因子 ──
    factors = {}

    # 1. timesfm_trend: 预测收益截面排名
    factors['timesfm_trend'] = _cross_sectional_rank(pred_return)

    # 2. timesfm_risk_adj: 预测收益 / 不确定性
    risk_adj = pred_return / pred_uncertainty
    risk_adj = risk_adj.clip(-10, 10)
    factors['timesfm_risk_adj'] = _cross_sectional_rank(risk_adj)

    # 3. timesfm_reversal: 预测方向 vs 近期动量分歧
    #    当模型预测上涨但近期下跌 (或反之)，信号更强
    mom3 = returns.shift(1).rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
    mom3_rank = _cross_sectional_rank(mom3)
    trend_rank = factors['timesfm_trend']
    # 分歧度 = trend 预测 - 动量方向 (正 = 模型看涨但近期跌 = 反转信号)
    divergence = trend_rank - mom3_rank
    factors['timesfm_reversal'] = _cross_sectional_rank(divergence)

    return factors


# ═══════════════════════════════════════════════════════════════════
#  1. TimesFMFactorStrategy — 三因子等权
# ═══════════════════════════════════════════════════════════════════

class TimesFMFactorStrategy(BaseStrategy):
    """TimesFM 三因子等权策略

    组合:
        - timesfm_trend (40%): 纯预测方向
        - timesfm_risk_adj (35%): 风险调整后信号
        - timesfm_reversal (25%): 动量分歧反转
    """
    name = "TimesFM Factor Composite"
    description = "TimesFM预测因子: 趋势40% + 风险调整35% + 反转分歧25%"

    w_trend: float = 0.40
    w_risk: float = 0.35
    w_reversal: float = 0.25

    def generate_signal(self, data: dict) -> pd.DataFrame:
        prices = data['prices']
        returns = data['returns']
        mktcap = data.get('mktcap')

        factors = compute_timesfm_factors(prices, returns, mktcap=mktcap)

        signal = (
            self.w_trend * factors['timesfm_trend']
            + self.w_risk * factors['timesfm_risk_adj']
            + self.w_reversal * factors['timesfm_reversal']
        )
        return _cross_sectional_rank(signal)


# ═══════════════════════════════════════════════════════════════════
#  2. TimesFMAlphaStrategy — TimesFM + 传统因子集成
# ═══════════════════════════════════════════════════════════════════

class TimesFMAlphaStrategy(BaseStrategy):
    """TimesFM + 传统因子集成策略

    集成 TimesFM 预测因子与传统动量/价值/质量因子:
        - TimesFM 风险调整因子 (30%)
        - 12-1 动量 (25%)
        - 3个月动量加速度 (15%)
        - 波动率反转 (15%)
        - TimesFM 反转分歧 (15%)
    """
    name = "TimesFM Alpha Ensemble"
    description = "TimesFM + 动量 + 波动率集成"

    def generate_signal(self, data: dict) -> pd.DataFrame:
        prices = data['prices']
        returns = data['returns']
        mktcap = data.get('mktcap')

        tfm_factors = compute_timesfm_factors(prices, returns, mktcap=mktcap)

        from qf.signals import SignalGenerator as SG

        mom12 = SG.multi_timeframe_momentum(returns)
        mom12_rank = _cross_sectional_rank(mom12)

        accel = SG.momentum_acceleration(returns)
        accel_rank = _cross_sectional_rank(accel)

        vol = SG.volatility(returns)
        lowvol_rank = _cross_sectional_rank(-vol)

        signal = (
            0.30 * tfm_factors['timesfm_risk_adj']
            + 0.25 * mom12_rank
            + 0.15 * accel_rank
            + 0.15 * lowvol_rank
            + 0.15 * tfm_factors['timesfm_reversal']
        )
        return _cross_sectional_rank(signal)


# ═══════════════════════════════════════════════════════════════════
#  3. TimesFMPureTrendStrategy — 纯趋势
# ═══════════════════════════════════════════════════════════════════

class TimesFMPureTrendStrategy(BaseStrategy):
    """纯 TimesFM 趋势预测因子策略

    最简单的用法: 只用 TimesFM 风险调整后的预测收益做排名。
    适合检验 TimesFM 单因子的 alpha。
    """
    name = "TimesFM Pure Trend"
    description = "纯TimesFM风险调整预测 → 截面排名"

    def generate_signal(self, data: dict) -> pd.DataFrame:
        prices = data['prices']
        returns = data['returns']
        mktcap = data.get('mktcap')
        factors = compute_timesfm_factors(prices, returns, mktcap=mktcap)
        return factors['timesfm_risk_adj']
