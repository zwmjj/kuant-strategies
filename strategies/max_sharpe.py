"""
最大夏普策略 — 目标 Sharpe 1.50+
================================
核心思路: 夏普1.5不靠神奇信号, 靠风险管理。
  Sharpe = mean(excess_return) / std(excess_return) * sqrt(12)
  从1.19到1.50需要: 提高月收益26% 或 降低月波动21% 或组合。

七层优化:
  1. 激进波动率目标 (6%年化, 而非10%) — 直接压缩波动率40%
  2. 更紧回撤控制 (8%/15%/20% 触发阈值)
  3. 多信号Alpha叠加 (regime+orthogonal+interaction, 非相关)
  4. 高换手惩罚 (0.50) — 减少交易摩擦
  5. 集中持仓 (多10空5) — 更高信念度
  6. 市场状态过滤 — 高波/崩盘时大幅减仓
  7. 质量过滤 — 仅交易大市值低波动股票
"""
import numpy as np
import pandas as pd
from qf.strategy import BaseStrategy
from qf.optimizer import build_combo_signal, vol_target_scale, drawdown_scale


class MaxSharpeStrategy(BaseStrategy):
    """
    最大夏普策略 — 通过风险管理将Sharpe从1.19提升至1.50+

    信号层:
      - 50% regime_blend (择时四因子, risk_parity)
      - 30% orthogonal_blend (正交化三因子, 剥离FF5暴露 -> 纯Alpha)
      - 20% interaction_blend (交互确认信号)

    风控层:
      - 波动率目标 6%年化 (激进压缩波动)
      - 回撤控制 8%/15%/20% 阈值 (更早减仓)
      - 市场状态过滤 (高波/崩盘大幅减仓)
      - 质量过滤 (大市值 + 低波动率)
      - 换手惩罚 0.50 (减少不必要交易)
    """
    name = "MaxSharpe (目标1.50+)"
    description = "七层优化: 激进vol目标+紧回撤+多信号叠加+集中持仓+市场过滤+质量过滤"

    # ── 持仓参数: 集中持仓, 高信念度 ──
    long_n: int = 10
    short_n: int = 5
    long_pct: float = 1.10          # 略低于默认, 减少杠杆风险
    short_pct: float = 0.10
    turnover_penalty: float = 0.50  # 高换手惩罚, 减少摩擦
    weight_mode: str = 'inv_vol'

    # ── 波动率目标: 激进压缩 ──
    target_vol: float = 0.06        # 6%年化 (而非默认10%)
    max_leverage: float = 1.2       # 更保守的最大杠杆
    vol_lookback: int = 6

    # ── 回撤控制: 更紧阈值 ──
    dd_threshold_1: float = 0.08    # 8% -> 缩至70% (默认10%->80%)
    dd_threshold_2: float = 0.15    # 15% -> 缩至40% (默认20%->50%)
    dd_threshold_3: float = 0.20    # 20% -> 缩至20% (默认25%->25%)
    dd_scale_1: float = 0.70
    dd_scale_2: float = 0.40
    dd_scale_3: float = 0.20

    # ── 信号叠加权重 ──
    regime_weight: float = 0.50
    orthogonal_weight: float = 0.30
    interaction_weight: float = 0.20

    # ── 质量过滤 ──
    cap_quantile: float = 0.75      # 仅前25%市值股票
    max_stock_vol: float = 0.50     # 排除年化波动>50%的股票

    # ── 市场状态过滤 ──
    high_vol_threshold_std: float = 1.0     # VIX代理 > 均值+1std
    crash_threshold: float = -0.10          # 3个月跌幅>10%
    high_vol_equity_scale: float = 0.50     # 高波时仓位缩至50%
    crash_equity_scale: float = 0.30        # 崩盘时仓位缩至30%

    def __init__(self, target_vol=0.06, long_n=10, short_n=5):
        """
        初始化最大夏普策略

        Parameters
        ----------
        target_vol : float
            年化目标波动率, 默认6% (激进压缩)
        long_n : int
            多头持仓数, 默认10 (集中)
        short_n : int
            空头持仓数, 默认5 (集中)
        """
        self.target_vol = target_vol
        self.long_n = long_n
        self.short_n = short_n
        self._returns_history = []
        self._pv_history = []

    def generate_signal(self, data):
        """
        生成最大夏普信号 — 三层Alpha叠加 + 质量过滤

        流程:
          1. 构建三路信号 (regime / orthogonal / interaction)
          2. 按权重叠加: 50% / 30% / 20%
          3. 质量过滤: 仅保留大市值 + 低波动率股票
          4. 重新截面排名

        Parameters
        ----------
        data : dict
            prepare_data() 返回的数据字典

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x permno), 值越高越看多
        """
        # ── 步骤1: 构建三路非相关信号 ──
        signals = {}

        # 主信号: regime_blend (择时四因子, risk_parity)
        try:
            signals['regime'] = build_combo_signal(
                'regime_blend', data, cap_quantile=self.cap_quantile, verbose=False
            )
        except Exception as e:
            print(f"  [MaxSharpe] regime_blend 失败: {e}")

        # 正交信号: 剥离FF5因子暴露, 保留纯Alpha
        try:
            signals['orthogonal'] = build_combo_signal(
                'orthogonal_blend', data, cap_quantile=self.cap_quantile, verbose=False
            )
        except Exception as e:
            print(f"  [MaxSharpe] orthogonal_blend 失败: {e}")

        # 交互信号: 双重确认 (质量×动量 等)
        try:
            signals['interaction'] = build_combo_signal(
                'interaction_blend', data, cap_quantile=self.cap_quantile, verbose=False
            )
        except Exception as e:
            print(f"  [MaxSharpe] interaction_blend 失败: {e}")

        if not signals:
            raise ValueError("所有信号层均失败, 无法生成MaxSharpe信号")

        # ── 步骤2: 对齐并叠加 ──
        weight_map = {
            'regime': self.regime_weight,
            'orthogonal': self.orthogonal_weight,
            'interaction': self.interaction_weight,
        }

        # 对齐到公共日期和股票
        common_idx = None
        common_cols = None
        for sig in signals.values():
            if common_idx is None:
                common_idx = sig.index
                common_cols = sig.columns
            else:
                common_idx = common_idx.intersection(sig.index)
                common_cols = common_cols.intersection(sig.columns)

        aligned = {k: v.loc[common_idx, common_cols].astype(float)
                   for k, v in signals.items()}

        # 权重归一化 (仅使用成功生成的信号)
        active_weights = {k: weight_map[k] for k in aligned}
        total_w = sum(active_weights.values())
        active_weights = {k: v / total_w for k, v in active_weights.items()}

        # 加权叠加
        stacked = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        for key, sig in aligned.items():
            stacked += active_weights[key] * sig.fillna(0)

        print(f"  [MaxSharpe] 信号叠加: "
              + ", ".join(f"{k}={v:.0%}" for k, v in active_weights.items())
              + f" | {len(common_idx)}期 x {len(common_cols)}只")

        # ── 步骤3: 质量过滤 ──
        stacked = self._apply_quality_filter(stacked, data)

        # ── 步骤4: 截面排名归一化 ──
        from qf.signals import SignalGenerator
        sg = SignalGenerator()
        stacked = sg.cross_sectional_rank(stacked)

        return stacked

    def _apply_quality_filter(self, signal, data):
        """
        质量过滤: 排除低质量股票, 降低噪音

        规则:
          1. 仅保留市值前25%的股票 (cap_quantile=0.75)
             -> 已在build_combo_signal中通过cap_quantile实现
          2. 排除年化波动率>50%的股票 (彩票型)
             -> 将其信号设为NaN, 不参与排名

        Parameters
        ----------
        signal : pd.DataFrame
            原始信号矩阵
        data : dict
            数据字典, 需要 'returns' 键

        Returns
        -------
        pd.DataFrame
            过滤后的信号矩阵
        """
        if 'returns' not in data:
            return signal

        returns = data['returns']
        # 滚动12个月年化波动率
        ann_vol = returns.rolling(12, min_periods=6).std() * np.sqrt(12)

        # 对齐
        common_idx = signal.index.intersection(ann_vol.index)
        common_cols = signal.columns.intersection(ann_vol.columns)

        if len(common_idx) == 0 or len(common_cols) == 0:
            return signal

        vol_aligned = ann_vol.loc[common_idx, common_cols]
        sig_aligned = signal.loc[common_idx, common_cols].copy()

        # 排除高波动股票: 波动率>50%的设为NaN
        high_vol_mask = vol_aligned > self.max_stock_vol
        n_filtered = high_vol_mask.sum().sum()
        sig_aligned[high_vol_mask] = np.nan

        # 保留信号中未被过滤的部分
        signal = signal.copy()
        signal.loc[common_idx, common_cols] = sig_aligned

        print(f"  [MaxSharpe] 质量过滤: 排除{n_filtered}个高波动观测 (>{self.max_stock_vol:.0%}年化)")
        return signal

    def compute_market_regime_scale(self, data):
        """
        市场状态过滤 — 通过市场波动率/趋势判断减仓

        规则:
          - 高波动状态 (波动率 > 均值 + 1std): 仓位缩至50%
          - 崩盘状态 (3个月跌幅 > 10%): 仓位缩至30%
          - 正常状态: 仓位100%

        Parameters
        ----------
        data : dict
            数据字典, 需要 'spy_ret' 键 (市场基准收益)

        Returns
        -------
        float
            市场状态仓位缩放因子 (0.30 ~ 1.00)
        """
        if 'spy_ret' not in data:
            return 1.0

        spy_ret = data['spy_ret'].squeeze()
        if len(spy_ret) < 12:
            return 1.0

        # VIX代理: 用SPY收益率的滚动波动率
        rolling_vol = spy_ret.rolling(6).std() * np.sqrt(12)
        if len(rolling_vol.dropna()) < 12:
            return 1.0

        current_vol = rolling_vol.iloc[-1]
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        # 3个月市场收益
        recent_3m = (1 + spy_ret.iloc[-3:]).prod() - 1 if len(spy_ret) >= 3 else 0

        # 崩盘检测 (优先级最高)
        if recent_3m < self.crash_threshold:
            return self.crash_equity_scale

        # 高波动检测
        if current_vol > vol_mean + self.high_vol_threshold_std * vol_std:
            return self.high_vol_equity_scale

        return 1.0

    def compute_drawdown_scale(self, pv_history):
        """
        更紧的回撤控制 — 更早减仓, 保护资本

        阈值:
          - 8%回撤 -> 70% (默认10%->80%)
          - 15%回撤 -> 40% (默认20%->50%)
          - 20%回撤 -> 20% (默认25%->25%)

        Parameters
        ----------
        pv_history : list
            组合净值历史

        Returns
        -------
        float
            回撤仓位缩放因子 (0.20 ~ 1.00)
        """
        if len(pv_history) < 3:
            return 1.0

        peak = max(pv_history)
        current = pv_history[-1]
        dd = (current - peak) / peak  # 负数

        if dd < -self.dd_threshold_3:
            return self.dd_scale_3
        elif dd < -self.dd_threshold_2:
            return self.dd_scale_2
        elif dd < -self.dd_threshold_1:
            return self.dd_scale_1
        return 1.0

    def compute_position_scale(self, returns_history=None, pv_history=None, data=None):
        """
        综合仓位缩放 — 三层叠加

        缩放因子 = vol_target_scale × drawdown_scale × market_regime_scale

        Parameters
        ----------
        returns_history : list
            月收益率历史
        pv_history : list
            组合净值历史
        data : dict, optional
            数据字典 (用于市场状态判断)

        Returns
        -------
        float
            综合仓位缩放因子
        """
        ret_hist = returns_history or self._returns_history
        pv_hist = pv_history or self._pv_history

        # 层1: 波动率目标 (6%年化)
        v_scale = vol_target_scale(
            ret_hist,
            target_vol=self.target_vol,
            lookback=self.vol_lookback,
            max_leverage=self.max_leverage,
        )

        # 层2: 回撤控制 (更紧阈值)
        d_scale = self.compute_drawdown_scale(pv_hist)

        # 层3: 市场状态
        m_scale = self.compute_market_regime_scale(data) if data is not None else 1.0

        return v_scale * d_scale * m_scale

    def get_params(self) -> dict:
        """返回策略全部参数"""
        params = super().get_params()
        params.update({
            'target_vol': self.target_vol,
            'max_leverage': self.max_leverage,
            'dd_thresholds': f'{self.dd_threshold_1}/{self.dd_threshold_2}/{self.dd_threshold_3}',
            'dd_scales': f'{self.dd_scale_1}/{self.dd_scale_2}/{self.dd_scale_3}',
            'signal_weights': f'regime={self.regime_weight}, orth={self.orthogonal_weight}, interact={self.interaction_weight}',
            'quality_filter': f'cap_q={self.cap_quantile}, max_vol={self.max_stock_vol}',
            'market_filter': f'high_vol={self.high_vol_equity_scale}, crash={self.crash_equity_scale}',
            'base_signal': 'regime(50%) + orthogonal(30%) + interaction(20%)',
        })
        return params


def run_max_sharpe_backtest(data, verbose=True):
    """
    运行最大夏普回测 — 完整流程

    流程:
      1. 构建三层叠加信号
      2. 运行事件驱动回测 (激进vol目标 + 紧回撤控制 + 市场过滤)
      3. 打印结果
      4. 返回 (BacktestResult, metrics_dict)

    Parameters
    ----------
    data : dict
        prepare_data() 返回的数据字典
    verbose : bool
        是否打印详细信息

    Returns
    -------
    tuple
        (BacktestResult, dict) — 回测结果对象和指标字典
    """
    from qf.backtest import DataHandler, Portfolio, BacktestResult
    from qf.costs import ExecutionHandler, SignalEvent

    strategy = MaxSharpeStrategy()

    if verbose:
        print("=" * 60)
        print("MaxSharpe策略 — 目标Sharpe 1.50+")
        print("=" * 60)
        params = strategy.get_params()
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

    # ── 步骤1: 生成信号 ──
    if verbose:
        print("[1/3] 生成三层叠加信号...")
    signal = strategy.generate_signal(data)

    # ── 步骤2: 事件驱动回测 (带动态仓位管理) ──
    if verbose:
        print(f"\n[2/3] 运行事件驱动回测 (vol_target={strategy.target_vol:.0%})...")

    inv_vol = 1.0 / data['returns'].rolling(12).std().replace(0, np.nan)
    dh = DataHandler(data['prices'], data['returns'],
                     data.get('volume'), data.get('adv_dollar'))
    port = Portfolio(initial_capital=10000)
    exe = ExecutionHandler(cost_model='sqrt')

    dates_with_ret = []

    for bar in dh.iter_bars():
        date = bar['date']
        if date not in signal.index:
            continue

        sig_row = signal.loc[date].dropna()
        valid = sig_row[sig_row.index.isin(bar['tradable'])]
        if len(valid) < strategy.long_n + strategy.short_n:
            continue

        # ── 动态仓位缩放: vol目标 × 回撤控制 × 市场状态 ──
        v_scale = vol_target_scale(
            port.return_history,
            target_vol=strategy.target_vol,
            lookback=strategy.vol_lookback,
            max_leverage=strategy.max_leverage,
        )
        d_scale = strategy.compute_drawdown_scale(port.pv_history)
        m_scale = strategy.compute_market_regime_scale(data)
        scale = v_scale * d_scale * m_scale

        # 调整后的敞口
        adj_long_pct = strategy.long_pct * scale
        adj_short_pct = strategy.short_pct * scale

        # ── 换手惩罚: 倾向保持现有持仓 ──
        adj = valid.copy()
        if strategy.turnover_penalty > 0:
            for p in getattr(port, '_prev_long', set()):
                if p in adj.index:
                    adj[p] += strategy.turnover_penalty
            for p in getattr(port, '_prev_short', set()):
                if p in adj.index:
                    adj[p] -= strategy.turnover_penalty

        # ── 集中持仓: 多10空5 ──
        long_list = adj.nlargest(strategy.long_n).index.tolist()
        short_list = adj.nsmallest(strategy.short_n).index.tolist()

        # ── 反波动率加权 ──
        if strategy.weight_mode == 'inv_vol' and date in inv_vol.index:
            iv_vals = inv_vol.loc[date, long_list].dropna()
            if len(iv_vals) > 0:
                long_w = (iv_vals / iv_vals.sum() * adj_long_pct).to_dict()
            else:
                long_w = {t: adj_long_pct / strategy.long_n for t in long_list}
        else:
            long_w = {t: adj_long_pct / strategy.long_n for t in long_list}
        short_w = {t: adj_short_pct / strategy.short_n for t in short_list}

        port._prev_long = set(long_w.keys())
        port._prev_short = set(short_w.keys())

        sig_evt = SignalEvent(date=date, long_targets=long_w, short_targets=short_w)
        order = port.on_signal(sig_evt, bar)
        _, month_ret = exe.on_order(order, bar, port)
        port.update_pv(month_ret)
        dates_with_ret.append(date)

    # ── 步骤3: 结果 ──
    pv = pd.Series(
        [port.initial_capital] + port.pv_history,
        index=[dh.dates[0]] + dates_with_ret,
    )
    rets = pd.Series(port.return_history, index=dates_with_ret)
    result = BacktestResult(pv, rets, pd.DataFrame())
    metrics = result.metrics(rf=data['rf'], benchmark_returns=data['spy_ret'])

    if verbose:
        print(f"\n[3/3] 回测结果")
        print("=" * 60)
        print(f"  总收益:     {metrics.get('total_return', 0):.1%}")
        print(f"  CAGR:       {metrics.get('cagr', 0):.1%}")
        print(f"  Sharpe:     {metrics.get('sharpe', 0):.2f}")
        print(f"  Sortino:    {metrics.get('sortino', 0):.2f}")
        print(f"  最大回撤:   {metrics.get('max_drawdown', 0):.1%}")
        print(f"  胜率:       {metrics.get('win_rate', 0):.1%}")
        print(f"  Beta:       {metrics.get('beta', 'N/A')}")
        print(f"  Alpha:      {metrics.get('alpha', 'N/A')}")
        print(f"  月数:       {metrics.get('n_months', 0)}")
        print(f"  终值:       ${metrics.get('final_value', 0):,.0f}")
        print("=" * 60)

        sharpe = metrics.get('sharpe', 0)
        if sharpe >= 1.50:
            print(f"  >>> 目标达成! Sharpe {sharpe:.2f} >= 1.50 <<<")
        else:
            gap = 1.50 - sharpe
            print(f"  >>> 距目标差 {gap:.2f}, 需进一步优化 <<<")

    return result, metrics
