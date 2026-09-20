"""
Maximum Sharpe strategy — target Sharpe 1.50+
================================
Core idea: a Sharpe of 1.5 does not come from a magic signal, it comes from risk management.
  Sharpe = mean(excess_return) / std(excess_return) * sqrt(12)
  Going from 1.19 to 1.50 requires: raising monthly return by 26%, cutting monthly volatility by 21%, or a mix.

Seven layers of optimization:
  1. Aggressive volatility target (6% annualized instead of 10%) — cuts volatility by 40% directly
  2. Tighter drawdown control (8%/15%/20% trigger thresholds)
  3. Stacked multi-signal alpha (regime+orthogonal+interaction, uncorrelated)
  4. High turnover penalty (0.50) — reduces trading friction
  5. Concentrated positions (10 long, 5 short) — higher conviction
  6. Market regime filter — cut exposure sharply in high-volatility/crash regimes
  7. Quality filter — trade only large-cap, low-volatility stocks
"""
import numpy as np
import pandas as pd
from qf.strategy import BaseStrategy
from qf.optimizer import build_combo_signal, vol_target_scale, drawdown_scale


class MaxSharpeStrategy(BaseStrategy):
    """
    Maximum Sharpe strategy — lifting Sharpe from 1.19 to 1.50+ through risk management

    Signal layer:
      - 50% regime_blend (four timing factors, risk_parity)
      - 30% orthogonal_blend (three orthogonalized factors, FF5 exposure stripped out -> pure alpha)
      - 20% interaction_blend (interaction confirmation signal)

    Risk control layer:
      - Volatility target of 6% annualized (aggressive volatility compression)
      - Drawdown control at 8%/15%/20% thresholds (de-risk earlier)
      - Market regime filter (cut exposure sharply in high-volatility/crash regimes)
      - Quality filter (large cap + low volatility)
      - Turnover penalty of 0.50 (avoid unnecessary trading)
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
        Generate the maximum Sharpe signal — three-layer alpha stack + quality filter

        Steps:
          1. Build the three signal streams (regime / orthogonal / interaction)
          2. Stack them by weight: 50% / 30% / 20%
          3. Quality filter: keep only large-cap, low-volatility stocks
          4. Re-rank cross-sectionally

        Parameters
        ----------
        data : dict
            Data dictionary returned by prepare_data()

        Returns
        -------
        pd.DataFrame
            Signal matrix (date x permno); higher values are more bullish
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
        Market regime filter — de-risk based on market volatility/trend

        Rules:
          - High-volatility regime (volatility > mean + 1std): scale positions to 50%
          - Crash regime (3-month decline > 10%): scale positions to 30%
          - Normal regime: positions at 100%

        Parameters
        ----------
        data : dict
            Data dictionary; requires the 'spy_ret' key (market benchmark returns)

        Returns
        -------
        float
            Market regime position scaling factor (0.30 ~ 1.00)
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
        Tighter drawdown control — de-risk earlier to protect capital

        Thresholds:
          - 8% drawdown -> 70% (default 10%->80%)
          - 15% drawdown -> 40% (default 20%->50%)
          - 20% drawdown -> 20% (default 25%->25%)

        Parameters
        ----------
        pv_history : list
            Portfolio net asset value history

        Returns
        -------
        float
            Drawdown position scaling factor (0.20 ~ 1.00)
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
        Combined position scaling — three layers stacked

        Scaling factor = vol_target_scale x drawdown_scale x market_regime_scale

        Parameters
        ----------
        returns_history : list
            Monthly return history
        pv_history : list
            Portfolio net asset value history
        data : dict, optional
            Data dictionary (used for the market regime assessment)

        Returns
        -------
        float
            Combined position scaling factor
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
        """Return all strategy parameters"""
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
    Run the maximum Sharpe backtest — full pipeline

    Steps:
      1. Build the three-layer stacked signal
      2. Run the event-driven backtest (aggressive vol target + tight drawdown control + market filter)
      3. Print the results
      4. Return (BacktestResult, metrics_dict)

    Parameters
    ----------
    data : dict
        Data dictionary returned by prepare_data()
    verbose : bool
        Whether to print detailed information

    Returns
    -------
    tuple
        (BacktestResult, dict) — the backtest result object and the metrics dictionary
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
