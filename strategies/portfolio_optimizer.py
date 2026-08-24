"""
Portfolio optimizer - portfolio construction module for multi-asset strategies

Provides several weighting schemes: equal weight, risk parity, maximum Sharpe, minimum variance,
dynamic risk parity, regime-based allocation, volatility target overlay, drawdown overlay, and more.
Supports walk-forward validation and a combined evaluation report.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PortfolioOptimizer:
    """
    Portfolio optimizer

    Takes the monthly return series of several sub-strategies, provides multiple weighting
    schemes, and outputs a combined performance comparison table.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping from sub-strategy name to monthly return series. All series should share the same date index.
    rf : float
        Annualized risk-free rate, default 0.0
    """

    def __init__(self, returns_dict: Dict[str, pd.Series], rf: float = 0.0):
        # 对齐所有子策略的日期索引
        self.returns = pd.DataFrame(returns_dict).dropna()
        self.asset_names = list(self.returns.columns)
        self.n = len(self.asset_names)
        self.rf = rf
        # 年化因子（月度 -> 年度）
        self._ann = 12

    # ------------------------------------------------------------------
    # 1. 等权分配
    # ------------------------------------------------------------------
    def equal_weight(self) -> pd.Series:
        """
        Equal weighting: each sub-strategy gets a weight of 1/N

        Returns
        -------
        pd.Series : asset weights
        """
        w = np.ones(self.n) / self.n
        return pd.Series(w, index=self.asset_names, name="equal_weight")

    # ------------------------------------------------------------------
    # 2. 风险平价
    # ------------------------------------------------------------------
    def risk_parity(self, returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Risk parity: weights are inversely proportional to volatility, normalized to sum to 1

        Parameters
        ----------
        returns : pd.DataFrame, optional
            If provided, volatility is computed from this data; otherwise the full sample is used.

        Returns
        -------
        pd.Series : asset weights
        """
        df = returns if returns is not None else self.returns
        vol = df.std()
        # 处理零波动率的情况
        vol = vol.replace(0, np.nan)
        inv_vol = 1.0 / vol
        inv_vol = inv_vol.fillna(0)
        total = inv_vol.sum()
        if total == 0:
            w = np.ones(self.n) / self.n
        else:
            w = (inv_vol / total).values
        return pd.Series(w, index=self.asset_names, name="risk_parity")

    # ------------------------------------------------------------------
    # 3. 最大夏普比率
    # ------------------------------------------------------------------
    def max_sharpe(self, returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Maximum Sharpe ratio optimization: maximize the Sharpe ratio in a mean-variance framework

        Constraints: long-only, weights sum to 1.
        Solved with scipy.optimize.minimize.

        Parameters
        ----------
        returns : pd.DataFrame, optional
            If provided, the optimization uses this data; otherwise the full sample is used.

        Returns
        -------
        pd.Series : asset weights
        """
        df = returns if returns is not None else self.returns
        mu = df.mean().values * self._ann
        cov = df.cov().values * self._ann
        rf = self.rf
        n = self.n

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            return -(port_ret - rf) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = result.x if result.success else w0
        w = np.maximum(w, 0)
        w /= w.sum()
        return pd.Series(w, index=self.asset_names, name="max_sharpe")

    # ------------------------------------------------------------------
    # 4. 最小方差
    # ------------------------------------------------------------------
    def min_variance(self, returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Minimum variance optimization: minimize portfolio variance subject to the given constraints

        Constraints: long-only, weights sum to 1.
        Solved with scipy.optimize.minimize.

        Parameters
        ----------
        returns : pd.DataFrame, optional
            If provided, the optimization uses this data; otherwise the full sample is used.

        Returns
        -------
        pd.Series : asset weights
        """
        df = returns if returns is not None else self.returns
        cov = df.cov().values * self._ann
        n = self.n

        def port_var(w):
            return w @ cov @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(
            port_var, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = result.x if result.success else w0
        w = np.maximum(w, 0)
        w /= w.sum()
        return pd.Series(w, index=self.asset_names, name="min_variance")

    # ------------------------------------------------------------------
    # 5. 动态风险平价
    # ------------------------------------------------------------------
    def dynamic_risk_parity(self, lookback: int = 6) -> pd.DataFrame:
        """
        Dynamic risk parity: volatility is estimated on a rolling window and weights are updated monthly

        Parameters
        ----------
        lookback : int
            Lookback in months, default 6

        Returns
        -------
        pd.DataFrame : weight matrix as a time series (index=date, columns=asset name)
        """
        weight_records = []
        dates = self.returns.index

        for i in range(lookback, len(dates)):
            window = self.returns.iloc[i - lookback: i]
            w = self.risk_parity(returns=window)
            weight_records.append(w.values)

        weights_df = pd.DataFrame(
            weight_records,
            index=dates[lookback:],
            columns=self.asset_names,
        )
        weights_df.name = "dynamic_risk_parity"
        return weights_df

    # ------------------------------------------------------------------
    # 6. 市场环境分配
    # ------------------------------------------------------------------
    def regime_allocation(self, spy_monthly: pd.Series) -> pd.DataFrame:
        """
        Regime-based allocation: classify bull/bear markets from SPY monthly returns and adjust weights dynamically

        Bull market test: SPY cumulative return over the past 3 months > 0
        - Bull market: overweight stocks and vol_arb style assets
        - Bear market: overweight gold and macro style assets

        Asset classification logic:
        - Name contains 'stock'/'equity'/'momentum'/'factor' -> offensive
        - Name contains 'gold'/'bond'/'macro'/'hedge' -> defensive
        - Name contains 'vol'/'vix'/'arb' -> volatility arbitrage
        - Otherwise -> neutral

        Parameters
        ----------
        spy_monthly : pd.Series
            SPY monthly returns; the index must align with the sub-strategies

        Returns
        -------
        pd.DataFrame : time series weight matrix
        """

        def _classify(name: str) -> str:
            """根据资产名称关键词分类"""
            nl = name.lower()
            if any(k in nl for k in ["stock", "equity", "momentum", "factor"]):
                return "offensive"
            if any(k in nl for k in ["gold", "bond", "macro", "hedge"]):
                return "defensive"
            if any(k in nl for k in ["vol", "vix", "arb"]):
                return "vol_arb"
            return "neutral"

        categories = {a: _classify(a) for a in self.asset_names}

        # 计算滚动 3 个月累计收益
        spy_aligned = spy_monthly.reindex(self.returns.index).fillna(0)
        cum3 = spy_aligned.rolling(3).sum()

        weight_records = []
        for dt in self.returns.index:
            is_bull = cum3.loc[dt] > 0 if not np.isnan(cum3.loc[dt]) else True

            raw = {}
            for asset in self.asset_names:
                cat = categories[asset]
                if is_bull:
                    # 牛市：进攻型和波动率套利型加权
                    if cat == "offensive":
                        raw[asset] = 2.0
                    elif cat == "vol_arb":
                        raw[asset] = 1.5
                    elif cat == "defensive":
                        raw[asset] = 0.5
                    else:
                        raw[asset] = 1.0
                else:
                    # 熊市：防御型和波动率套利型加权
                    if cat == "defensive":
                        raw[asset] = 2.0
                    elif cat == "vol_arb":
                        raw[asset] = 1.5
                    elif cat == "offensive":
                        raw[asset] = 0.5
                    else:
                        raw[asset] = 1.0

            total = sum(raw.values())
            normed = {k: v / total for k, v in raw.items()}
            weight_records.append(normed)

        weights_df = pd.DataFrame(weight_records, index=self.returns.index)
        weights_df.name = "regime_allocation"
        return weights_df

    # ------------------------------------------------------------------
    # 7. 波动率目标覆盖
    # ------------------------------------------------------------------
    @staticmethod
    def vol_target_overlay(
        portfolio_ret: pd.Series,
        target: float = 0.10,
        lookback: int = 3,
    ) -> pd.Series:
        """
        Volatility target overlay: scale positions by the ratio of target to realized volatility

        scale = target_vol / realized_vol, clipped to [0.1, 2.0]

        Parameters
        ----------
        portfolio_ret : pd.Series
            Portfolio monthly returns
        target : float
            Annualized target volatility, default 0.10 (10%)
        lookback : int
            Lookback in months used to compute realized volatility, default 3

        Returns
        -------
        pd.Series : portfolio returns after volatility scaling
        """
        rolling_vol = portfolio_ret.rolling(lookback).std() * np.sqrt(12)
        scale = target / rolling_vol
        # 限制缩放倍数
        scale = scale.clip(0.1, 2.0)
        adjusted = portfolio_ret * scale.shift(1)  # 使用上期信号，避免前视偏差
        adjusted.name = "vol_target_overlay"
        return adjusted.dropna()

    # ------------------------------------------------------------------
    # 8. 回撤覆盖
    # ------------------------------------------------------------------
    @staticmethod
    def drawdown_overlay(
        portfolio_ret: pd.Series,
        threshold: float = 0.05,
    ) -> pd.Series:
        """
        Drawdown overlay: cut positions proportionally once the drawdown exceeds a threshold

        When the drawdown > threshold, position size is scaled to threshold / drawdown,
        with a floor of 20% of the position.

        Parameters
        ----------
        portfolio_ret : pd.Series
            Portfolio monthly returns
        threshold : float
            Drawdown threshold that triggers de-risking, default 0.05 (5%)

        Returns
        -------
        pd.Series : portfolio returns after the drawdown overlay
        """
        cum = (1 + portfolio_ret).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max  # 负值

        scale = pd.Series(1.0, index=portfolio_ret.index)
        breach = dd.abs() > threshold
        scale[breach] = threshold / dd[breach].abs()
        scale = scale.clip(0.2, 1.0)

        adjusted = portfolio_ret * scale.shift(1).fillna(1.0)
        adjusted.name = "drawdown_overlay"
        return adjusted

    # ------------------------------------------------------------------
    # 9. 评估
    # ------------------------------------------------------------------
    def evaluate(self, weights, label: str = "") -> dict:
        """
        Evaluate portfolio performance metrics for the given weights

        Parameters
        ----------
        weights : pd.Series, pd.DataFrame, or array-like
            - pd.Series : static weights
            - pd.DataFrame : time series weight matrix (rows=dates, columns=assets)
            - array-like : same as pd.Series

        label : str
            Strategy label used to identify the output

        Returns
        -------
        dict : contains sharpe, cagr, mdd, sortino, calmar, annual_vol and other metrics
        """
        if isinstance(weights, pd.DataFrame):
            # 时序权重：逐月计算加权收益
            common = weights.index.intersection(self.returns.index)
            port_ret = (self.returns.loc[common] * weights.loc[common]).sum(axis=1)
        else:
            if isinstance(weights, pd.Series):
                w = weights.values
            else:
                w = np.array(weights)
            port_ret = (self.returns * w).sum(axis=1)

        return self._calc_metrics(port_ret, label)

    def _calc_metrics(self, port_ret: pd.Series, label: str = "") -> dict:
        """
        根据组合收益率序列计算绩效指标

        Parameters
        ----------
        port_ret : pd.Series
            组合月度收益率
        label : str
            策略标签名

        Returns
        -------
        dict : 绩效指标字典
        """
        if len(port_ret) < 2:
            return {
                "label": label, "sharpe": 0, "cagr": 0,
                "mdd": 0, "sortino": 0, "calmar": 0, "annual_vol": 0,
            }

        ann_ret = port_ret.mean() * self._ann
        ann_vol = port_ret.std() * np.sqrt(self._ann)

        # 夏普比率
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 1e-12 else 0.0

        # CAGR
        cum = (1 + port_ret).cumprod()
        n_years = len(port_ret) / self._ann
        if n_years > 0 and cum.iloc[-1] > 0:
            cagr = cum.iloc[-1] ** (1 / n_years) - 1
        else:
            cagr = 0.0

        # 最大回撤
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        mdd = dd.min()  # 负值

        # Sortino
        downside = port_ret[port_ret < 0]
        down_vol = downside.std() * np.sqrt(self._ann) if len(downside) > 1 else 1e-12
        sortino = (ann_ret - self.rf) / down_vol if down_vol > 1e-12 else 0.0

        # Calmar
        calmar = cagr / abs(mdd) if abs(mdd) > 1e-12 else 0.0

        return {
            "label": label,
            "sharpe": round(sharpe, 3),
            "cagr": round(cagr, 4),
            "mdd": round(mdd, 4),
            "sortino": round(sortino, 3),
            "calmar": round(calmar, 3),
            "annual_vol": round(ann_vol, 4),
        }

    # ------------------------------------------------------------------
    # 10. 滚动窗口验证
    # ------------------------------------------------------------------
    def walk_forward(
        self,
        is_months: int = 36,
        oos_months: int = 12,
    ) -> pd.Series:
        """
        Walk-forward optimization

        Optimize the weights in-sample (IS) with max_sharpe, then apply those weights
        out-of-sample (OOS), rolling forward step by step.

        Parameters
        ----------
        is_months : int
            In-sample window in months, default 36
        oos_months : int
            Out-of-sample window in months, default 12

        Returns
        -------
        pd.Series : concatenated out-of-sample (OOS) portfolio returns
        """
        dates = self.returns.index
        total = len(dates)
        oos_returns = []

        start = 0
        while start + is_months < total:
            is_end = start + is_months
            oos_end = min(is_end + oos_months, total)

            if oos_end <= is_end:
                break

            # 样本内优化
            is_data = self.returns.iloc[start:is_end]
            w = self.max_sharpe(returns=is_data)

            # 样本外应用
            oos_data = self.returns.iloc[is_end:oos_end]
            oos_ret = (oos_data * w.values).sum(axis=1)
            oos_returns.append(oos_ret)

            start += oos_months

        if not oos_returns:
            return pd.Series(dtype=float, name="walk_forward")

        result = pd.concat(oos_returns)
        result.name = "walk_forward"
        return result

    # ------------------------------------------------------------------
    # 11. 运行全部方法并输出对比表
    # ------------------------------------------------------------------
    def run_all(self, spy_monthly: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Run every optimization method and print the combined performance comparison table

        Parameters
        ----------
        spy_monthly : pd.Series, optional
            SPY monthly returns, used by regime_allocation.
            The method is skipped if this is not provided.

        Returns
        -------
        pd.DataFrame : performance comparison table across all methods
        """
        results = []

        # --- 静态权重方法 ---
        static_methods = {
            "equal_weight": self.equal_weight,
            "risk_parity": self.risk_parity,
            "max_sharpe": self.max_sharpe,
            "min_variance": self.min_variance,
        }
        for name, method in static_methods.items():
            w = method()
            metrics = self.evaluate(w, label=name)
            results.append(metrics)

        # --- 动态风险平价 ---
        drp_weights = self.dynamic_risk_parity(lookback=6)
        drp_metrics = self.evaluate(drp_weights, label="dynamic_risk_parity")
        results.append(drp_metrics)

        # --- 市场环境分配 ---
        if spy_monthly is not None:
            regime_weights = self.regime_allocation(spy_monthly)
            regime_metrics = self.evaluate(regime_weights, label="regime_allocation")
            results.append(regime_metrics)

        # --- 覆盖策略（基于等权组合） ---
        eq_w = self.equal_weight()
        base_ret = (self.returns * eq_w.values).sum(axis=1)

        vt_ret = self.vol_target_overlay(base_ret, target=0.10)
        vt_metrics = self._calc_metrics(vt_ret, label="vol_target_overlay")
        results.append(vt_metrics)

        dd_ret = self.drawdown_overlay(base_ret, threshold=0.05)
        dd_metrics = self._calc_metrics(dd_ret, label="drawdown_overlay")
        results.append(dd_metrics)

        # --- Walk-Forward ---
        wf_ret = self.walk_forward(is_months=36, oos_months=12)
        if len(wf_ret) > 0:
            wf_metrics = self._calc_metrics(wf_ret, label="walk_forward")
            results.append(wf_metrics)

        # --- 输出对比表 ---
        df = pd.DataFrame(results).set_index("label")
        df.index.name = "方法"

        print("\n" + "=" * 72)
        print("  投资组合优化 - 综合绩效对比表")
        print("=" * 72)
        print(df.to_string())
        print("=" * 72 + "\n")

        return df


# ======================================================================
# 独立运行示例
# ======================================================================
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2015-01-31", periods=120, freq="ME")

    # 模拟 5 个子策略的月度收益率
    mock_returns = {
        "stock_momentum": pd.Series(
            np.random.normal(0.008, 0.04, 120), index=dates
        ),
        "gold_macro": pd.Series(
            np.random.normal(0.005, 0.025, 120), index=dates
        ),
        "vol_arb": pd.Series(
            np.random.normal(0.006, 0.03, 120), index=dates
        ),
        "bond_hedge": pd.Series(
            np.random.normal(0.003, 0.015, 120), index=dates
        ),
        "equity_factor": pd.Series(
            np.random.normal(0.007, 0.035, 120), index=dates
        ),
    }

    spy_mock = pd.Series(np.random.normal(0.007, 0.04, 120), index=dates, name="SPY")

    opt = PortfolioOptimizer(mock_returns, rf=0.02)
    comparison = opt.run_all(spy_monthly=spy_mock)
