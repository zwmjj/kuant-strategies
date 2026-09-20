"""Machine learning strategies — use existing factor signals as features to predict forward returns and rank instruments

Strategy list:
    1. LinearFactorStrategy   — Ridge regression, expanding-window training
    2. RandomForestStrategy    — random forest, with feature-importance tracking
    3. GradientBoostStrategy   — LightGBM gradient boosting, with early stopping
    4. EnsembleStrategy        — equal-weight ensemble of the three models
    5. FeatureSelectionStrategy — rolling-IC factor selection + Ridge regression
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from qf.strategy import BaseStrategy
from qf.signals_daily import DailySignalGenerator

# LightGBM 可选依赖
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    warnings.warn("lightgbm 未安装，GradientBoostStrategy 将不可用。"
                   "请运行: pip install lightgbm")


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def _compute_all_factors(data: dict) -> dict[str, pd.DataFrame]:
    """从OHLCV数据字典计算全部可用日频因子信号

    Parameters
    ----------
    data : dict
        必须包含 'close', 'volume'；可选 'open', 'high', 'low'

    Returns
    -------
    dict
        因子名 -> DataFrame (date x symbol)，值域 [-1, 1]
    """
    sg = DailySignalGenerator
    close = data['close']
    volume = data['volume']
    high = data.get('high')
    low = data.get('low')
    open_prices = data.get('open')

    returns = close.pct_change()
    dollar_volume = close * volume

    factors = {}

    # 短期反转 / 动量
    factors['rev5'] = sg.momentum_reversal_5d(returns)
    factors['mom5'] = sg.momentum_5d(close)
    factors['mom20'] = sg.momentum_20d(close)

    # 中期动量 (60日)
    raw_mom60 = close / close.shift(60) - 1
    factors['mom60'] = sg.cross_sectional_rank(raw_mom60)

    # 10日反转
    raw_rev10 = -(returns.rolling(10).sum())
    factors['rev10'] = sg.cross_sectional_rank(raw_rev10)

    # 20日反转
    raw_rev20 = -(returns.rolling(20).sum())
    factors['rev20'] = sg.cross_sectional_rank(raw_rev20)

    # 成交量
    factors['vol_surge'] = sg.volume_surge(volume)
    factors['dvol_rank'] = sg.dollar_volume_rank(volume, close)
    factors['vpt'] = sg.volume_price_trend(close, volume)

    # 波动率
    factors['lowvol'] = sg.realized_vol_20d(returns)
    if high is not None and low is not None:
        factors['vol_breakout'] = sg.vol_breakout_hl(high, low, close)
    else:
        factors['vol_breakout'] = sg.vol_breakout(close)

    # 微观结构
    factors['amihud'] = sg.amihud_illiquidity(returns, dollar_volume)
    if high is not None and low is not None:
        factors['realized_spread'] = sg.high_low_spread(high, low, close)

    # 隔夜跳空
    if open_prices is not None:
        factors['overnight_gap'] = sg.overnight_gap(open_prices, close.shift(1))

    return factors


def _compute_lagged_returns(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """计算滞后收益率作为额外特征

    Returns
    -------
    dict
        'lag_ret_1d', 'lag_ret_5d', 'lag_ret_20d' -> DataFrame
    """
    sg = DailySignalGenerator
    returns = close.pct_change()
    lagged = {}
    lagged['lag_ret_1d'] = sg.cross_sectional_rank(returns.shift(1))
    lagged['lag_ret_5d'] = sg.cross_sectional_rank(
        (close.shift(1) / close.shift(6) - 1)
    )
    lagged['lag_ret_20d'] = sg.cross_sectional_rank(
        (close.shift(1) / close.shift(21) - 1)
    )
    return lagged


def _build_panel(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将因子字典转为长面板 (date, symbol, factor1, factor2, ...)

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, symbol)，每列一个因子
    """
    pieces = {}
    for name, df in factors.items():
        stacked = df.stack()
        stacked.name = name
        pieces[name] = stacked

    panel = pd.DataFrame(pieces)
    panel.index.names = ['date', 'symbol']
    return panel


def _add_forward_returns(panel: pd.DataFrame, close: pd.DataFrame,
                         horizon: int = 10) -> pd.DataFrame:
    """添加未来N日收益 (截面排名) 作为目标变量

    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex (date, symbol) 的因子面板
    close : pd.DataFrame
        date x symbol 的收盘价
    horizon : int
        预测期限 (交易日)

    Returns
    -------
    pd.DataFrame
        添加了 'fwd_ret_rank' 列的面板
    """
    sg = DailySignalGenerator
    fwd_ret = close.shift(-horizon) / close - 1
    fwd_rank = sg.cross_sectional_rank(fwd_ret)
    fwd_stacked = fwd_rank.stack()
    fwd_stacked.name = 'fwd_ret_rank'

    panel = panel.join(fwd_stacked, how='left')
    return panel


def _cross_sectional_rank_series(series: pd.Series, dates: pd.Index) -> pd.Series:
    """对预测结果做截面排名，映射到 [-1, 1]"""
    result = series.copy()
    for date in dates:
        mask = result.index.get_level_values('date') == date
        vals = result.loc[mask].dropna()
        if len(vals) < 5:
            continue
        ranked = vals.rank(pct=True) * 2 - 1
        result.loc[ranked.index] = ranked
    return result


# ═══════════════════════════════════════════════════════════════════
#  1. LinearFactorStrategy — Ridge回归
# ═══════════════════════════════════════════════════════════════════

class LinearFactorStrategy(BaseStrategy):
    """Ridge regression factor strategy — predicts the 10-day forward cross-sectional return ranking from the full factor set

    Training: expanding window (at least 252 days), retrained every 60 days.
    Signal: the model's predicted ranking, emitted after cross-sectional ranking.
    """
    name = "ML Linear (Ridge)"
    description = "Ridge回归：全因子 → 预测10日收益排名"

    retrain_freq: int = 60
    min_train_days: int = 252
    train_window: int = 252
    horizon: int = 10
    alpha: float = 1.0

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the Ridge regression prediction signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'; optionally 'open', 'high', 'low'

        Returns
        -------
        pd.DataFrame
            Prediction signal (date x symbol), in [-1, 1]
        """
        close = data['close']
        factors = _compute_all_factors(data)
        feature_names = sorted(factors.keys())

        panel = _build_panel(factors)
        panel = _add_forward_returns(panel, close, self.horizon)

        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = None
        scaler = StandardScaler()
        last_train_idx = -self.retrain_freq  # 强制首次训练

        for i in range(self.min_train_days, len(dates)):
            today = dates[i]

            # 判断是否需要重新训练
            if i - last_train_idx >= self.retrain_freq or model is None:
                train_start = max(0, i - self.train_window)
                train_dates = dates[train_start:i - self.horizon]  # 排除预测期

                train_mask = panel.index.get_level_values('date').isin(train_dates)
                train_data = panel.loc[train_mask].dropna()

                if len(train_data) < 100:
                    continue

                X_train = train_data[feature_names].values
                y_train = train_data['fwd_ret_rank'].values

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)

                model = Ridge(alpha=self.alpha)
                model.fit(X_train_sc, y_train)
                last_train_idx = i

            # 预测今日截面
            if model is not None:
                today_mask = panel.index.get_level_values('date') == today
                today_data = panel.loc[today_mask, feature_names].dropna()

                if len(today_data) < 5:
                    continue

                X_pred = scaler.transform(today_data.values)
                preds = model.predict(X_pred)

                pred_series = pd.Series(preds, index=today_data.index.get_level_values('symbol'))
                # 截面排名
                ranked = pred_series.rank(pct=True) * 2 - 1
                signal_df.loc[today, ranked.index] = ranked.values

        return signal_df


# ═══════════════════════════════════════════════════════════════════
#  2. RandomForestStrategy — 随机森林
# ═══════════════════════════════════════════════════════════════════

class RandomForestStrategy(BaseStrategy):
    """Random forest factor strategy — captures nonlinear factor interactions

    Model: RandomForestRegressor (max_depth=5, n_estimators=100)
    Tracks feature importances, available through the feature_importances_ attribute.
    """
    name = "ML Random Forest"
    description = "随机森林：全因子非线性交互 → 预测10日收益排名"

    retrain_freq: int = 60
    min_train_days: int = 252
    train_window: int = 252
    horizon: int = 10
    n_estimators: int = 100
    max_depth: int = 5

    def __init__(self):
        self.feature_importances_ = {}  # date -> {feature: importance}
        self._feature_names = None

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the random forest prediction signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            Prediction signal (date x symbol)
        """
        close = data['close']
        factors = _compute_all_factors(data)
        feature_names = sorted(factors.keys())
        self._feature_names = feature_names

        panel = _build_panel(factors)
        panel = _add_forward_returns(panel, close, self.horizon)

        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = None
        last_train_idx = -self.retrain_freq

        for i in range(self.min_train_days, len(dates)):
            today = dates[i]

            if i - last_train_idx >= self.retrain_freq or model is None:
                train_start = max(0, i - self.train_window)
                train_dates = dates[train_start:i - self.horizon]

                train_mask = panel.index.get_level_values('date').isin(train_dates)
                train_data = panel.loc[train_mask].dropna()

                if len(train_data) < 100:
                    continue

                X_train = train_data[feature_names].values
                y_train = train_data['fwd_ret_rank'].values

                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                last_train_idx = i

                # 记录特征重要性
                self.feature_importances_[today] = dict(
                    zip(feature_names, model.feature_importances_)
                )

            if model is not None:
                today_mask = panel.index.get_level_values('date') == today
                today_data = panel.loc[today_mask, feature_names].dropna()

                if len(today_data) < 5:
                    continue

                preds = model.predict(today_data.values)

                pred_series = pd.Series(preds, index=today_data.index.get_level_values('symbol'))
                ranked = pred_series.rank(pct=True) * 2 - 1
                signal_df.loc[today, ranked.index] = ranked.values

        return signal_df

    def get_importance_summary(self) -> pd.DataFrame:
        """Get the feature importance summary table

        Returns
        -------
        pd.DataFrame
            Feature importances from each training run; index = date, columns = factor name
        """
        if not self.feature_importances_:
            return pd.DataFrame()
        df = pd.DataFrame(self.feature_importances_).T
        df.index.name = 'train_date'
        return df


# ═══════════════════════════════════════════════════════════════════
#  3. GradientBoostStrategy — LightGBM
# ═══════════════════════════════════════════════════════════════════

class GradientBoostStrategy(BaseStrategy):
    """LightGBM gradient boosting strategy — all factors plus lagged return features

    Uses early stopping to guard against overfitting.
    The last 20% of the training data serves as the validation set.
    """
    name = "ML GradientBoost (LGBM)"
    description = "LightGBM：全因子+滞后收益 → 预测10日收益排名，含早停"

    retrain_freq: int = 60
    min_train_days: int = 252
    train_window: int = 252
    horizon: int = 10
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8

    def __init__(self):
        if not HAS_LGBM:
            raise ImportError(
                "GradientBoostStrategy 需要 lightgbm。请运行: pip install lightgbm"
            )
        self.feature_importances_ = {}
        self._feature_names = None

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the LightGBM prediction signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            Prediction signal (date x symbol)
        """
        close = data['close']
        factors = _compute_all_factors(data)
        lagged = _compute_lagged_returns(close)
        all_features = {**factors, **lagged}
        feature_names = sorted(all_features.keys())
        self._feature_names = feature_names

        panel = _build_panel(all_features)
        panel = _add_forward_returns(panel, close, self.horizon)

        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = None
        last_train_idx = -self.retrain_freq

        for i in range(self.min_train_days, len(dates)):
            today = dates[i]

            if i - last_train_idx >= self.retrain_freq or model is None:
                train_start = max(0, i - self.train_window)
                train_dates = dates[train_start:i - self.horizon]

                train_mask = panel.index.get_level_values('date').isin(train_dates)
                train_data = panel.loc[train_mask].dropna()

                if len(train_data) < 100:
                    continue

                X_all = train_data[feature_names].values
                y_all = train_data['fwd_ret_rank'].values

                # 训练/验证分割 (最后20%作为验证集)
                split_idx = int(len(X_all) * 0.8)
                X_train, X_val = X_all[:split_idx], X_all[split_idx:]
                y_train, y_val = y_all[:split_idx], y_all[split_idx:]

                model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    random_state=42,
                    verbosity=-1,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20, verbose=False),
                            lgb.log_evaluation(period=-1),
                        ],
                    )

                last_train_idx = i

                # 记录特征重要性
                self.feature_importances_[today] = dict(
                    zip(feature_names, model.feature_importances_)
                )

            if model is not None:
                today_mask = panel.index.get_level_values('date') == today
                today_data = panel.loc[today_mask, feature_names].dropna()

                if len(today_data) < 5:
                    continue

                preds = model.predict(today_data.values)

                pred_series = pd.Series(preds, index=today_data.index.get_level_values('symbol'))
                ranked = pred_series.rank(pct=True) * 2 - 1
                signal_df.loc[today, ranked.index] = ranked.values

        return signal_df

    def get_importance_summary(self) -> pd.DataFrame:
        """Get the LightGBM feature importance summary table

        Returns
        -------
        pd.DataFrame
            Feature importances from each training run (split-based); index = date, columns = factor name
        """
        if not self.feature_importances_:
            return pd.DataFrame()
        df = pd.DataFrame(self.feature_importances_).T
        df.index.name = 'train_date'
        return df


# ═══════════════════════════════════════════════════════════════════
#  4. EnsembleStrategy — 三模型等权集成
# ═══════════════════════════════════════════════════════════════════

class EnsembleStrategy(BaseStrategy):
    """Equal-weight three-model ensemble — Ridge + RandomForest + LightGBM

    Each sub-model is trained and predicts independently; the final signal is the cross-sectional
    ranking of the averaged predictions. More robust than any single model.
    """
    name = "ML Ensemble (3-Model)"
    description = "Ridge + RF + LGBM 等权集成，稳健预测"

    def __init__(self):
        self._linear = LinearFactorStrategy()
        self._rf = RandomForestStrategy()
        self._lgbm = None
        if HAS_LGBM:
            self._lgbm = GradientBoostStrategy()

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the ensemble prediction signal

        Averages the signals of the three sub-models (or two, when LightGBM is unavailable) at
        equal weight, then ranks them cross-sectionally.

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            Ensemble signal (date x symbol)
        """
        sg = DailySignalGenerator

        sig_linear = self._linear.generate_signal(data)
        sig_rf = self._rf.generate_signal(data)

        if self._lgbm is not None:
            sig_lgbm = self._lgbm.generate_signal(data)
            # 等权平均 (忽略NaN)
            combined = pd.concat(
                [sig_linear, sig_rf, sig_lgbm],
                keys=['linear', 'rf', 'lgbm'],
            )
            avg = combined.groupby(level=1).mean()
        else:
            warnings.warn("lightgbm 不可用，Ensemble仅使用 Ridge + RandomForest")
            combined = pd.concat(
                [sig_linear, sig_rf],
                keys=['linear', 'rf'],
            )
            avg = combined.groupby(level=1).mean()

        return sg.cross_sectional_rank(avg)

    @property
    def feature_importances_rf(self) -> pd.DataFrame:
        """Random forest feature importances"""
        return self._rf.get_importance_summary()

    @property
    def feature_importances_lgbm(self) -> pd.DataFrame:
        """LightGBM feature importances"""
        if self._lgbm is not None:
            return self._lgbm.get_importance_summary()
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  5. FeatureSelectionStrategy — 滚动IC选因子 + Ridge
# ═══════════════════════════════════════════════════════════════════

class FeatureSelectionStrategy(BaseStrategy):
    """Adaptive factor selection strategy — selects the best factors by rolling IC, then fits a Ridge regression

    Steps:
        1. Compute each factor's cross-sectional IC (information coefficient) against the 10-day
           forward return over the trailing 252 days
        2. Keep the 5 factors with the highest absolute IC
        3. Train a Ridge regression on the selected factors
        4. Reselect factors and retrain every 60 days
    """
    name = "ML Feature Selection (IC)"
    description = "滚动IC选Top-5因子 + Ridge回归，自适应特征选择"

    retrain_freq: int = 60
    min_train_days: int = 252
    train_window: int = 252
    horizon: int = 10
    top_k: int = 5
    alpha: float = 1.0

    def __init__(self):
        self.selected_features_history = {}  # date -> list of selected features

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """Generate the adaptive factor selection prediction signal

        Parameters
        ----------
        data : dict
            Must contain 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            Prediction signal (date x symbol)
        """
        close = data['close']
        factors = _compute_all_factors(data)
        feature_names = sorted(factors.keys())

        panel = _build_panel(factors)
        panel = _add_forward_returns(panel, close, self.horizon)

        dates = close.index
        symbols = close.columns
        signal_df = pd.DataFrame(np.nan, index=dates, columns=symbols)

        model = None
        scaler = StandardScaler()
        selected = feature_names[:self.top_k]  # 初始默认
        last_train_idx = -self.retrain_freq

        for i in range(self.min_train_days, len(dates)):
            today = dates[i]

            if i - last_train_idx >= self.retrain_freq or model is None:
                train_start = max(0, i - self.train_window)
                train_dates = dates[train_start:i - self.horizon]

                train_mask = panel.index.get_level_values('date').isin(train_dates)
                train_data = panel.loc[train_mask].dropna()

                if len(train_data) < 100:
                    continue

                # 计算每个因子与目标的截面IC
                ic_scores = {}
                for fname in feature_names:
                    # 按日期分组计算Spearman相关
                    ic_by_date = []
                    for d in train_dates:
                        d_mask = train_data.index.get_level_values('date') == d
                        d_data = train_data.loc[d_mask]
                        if len(d_data) < 10:
                            continue
                        x = d_data[fname]
                        y = d_data['fwd_ret_rank']
                        if x.std() == 0 or y.std() == 0:
                            continue
                        ic = x.corr(y, method='spearman')
                        if not np.isnan(ic):
                            ic_by_date.append(ic)
                    ic_scores[fname] = np.mean(ic_by_date) if ic_by_date else 0.0

                # 选取IC绝对值最大的Top-K因子
                sorted_by_ic = sorted(ic_scores.items(),
                                      key=lambda kv: abs(kv[1]), reverse=True)
                selected = [kv[0] for kv in sorted_by_ic[:self.top_k]]
                self.selected_features_history[today] = selected.copy()

                # 用选定因子训练Ridge
                X_train = train_data[selected].values
                y_train = train_data['fwd_ret_rank'].values

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)

                model = Ridge(alpha=self.alpha)
                model.fit(X_train_sc, y_train)
                last_train_idx = i

            if model is not None:
                today_mask = panel.index.get_level_values('date') == today
                today_data = panel.loc[today_mask, selected].dropna()

                if len(today_data) < 5:
                    continue

                X_pred = scaler.transform(today_data.values)
                preds = model.predict(X_pred)

                pred_series = pd.Series(preds, index=today_data.index.get_level_values('symbol'))
                ranked = pred_series.rank(pct=True) * 2 - 1
                signal_df.loc[today, ranked.index] = ranked.values

        return signal_df

    def get_selection_history(self) -> pd.DataFrame:
        """Get the factor selection history

        Returns
        -------
        pd.DataFrame
            The factors selected at each reselection
        """
        if not self.selected_features_history:
            return pd.DataFrame()
        rows = []
        for date, feats in self.selected_features_history.items():
            row = {'date': date}
            for rank_i, f in enumerate(feats):
                row[f'rank_{rank_i+1}'] = f
            rows.append(row)
        return pd.DataFrame(rows).set_index('date')


# ═══════════════════════════════════════════════════════════════════
#  策略注册表
# ═══════════════════════════════════════════════════════════════════

ML_STRATEGY_REGISTRY = {
    'ml_linear':    LinearFactorStrategy,
    'ml_rf':        RandomForestStrategy,
    'ml_ensemble':  EnsembleStrategy,
    'ml_featsel':   FeatureSelectionStrategy,
}

if HAS_LGBM:
    ML_STRATEGY_REGISTRY['ml_lgbm'] = GradientBoostStrategy


# ═══════════════════════════════════════════════════════════════════
#  回测入口
# ═══════════════════════════════════════════════════════════════════

def _print_importance_summary(strategy, label: str):
    """打印特征重要性汇总"""
    if hasattr(strategy, 'get_importance_summary'):
        imp_df = strategy.get_importance_summary()
        if not imp_df.empty:
            mean_imp = imp_df.mean().sort_values(ascending=False)
            print(f"\n{'='*60}")
            print(f"  {label} — 平均特征重要性")
            print(f"{'='*60}")
            for feat, val in mean_imp.items():
                bar = '#' * int(val / mean_imp.max() * 30)
                print(f"  {feat:<22s} {val:8.4f}  {bar}")


def run_ml_backtests(start: str = '2022-01-01', end: str = '2025-12-31',
                     tickers: list[str] | None = None):
    """Download the data, run every ML strategy and print the comparison

    Parameters
    ----------
    start : str
        Start date YYYY-MM-DD
    end : str
        End date YYYY-MM-DD
    tickers : list[str] or None
        Instrument list; defaults to a subset of S&P 500 large caps
    """
    import yfinance as yf

    if tickers is None:
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
            'DIS', 'BAC', 'XOM', 'ABBV', 'KO', 'PEP', 'COST',
            'MRK', 'TMO', 'AVGO', 'LLY', 'ORCL', 'ACN', 'MCD', 'CRM',
        ]

    print(f"下载数据: {len(tickers)}只股票, {start} ~ {end}")

    # 额外往前拉300天用于因子warm-up
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    warmup_start = (start_dt - timedelta(days=450)).strftime('%Y-%m-%d')

    raw = yf.download(tickers, start=warmup_start, end=end, auto_adjust=True)

    # 构建data字典
    data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        data['close'] = raw['Close']
        data['open'] = raw['Open']
        data['high'] = raw['High']
        data['low'] = raw['Low']
        data['volume'] = raw['Volume']
    else:
        # 单只股票
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            data[col.lower()] = raw[[col]].rename(columns={col: tickers[0]})

    print(f"数据加载完成: {data['close'].shape[0]}天 x {data['close'].shape[1]}股")

    # 构建策略列表
    strategies = [
        ('Ridge回归', LinearFactorStrategy()),
        ('随机森林', RandomForestStrategy()),
        ('IC因子选择', FeatureSelectionStrategy()),
    ]

    if HAS_LGBM:
        strategies.append(('LightGBM', GradientBoostStrategy()))
        strategies.append(('三模型集成', EnsembleStrategy()))
    else:
        print("\n[警告] lightgbm 未安装，跳过 GradientBoost 和 Ensemble 策略")
        strategies.append(('双模型集成', EnsembleStrategy()))

    # 运行策略
    results = {}
    for label, strat in strategies:
        print(f"\n运行策略: {label} ...")
        try:
            signal = strat.generate_signal(data)

            # 简单绩效评估: 做多Top-20%，做空Bottom-20%的每日收益
            returns = data['close'].pct_change()
            # 只评估start之后的
            eval_mask = signal.index >= pd.Timestamp(start)
            signal_eval = signal.loc[eval_mask]
            returns_eval = returns.reindex(signal_eval.index)

            daily_rets = []
            for date in signal_eval.index:
                sig_row = signal_eval.loc[date].dropna()
                ret_row = returns_eval.loc[date].reindex(sig_row.index).dropna()
                common = sig_row.index.intersection(ret_row.index)
                if len(common) < 10:
                    continue
                sig_row = sig_row[common]
                ret_row = ret_row[common]

                q80 = sig_row.quantile(0.8)
                q20 = sig_row.quantile(0.2)

                long_ret = ret_row[sig_row >= q80].mean()
                short_ret = ret_row[sig_row <= q20].mean()
                ls_ret = long_ret - short_ret
                daily_rets.append({'date': date, 'long': long_ret,
                                   'short': short_ret, 'ls': ls_ret})

            if daily_rets:
                ret_df = pd.DataFrame(daily_rets).set_index('date')
                cum_ls = (1 + ret_df['ls'].fillna(0)).cumprod()

                ann_ret = ret_df['ls'].mean() * 252
                ann_vol = ret_df['ls'].std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                max_dd = (cum_ls / cum_ls.cummax() - 1).min()

                results[label] = {
                    'strategy': strat,
                    'signal': signal,
                    'ret_df': ret_df,
                    'ann_ret': ann_ret,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'cum_ls': cum_ls,
                }
            else:
                results[label] = {'ann_ret': 0, 'sharpe': 0, 'max_dd': 0}

        except Exception as e:
            print(f"  [错误] {label}: {e}")
            results[label] = {'ann_ret': 0, 'sharpe': 0, 'max_dd': 0}

    # 打印对比表
    print(f"\n{'='*70}")
    print(f"  ML策略对比 — 多空收益 ({start} ~ {end})")
    print(f"{'='*70}")
    print(f"  {'策略':<20s} {'年化收益':>10s} {'Sharpe':>10s} {'最大回撤':>10s}")
    print(f"  {'-'*50}")
    for label, res in results.items():
        print(f"  {label:<20s} {res['ann_ret']:>9.2%} {res['sharpe']:>10.2f} "
              f"{res['max_dd']:>9.2%}")

    # 打印特征重要性
    for label, res in results.items():
        strat = res.get('strategy')
        if strat is None:
            continue
        if isinstance(strat, RandomForestStrategy):
            _print_importance_summary(strat, f"随机森林 ({label})")
        elif isinstance(strat, GradientBoostStrategy):
            _print_importance_summary(strat, f"LightGBM ({label})")
        elif isinstance(strat, EnsembleStrategy):
            if hasattr(strat, '_rf'):
                _print_importance_summary(strat._rf, f"Ensemble-RF子模型")
            if hasattr(strat, '_lgbm') and strat._lgbm is not None:
                _print_importance_summary(strat._lgbm, f"Ensemble-LGBM子模型")

    # 打印因子选择历史
    for label, res in results.items():
        strat = res.get('strategy')
        if isinstance(strat, FeatureSelectionStrategy):
            hist = strat.get_selection_history()
            if not hist.empty:
                print(f"\n{'='*60}")
                print(f"  IC因子选择历史 (最近5次)")
                print(f"{'='*60}")
                print(hist.tail().to_string())

    return results


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_ml_backtests(start='2023-01-01', end='2025-12-31')
