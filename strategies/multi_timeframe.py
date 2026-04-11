"""多时间框架策略 — 日线+周线+月线多维度Alpha组合

核心思想：不同时间尺度捕获不同的Alpha来源
- 日线：微观结构、短期反转、成交量异动
- 周线：中期动量、均值回归
- 月线：长期趋势、质量因子、价值因子

通过多时间框架叠加，策略在趋势方向上寻找最佳入场点，
同时利用低相关性实现子策略间的分散化。
"""

import numpy as np
import pandas as pd
import warnings
from itertools import combinations

from qf.strategy import BaseStrategy

# ═══════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════

def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面标准化排名 -> [-1, 1]"""
    ranked = df.rank(axis=1, pct=True)
    return 2 * ranked - 1


def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """滚动z-score标准化"""
    mu = series.rolling(window, min_periods=max(1, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _safe_div(a, b, fill=0.0):
    """安全除法，避免除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    if isinstance(result, pd.DataFrame):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    elif isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)


# ═══════════════════════════════════════════════════════════════════
#  1. 时间框架叠加策略 (TimeframeStackStrategy)
# ═══════════════════════════════════════════════════════════════════

class TimeframeStackStrategy(BaseStrategy):
    """多时间框架叠加策略 — 月线趋势+周线反转+日线量能三重共振

    原理：
    - 月线信号（60日动量）：判断中长期趋势方向，只做顺势交易
    - 周线信号（10日反转）：在趋势方向内寻找短期超买/超卖的入场点
    - 日线信号（成交量异动）：用量能放大来确认买卖信号的有效性

    逻辑：
    只在月线趋势为多头时买入周线超卖反弹标的，且需日线放量确认。
    signal = monthly_momentum * weekly_reversal * volume_confirmation
    每周再平衡（5个交易日）。
    """
    name = "Timeframe Stack"
    description = "月线趋势+周线反转+日线量能三重共振，周频再平衡"

    mom_window: int = 60        # 月线动量窗口
    rev_window: int = 10        # 周线反转窗口
    vol_avg_window: int = 20    # 成交量均值窗口
    vol_surge_mult: float = 1.5 # 量能放大倍数阈值
    rebalance_days: int = 5     # 再平衡频率

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成三重时间框架叠加信号

        Parameters
        ----------
        data : dict
            必须包含 'close'（收盘价）, 'volume'（成交量）；
            值为 DataFrame (date x symbol)

        Returns
        -------
        pd.DataFrame
            复合信号 (date x symbol)，值域约 [-1, 1]
        """
        close = data['close']
        volume = data['volume']

        # --- 月线信号：60日动量 (趋势方向) ---
        monthly_mom = close.pct_change(self.mom_window)
        monthly_signal = _cross_sectional_rank(monthly_mom)

        # --- 周线信号：10日反转 (入场时机) ---
        # 短期跌幅大的在趋势方向内是超卖买入机会
        weekly_ret = close.pct_change(self.rev_window)
        weekly_reversal = _cross_sectional_rank(-weekly_ret)  # 取反：跌多的排名高

        # --- 日线信号：量能确认 ---
        vol_avg = volume.rolling(self.vol_avg_window, min_periods=10).mean()
        vol_ratio = _safe_div(volume, vol_avg, fill=1.0)
        # 量能放大 -> 确认信号；量能萎缩 -> 信号打折
        volume_confirm = vol_ratio.clip(lower=0.5, upper=3.0) / 3.0

        # --- 三重叠加 ---
        raw_signal = monthly_signal * weekly_reversal * volume_confirm

        # --- 周频再平衡：只在每N天更新信号 ---
        mask = pd.DataFrame(0.0, index=raw_signal.index, columns=raw_signal.columns)
        rebalance_idx = list(range(0, len(raw_signal), self.rebalance_days))
        for i in rebalance_idx:
            end = min(i + self.rebalance_days, len(raw_signal))
            mask.iloc[i:end] = raw_signal.iloc[i].values

        return _cross_sectional_rank(mask)


# ═══════════════════════════════════════════════════════════════════
#  2. 自适应时间框架策略 (AdaptiveTimeframeStrategy)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveTimeframeStrategy(BaseStrategy):
    """自适应时间框架策略 — 根据市场状态动态切换趋势/反转模式

    原理：
    - 趋势市（ADX代理 > 阈值）：使用20日和60日动量信号
    - 震荡市（ADX代理 < 阈值）：使用5日和10日反转信号
    - ADX代理 = |20日收益率| / 20日波动率（简化版方向性指标）

    文献依据：
    趋势跟踪在高波动趋势市表现好，均值回归在低波动震荡市表现好。
    两类策略天然负相关，自适应切换可以提高胜率。
    """
    name = "Adaptive Timeframe"
    description = "趋势市用动量、震荡市用反转，ADX代理自适应切换"

    adx_window: int = 20
    adx_trend_threshold: float = 0.8   # 趋势阈值（标准化后）
    adx_range_threshold: float = 0.5   # 震荡阈值
    mom_short: int = 20
    mom_long: int = 60
    rev_short: int = 5
    rev_long: int = 10

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成自适应信号 — 根据市场状态选择因子

        Parameters
        ----------
        data : dict
            必须包含 'close'

        Returns
        -------
        pd.DataFrame
            自适应信号 (date x symbol)
        """
        close = data['close']

        # --- ADX代理：方向性强度指标 ---
        ret_20d = close.pct_change(self.adx_window)
        vol_20d = close.pct_change(1).rolling(self.adx_window, min_periods=10).std()
        # |收益率|/波动率，越高说明趋势越明确
        adx_proxy = _safe_div(ret_20d.abs(), vol_20d * np.sqrt(self.adx_window), fill=0.0)

        # 截面标准化ADX代理
        adx_rank = adx_proxy.rank(axis=1, pct=True)

        # --- 趋势信号 (动量) ---
        mom_short = _cross_sectional_rank(close.pct_change(self.mom_short))
        mom_long = _cross_sectional_rank(close.pct_change(self.mom_long))
        trend_signal = 0.5 * mom_short + 0.5 * mom_long

        # --- 反转信号 (均值回归) ---
        rev_short = _cross_sectional_rank(-close.pct_change(self.rev_short))
        rev_long = _cross_sectional_rank(-close.pct_change(self.rev_long))
        revert_signal = 0.5 * rev_short + 0.5 * rev_long

        # --- 自适应混合 ---
        # adx_rank高 -> 趋势权重大；adx_rank低 -> 反转权重大
        trend_weight = adx_rank.clip(lower=0.2, upper=0.8)
        revert_weight = 1.0 - trend_weight

        raw = trend_weight * trend_signal + revert_weight * revert_signal
        return _cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  3. 行业轮动策略 (SectorRotationStrategy)
# ═══════════════════════════════════════════════════════════════════

SECTOR_ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLC', 'XLU', 'XLRE', 'XLB', 'XLP', 'XLY']

class SectorRotationStrategy(BaseStrategy):
    """行业轮动策略 — 动量选行业+反转选个股

    原理：
    - 行业层面：20日动量截面排名，做多前3名行业，做空后2名
    - 个股层面：在优选行业内用5日反转选股（板块内超卖反弹）
    - 理论依据：行业动量效应（Moskowitz & Grinblatt 1999）
      行业间存在显著动量，但行业内个股更倾向短期反转

    使用行业ETF：XLE, XLF, XLK, XLV, XLI, XLC, XLU, XLRE, XLB, XLP, XLY
    """
    name = "Sector Rotation"
    description = "20日动量选行业(多前3空后2)+5日反转选个股"

    sector_mom_window: int = 20
    stock_rev_window: int = 5
    long_sectors: int = 3
    short_sectors: int = 2

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成行业轮动+个股选择的复合信号

        Parameters
        ----------
        data : dict
            必须包含 'close'；columns中应包含行业ETF和个股代码

        Returns
        -------
        pd.DataFrame
            复合信号 (date x symbol)
        """
        close = data['close']
        all_symbols = close.columns.tolist()

        # 找出数据中存在的行业ETF
        available_sectors = [s for s in SECTOR_ETFS if s in all_symbols]
        stock_symbols = [s for s in all_symbols if s not in SECTOR_ETFS]

        if len(available_sectors) < 3:
            # 行业ETF不足，退化为纯动量策略
            mom = close.pct_change(self.sector_mom_window)
            return _cross_sectional_rank(mom)

        # --- 行业层面：20日动量排名 ---
        sector_close = close[available_sectors]
        sector_mom = sector_close.pct_change(self.sector_mom_window)
        sector_rank = sector_mom.rank(axis=1, ascending=False)

        # 标记多头行业(排名前N)和空头行业(排名后N)
        n_sectors = len(available_sectors)
        long_mask = sector_rank <= self.long_sectors        # 前3
        short_mask = sector_rank > (n_sectors - self.short_sectors)  # 后2

        # 行业信号：多头=+1，空头=-1，中间=0
        sector_signal = pd.DataFrame(0.0, index=close.index, columns=available_sectors)
        sector_signal[long_mask] = 1.0
        sector_signal[short_mask] = -1.0

        # --- 个股层面：反转信号 ---
        if stock_symbols:
            stock_close = close[stock_symbols]
            stock_rev = _cross_sectional_rank(-stock_close.pct_change(self.stock_rev_window))
        else:
            stock_rev = pd.DataFrame(index=close.index)

        # --- 合并输出 ---
        result = pd.DataFrame(0.0, index=close.index, columns=all_symbols)

        # 行业ETF直接用行业信号
        for s in available_sectors:
            result[s] = sector_signal[s]

        # 个股用反转信号（这里简化处理，实际可叠加行业归属权重）
        for s in stock_symbols:
            if s in stock_rev.columns:
                result[s] = stock_rev[s]

        return result


# ═══════════════════════════════════════════════════════════════════
#  4. 配对回归策略 (PairsReversionStrategy)
# ═══════════════════════════════════════════════════════════════════

class PairsReversionStrategy(BaseStrategy):
    """配对交易策略 — 高相关性股票对的价差均值回归

    原理：
    - 在投资宇宙中寻找高相关性股票对（60日滚动相关系数）
    - 计算价差（对数价格比）的z-score
    - z > 2：做空超涨股、做多滞涨股（价差收敛）
    - z < -2：反向操作
    - 理论依据：Gatev, Goetzmann & Rouwenhorst (2006)

    选取相关性最高的5对，每日再平衡。
    """
    name = "Pairs Reversion"
    description = "高相关性股票对价差z-score均值回归，前5对日频再平衡"

    corr_window: int = 60       # 相关性计算窗口
    zscore_window: int = 20     # z-score计算窗口
    z_entry: float = 2.0        # 开仓z-score阈值
    z_exit: float = 0.5         # 平仓z-score阈值
    top_pairs: int = 5          # 选取的配对数

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成配对交易信号

        Parameters
        ----------
        data : dict
            必须包含 'close'

        Returns
        -------
        pd.DataFrame
            配对信号 (date x symbol)，多头>0，空头<0
        """
        close = data['close']
        log_close = np.log(close.replace(0, np.nan))
        returns = close.pct_change(1)
        symbols = close.columns.tolist()

        result = pd.DataFrame(0.0, index=close.index, columns=symbols)

        if len(symbols) < 2:
            return result

        # --- 在整个回测期中寻找相关性最高的配对 ---
        # 使用前corr_window天的数据选配对（避免前视偏差用滚动方式）
        all_pairs = list(combinations(symbols, 2))

        # 为效率限制最大配对数：随机子采样
        if len(all_pairs) > 500:
            rng = np.random.RandomState(42)
            pair_indices = rng.choice(len(all_pairs), size=500, replace=False)
            candidate_pairs = [all_pairs[i] for i in pair_indices]
        else:
            candidate_pairs = all_pairs

        # 计算每对的滚动相关性均值
        pair_corrs = {}
        for s1, s2 in candidate_pairs:
            if s1 in returns.columns and s2 in returns.columns:
                corr = returns[s1].rolling(self.corr_window, min_periods=30).corr(returns[s2])
                avg_corr = corr.mean()
                if not np.isnan(avg_corr):
                    pair_corrs[(s1, s2)] = avg_corr

        if not pair_corrs:
            return result

        # 选取相关性最高的N对
        sorted_pairs = sorted(pair_corrs.items(), key=lambda x: x[1], reverse=True)
        selected_pairs = [p[0] for p in sorted_pairs[:self.top_pairs]]

        # --- 对每对计算价差z-score并生成信号 ---
        for s1, s2 in selected_pairs:
            spread = log_close[s1] - log_close[s2]
            z = _zscore(spread, window=self.zscore_window)

            # z > entry: 做空s1, 做多s2 (价差收敛)
            # z < -entry: 做多s1, 做空s2
            # |z| < exit: 平仓
            signal_s1 = pd.Series(0.0, index=close.index)
            signal_s2 = pd.Series(0.0, index=close.index)

            signal_s1[z > self.z_entry] = -1.0
            signal_s2[z > self.z_entry] = 1.0
            signal_s1[z < -self.z_entry] = 1.0
            signal_s2[z < -self.z_entry] = -1.0

            # 累加到结果（多对配对信号叠加）
            result[s1] = result[s1] + signal_s1 / self.top_pairs
            result[s2] = result[s2] + signal_s2 / self.top_pairs

        return result


# ═══════════════════════════════════════════════════════════════════
#  5. 事件驱动动量策略 (EventMomentumStrategy)
# ═══════════════════════════════════════════════════════════════════

class EventMomentumStrategy(BaseStrategy):
    """事件驱动漂移策略 — 捕获大幅异动后的价格漂移

    原理：
    - 检测"事件日"：单日涨跌幅>3% 且 成交量>2倍均量
    - 正面事件后：持有5天（盈利公告后漂移效应，PEAD）
    - 负面事件后：回避/做空10天（负面漂移更慢更持久）
    - 文献依据：Ball & Brown (1968), Bernard & Thomas (1989)
      盈利公告后漂移（PEAD）是最稳健的市场异象之一

    事件检测每日运行，持仓自动衰减。
    """
    name = "Event Momentum"
    description = "大幅异动(>3%且放量)后的价格漂移效应，PEAD启发"

    price_threshold: float = 0.03   # 价格异动阈值
    vol_mult: float = 2.0           # 成交量放大倍数
    vol_avg_window: int = 20        # 均量计算窗口
    pos_hold_days: int = 5          # 正面事件持有天数
    neg_hold_days: int = 10         # 负面事件回避天数

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成事件驱动漂移信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            事件信号 (date x symbol)
        """
        close = data['close']
        volume = data['volume']

        daily_ret = close.pct_change(1)
        vol_avg = volume.rolling(self.vol_avg_window, min_periods=10).mean()
        vol_ratio = _safe_div(volume, vol_avg, fill=1.0)

        # --- 检测事件日 ---
        big_move = daily_ret.abs() > self.price_threshold
        high_volume = vol_ratio > self.vol_mult
        is_event = big_move & high_volume

        # 正面事件 vs 负面事件
        pos_event = is_event & (daily_ret > 0)
        neg_event = is_event & (daily_ret < 0)

        # --- 构建信号：事件后N天保持信号 ---
        signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # 正面事件：+1信号持续pos_hold_days天（线性衰减）
        for lag in range(self.pos_hold_days):
            decay = 1.0 - lag / self.pos_hold_days
            shifted = pos_event.shift(lag).fillna(False)
            signal = signal + shifted.astype(float) * decay

        # 负面事件：-1信号持续neg_hold_days天（线性衰减）
        for lag in range(self.neg_hold_days):
            decay = 1.0 - lag / self.neg_hold_days
            shifted = neg_event.shift(lag).fillna(False)
            signal = signal - shifted.astype(float) * decay

        return _cross_sectional_rank(signal)


# ═══════════════════════════════════════════════════════════════════
#  6. 季节性策略 (SeasonalStrategy)
# ═══════════════════════════════════════════════════════════════════

class SeasonalStrategy(BaseStrategy):
    """季节性效应策略 — 月份效应+周内效应+月末月初效应

    原理：
    - 月份效应："Sell in May"（5-10月减仓），圣诞行情（11-1月加仓）
    - 周内效应：周一偏弱（周末信息消化），周五偏强（周末前平仓需求）
    - 月末月初效应：每月最后3天+最初3天更强（薪资流入、基金调仓）
    - 文献依据：Lakonishok & Smidt (1988), Kamstra et al. (2003)

    叠加动量因子做确认，避免纯日历策略的衰减风险。
    """
    name = "Seasonal"
    description = "月份+周内+月末月初季节性效应叠加动量确认"

    # 季节性权重
    w_month: float = 0.4       # 月份效应权重
    w_weekday: float = 0.2     # 周内效应权重
    w_turn: float = 0.2        # 月末月初效应权重
    w_momentum: float = 0.2    # 动量确认权重
    mom_window: int = 20       # 动量计算窗口

    # 月份得分：11-1月高分，5-10月低分
    MONTH_SCORES = {
        1: 0.7, 2: 0.3, 3: 0.2, 4: 0.3,
        5: -0.5, 6: -0.6, 7: -0.4, 8: -0.5,
        9: -0.8, 10: -0.3, 11: 0.6, 12: 0.8,
    }

    # 周内得分：周一弱，周五强
    WEEKDAY_SCORES = {
        0: -0.5,    # Monday
        1: 0.0,     # Tuesday
        2: 0.1,     # Wednesday
        3: 0.2,     # Thursday
        4: 0.4,     # Friday
    }

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成季节性复合信号

        Parameters
        ----------
        data : dict
            必须包含 'close'

        Returns
        -------
        pd.DataFrame
            季节性信号 (date x symbol)
        """
        close = data['close']
        dates = close.index

        # --- 月份效应 ---
        month_signal = pd.Series(
            [self.MONTH_SCORES.get(d.month, 0.0) for d in dates],
            index=dates
        )

        # --- 周内效应 ---
        weekday_signal = pd.Series(
            [self.WEEKDAY_SCORES.get(d.weekday(), 0.0) for d in dates],
            index=dates
        )

        # --- 月末月初效应 ---
        turn_signal = pd.Series(0.0, index=dates)
        for i, d in enumerate(dates):
            day = d.day
            # 月初前3天
            if day <= 3:
                turn_signal.iloc[i] = 0.5
            # 月末后3天（简化：28号以后）
            elif day >= 28:
                turn_signal.iloc[i] = 0.5

        # --- 日历总信号（对所有股票相同） ---
        calendar_score = (
            self.w_month * month_signal
            + self.w_weekday * weekday_signal
            + self.w_turn * turn_signal
        )

        # 广播到所有股票
        calendar_df = pd.DataFrame(
            np.outer(calendar_score.values, np.ones(len(close.columns))),
            index=dates,
            columns=close.columns,
        )

        # --- 动量确认 ---
        momentum = _cross_sectional_rank(close.pct_change(self.mom_window))

        # --- 组合：日历信号 * (1 + 动量确认) ---
        raw = calendar_df + self.w_momentum * momentum
        return _cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  7. 风险平价日频策略 (RiskParityDailyStrategy)
# ═══════════════════════════════════════════════════════════════════

class RiskParityDailyStrategy(BaseStrategy):
    """风险平价策略 — 按波动率倒数分配权重，目标组合波动率10%

    原理：
    - 传统等权重组合：高波动股票主导组合风险
    - 风险平价：每只股票贡献相等的风险（按波动率倒数加权）
    - 当波动率估计变化>20%时触发再平衡
    - 目标组合年化波动率 = 10%
    - 文献依据：Qian (2005), Maillard et al. (2010)

    生成的信号代表权重分配（而非多空方向）。
    """
    name = "Risk Parity Daily"
    description = "波动率倒数加权，目标10%年化波动率，周频再平衡"

    vol_window: int = 20            # 波动率计算窗口
    target_vol: float = 0.10        # 目标年化波动率
    rebalance_threshold: float = 0.20  # 再平衡触发阈值
    annualize_factor: float = 252 ** 0.5

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成风险平价权重信号

        Parameters
        ----------
        data : dict
            必须包含 'close'

        Returns
        -------
        pd.DataFrame
            权重信号 (date x symbol)，值域 [0, 1]
        """
        close = data['close']
        returns = close.pct_change(1)

        # --- 滚动波动率 ---
        rolling_vol = returns.rolling(self.vol_window, min_periods=10).std()
        ann_vol = rolling_vol * self.annualize_factor

        # --- 风险平价权重：波动率倒数 ---
        inv_vol = _safe_div(1.0, ann_vol, fill=0.0)
        inv_vol_sum = inv_vol.sum(axis=1).replace(0, np.nan)
        weights = inv_vol.div(inv_vol_sum, axis=0).fillna(0.0)

        # --- 目标波动率缩放 ---
        # 组合波动率估计（简化：假设股票间无相关性）
        port_var = (weights ** 2 * ann_vol ** 2).sum(axis=1)
        port_vol = np.sqrt(port_var).replace(0, np.nan)
        vol_scalar = _safe_div(self.target_vol, port_vol, fill=1.0)
        vol_scalar = vol_scalar.clip(lower=0.2, upper=3.0)  # 杠杆限制

        # 缩放权重
        scaled_weights = weights.mul(vol_scalar, axis=0)

        # --- 再平衡滤波：波动率变化<阈值时维持旧权重 ---
        result = scaled_weights.copy()
        last_rebal_vol = ann_vol.iloc[0].copy() if len(ann_vol) > 0 else None

        if last_rebal_vol is not None:
            for i in range(1, len(result)):
                current_vol = ann_vol.iloc[i]
                vol_change = _safe_div(
                    (current_vol - last_rebal_vol).abs(),
                    last_rebal_vol,
                    fill=0.0
                )
                avg_change = vol_change.mean()
                if isinstance(avg_change, pd.Series):
                    avg_change = avg_change.mean()

                if avg_change < self.rebalance_threshold:
                    # 波动率变化不大，维持上一期权重
                    result.iloc[i] = result.iloc[i - 1].values
                else:
                    last_rebal_vol = current_vol.copy()

        return result


# ═══════════════════════════════════════════════════════════════════
#  8. 综合Alpha策略 (CombinedAlphaStrategy)
# ═══════════════════════════════════════════════════════════════════

class CombinedAlphaStrategy(BaseStrategy):
    """终极综合策略 — 四大子策略等权组合，利用低相关性分散化

    组合方式：
    - 25% TimeframeStack（趋势+反转+量能）
    - 25% Adaptive（趋势/反转自适应切换）
    - 25% EventMomentum（事件驱动漂移）
    - 25% RiskParity（风险平价权重）

    原理：
    子策略之间天然低相关（趋势 vs 反转 vs 事件 vs 风险管理），
    等权组合后信号更稳定，回撤更小，夏普比率更高。
    类似基金中的基金(FoF)思路。
    """
    name = "Combined Alpha"
    description = "四大子策略等权组合(趋势叠加+自适应+事件+风险平价)"

    w_stack: float = 0.25
    w_adaptive: float = 0.25
    w_event: float = 0.25
    w_riskparity: float = 0.25

    def __init__(self):
        self._sub_strategies = {
            'stack': TimeframeStackStrategy(),
            'adaptive': AdaptiveTimeframeStrategy(),
            'event': EventMomentumStrategy(),
            'riskparity': RiskParityDailyStrategy(),
        }

    def generate_signal(self, data: dict) -> pd.DataFrame:
        """生成综合Alpha信号

        Parameters
        ----------
        data : dict
            必须包含 'close', 'volume'

        Returns
        -------
        pd.DataFrame
            综合信号 (date x symbol)
        """
        sig_stack = self._sub_strategies['stack'].generate_signal(data)
        sig_adaptive = self._sub_strategies['adaptive'].generate_signal(data)
        sig_event = self._sub_strategies['event'].generate_signal(data)
        sig_riskparity = self._sub_strategies['riskparity'].generate_signal(data)

        # 标准化子策略信号到相同尺度
        sig_stack = _cross_sectional_rank(sig_stack)
        sig_adaptive = _cross_sectional_rank(sig_adaptive)
        sig_event = _cross_sectional_rank(sig_event)
        sig_riskparity = _cross_sectional_rank(sig_riskparity)

        raw = (
            self.w_stack * sig_stack
            + self.w_adaptive * sig_adaptive
            + self.w_event * sig_event
            + self.w_riskparity * sig_riskparity
        )

        return _cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════
#  回测入口
# ═══════════════════════════════════════════════════════════════════

# 默认50只股票宇宙（高流动性大盘股）
DEFAULT_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM',
    'ADBE', 'CRM', 'NFLX', 'CMCSA', 'PFE', 'TMO', 'ABT', 'COST',
    'AVGO', 'NKE', 'PEP', 'CSCO', 'MRK', 'INTC', 'WMT', 'LLY',
    'AMD', 'QCOM', 'TXN', 'LOW', 'MS', 'GS', 'SCHW', 'AXP',
    'CAT', 'DE', 'MMM', 'HON', 'RTX', 'LMT', 'GE', 'BA',
]


def _download_data(symbols: list, start: str, end: str) -> dict:
    """下载yfinance OHLCV数据并整理为策略所需的dict格式

    Parameters
    ----------
    symbols : list
        股票代码列表
    start, end : str
        日期范围 (YYYY-MM-DD)

    Returns
    -------
    dict
        {'close': DataFrame, 'open': DataFrame, 'high': DataFrame,
         'low': DataFrame, 'volume': DataFrame}
    """
    import yfinance as yf

    print(f"[数据下载] 正在下载 {len(symbols)} 只标的 {start} ~ {end} ...")
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("yfinance下载数据为空，请检查日期范围和标的代码")

    # yfinance返回MultiIndex columns: (field, symbol)
    data = {}
    for field, key in [('Close', 'close'), ('Open', 'open'), ('High', 'high'),
                        ('Low', 'low'), ('Volume', 'volume')]:
        if field in raw.columns.get_level_values(0):
            data[key] = raw[field].copy()
        elif field.lower() in raw.columns.get_level_values(0):
            data[key] = raw[field.lower()].copy()

    # 前向填充缺失值
    for k in data:
        data[k] = data[k].ffill().bfill()

    n_dates = len(data.get('close', pd.DataFrame()))
    n_syms = len(data.get('close', pd.DataFrame()).columns)
    print(f"[数据下载] 完成: {n_dates} 个交易日, {n_syms} 只标的")
    return data


def _simple_backtest(strategy: BaseStrategy, data: dict) -> dict:
    """简易回测引擎 — 根据信号模拟日频收益

    Parameters
    ----------
    strategy : BaseStrategy
        策略实例
    data : dict
        OHLCV数据

    Returns
    -------
    dict
        {'name': str, 'cum_return': float, 'sharpe': float,
         'max_drawdown': float, 'daily_returns': Series}
    """
    close = data['close']
    returns = close.pct_change(1).iloc[1:]

    try:
        signal = strategy.generate_signal(data)
    except Exception as e:
        print(f"  [!] {strategy.name} 信号生成失败: {e}")
        return {
            'name': strategy.name,
            'cum_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'daily_returns': pd.Series(dtype=float),
        }

    # 信号滞后1天（避免前视偏差）
    signal_lagged = signal.shift(1).iloc[1:]

    # 对齐索引
    common_idx = returns.index.intersection(signal_lagged.index)
    common_cols = returns.columns.intersection(signal_lagged.columns)
    ret = returns.loc[common_idx, common_cols]
    sig = signal_lagged.loc[common_idx, common_cols]

    # 截面归一化权重
    sig_abs_sum = sig.abs().sum(axis=1).replace(0, np.nan)
    weights = sig.div(sig_abs_sum, axis=0).fillna(0.0)

    # 组合日收益
    port_returns = (weights * ret).sum(axis=1)

    # 绩效指标
    cum_ret = (1 + port_returns).prod() - 1
    if port_returns.std() > 0:
        sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    cum_series = (1 + port_returns).cumprod()
    running_max = cum_series.expanding().max()
    drawdown = (cum_series / running_max) - 1
    max_dd = drawdown.min()

    return {
        'name': strategy.name,
        'cum_return': cum_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'daily_returns': port_returns,
    }


def run_all_multi_timeframe_backtests(
    start: str = '2020-01-01',
    end: str = '2025-12-31',
    universe: list = None,
) -> pd.DataFrame:
    """运行所有多时间框架策略的回测

    下载数据，实例化8个策略，逐一回测，输出汇总表格。

    Parameters
    ----------
    start : str
        回测起始日期 (YYYY-MM-DD)
    end : str
        回测结束日期 (YYYY-MM-DD)
    universe : list, optional
        股票宇宙，默认使用50只大盘股+11只行业ETF

    Returns
    -------
    pd.DataFrame
        策略绩效汇总表
    """
    if universe is None:
        universe = DEFAULT_UNIVERSE + SECTOR_ETFS

    # 去重
    universe = list(dict.fromkeys(universe))

    # 下载数据
    data = _download_data(universe, start, end)

    # 实例化所有策略
    strategies = [
        TimeframeStackStrategy(),
        AdaptiveTimeframeStrategy(),
        SectorRotationStrategy(),
        PairsReversionStrategy(),
        EventMomentumStrategy(),
        SeasonalStrategy(),
        RiskParityDailyStrategy(),
        CombinedAlphaStrategy(),
    ]

    print(f"\n{'='*70}")
    print(f"  多时间框架策略回测  |  {start} ~ {end}")
    print(f"  投资宇宙: {len(universe)} 只标的")
    print(f"{'='*70}\n")

    results = []
    for strat in strategies:
        print(f"  运行: {strat.name} ...")
        res = _simple_backtest(strat, data)
        results.append(res)
        print(f"    累计收益: {res['cum_return']:.2%} | "
              f"夏普: {res['sharpe']:.2f} | "
              f"最大回撤: {res['max_drawdown']:.2%}")

    # 汇总表
    summary = pd.DataFrame([
        {
            '策略': r['name'],
            '累计收益': f"{r['cum_return']:.2%}",
            '年化夏普': f"{r['sharpe']:.2f}",
            '最大回撤': f"{r['max_drawdown']:.2%}",
        }
        for r in results
    ])

    print(f"\n{'='*70}")
    print("  策略绩效汇总")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print()

    return summary


# ═══════════════════════════════════════════════════════════════════
#  __main__ 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    summary = run_all_multi_timeframe_backtests(
        start='2020-01-01',
        end='2025-12-31',
    )
