"""另类数据策略 — 基于Alpaca非价格数据的Alpha信号

利用Alpaca提供的另类数据源构建交易策略:
    1. NewsAlphaStrategy         — 新闻情绪+动量确认策略
    2. DividendEventStrategy     — 股息事件驱动策略
    3. SplitMomentumStrategy     — 拆股动量策略 (Ikenberry et al. 1996)
    4. OptionsSmartMoneyStrategy — 期权聪明钱逆向策略
    5. InstitutionalFlowStrategy — 机构资金流策略
    6. MultiAlternativeStrategy  — 多另类数据综合策略

每个策略支持两种模式:
    - 实盘模式: 传入 AlpacaDataLoader 实例获取实时数据
    - 回测模式: 使用价格数据生成代理信号
"""

import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy
from qf.signals_news import NewsSentimentGenerator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _momentum(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """计算滚动动量 (收益率), 返回截面排名 [-1, 1]"""
    raw = prices.pct_change(window)
    ranked = raw.rank(axis=1, pct=True) * 2 - 1
    return ranked


def _rank_signal(s: pd.Series) -> pd.Series:
    """截面百分位排名映射到 [-1, 1]"""
    valid = s.dropna()
    if len(valid) <= 1:
        return s.apply(lambda x: 0.0 if not np.isnan(x) else np.nan)
    ranked = s.rank(pct=True)
    return ranked * 2.0 - 1.0


def _safe_align(*frames: pd.DataFrame) -> list:
    """对齐多个DataFrame的索引和列"""
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)
    return [f.loc[idx, cols] for f in frames]


def _run_signal_backtest(
    signal_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    long_n: int = 20,
    short_n: int = 20,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """快速信号回测 — 根据信号做多/做空并计算收益

    Parameters
    ----------
    signal_df : pd.DataFrame
        信号矩阵 (date x symbol)，值越高越看多
    returns_df : pd.DataFrame
        日收益率矩阵 (date x symbol)
    long_n : int
        做多股票数量
    short_n : int
        做空股票数量
    cost_bps : float
        单边交易成本（基点）

    Returns
    -------
    pd.DataFrame
        含 long_ret, short_ret, ls_ret, ls_ret_net, cum_ret, turnover 列
    """
    signal_df, returns_df = _safe_align(signal_df, returns_df)

    # 信号滞后一期（t日信号 => t+1日持仓）
    signal_lag = signal_df.shift(1).iloc[1:]
    returns_aligned = returns_df.iloc[1:]

    results = []
    prev_long = set()
    prev_short = set()

    for dt in signal_lag.index:
        sig_row = signal_lag.loc[dt].dropna()
        ret_row = returns_aligned.loc[dt]

        if len(sig_row) < long_n + short_n:
            continue

        sorted_syms = sig_row.sort_values(ascending=False)
        long_syms = set(sorted_syms.index[:long_n])
        short_syms = set(sorted_syms.index[-short_n:])

        long_ret = ret_row.reindex(long_syms).mean() if long_syms else 0.0
        short_ret = ret_row.reindex(short_syms).mean() if short_syms else 0.0
        ls_ret = long_ret - short_ret

        long_to = len(long_syms - prev_long) / max(long_n, 1)
        short_to = len(short_syms - prev_short) / max(short_n, 1)
        turnover = (long_to + short_to) / 2

        cost = turnover * cost_bps / 10000 * 2
        ls_ret_net = ls_ret - cost

        results.append({
            'date': dt,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': ls_ret,
            'ls_ret_net': ls_ret_net,
            'turnover': turnover,
        })

        prev_long = long_syms
        prev_short = short_syms

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index('date')
    df['cum_ret'] = (1 + df['ls_ret_net']).cumprod() - 1

    n_days = len(df)
    ann = 252
    mean_ret = df['ls_ret_net'].mean()
    std_ret = df['ls_ret_net'].std()
    sharpe = mean_ret / std_ret * np.sqrt(ann) if std_ret > 0 else 0.0
    max_dd = (df['cum_ret'] - df['cum_ret'].cummax()).min()

    df.attrs['summary'] = {
        'n_days': n_days,
        'ann_return': mean_ret * ann,
        'ann_vol': std_ret * np.sqrt(ann),
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_turnover': df['turnover'].mean(),
    }

    return df


# ═══════════════════════════════════════════════════════════════════════
# 1. NewsAlphaStrategy — 新闻情绪+价格动量确认策略
# ═══════════════════════════════════════════════════════════════════════

class NewsAlphaStrategy(BaseStrategy):
    """新闻情绪Alpha策略 — 正面新闻+正向动量 = 买入，反之卖出

    逻辑:
        1. 基于关键词方法对新闻标题打情绪分 (来自 qf/signals_news.py)
        2. 与5日价格动量结合确认信号方向
        3. 买入: 正面新闻情绪 + 正向5日动量
        4. 卖出: 负面新闻情绪 + 负向动量
        5. 新闻效应在3-5天内衰减（指数衰减权重）

    回测模式:
        无实时新闻时，使用以下价格代理:
        - 异常成交量 → 新闻关注度代理
        - 日内波动率 → 新闻冲击代理
        - 与动量组合生成信号
    """

    name = "News Alpha"
    description = "新闻情绪+价格动量确认策略"

    # 参数
    momentum_window: int = 5           # 动量计算窗口
    news_decay_days: int = 3           # 新闻效应衰减天数
    sentiment_weight: float = 0.6      # 情绪信号权重
    momentum_weight: float = 0.4       # 动量信号权重
    volume_anomaly_window: int = 20    # 成交量异常检测窗口

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化新闻Alpha策略

        Parameters
        ----------
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器，为None时使用价格代理
        """
        self.alpaca_loader = alpaca_loader
        self.sentiment_gen = NewsSentimentGenerator()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _proxy_news_sentiment(self, data: dict) -> pd.DataFrame:
        """无实时新闻时，用价格数据生成情绪代理信号

        代理逻辑:
            - 异常高成交量 + 正收益 → 正面新闻代理
            - 异常高成交量 + 负收益 → 负面新闻代理
            - 成交量正常 → 无新闻 → 信号为0

        衰减: 使用指数移动平均模拟3-5天新闻效应衰减
        """
        close = data['close']
        volume = data.get('volume', close * 0 + 1)

        # 成交量z-score作为新闻关注度代理
        vol_ma = volume.rolling(self.volume_anomaly_window).mean()
        vol_std = volume.rolling(self.volume_anomaly_window).std()
        vol_zscore = (volume - vol_ma) / vol_std.replace(0, np.nan)
        vol_zscore = vol_zscore.clip(-3, 3)

        # 日收益率方向
        daily_ret = close.pct_change()

        # 情绪代理 = 成交量异常 × 收益方向
        raw_sentiment = vol_zscore * np.sign(daily_ret)

        # 指数衰减模拟新闻效应消退 (span=decay_days)
        alpha = 2.0 / (self.news_decay_days + 1)
        sentiment_decayed = raw_sentiment.ewm(span=self.news_decay_days).mean()

        # 截面排名
        ranked = sentiment_decayed.rank(axis=1, pct=True) * 2 - 1
        return ranked

    def _live_news_sentiment(self, data: dict) -> pd.DataFrame:
        """使用Alpaca实时新闻数据生成情绪信号"""
        close = data['close']
        symbols = list(close.columns)

        # 获取新闻
        try:
            news_list = self.alpaca_loader.get_news(symbols=symbols, limit=50)
        except Exception as e:
            logger.warning("获取新闻失败，回退到代理: %s", e)
            return self._proxy_news_sentiment(data)

        if not news_list:
            return self._proxy_news_sentiment(data)

        # 按标的分组
        news_by_symbol = {sym: [] for sym in symbols}
        for item in news_list:
            for sym in item.get('symbols', []):
                if sym in news_by_symbol:
                    news_by_symbol[sym].append(item)

        # 生成截面情绪信号
        sentiment_series = self.sentiment_gen.build_composite_news_signal(
            news_by_symbol
        )

        # 广播到所有日期（最新情绪信号应用到近期交易日）
        signal_df = pd.DataFrame(
            index=close.index,
            columns=close.columns,
            dtype=float,
        )
        signal_df.iloc[:] = 0.0

        # 最后news_decay_days天使用衰减后的情绪
        for i, dt in enumerate(close.index[-self.news_decay_days:]):
            decay = math.exp(-0.3 * (self.news_decay_days - 1 - i))
            for sym in symbols:
                if sym in sentiment_series.index:
                    signal_df.loc[dt, sym] = sentiment_series[sym] * decay

        ranked = signal_df.rank(axis=1, pct=True) * 2 - 1
        return ranked

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成新闻Alpha信号

        Parameters
        ----------
        data : dict
            含 'close' (和可选 'volume') 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            临时传入的Alpaca加载器（优先于__init__时的）

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)，值越高越看多
        """
        loader = alpaca_loader or self.alpaca_loader
        close = data['close']

        # 动量信号
        mom_signal = _momentum(close, self.momentum_window)

        # 情绪信号
        if loader is not None:
            self.alpaca_loader = loader
            sent_signal = self._live_news_sentiment(data)
        else:
            sent_signal = self._proxy_news_sentiment(data)

        # 对齐
        mom_signal, sent_signal = _safe_align(mom_signal, sent_signal)

        # 组合: 只在方向一致时给强信号
        combined = (
            self.sentiment_weight * sent_signal
            + self.momentum_weight * mom_signal
        )

        # 方向确认加强: 同向时信号放大，反向时信号缩小
        direction_agree = np.sign(sent_signal) * np.sign(mom_signal)
        confirmed = combined * (1 + 0.5 * direction_agree.clip(0, 1))

        return confirmed


# ═══════════════════════════════════════════════════════════════════════
# 2. DividendEventStrategy — 股息事件驱动策略
# ═══════════════════════════════════════════════════════════════════════

class DividendEventStrategy(BaseStrategy):
    """股息事件驱动策略 — 除息日前买入，除息日后卖出

    逻辑:
        1. 追踪即将到来的除息日（Alpaca企业行动数据）
        2. 除息日前3天买入，除息日后1天卖出
        3. 过滤: 股息收益率 > 0.5%，股票不在下跌趋势中
        4. 历史研究: 股息捕获策略年化增加约2-4%收益

    回测模式:
        使用价格数据模拟股息事件:
        - 高股息收益率代理: 低波动率 + 高均值回归倾向
        - 除息日代理: 每月固定日期模拟
    """

    name = "Dividend Event"
    description = "股息事件驱动: 除息日前买入，除息日后卖出"

    # 参数
    buy_days_before_ex: int = 3     # 除息日前几天买入
    sell_days_after_ex: int = 1     # 除息日后几天卖出
    min_yield_pct: float = 0.5      # 最低股息收益率 (%)
    trend_lookback: int = 20        # 趋势判断回看天数
    trend_threshold: float = -0.05  # 下跌趋势阈值

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化股息事件策略"""
        self.alpaca_loader = alpaca_loader
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _proxy_dividend_signal(self, data: dict) -> pd.DataFrame:
        """用价格数据模拟股息事件信号

        代理逻辑:
            1. 股息收益率代理: 低波动率 + 低估值(低市净率)的稳定股票
               → 用实际波动率的倒数 + 均值回归信号模拟
            2. 事件窗口: 每月15号附近模拟一次除息事件
            3. 趋势过滤: 20日涨幅低于-5%的排除
        """
        close = data['close']

        # 20日波动率（低波=高股息代理）
        ret = close.pct_change()
        vol_20 = ret.rolling(20).std()
        inv_vol = 1.0 / vol_20.replace(0, np.nan)  # 低波动=高得分

        # 均值回归信号（过去20天跌多了→反弹，模拟股息股特性）
        ma_20 = close.rolling(20).mean()
        mean_rev = (ma_20 - close) / close  # 偏离均线程度

        # 趋势过滤: 20日涨跌幅
        trend = close.pct_change(self.trend_lookback)
        trend_ok = (trend > self.trend_threshold).astype(float)

        # 事件窗口: 模拟除息日效应
        # 使用日期中的天数模拟: 10-17号附近为事件窗口
        event_mask = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for dt in close.index:
            if hasattr(dt, 'day'):
                day = dt.day
            else:
                day = pd.Timestamp(dt).day
            if 10 <= day <= 17:
                event_mask.loc[dt] = 1.0

        # 组合信号
        raw = inv_vol.rank(axis=1, pct=True) * 0.4 + \
              mean_rev.rank(axis=1, pct=True) * 0.3 + \
              event_mask * 0.3

        # 应用趋势过滤
        filtered = raw * trend_ok

        # 截面排名
        signal = filtered.rank(axis=1, pct=True) * 2 - 1
        return signal

    def _live_dividend_signal(self, data: dict) -> pd.DataFrame:
        """使用Alpaca企业行动数据生成股息信号"""
        close = data['close']
        symbols = list(close.columns)

        try:
            div_df = self.alpaca_loader.get_corporate_actions(
                ca_type='dividend',
                since=date.today() - timedelta(days=30),
                until=date.today() + timedelta(days=30),
            )
        except Exception as e:
            logger.warning("获取股息数据失败，回退到代理: %s", e)
            return self._proxy_dividend_signal(data)

        if div_df is None or div_df.empty:
            return self._proxy_dividend_signal(data)

        signal_df = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        today = date.today()

        for _, row in div_df.iterrows():
            sym = row.get('target_symbol', '')
            if sym not in close.columns:
                continue

            ex_dt = row.get('ex_date')
            if ex_dt is None:
                continue
            if isinstance(ex_dt, str):
                ex_dt = datetime.strptime(ex_dt, '%Y-%m-%d').date()

            cash = float(row.get('cash', 0) or 0)
            # 计算年化收益率（假设季度分红）
            last_price = close[sym].dropna().iloc[-1] if sym in close.columns else 0
            if last_price > 0:
                ann_yield = (cash * 4) / last_price * 100
            else:
                ann_yield = 0

            if ann_yield < self.min_yield_pct:
                continue

            # 趋势过滤
            if sym in close.columns:
                recent = close[sym].dropna()
                if len(recent) >= self.trend_lookback:
                    trend_ret = (recent.iloc[-1] / recent.iloc[-self.trend_lookback]) - 1
                    if trend_ret < self.trend_threshold:
                        continue

            # 事件窗口信号
            buy_start = ex_dt - timedelta(days=self.buy_days_before_ex + 2)
            sell_end = ex_dt + timedelta(days=self.sell_days_after_ex + 1)

            for dt in close.index:
                dt_date = dt.date() if hasattr(dt, 'date') else dt
                if buy_start <= dt_date <= sell_end:
                    # 信号强度与收益率成正比
                    signal_df.loc[dt, sym] = max(signal_df.loc[dt, sym], ann_yield)

        ranked = signal_df.rank(axis=1, pct=True) * 2 - 1
        return ranked

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成股息事件信号

        Parameters
        ----------
        data : dict
            含 'close' 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)
        """
        loader = alpaca_loader or self.alpaca_loader
        if loader is not None:
            self.alpaca_loader = loader
            return self._live_dividend_signal(data)
        return self._proxy_dividend_signal(data)


# ═══════════════════════════════════════════════════════════════════════
# 3. SplitMomentumStrategy — 拆股动量策略
# ═══════════════════════════════════════════════════════════════════════

class SplitMomentumStrategy(BaseStrategy):
    """拆股动量策略 — 拆股公告后买入持有60个交易日

    学术依据:
        Ikenberry, Rankine & Stice (1996): 拆股后股票倾向于在
        3-6个月内跑赢大盘。原因可能是管理层对公司前景的信心信号。

    逻辑:
        1. 追踪Alpaca企业行动中的拆股公告
        2. 拆股公告日买入
        3. 持有60个交易日后卖出
        4. 等权持仓

    回测模式:
        使用价格数据识别可能的拆股事件代理:
        - 股价处于历史高位 (拆股通常发生在高价时)
        - 短期快速上涨后 (管理层信心)
        - 成交量放大 (公告效应)
    """

    name = "Split Momentum"
    description = "拆股公告后动量效应: 买入持有60天"

    # 参数
    hold_days: int = 60             # 持有交易日数
    price_percentile: float = 0.9   # 高价代理阈值 (90百分位)

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化拆股动量策略"""
        self.alpaca_loader = alpaca_loader
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _proxy_split_signal(self, data: dict) -> pd.DataFrame:
        """用价格数据模拟拆股信号

        代理逻辑:
            1. 股价处于过去一年最高位附近 (>90百分位) → 可能拆股
            2. 过去20天涨幅显著 (>10%) → 管理层信心
            3. 模拟60天持有期: 用指数衰减窗口
        """
        close = data['close']

        # 过去252天百分位 (股价相对历史高位)
        rank_252 = close.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # 高价触发 = 处于90百分位以上
        high_price = (rank_252 > self.price_percentile).astype(float)

        # 短期动量确认: 20日涨幅
        mom_20 = close.pct_change(20)
        strong_mom = (mom_20 > 0.10).astype(float)

        # 拆股事件代理 = 高价 AND 强动量
        event_trigger = high_price * strong_mom

        # 模拟持有期: 过去hold_days内有触发则持仓
        # 使用rolling sum，任一天触发则信号=1
        hold_signal = event_trigger.rolling(self.hold_days, min_periods=1).max()

        # 截面排名
        signal = hold_signal.rank(axis=1, pct=True) * 2 - 1
        return signal

    def _live_split_signal(self, data: dict) -> pd.DataFrame:
        """使用Alpaca企业行动数据生成拆股信号"""
        close = data['close']

        try:
            split_df = self.alpaca_loader.get_corporate_actions(
                ca_type='split',
                since=date.today() - timedelta(days=180),
                until=date.today(),
            )
        except Exception as e:
            logger.warning("获取拆股数据失败，回退到代理: %s", e)
            return self._proxy_split_signal(data)

        if split_df is None or split_df.empty:
            return self._proxy_split_signal(data)

        signal_df = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for _, row in split_df.iterrows():
            sym = row.get('target_symbol', '')
            if sym not in close.columns:
                continue

            decl_date = row.get('declaration_date') or row.get('ex_date')
            if decl_date is None:
                continue
            if isinstance(decl_date, str):
                decl_date = datetime.strptime(decl_date, '%Y-%m-%d').date()

            # 从公告日起持有hold_days天
            hold_end = decl_date + timedelta(days=int(self.hold_days * 1.5))  # 日历日

            for dt in close.index:
                dt_date = dt.date() if hasattr(dt, 'date') else dt
                if decl_date <= dt_date <= hold_end:
                    signal_df.loc[dt, sym] = 1.0

        ranked = signal_df.rank(axis=1, pct=True) * 2 - 1
        return ranked

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成拆股动量信号

        Parameters
        ----------
        data : dict
            含 'close' 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)
        """
        loader = alpaca_loader or self.alpaca_loader
        if loader is not None:
            self.alpaca_loader = loader
            return self._live_split_signal(data)
        return self._proxy_split_signal(data)


# ═══════════════════════════════════════════════════════════════════════
# 4. OptionsSmartMoneyStrategy — 期权聪明钱逆向策略
# ═══════════════════════════════════════════════════════════════════════

class OptionsSmartMoneyStrategy(BaseStrategy):
    """期权聪明钱逆向策略 — 用看跌看涨比率作为逆向指标

    逻辑:
        1. 看跌看涨比率 (PCR) 作为市场情绪指标
        2. PCR > 1.5 = 极度恐惧 → 逆向买入
        3. PCR < 0.5 = 极度贪婪 → 逆向卖出
        4. 结合IV排名: PCR高 + IV排名高 = 最大恐惧 → 最强买入信号

    回测模式:
        PCR代理: 使用价格波动特征模拟
        - 急跌后反弹 → 高PCR代理
        - 高波动率 + 大跌 → 恐惧代理
        - 低波动率 + 连涨 → 贪婪代理
    """

    name = "Options Smart Money"
    description = "期权PCR逆向策略: 极度恐惧时买入，极度贪婪时卖出"

    # 参数
    pcr_buy_threshold: float = 1.5    # PCR买入阈值（高=恐惧）
    pcr_sell_threshold: float = 0.5   # PCR卖出阈值（低=贪婪）
    iv_rank_window: int = 252         # IV排名计算窗口
    pcr_weight: float = 0.6           # PCR信号权重
    iv_weight: float = 0.4            # IV排名信号权重

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化期权聪明钱策略"""
        self.alpaca_loader = alpaca_loader
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _proxy_pcr_signal(self, data: dict) -> pd.DataFrame:
        """用价格数据模拟PCR逆向信号

        代理逻辑:
            PCR代理 = 恐惧指标:
            - 实现波动率的z-score (高波动=恐惧)
            - 跌幅深度 (大跌=恐惧)
            - 逆向: 高恐惧→买入，低恐惧→卖出

            IV排名代理:
            - 过去252天实现波动率的百分位排名
        """
        close = data['close']
        ret = close.pct_change()

        # 实现波动率 (20日)
        realized_vol = ret.rolling(20).std() * np.sqrt(252)

        # 波动率z-score (截面)
        vol_zscore = realized_vol.apply(lambda row: (row - row.mean()) / row.std()
                                         if row.std() > 0 else row * 0, axis=1)

        # 跌幅深度: 过去10天累计收益（负=下跌=恐惧）
        cum_ret_10 = ret.rolling(10).sum()

        # 恐惧指标 = 高波动 + 大跌
        fear_index = vol_zscore * 0.5 + (-cum_ret_10).rank(axis=1, pct=True) * 0.5

        # IV排名代理: 当前波动率在过去252天中的百分位
        iv_rank = realized_vol.rolling(self.iv_rank_window, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # 逆向信号: 恐惧越高 → 买入信号越强
        pcr_signal = fear_index.rank(axis=1, pct=True) * 2 - 1
        iv_signal = iv_rank.rank(axis=1, pct=True) * 2 - 1

        # 组合: 高恐惧 + 高IV排名 = 最强买入
        combined = self.pcr_weight * pcr_signal + self.iv_weight * iv_signal
        return combined

    def _live_options_signal(self, data: dict) -> pd.DataFrame:
        """使用Alpaca期权数据生成信号"""
        close = data['close']
        symbols = list(close.columns)
        signal_df = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        pcr_scores = {}
        for sym in symbols:
            try:
                chain = self.alpaca_loader.get_option_chain(sym)
                if chain is None or (isinstance(chain, pd.DataFrame) and chain.empty):
                    continue

                if isinstance(chain, pd.DataFrame):
                    puts = chain[chain['type'] == 'put'] if 'type' in chain.columns else pd.DataFrame()
                    calls = chain[chain['type'] == 'call'] if 'type' in chain.columns else pd.DataFrame()

                    put_vol = puts['open_interest'].sum() if 'open_interest' in puts.columns and not puts.empty else 0
                    call_vol = calls['open_interest'].sum() if 'open_interest' in calls.columns and not calls.empty else 0

                    if call_vol > 0:
                        pcr = put_vol / call_vol
                    else:
                        pcr = 1.0

                    # 逆向: 高PCR = 买入
                    if pcr > self.pcr_buy_threshold:
                        pcr_scores[sym] = pcr  # 正向 (逆向买入)
                    elif pcr < self.pcr_sell_threshold:
                        pcr_scores[sym] = -1.0 / pcr  # 负向 (逆向卖出)
                    else:
                        pcr_scores[sym] = 0.0

            except Exception as e:
                logger.debug("获取 %s 期权链失败: %s", sym, e)
                continue

        if pcr_scores:
            pcr_series = _rank_signal(pd.Series(pcr_scores, dtype=float))
            for dt in close.index[-5:]:
                for sym in pcr_series.index:
                    if sym in signal_df.columns:
                        signal_df.loc[dt, sym] = pcr_series[sym]

        ranked = signal_df.rank(axis=1, pct=True) * 2 - 1
        return ranked

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成期权聪明钱逆向信号

        Parameters
        ----------
        data : dict
            含 'close' 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)
        """
        loader = alpaca_loader or self.alpaca_loader
        if loader is not None:
            self.alpaca_loader = loader
            return self._live_options_signal(data)
        return self._proxy_pcr_signal(data)


# ═══════════════════════════════════════════════════════════════════════
# 5. InstitutionalFlowStrategy — 机构资金流策略
# ═══════════════════════════════════════════════════════════════════════

class InstitutionalFlowStrategy(BaseStrategy):
    """机构资金流策略 — 追踪大单/机构买入信号

    逻辑:
        1. 成交笔数/成交量比率 (trade_count / volume) 作为机构流代理
        2. 低比率 = 大额块交易 = 机构交易
        3. 高比率 = 小额散户交易
        4. 跟随机构流: 检测到机构买入时买入

    微观结构特征:
        - 机构交易: 笔均金额大，比率低
        - 散户交易: 笔均金额小，比率高
        - 方向判断: 结合价格方向确定机构是买是卖

    回测模式:
        使用price × volume的变化检测大单:
        - 金额放大但笔数（代理: 波动率）降低 → 机构大单
    """

    name = "Institutional Flow"
    description = "机构资金流追踪: 检测并跟随机构买入"

    # 参数
    flow_window: int = 10            # 资金流检测窗口
    institutional_threshold: float = 0.3  # 机构流阈值 (低于30百分位)
    direction_weight: float = 0.5    # 方向确认权重

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化机构资金流策略"""
        self.alpaca_loader = alpaca_loader
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _proxy_institutional_flow(self, data: dict) -> pd.DataFrame:
        """用价格和成交量数据模拟机构资金流

        代理逻辑:
            1. 笔均金额代理: dollar_volume / volatility_proxy
               (高金额+低波动 = 机构有序执行)
            2. Amihud非流动性: |ret| / dollar_volume
               (机构交易时非流动性变化)
            3. 方向: 价格上涨时的机构流 → 机构买入
        """
        close = data['close']
        volume = data.get('volume', close * 0 + 1e6)

        ret = close.pct_change()
        dollar_vol = close * volume

        # 日内波动代理: 用日间收益的绝对值
        intraday_vol = ret.abs().rolling(self.flow_window).mean()

        # 笔均金额代理 (高=机构)
        # dollar_vol高 + 波动低 → 机构有序买入
        avg_trade_size = dollar_vol.rolling(self.flow_window).mean() / \
                         (intraday_vol.replace(0, np.nan) * 1e6)

        # Amihud非流动性倒数 (高流动性=机构参与)
        amihud = ret.abs() / (dollar_vol / 1e6).replace(0, np.nan)
        amihud_smooth = amihud.rolling(self.flow_window).mean()
        liquidity = 1.0 / amihud_smooth.replace(0, np.nan)

        # 机构流指标 = 大笔均金额 + 高流动性
        inst_flow = avg_trade_size.rank(axis=1, pct=True) * 0.5 + \
                    liquidity.rank(axis=1, pct=True) * 0.5

        # 方向确认: 价格上涨时的机构流 → 机构买入 → 看多
        price_direction = ret.rolling(self.flow_window).sum()
        direction_signal = np.sign(price_direction)

        # 最终信号: 机构流强度 × 方向
        raw_signal = inst_flow * (1 + self.direction_weight * direction_signal)

        # 截面排名
        signal = raw_signal.rank(axis=1, pct=True) * 2 - 1
        return signal

    def _live_institutional_flow(self, data: dict) -> pd.DataFrame:
        """使用Alpaca交易数据检测机构流"""
        close = data['close']
        symbols = list(close.columns)
        signal_df = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        inst_scores = {}
        for sym in symbols:
            try:
                # 获取最近交易数据
                trades = self.alpaca_loader.get_trades(
                    sym,
                    start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    limit=1000,
                )
                if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
                    continue

                if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                    # 计算笔均金额
                    if 'price' in trades.columns and 'size' in trades.columns:
                        total_dollar = (trades['price'] * trades['size']).sum()
                        trade_count = len(trades)
                        avg_size = total_dollar / trade_count if trade_count > 0 else 0

                        # 方向: 近期价格变化
                        recent_prices = close[sym].dropna()
                        if len(recent_prices) >= 5:
                            direction = 1 if recent_prices.iloc[-1] > recent_prices.iloc[-5] else -1
                        else:
                            direction = 0

                        inst_scores[sym] = avg_size * direction

            except Exception as e:
                logger.debug("获取 %s 交易数据失败: %s", sym, e)
                continue

        if inst_scores:
            ranked = _rank_signal(pd.Series(inst_scores, dtype=float))
            for dt in close.index[-5:]:
                for sym in ranked.index:
                    if sym in signal_df.columns:
                        signal_df.loc[dt, sym] = ranked[sym]

        final = signal_df.rank(axis=1, pct=True) * 2 - 1
        return final

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成机构资金流信号

        Parameters
        ----------
        data : dict
            含 'close' 和 'volume' 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)
        """
        loader = alpaca_loader or self.alpaca_loader
        if loader is not None:
            self.alpaca_loader = loader
            return self._live_institutional_flow(data)
        return self._proxy_institutional_flow(data)


# ═══════════════════════════════════════════════════════════════════════
# 6. MultiAlternativeStrategy — 多另类数据综合策略
# ═══════════════════════════════════════════════════════════════════════

class MultiAlternativeStrategy(BaseStrategy):
    """多另类数据综合策略 — 等权组合5个另类数据策略

    逻辑:
        将以下5个策略的信号等权组合:
        1. NewsAlpha        — 新闻情绪 + 动量确认
        2. DividendEvent    — 股息事件驱动
        3. SplitMomentum    — 拆股动量效应
        4. OptionsSmartMoney — 期权PCR逆向
        5. InstitutionalFlow — 机构资金流

    分散化优势:
        各策略依赖不同数据源和逻辑，相关性低，组合后:
        - 降低单一策略失效风险
        - 提高信号稳定性
        - 覆盖更多市场状态
    """

    name = "Multi Alternative"
    description = "5个另类数据策略等权综合"

    def __init__(self, alpaca_loader=None, **kwargs):
        """初始化多另类数据策略"""
        self.alpaca_loader = alpaca_loader
        self.sub_strategies = [
            NewsAlphaStrategy(alpaca_loader=alpaca_loader),
            DividendEventStrategy(alpaca_loader=alpaca_loader),
            SplitMomentumStrategy(alpaca_loader=alpaca_loader),
            OptionsSmartMoneyStrategy(alpaca_loader=alpaca_loader),
            InstitutionalFlowStrategy(alpaca_loader=alpaca_loader),
        ]
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def generate_signal(self, data: dict, alpaca_loader=None) -> pd.DataFrame:
        """生成多另类数据综合信号（等权平均）

        Parameters
        ----------
        data : dict
            含 'close' (和可选 'volume') 的数据字典
        alpaca_loader : AlpacaDataLoader, optional
            Alpaca数据加载器

        Returns
        -------
        pd.DataFrame
            信号矩阵 (date x symbol)
        """
        loader = alpaca_loader or self.alpaca_loader
        close = data['close']
        combined = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        n_valid = 0

        for strat in self.sub_strategies:
            try:
                sig = strat.generate_signal(data, alpaca_loader=loader)
                # 对齐
                common_idx = combined.index.intersection(sig.index)
                common_col = combined.columns.intersection(sig.columns)
                if len(common_idx) > 0 and len(common_col) > 0:
                    combined.loc[common_idx, common_col] += sig.loc[common_idx, common_col].fillna(0)
                    n_valid += 1
            except Exception as e:
                logger.warning("子策略 %s 失败: %s", strat.name, e)
                continue

        if n_valid > 0:
            combined = combined / n_valid

        # 最终截面排名
        signal = combined.rank(axis=1, pct=True) * 2 - 1
        return signal


# ═══════════════════════════════════════════════════════════════════════
# 回测入口
# ═══════════════════════════════════════════════════════════════════════

def run_alternative_backtests(
    symbols: Optional[List[str]] = None,
    start: str = '2022-01-01',
    end: str = '2025-12-31',
    long_n: int = 10,
    short_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """运行所有另类数据策略的回测（使用yfinance数据+代理信号）

    Parameters
    ----------
    symbols : list[str], optional
        股票列表，默认S&P 500子集
    start : str
        回测开始日期
    end : str
        回测结束日期
    long_n : int
        做多股票数
    short_n : int
        做空股票数

    Returns
    -------
    dict[str, pd.DataFrame]
        策略名 → 回测结果DataFrame
    """
    import yfinance as yf

    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA',
            'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'CSCO', 'PFE',
            'ABT', 'KO', 'PEP', 'TMO', 'COST', 'AVGO', 'MRK', 'NKE',
            'CVX', 'XOM', 'LLY', 'ABBV', 'DHR', 'TXN', 'QCOM', 'LOW',
            'NEE', 'MDT', 'HON', 'UPS', 'MS', 'GS', 'BLK', 'AMGN',
            'ISRG', 'SYK', 'GILD', 'ADP',
        ]

    print(f"下载 {len(symbols)} 只股票数据 ({start} ~ {end})...")
    raw = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker')

    # 解析价格和成交量
    close_dict = {}
    volume_dict = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                close_dict[sym] = raw['Close']
                volume_dict[sym] = raw['Volume']
            else:
                close_dict[sym] = raw[(sym, 'Close')]
                volume_dict[sym] = raw[(sym, 'Volume')]
        except (KeyError, TypeError):
            continue

    close_df = pd.DataFrame(close_dict).dropna(how='all')
    volume_df = pd.DataFrame(volume_dict).reindex(close_df.index).fillna(0)
    returns_df = close_df.pct_change().iloc[1:]

    # 去掉缺失太多的列
    valid_cols = close_df.columns[close_df.notna().sum() > len(close_df) * 0.5]
    close_df = close_df[valid_cols]
    volume_df = volume_df[valid_cols]
    returns_df = returns_df[valid_cols]

    data = {'close': close_df, 'volume': volume_df}

    print(f"数据准备完成: {len(close_df)} 天 × {len(valid_cols)} 只股票")
    print("=" * 70)

    # 定义所有策略 (回测模式，不传alpaca_loader)
    strategies = {
        'NewsAlpha': NewsAlphaStrategy(),
        'DividendEvent': DividendEventStrategy(),
        'SplitMomentum': SplitMomentumStrategy(),
        'OptionsSmartMoney': OptionsSmartMoneyStrategy(),
        'InstitutionalFlow': InstitutionalFlowStrategy(),
        'MultiAlternative': MultiAlternativeStrategy(),
    }

    results = {}
    for name, strat in strategies.items():
        print(f"\n{'─' * 50}")
        print(f"回测策略: {strat.name} ({strat.description})")
        print(f"{'─' * 50}")

        try:
            signal = strat.generate_signal(data)
            bt = _run_signal_backtest(
                signal, returns_df,
                long_n=long_n, short_n=short_n,
            )

            if bt.empty:
                print(f"  {name}: 无足够数据生成回测结果")
                continue

            summary = bt.attrs.get('summary', {})
            print(f"  年化收益: {summary.get('ann_return', 0):.2%}")
            print(f"  年化波动: {summary.get('ann_vol', 0):.2%}")
            print(f"  夏普比率: {summary.get('sharpe', 0):.2f}")
            print(f"  最大回撤: {summary.get('max_drawdown', 0):.2%}")
            print(f"  平均换手: {summary.get('avg_turnover', 0):.2%}")
            print(f"  交易天数: {summary.get('n_days', 0)}")

            results[name] = bt

        except Exception as e:
            print(f"  {name} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总比较
    if results:
        print(f"\n{'=' * 70}")
        print("策略比较汇总")
        print(f"{'=' * 70}")
        comparison = []
        for name, bt in results.items():
            s = bt.attrs.get('summary', {})
            comparison.append({
                '策略': name,
                '年化收益': f"{s.get('ann_return', 0):.2%}",
                '年化波动': f"{s.get('ann_vol', 0):.2%}",
                '夏普比率': f"{s.get('sharpe', 0):.2f}",
                '最大回撤': f"{s.get('max_drawdown', 0):.2%}",
                '换手率': f"{s.get('avg_turnover', 0):.2%}",
            })
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))

    return results


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_alternative_backtests()
