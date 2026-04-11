"""
优化实盘策略 — 基于审计结果的最佳组合 + 实时过滤器
=================================================
核心: regime_blend 组合 (Sharpe 1.19, 审计最佳)
叠加: VWAP过滤 + 新闻情绪 + IV Rank + 动态仓位管理

用于 Alpaca 纸面盘/实盘的主策略。
"""
import numpy as np
import pandas as pd
from qf.strategy import BaseStrategy
from qf.optimizer import build_combo_signal, vol_target_scale, drawdown_scale


class OptimizedLiveStrategy(BaseStrategy):
    """
    优化实盘策略 — 多层信号融合 + 实时风控

    信号生成流程:
      1. regime_blend 四因子择时信号 (GPA+ROE+MOM12+FF5Alpha, risk_parity)
      2. VWAP 过滤: 仅做多日内价格 > VWAP 的股票
      3. 新闻情绪叠加: 正面新闻加分, 负面新闻减分
      4. IV Rank 叠加: 低IV Rank (期权便宜=恐慌低) 加分
      5. 动态仓位: 波动率目标10%年化 + 回撤控制
    """
    name = "Optimized Live (regime_blend + overlays)"
    description = "审计最佳组合 Sharpe 1.19 + VWAP/情绪/IV过滤 + 动态仓位"

    # 策略参数 — $100K 账户优化
    long_n: int = 30
    short_n: int = 10
    long_pct: float = 1.15
    short_pct: float = 0.15
    turnover_penalty: float = 0.25
    weight_mode: str = 'inv_vol'

    # 波动率目标参数
    target_vol: float = 0.10        # 10%年化目标波动率
    max_leverage: float = 1.5
    dd_control: bool = True

    # 叠加层权重
    vwap_filter_enabled: bool = True
    sentiment_overlay_weight: float = 0.10   # 情绪信号占比
    iv_rank_overlay_weight: float = 0.05     # IV Rank占比

    def __init__(self, alpaca_api=None, target_vol=0.10, long_n=30, short_n=10):
        """
        初始化优化实盘策略

        Parameters
        ----------
        alpaca_api : TradingClient, optional
            Alpaca API连接, 用于获取实时VWAP和账户信息
        target_vol : float
            年化目标波动率 (默认10%)
        long_n : int
            多头持仓数 (默认30, 适配$100K账户)
        short_n : int
            空头持仓数 (默认10)
        """
        self.alpaca_api = alpaca_api
        self.target_vol = target_vol
        self.long_n = long_n
        self.short_n = short_n
        self._returns_history = []
        self._pv_history = []

    def generate_signal(self, data):
        """
        生成优化信号 — 多层融合

        优先使用 regime_blend (Sharpe 1.19), 失败时回退到 GPA 或 composite。
        """
        signal = self._build_base_signal(data)
        signal = self._apply_vwap_filter(signal, data)
        signal = self._apply_sentiment_overlay(signal, data)
        signal = self._apply_iv_rank_overlay(signal, data)
        return signal

    def _build_base_signal(self, data):
        """
        构建基础信号 — 优先regime_blend, 回退GPA, 最终回退composite

        regime_blend: 择时四因子 (GPA+ROE+MOM12+FF5Alpha), risk_parity加权
        审计结果 Sharpe 1.19, 是所有组合中最优。
        """
        # 第一优先: regime_blend (Sharpe 1.19)
        try:
            signal = build_combo_signal('regime_blend', data, verbose=False)
            print("  [信号] regime_blend 择时四因子 (Sharpe 1.19)")
            return signal
        except Exception as e:
            print(f"  [警告] regime_blend 生成失败: {e}")

        # 第二优先: GPA单因子 (Sharpe 1.01)
        try:
            from strategies.factors import GPAStrategy
            gpa = GPAStrategy()
            signal = gpa.generate_signal(data)
            print("  [信号] 回退到 GPA 单因子 (Sharpe 1.01)")
            return signal
        except Exception as e:
            print(f"  [警告] GPA 生成失败: {e}")

        # 最终回退: composite (动量+质量)
        from qf.signals import build_signal
        signal = build_signal(
            data['returns'], data['prices'], data['mktcap'], data.get('ccm_fund'),
            w_mom=0.50, w_accel=0.20, w_quality=0.20, w_vol=0.10,
        )
        print("  [信号] 回退到 composite 动量+质量")
        return signal

    def _apply_vwap_filter(self, signal, data):
        """
        VWAP过滤: 仅做多日内价格在VWAP之上的股票

        原理: 价格 > VWAP 表示买方主导, 趋势向上。
        对多头信号, 如果价格低于VWAP则惩罚信号值。
        """
        if not self.vwap_filter_enabled:
            return signal

        vwap_data = self._fetch_vwap_data(data)
        if vwap_data is None:
            return signal

        # 价格低于VWAP的股票, 多头信号打折
        last_date = signal.index[-1]
        if last_date not in data['prices'].index:
            return signal

        prices_row = data['prices'].loc[last_date]
        vwap_row = vwap_data

        for col in signal.columns:
            price = prices_row.get(col, np.nan)
            vwap = vwap_row.get(col, np.nan)
            if pd.notna(price) and pd.notna(vwap) and vwap > 0:
                if price < vwap:
                    # 价格低于VWAP, 惩罚多头信号 (减20%)
                    if signal.loc[last_date, col] > 0.5:
                        signal.loc[last_date, col] *= 0.80

        return signal

    def _apply_sentiment_overlay(self, signal, data):
        """
        新闻情绪叠加: 正面新闻提升信号, 负面新闻压低信号

        情绪来源优先级:
          1. Alpaca news API (实盘)
          2. data字典中预计算的sentiment (回测)
          3. 无情绪数据时跳过
        """
        sentiment = self._fetch_sentiment(data)
        if sentiment is None:
            return signal

        last_date = signal.index[-1]
        weight = self.sentiment_overlay_weight

        for col in signal.columns:
            sent_score = sentiment.get(col, 0.0)
            if sent_score != 0.0:
                # 情绪分数 [-1, 1] 映射到信号调整
                # 正面情绪: 提升信号; 负面情绪: 压低信号
                adjustment = 1.0 + weight * sent_score
                signal.loc[last_date, col] *= adjustment

        return signal

    def _apply_iv_rank_overlay(self, signal, data):
        """
        IV Rank叠加: 低IV Rank = 市场恐慌低 = 更适合做多

        IV Rank 0-100, 低值表示当前隐含波动率相对历史较低。
        低IV Rank的股票获得信号加分 (期权便宜, 恐慌度低)。
        """
        iv_rank = self._fetch_iv_rank(data)
        if iv_rank is None:
            return signal

        last_date = signal.index[-1]
        weight = self.iv_rank_overlay_weight

        for col in signal.columns:
            ivr = iv_rank.get(col, 50.0)  # 默认中性
            # IV Rank 归一化到 [-1, 1]: 低IV=+1, 高IV=-1
            ivr_score = (50.0 - ivr) / 50.0
            adjustment = 1.0 + weight * ivr_score
            signal.loc[last_date, col] *= adjustment

        return signal

    def _fetch_vwap_data(self, data):
        """
        获取VWAP数据

        实盘: 通过Alpaca API获取日内VWAP
        回测: 使用 volume-weighted average 近似
        """
        if self.alpaca_api is not None:
            try:
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                from alpaca.data.historical import StockHistoricalDataClient
                # 实盘模式下从Alpaca获取VWAP
                # 这里返回近似值: 用日内bar的vwap字段
                return None  # 需要data client, 后续扩展
            except Exception:
                pass

        # 回测/离线模式: 用成交量加权价格近似
        if 'volume' in data and 'prices' in data:
            try:
                prices = data['prices']
                volume = data['volume']
                last_date = prices.index[-1]
                # 简单近似: 最近5期的量价加权均值
                lookback = min(5, len(prices))
                recent_prices = prices.iloc[-lookback:]
                recent_vol = volume.iloc[-lookback:].fillna(0)
                total_vol = recent_vol.sum()
                total_vol = total_vol.replace(0, np.nan)
                vwap = (recent_prices * recent_vol).sum() / total_vol
                return vwap
            except Exception:
                pass

        return None

    def _fetch_sentiment(self, data):
        """
        获取新闻情绪数据

        实盘: 通过Alpaca news API获取
        回测: 使用data字典中的预计算情绪
        返回: dict {permno: score} where score in [-1, 1]
        """
        # 检查data中是否有预计算的情绪数据
        if 'sentiment' in data:
            return data['sentiment']

        # 实盘模式: 通过Alpaca获取新闻情绪
        if self.alpaca_api is not None:
            try:
                # Alpaca News API 返回新闻, 需要外部NLP打分
                # 这里预留接口, 返回None表示无情绪数据
                return None
            except Exception:
                pass

        return None

    def _fetch_iv_rank(self, data):
        """
        获取IV Rank数据

        IV Rank = (当前IV - 52周最低IV) / (52周最高IV - 52周最低IV) * 100
        返回: dict {permno: iv_rank} where iv_rank in [0, 100]
        """
        if 'iv_rank' in data:
            return data['iv_rank']

        # 用历史波动率近似IV Rank
        if 'returns' in data:
            try:
                returns = data['returns']
                if len(returns) < 12:
                    return None
                # 当前实现波动率 (6个月)
                current_vol = returns.iloc[-6:].std() * np.sqrt(12)
                # 52周(12个月) 最高/最低波动率
                rolling_vol = returns.rolling(6).std() * np.sqrt(12)
                vol_max = rolling_vol.iloc[-12:].max()
                vol_min = rolling_vol.iloc[-12:].min()
                vol_range = vol_max - vol_min
                vol_range = vol_range.replace(0, np.nan)
                iv_rank = ((current_vol - vol_min) / vol_range * 100).fillna(50)
                return iv_rank.to_dict()
            except Exception:
                pass

        return None

    def compute_position_scale(self, returns_history=None, pv_history=None):
        """
        动态仓位缩放 — 基于波动率目标(10%年化) + 回撤控制

        Returns
        -------
        float
            仓位缩放因子 (0.25 ~ 1.5)
        """
        ret_hist = returns_history or self._returns_history
        pv_hist = pv_history or self._pv_history

        v_scale = vol_target_scale(
            ret_hist,
            target_vol=self.target_vol,
            lookback=6,
            max_leverage=self.max_leverage,
        )

        d_scale = 1.0
        if self.dd_control and len(pv_hist) >= 3:
            d_scale = drawdown_scale(pv_hist)

        return v_scale * d_scale

    def get_params(self) -> dict:
        """返回策略全部参数"""
        params = super().get_params()
        params.update({
            'target_vol': self.target_vol,
            'max_leverage': self.max_leverage,
            'dd_control': self.dd_control,
            'vwap_filter': self.vwap_filter_enabled,
            'sentiment_weight': self.sentiment_overlay_weight,
            'iv_rank_weight': self.iv_rank_overlay_weight,
            'base_signal': 'regime_blend (Sharpe 1.19)',
            'fallback_1': 'GPA (Sharpe 1.01)',
            'fallback_2': 'composite (mom+quality)',
        })
        return params
