"""动量策略 — 默认策略"""
from qf.strategy import BaseStrategy
from qf.signals import build_signal


class MomentumStrategy(BaseStrategy):
    """
    多时间框架动量 + 加速度
    115/15 多空, 逆波动率加权
    """
    name = "Momentum 115/15"
    description = "Multi-timeframe momentum + acceleration, inverse-vol weighted"

    def __init__(self, w_mom=0.75, w_accel=0.25, w_quality=0.0, w_vol=0.0):
        self.w_mom = w_mom
        self.w_accel = w_accel
        self.w_quality = w_quality
        self.w_vol = w_vol

    def generate_signal(self, data):
        return build_signal(
            data['returns'], data['prices'], data['mktcap'], data['ccm_fund'],
            w_mom=self.w_mom, w_accel=self.w_accel,
            w_quality=self.w_quality, w_vol=self.w_vol,
        )


class MomentumQualityStrategy(BaseStrategy):
    """动量 + 质量 复合策略"""
    name = "Momentum + Quality"

    def __init__(self):
        self.long_n = 20
        self.short_n = 20

    def generate_signal(self, data):
        return build_signal(
            data['returns'], data['prices'], data['mktcap'], data['ccm_fund'],
            w_mom=0.50, w_accel=0.20, w_quality=0.20, w_vol=0.10,
        )


class PureMomentumStrategy(BaseStrategy):
    """纯12-1动量 (学术基准)"""
    name = "Pure 12-1 Momentum"
    long_pct = 1.0
    short_pct = 0.0
    short_n = 1

    def generate_signal(self, data):
        return build_signal(
            data['returns'], data['prices'], data['mktcap'], data['ccm_fund'],
            w_mom=1.0, w_accel=0.0, w_quality=0.0, w_vol=0.0,
        )
