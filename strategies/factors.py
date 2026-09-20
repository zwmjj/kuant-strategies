"""Standalone factor strategies — one class per academic factor"""
from qf.strategy import BaseStrategy
from qf.signals import build_factor_signal, build_signal


class MomentumStrategy(BaseStrategy):
    """Multi-timeframe momentum — Jegadeesh & Titman (1993)"""
    name = "Momentum 12-1"
    description = "Multi-timeframe momentum (3/6/12M), vol-adjusted"

    def generate_signal(self, data):
        return build_factor_signal('mom12', data)


class AccelerationStrategy(BaseStrategy):
    """Momentum acceleration — Gettleman & Marks (2006)"""
    name = "Momentum Acceleration"
    description = "Recent vs older 3M momentum difference"

    def generate_signal(self, data):
        return build_factor_signal('accel', data)


class High52Strategy(BaseStrategy):
    """Proximity to the 52-week high — George & Hwang (2004)"""
    name = "52-Week High"
    description = "Proximity to 12-month high price"

    def generate_signal(self, data):
        return build_factor_signal('high52', data)


class BookToMarketStrategy(BaseStrategy):
    """Book-to-market — Fama & French (1992)"""
    name = "Book-to-Market"
    description = "Value factor: high book equity / market equity"

    def generate_signal(self, data):
        return build_factor_signal('bm', data)


class EarningsPriceStrategy(BaseStrategy):
    """Earnings-to-price — Basu (1977)"""
    name = "Earnings-to-Price"
    description = "Value factor: high net income / market cap"

    def generate_signal(self, data):
        return build_factor_signal('ep', data)


class ROEStrategy(BaseStrategy):
    """Return on equity — Hou, Xue, Zhang (2015)"""
    name = "Return on Equity"
    description = "Profitability factor: high ROE"

    def generate_signal(self, data):
        return build_factor_signal('roe', data)


class GPAStrategy(BaseStrategy):
    """Gross profits-to-assets — Novy-Marx (2013)"""
    name = "Gross Profitability"
    description = "Quality factor: high gross profit / total assets"

    def generate_signal(self, data):
        return build_factor_signal('gpa', data)


class AssetGrowthStrategy(BaseStrategy):
    """Asset growth — Cooper, Gulen, Schill (2008)"""
    name = "Asset Growth (Inv.)"
    description = "Investment anomaly: low asset growth = higher returns"

    def generate_signal(self, data):
        return build_factor_signal('ag', data)


class IVolStrategy(BaseStrategy):
    """Idiosyncratic volatility — Ang et al. (2006)"""
    name = "Idiosyncratic Volatility"
    description = "Low IVOL anomaly: low residual vol = higher returns"

    def generate_signal(self, data):
        return build_factor_signal('ivol', data)


class FF5AlphaStrategy(BaseStrategy):
    """FF5 rolling alpha"""
    name = "FF5 Rolling Alpha"
    description = "36M rolling alpha from Fama-French 5-factor regression"

    def generate_signal(self, data):
        return build_factor_signal('ff5alpha', data)


class CompositeStrategy(BaseStrategy):
    """Composite momentum + quality strategy (the original default)"""
    name = "Composite Mom+Quality"
    description = "50% momentum + 20% accel + 20% quality + 10% vol"

    def generate_signal(self, data):
        return build_signal(
            data['returns'], data['prices'], data['mktcap'], data['ccm_fund'],
            w_mom=0.50, w_accel=0.20, w_quality=0.20, w_vol=0.10,
        )


# 策略注册表
STRATEGY_REGISTRY = {
    'mom12':     MomentumStrategy,
    'accel':     AccelerationStrategy,
    'high52':    High52Strategy,
    'bm':        BookToMarketStrategy,
    'ep':        EarningsPriceStrategy,
    'roe':       ROEStrategy,
    'gpa':       GPAStrategy,
    'ag':        AssetGrowthStrategy,
    'ivol':      IVolStrategy,
    'ff5alpha':  FF5AlphaStrategy,
    'composite': CompositeStrategy,
}


def get_strategy(factor_id):
    """Get a strategy instance by ID"""
    if factor_id not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {factor_id}. 可用: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[factor_id]()
