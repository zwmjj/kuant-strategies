"""期权交易策略集合 — 基于 Alpaca 期权数据的综合策略模块

包含8种期权策略 + 组合管理器 + 扫描入口函数:
    1. CoveredCallStrategy        — 备兑看涨 (持股+卖Call)
    2. CashSecuredPutStrategy     — 现金担保卖Put
    3. IVCrushStrategy            — 财报IV碾压 (卖跨式/宽跨式)
    4. VolatilityArbitrageStrategy — 波动率套利 (IV vs RV)
    5. PutSpreadIncomeStrategy    — 牛市看跌价差收入
    6. IronCondorStrategy         — 铁鹰策略 (双向价差)
    7. ProtectivePutStrategy      — 保护性看跌 (尾部对冲)
    8. GammaScalpingStrategy      — Gamma剥头皮

以及:
    - OptionsPortfolioManager     — 多策略组合管理 (Greeks汇总/仓位限制/P&L场景)
    - run_options_scan()          — 全策略扫描并按Sharpe排序
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qf.signals_options import OptionsSignalGenerator

logger = logging.getLogger(__name__)


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class TradeRecommendation:
    """单笔期权交易建议"""
    strategy: str           # 策略名称
    underlying: str         # 标的股票
    action: str             # 'sell_call', 'buy_put', 'sell_put', 'buy_call' 等
    contract_symbol: str    # Alpaca 期权合约符号
    strike: float           # 行权价
    expiry: str             # 到期日 YYYY-MM-DD
    option_type: str        # 'call' / 'put'
    quantity: int           # 合约数量 (正=买, 负=卖)
    premium: float          # 每股权利金
    underlying_price: float # 标的当前价
    dte: int                # 距到期天数
    # 计算指标
    expected_return: float = 0.0   # 预期收益率 (年化)
    max_loss: float = 0.0          # 最大亏损 (总额)
    prob_profit: float = 0.0       # 盈利概率
    sharpe_est: float = 0.0        # 预估Sharpe
    greeks: Dict = field(default_factory=dict)  # delta/gamma/theta/vega


@dataclass
class OrderDict:
    """Alpaca 执行订单格式"""
    symbol: str             # 期权合约符号
    qty: int                # 数量
    side: str               # 'buy' / 'sell'
    type: str               # 'limit' / 'market'
    time_in_force: str      # 'day' / 'gtc'
    limit_price: float = 0.0
    order_class: str = ''   # 'bracket' / 'oto' / ''
    legs: List = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为 Alpaca API 字典"""
        d = {
            'symbol': self.symbol,
            'qty': self.qty,
            'side': self.side,
            'type': self.type,
            'time_in_force': self.time_in_force,
        }
        if self.limit_price > 0:
            d['limit_price'] = round(self.limit_price, 2)
        if self.order_class:
            d['order_class'] = self.order_class
        if self.legs:
            d['legs'] = self.legs
        return d


# =====================================================================
# 工具函数
# =====================================================================

def _get_underlying_price(loader, symbol: str) -> Optional[float]:
    """获取标的当前价格"""
    try:
        snaps = loader.get_snapshots([symbol])
        return snaps.get(symbol, {}).get('last', None)
    except Exception as e:
        logger.warning("获取 %s 价格失败: %s", symbol, e)
        return None


def _get_chain_df(loader, symbol: str, sig_gen: OptionsSignalGenerator,
                  price: float) -> Optional[pd.DataFrame]:
    """获取并标准化期权链 DataFrame"""
    try:
        chain = loader.get_option_chain(symbol)
        if chain is None or (isinstance(chain, pd.DataFrame) and chain.empty):
            return None
        df = sig_gen.process_chain(chain, price)
        return df if not df.empty else None
    except Exception as e:
        logger.warning("获取 %s 期权链失败: %s", symbol, e)
        return None


def _get_historical_vol(loader, symbol: str, window: int = 20) -> float:
    """计算已实现波动率 (年化)"""
    try:
        end = date.today()
        start = end - timedelta(days=window * 2)
        bars = loader.get_daily_bars([symbol], start.isoformat(), end.isoformat())
        if bars is None or bars.empty:
            return np.nan
        # bars 可能包含多标的, 取出目标
        if 'symbol' in bars.columns:
            bars = bars[bars['symbol'] == symbol]
        if 'close' in bars.columns:
            prices = bars['close'].values
        elif isinstance(bars.index, pd.MultiIndex):
            prices = bars.xs(symbol, level='symbol')['close'].values if symbol in bars.index.get_level_values('symbol') else bars['close'].values
        else:
            prices = bars['close'].values
        if len(prices) < 5:
            return np.nan
        log_ret = np.diff(np.log(prices))
        return float(np.std(log_ret) * np.sqrt(252))
    except Exception:
        return np.nan


def _bs_delta(S: float, K: float, T: float, sigma: float,
              r: float = 0.05, opt_type: str = 'call') -> float:
    """Black-Scholes Delta 近似"""
    S, K, T, sigma, r = float(S), float(K), float(T), float(sigma), float(r)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    from math import log, sqrt, exp
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0
    # 标准正态CDF近似
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    if opt_type == 'call':
        return nd1
    else:
        return nd1 - 1.0


def _bs_gamma(S: float, K: float, T: float, sigma: float,
              r: float = 0.05) -> float:
    """Black-Scholes Gamma"""
    S, K, T, sigma, r = float(S), float(K), float(T), float(sigma), float(r)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    nd1_pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return nd1_pdf / (S * sigma * math.sqrt(T))


def _bs_theta(S: float, K: float, T: float, sigma: float,
              r: float = 0.05, opt_type: str = 'call') -> float:
    """Black-Scholes Theta (每天)"""
    S, K, T, sigma, r = float(S), float(K), float(T), float(sigma), float(r)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1_pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    # 每日theta
    theta = -(S * nd1_pdf * sigma) / (2 * math.sqrt(T))
    nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    if opt_type == 'call':
        theta -= r * K * math.exp(-r * T) * nd2
    else:
        theta += r * K * math.exp(-r * T) * (1 - nd2)
    return theta / 365.0


def _bs_vega(S: float, K: float, T: float, sigma: float,
             r: float = 0.05) -> float:
    """Black-Scholes Vega (per 1% vol change)"""
    S, K, T, sigma, r = float(S), float(K), float(T), float(sigma), float(r)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    nd1_pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return S * nd1_pdf * math.sqrt(T) * 0.01


def _select_contract(chain_df: pd.DataFrame, opt_type: str,
                     moneyness_range: Tuple[float, float],
                     dte_range: Tuple[int, int]) -> Optional[pd.Series]:
    """从期权链中选择最佳合约 (最接近目标moneyness中点, 流动性最好)"""
    if chain_df is None or chain_df.empty:
        return None
    # 确保数值类型
    chain_df = chain_df.copy()
    for col in ['strike', 'bid', 'ask', 'mid', 'moneyness', 'dte']:
        if col in chain_df.columns:
            chain_df[col] = pd.to_numeric(chain_df[col], errors='coerce')
    # 标准化type列 (可能是 'call'/'put' 或 'ContractType.CALL')
    if 'type' in chain_df.columns:
        chain_df['type'] = chain_df['type'].astype(str).str.lower()
        chain_df['type'] = chain_df['type'].str.replace('contracttype.', '')
    opt_type = opt_type.lower().replace('contracttype.', '')
    mask = (
        (chain_df['type'].str.contains(opt_type))
        & (chain_df['moneyness'] >= moneyness_range[0])
        & (chain_df['moneyness'] <= moneyness_range[1])
        & (chain_df['dte'] >= dte_range[0])
        & (chain_df['dte'] <= dte_range[1])
    )
    filtered = chain_df.loc[mask].copy()
    if filtered.empty:
        return None
    # 选最接近moneyness目标中点的
    target = (moneyness_range[0] + moneyness_range[1]) / 2
    filtered['_dist'] = abs(filtered['moneyness'] - target)
    best = filtered.sort_values(['_dist', 'bid'], ascending=[True, False]).iloc[0]
    return best


def _build_contract_symbol(underlying: str, expiry: str,
                           opt_type: str, strike: float) -> str:
    """构建 OCC 格式期权符号: AAPL260424C00170000"""
    # expiry: YYYY-MM-DD -> YYMMDD
    parts = expiry.split('-')
    date_str = parts[0][2:] + parts[1] + parts[2]
    type_char = 'C' if opt_type == 'call' else 'P'
    strike_int = int(strike * 1000)
    return f"{underlying}{date_str}{type_char}{strike_int:08d}"


# =====================================================================
# 策略基类
# =====================================================================

class OptionsStrategy(ABC):
    """期权策略基类 — 所有期权策略实现此接口"""

    name: str = "未命名期权策略"
    description: str = ""

    def __init__(self, loader, symbols: List[str]):
        """初始化期权策略

        Parameters
        ----------
        loader : AlpacaDataLoader
            Alpaca 数据加载器实例
        symbols : list
            标的股票列表
        """
        self.loader = loader
        self.symbols = symbols
        self.sig_gen = OptionsSignalGenerator()
        self._prices: Dict[str, float] = {}
        self._chains: Dict[str, pd.DataFrame] = {}

    def _load_data(self):
        """加载所有标的的价格和期权链"""
        for sym in self.symbols:
            price = _get_underlying_price(self.loader, sym)
            if price is None or price <= 0:
                logger.warning("跳过 %s: 无法获取价格", sym)
                continue
            self._prices[sym] = price
            chain = _get_chain_df(self.loader, sym, self.sig_gen, price)
            if chain is not None:
                self._chains[sym] = chain

    @abstractmethod
    def scan(self) -> List[TradeRecommendation]:
        """扫描并返回交易建议列表"""
        raise NotImplementedError

    @abstractmethod
    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算交易的预期收益/最大亏损/盈利概率"""
        raise NotImplementedError

    def generate_orders(self) -> List[dict]:
        """生成 Alpaca 可执行订单列表"""
        trades = self.scan()
        orders = []
        for t in trades:
            t = self.calc_metrics(t)
            side = 'sell' if t.quantity < 0 else 'buy'
            order = OrderDict(
                symbol=t.contract_symbol,
                qty=abs(t.quantity),
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=t.premium,
            )
            orders.append(order.to_dict())
        return orders


# =====================================================================
# 1. CoveredCallStrategy — 备兑看涨
# =====================================================================

class CoveredCallStrategy(OptionsStrategy):
    """备兑看涨策略 — 持有股票 + 卖出虚值Call收取权利金

    选股逻辑: 高IV排名 (权利金贵) + 正动量 (不怕卖飞)
    合约选择: 5-10% OTM, 30-45 DTE
    滚仓: DTE < 7 或 delta > 0.7 时滚动到下一周期
    预期: 降低波动率, 稳定收入, Sharpe提升
    """

    name = "备兑看涨策略"
    description = "持有股票 + 卖出OTM Call, 赚取权利金"

    # 参数
    otm_min: float = 1.05       # 最小moneyness (5% OTM)
    otm_max: float = 1.10       # 最大moneyness (10% OTM)
    dte_min: int = 30           # 最短到期天数
    dte_max: int = 45           # 最长到期天数
    roll_dte: int = 7           # 滚仓触发: DTE阈值
    roll_delta: float = 0.70    # 滚仓触发: delta阈值
    contracts_per_100: int = 1  # 每100股卖1手

    def scan(self) -> List[TradeRecommendation]:
        """扫描所有标的, 筛选适合备兑看涨的机会"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            # 计算IV排名 — 用当前ATM IV vs 近似历史
            atm_iv = self.sig_gen.atm_iv(chain, price, days_range=(20, 60))
            if np.isnan(atm_iv):
                continue

            # 选择OTM Call
            contract = _select_contract(
                chain, 'call',
                moneyness_range=(self.otm_min, self.otm_max),
                dte_range=(self.dte_min, self.dte_max),
            )
            if contract is None:
                continue

            strike = contract['strike']
            premium = contract.get('mid', contract.get('bid', 0))
            dte = int(contract['dte'])
            expiry = str(contract['expiry'])[:10]

            if premium <= 0:
                continue

            contract_sym = contract.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', strike)

            T = dte / 365.0
            delta = _bs_delta(price, strike, T, atm_iv, opt_type='call')

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='sell_call',
                contract_symbol=contract_sym,
                strike=strike,
                expiry=expiry,
                option_type='call',
                quantity=-self.contracts_per_100,
                premium=premium,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': -delta,  # 卖Call的delta是负的
                    'gamma': -_bs_gamma(price, strike, T, atm_iv),
                    'theta': -_bs_theta(price, strike, T, atm_iv, opt_type='call'),
                    'vega': -_bs_vega(price, strike, T, atm_iv),
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算备兑看涨的预期收益、最大亏损、盈利概率"""
        S = trade.underlying_price
        K = trade.strike
        prem = trade.premium
        T = trade.dte / 365.0

        # 最大收益 = 权利金 + (K - S) — 若股价涨到行权价
        max_profit = prem * 100 + (K - S) * 100
        # 盈亏平衡 = S - 权利金 (持有股票成本减少)
        breakeven = S - prem
        # 最大亏损 = 理论上股价归零 (但有权利金缓冲)
        trade.max_loss = (S - prem) * 100  # 持有100股的最大亏损
        # 预期收益 (年化): 权利金/股价, 年化
        if T > 0:
            trade.expected_return = (prem / S) / T
        # 盈利概率: 近似 = 1 - delta (卖Call盈利 ≈ 股价不超过K)
        delta_abs = abs(trade.greeks.get('delta', 0.3))
        trade.prob_profit = 1.0 - delta_abs
        # 估算Sharpe: (年化收益 - 无风险) / 年化vol
        rv = 0.25  # 保守估计
        if trade.expected_return > 0 and rv > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / rv

        return trade

    def should_roll(self, current_dte: int, current_delta: float) -> bool:
        """判断是否需要滚仓"""
        return current_dte < self.roll_dte or abs(current_delta) > self.roll_delta


# =====================================================================
# 2. CashSecuredPutStrategy — 现金担保卖Put
# =====================================================================

class CashSecuredPutStrategy(OptionsStrategy):
    """现金担保卖Put策略 — 卖出虚值Put收取权利金, 如被行权则以折扣价买入

    选股逻辑: 低IV排名 (权利金便宜 = 标的被低估) + 高质量 (GPA/ROE)
    合约选择: 5-10% OTM Put, 30-45 DTE
    管理: 50%利润平仓, 7DTE滚仓
    预期: 收入 + 以折扣价买入优质股票
    """

    name = "现金担保卖Put策略"
    description = "卖出OTM Put, 收取权利金或以折扣价买入股票"

    otm_min: float = 0.90
    otm_max: float = 0.95
    dte_min: int = 30
    dte_max: int = 45
    profit_take: float = 0.50   # 50%利润平仓
    roll_dte: int = 7

    def scan(self) -> List[TradeRecommendation]:
        """扫描适合卖Put的标的"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            if np.isnan(atm_iv):
                continue

            # 选择OTM Put
            contract = _select_contract(
                chain, 'put',
                moneyness_range=(self.otm_min, self.otm_max),
                dte_range=(self.dte_min, self.dte_max),
            )
            if contract is None:
                continue

            strike = contract['strike']
            premium = contract.get('mid', contract.get('bid', 0))
            dte = int(contract['dte'])
            expiry = str(contract['expiry'])[:10]

            if premium <= 0:
                continue

            contract_sym = contract.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', strike)

            T = dte / 365.0
            delta = _bs_delta(price, strike, T, atm_iv, opt_type='put')

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='sell_put',
                contract_symbol=contract_sym,
                strike=strike,
                expiry=expiry,
                option_type='put',
                quantity=-1,
                premium=premium,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': -delta,  # 卖Put的delta是正的 (因为put delta为负)
                    'gamma': -_bs_gamma(price, strike, T, atm_iv),
                    'theta': -_bs_theta(price, strike, T, atm_iv, opt_type='put'),
                    'vega': -_bs_vega(price, strike, T, atm_iv),
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算现金担保卖Put的指标"""
        K = trade.strike
        prem = trade.premium
        T = trade.dte / 365.0

        # 最大收益 = 全部权利金 (每手100股)
        max_profit = prem * 100
        # 最大亏损 = (行权价 - 权利金) * 100 (股价归零时)
        trade.max_loss = (K - prem) * 100
        # 盈亏平衡 = K - 权利金
        breakeven = K - prem
        # 年化收益 = 权利金 / 现金担保额
        cash_secured = K * 100
        if T > 0 and cash_secured > 0:
            trade.expected_return = (max_profit / cash_secured) / T
        # 盈利概率: ≈ 1 - |delta| (因为卖的是OTM put)
        delta_abs = abs(trade.greeks.get('delta', 0.2))
        trade.prob_profit = 1.0 - delta_abs
        # Sharpe估算
        rv = 0.20
        if trade.expected_return > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / rv

        return trade

    def should_close(self, current_premium: float, entry_premium: float) -> bool:
        """判断是否应该平仓 (50%利润)"""
        profit_pct = 1.0 - (current_premium / entry_premium)
        return profit_pct >= self.profit_take


# =====================================================================
# 3. IVCrushStrategy — 财报IV碾压
# =====================================================================

class IVCrushStrategy(OptionsStrategy):
    """财报IV碾压策略 — 利用财报前IV泵升, 财报后IV骤降获利

    原理: 财报前IV通常被高估 (市场过度定价不确定性),
          实际波动幅度 < 隐含波动幅度 约75%的时间
    操作: 财报前卖出跨式/宽跨式, 财报后IV碾压时买回
    风控: 最大亏损 = 2x 收取的权利金
    """

    name = "财报IV碾压策略"
    description = "财报前卖跨式/宽跨式, 赚取IV碾压收益"

    # 宽跨式参数
    strangle_call_otm: float = 1.05   # Call OTM 5%
    strangle_put_otm: float = 0.95    # Put OTM 5%
    target_dte: Tuple[int, int] = (5, 15)  # 财报前1-2周
    iv_rank_threshold: float = 0.70   # IV排名 > 70% 才做
    max_loss_multiplier: float = 2.0  # 止损 = 2x权利金

    def scan(self) -> List[TradeRecommendation]:
        """扫描高IV (疑似临近财报) 标的"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            if np.isnan(atm_iv):
                continue

            # 用历史波动率近似IV排名 — IV远高于RV说明可能有事件
            rv = _get_historical_vol(self.loader, sym, window=20)
            if np.isnan(rv) or rv <= 0:
                continue

            iv_rv_ratio = atm_iv / rv
            # IV/RV > 1.5 认为可能临近财报 (IV泵升)
            if iv_rv_ratio < 1.5:
                continue

            # 选Call腿 (OTM)
            call_contract = _select_contract(
                chain, 'call',
                moneyness_range=(self.strangle_call_otm, self.strangle_call_otm + 0.05),
                dte_range=self.target_dte,
            )
            # 选Put腿 (OTM)
            put_contract = _select_contract(
                chain, 'put',
                moneyness_range=(self.strangle_put_otm - 0.05, self.strangle_put_otm),
                dte_range=self.target_dte,
            )

            if call_contract is None or put_contract is None:
                continue

            call_prem = call_contract.get('mid', call_contract.get('bid', 0))
            put_prem = put_contract.get('mid', put_contract.get('bid', 0))
            total_prem = call_prem + put_prem

            if total_prem <= 0:
                continue

            call_strike = call_contract['strike']
            put_strike = put_contract['strike']
            dte = int(call_contract['dte'])
            expiry = str(call_contract['expiry'])[:10]

            T = dte / 365.0

            # 构建两腿交易 — 用Call腿作为主合约记录
            call_sym = call_contract.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', call_strike)
            put_sym = put_contract.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', put_strike)

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='sell_strangle',
                contract_symbol=f"{call_sym}|{put_sym}",
                strike=call_strike,  # 记录Call strike, Put strike在描述中
                expiry=expiry,
                option_type='strangle',
                quantity=-1,
                premium=total_prem,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': -(_bs_delta(price, call_strike, T, atm_iv, opt_type='call')
                               + _bs_delta(price, put_strike, T, atm_iv, opt_type='put')),
                    'gamma': -(_bs_gamma(price, call_strike, T, atm_iv)
                               + _bs_gamma(price, put_strike, T, atm_iv)),
                    'theta': -(_bs_theta(price, call_strike, T, atm_iv, opt_type='call')
                               + _bs_theta(price, put_strike, T, atm_iv, opt_type='put')),
                    'vega': -(_bs_vega(price, call_strike, T, atm_iv)
                              + _bs_vega(price, put_strike, T, atm_iv)),
                    'iv_rv_ratio': iv_rv_ratio,
                    'put_strike': put_strike,
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算IV碾压策略指标"""
        prem = trade.premium
        S = trade.underlying_price
        call_K = trade.strike
        put_K = trade.greeks.get('put_strike', trade.strike * 0.95)

        # 最大亏损 = 止损线 (2x权利金)
        trade.max_loss = self.max_loss_multiplier * prem * 100
        # 盈亏平衡 = Call_K + prem 或 Put_K - prem
        upper_be = call_K + prem
        lower_be = put_K - prem
        # 隐含波动范围
        implied_range = (upper_be - lower_be) / S
        # 盈利概率: 经验值 ~70-80% (IV通常高估)
        trade.prob_profit = 0.75
        # 预期收益: IV碾压后 ~50-70% 权利金利润
        expected_profit = 0.60 * prem * 100
        capital_at_risk = trade.max_loss
        T = trade.dte / 365.0
        if T > 0 and capital_at_risk > 0:
            trade.expected_return = (expected_profit / capital_at_risk) / T
        # Sharpe
        if trade.expected_return > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / 0.40

        return trade

    def generate_orders(self) -> List[dict]:
        """生成宽跨式两腿订单"""
        trades = self.scan()
        orders = []
        for t in trades:
            t = self.calc_metrics(t)
            syms = t.contract_symbol.split('|')
            if len(syms) == 2:
                # 卖Call腿
                orders.append(OrderDict(
                    symbol=syms[0], qty=1, side='sell',
                    type='limit', time_in_force='day',
                    limit_price=t.premium * 0.55,  # 近似Call腿权利金
                ).to_dict())
                # 卖Put腿
                orders.append(OrderDict(
                    symbol=syms[1], qty=1, side='sell',
                    type='limit', time_in_force='day',
                    limit_price=t.premium * 0.45,  # 近似Put腿权利金
                ).to_dict())
        return orders


# =====================================================================
# 4. VolatilityArbitrageStrategy — 波动率套利
# =====================================================================

class VolatilityArbitrageStrategy(OptionsStrategy):
    """波动率套利策略 — 交易IV与RV之间的价差

    核心指标: IV/RV 比率
        - IV/RV > 1.3: IV高估 → 卖期权 (IV会均值回归下降)
        - IV/RV < 0.7: IV低估 → 买期权 (IV会均值回归上升)
    Delta对冲: 隔离纯波动率交易 (消除方向性风险)
    """

    name = "波动率套利策略"
    description = "IV vs RV 套利, delta对冲隔离纯vol交易"

    iv_rv_sell_threshold: float = 1.30   # 卖出阈值
    iv_rv_buy_threshold: float = 0.70    # 买入阈值
    dte_min: int = 20
    dte_max: int = 45
    hedge_frequency: str = 'daily'  # delta对冲频率

    def scan(self) -> List[TradeRecommendation]:
        """扫描IV/RV偏离的标的"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            rv = _get_historical_vol(self.loader, sym, window=20)

            if np.isnan(atm_iv) or np.isnan(rv) or rv <= 0:
                continue

            ratio = atm_iv / rv

            if ratio > self.iv_rv_sell_threshold:
                # IV高估 — 卖ATM跨式
                direction = 'sell'
                action = 'sell_straddle'
                qty = -1
            elif ratio < self.iv_rv_buy_threshold:
                # IV低估 — 买ATM跨式
                direction = 'buy'
                action = 'buy_straddle'
                qty = 1
            else:
                continue  # 无信号

            # 选ATM Call + ATM Put
            call = _select_contract(
                chain, 'call',
                moneyness_range=(0.97, 1.03),
                dte_range=(self.dte_min, self.dte_max),
            )
            put = _select_contract(
                chain, 'put',
                moneyness_range=(0.97, 1.03),
                dte_range=(self.dte_min, self.dte_max),
            )

            if call is None or put is None:
                continue

            call_prem = call.get('mid', 0)
            put_prem = put.get('mid', 0)
            total_prem = call_prem + put_prem

            if total_prem <= 0:
                continue

            dte = int(call['dte'])
            expiry = str(call['expiry'])[:10]
            T = dte / 365.0

            call_sym = call.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', call['strike'])
            put_sym = put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', put['strike'])

            # Delta对冲: 初始delta
            call_delta = _bs_delta(price, call['strike'], T, atm_iv, opt_type='call')
            put_delta = _bs_delta(price, put['strike'], T, atm_iv, opt_type='put')
            net_delta = (call_delta + put_delta) * qty
            hedge_shares = -int(round(net_delta * 100))  # 需要对冲的股数

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action=action,
                contract_symbol=f"{call_sym}|{put_sym}",
                strike=call['strike'],
                expiry=expiry,
                option_type='straddle',
                quantity=qty,
                premium=total_prem,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': net_delta,
                    'gamma': (_bs_gamma(price, call['strike'], T, atm_iv)
                              + _bs_gamma(price, put['strike'], T, atm_iv)) * qty,
                    'theta': (_bs_theta(price, call['strike'], T, atm_iv, opt_type='call')
                              + _bs_theta(price, put['strike'], T, atm_iv, opt_type='put')) * qty,
                    'vega': (_bs_vega(price, call['strike'], T, atm_iv)
                             + _bs_vega(price, put['strike'], T, atm_iv)) * qty,
                    'iv': atm_iv,
                    'rv': rv,
                    'iv_rv_ratio': ratio,
                    'hedge_shares': hedge_shares,
                    'put_strike': put['strike'],
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算波动率套利策略指标"""
        prem = trade.premium
        S = trade.underlying_price
        T = trade.dte / 365.0
        iv = trade.greeks.get('iv', 0.30)
        rv = trade.greeks.get('rv', 0.25)

        if trade.quantity < 0:
            # 卖跨式: 赚IV下降
            # 预期P&L ≈ vega * (IV - RV) * sqrt(T) (简化)
            iv_diff = iv - rv
            expected_pnl = abs(trade.greeks.get('vega', 0)) * iv_diff * 100 * 100
            trade.max_loss = prem * 100 * 2  # 止损2x
            trade.prob_profit = 0.65 if iv / rv > 1.3 else 0.50
        else:
            # 买跨式: 赚IV上升 + gamma
            iv_diff = rv - iv
            expected_pnl = abs(trade.greeks.get('vega', 0)) * iv_diff * 100 * 100
            trade.max_loss = prem * 100  # 最大亏损=全部权利金
            trade.prob_profit = 0.55 if rv / iv > 1.3 else 0.40

        if T > 0 and trade.max_loss > 0:
            trade.expected_return = (expected_pnl / trade.max_loss) / T
        trade.sharpe_est = (trade.expected_return - 0.05) / 0.35 if trade.expected_return > 0 else 0

        return trade


# =====================================================================
# 5. PutSpreadIncomeStrategy — 牛市看跌价差
# =====================================================================

class PutSpreadIncomeStrategy(OptionsStrategy):
    """牛市看跌价差收入策略 — 卖高行权Put + 买低行权Put (限定亏损)

    选股: 强支撑 (20日低点) + 正动量
    目标: 30-delta 短腿 (约70%盈利概率)
    风控: 最大亏损 = 价差宽度 - 权利金
    """

    name = "牛市看跌价差收入策略"
    description = "卖高行权Put + 买低行权Put, 定义风险收取权利金"

    short_put_moneyness: Tuple[float, float] = (0.92, 0.97)  # ~30 delta
    spread_width_pct: float = 0.05   # 价差宽度 = 5%
    dte_min: int = 30
    dte_max: int = 45

    def scan(self) -> List[TradeRecommendation]:
        """扫描看跌价差机会"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            if np.isnan(atm_iv):
                continue

            # 选短腿 (卖的Put, 较高strike)
            short_put = _select_contract(
                chain, 'put',
                moneyness_range=self.short_put_moneyness,
                dte_range=(self.dte_min, self.dte_max),
            )
            if short_put is None:
                continue

            short_strike = short_put['strike']
            short_prem = short_put.get('mid', short_put.get('bid', 0))

            # 长腿 (买的Put, 较低strike)
            long_strike_target = short_strike * (1 - self.spread_width_pct)
            long_put = _select_contract(
                chain, 'put',
                moneyness_range=(
                    long_strike_target / price - 0.02,
                    long_strike_target / price + 0.02,
                ),
                dte_range=(self.dte_min, self.dte_max),
            )
            if long_put is None:
                continue

            long_strike = long_put['strike']
            long_prem = long_put.get('mid', long_put.get('ask', 0))

            # 净权利金 = 卖短腿 - 买长腿
            net_prem = short_prem - long_prem
            if net_prem <= 0:
                continue

            dte = int(short_put['dte'])
            expiry = str(short_put['expiry'])[:10]
            T = dte / 365.0

            short_sym = short_put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', short_strike)
            long_sym = long_put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', long_strike)

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='bull_put_spread',
                contract_symbol=f"{short_sym}|{long_sym}",
                strike=short_strike,
                expiry=expiry,
                option_type='put_spread',
                quantity=-1,
                premium=net_prem,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': -(_bs_delta(price, short_strike, T, atm_iv, opt_type='put')
                               - _bs_delta(price, long_strike, T, atm_iv, opt_type='put')),
                    'gamma': -(_bs_gamma(price, short_strike, T, atm_iv)
                               - _bs_gamma(price, long_strike, T, atm_iv)),
                    'theta': -(_bs_theta(price, short_strike, T, atm_iv, opt_type='put')
                               - _bs_theta(price, long_strike, T, atm_iv, opt_type='put')),
                    'vega': -(_bs_vega(price, short_strike, T, atm_iv)
                              - _bs_vega(price, long_strike, T, atm_iv)),
                    'long_strike': long_strike,
                    'short_premium': short_prem,
                    'long_premium': long_prem,
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算牛市看跌价差指标"""
        net_prem = trade.premium
        short_K = trade.strike
        long_K = trade.greeks.get('long_strike', short_K * 0.95)
        T = trade.dte / 365.0

        spread_width = short_K - long_K
        # 最大亏损 = (价差宽度 - 净权利金) * 100
        trade.max_loss = (spread_width - net_prem) * 100
        # 最大收益 = 净权利金 * 100
        max_profit = net_prem * 100
        # 盈利概率 ≈ 70% (30-delta短腿)
        trade.prob_profit = 0.70
        # 年化收益
        if T > 0 and trade.max_loss > 0:
            expected_pnl = max_profit * trade.prob_profit - trade.max_loss * (1 - trade.prob_profit)
            trade.expected_return = (expected_pnl / trade.max_loss) / T
        # Sharpe
        if trade.expected_return > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / 0.30

        return trade

    def generate_orders(self) -> List[dict]:
        """生成价差两腿订单"""
        trades = self.scan()
        orders = []
        for t in trades:
            t = self.calc_metrics(t)
            syms = t.contract_symbol.split('|')
            if len(syms) == 2:
                # 卖高strike Put
                orders.append(OrderDict(
                    symbol=syms[0], qty=1, side='sell',
                    type='limit', time_in_force='day',
                    limit_price=t.greeks.get('short_premium', t.premium * 0.6),
                ).to_dict())
                # 买低strike Put
                orders.append(OrderDict(
                    symbol=syms[1], qty=1, side='buy',
                    type='limit', time_in_force='day',
                    limit_price=t.greeks.get('long_premium', t.premium * 0.4),
                ).to_dict())
        return orders


# =====================================================================
# 6. IronCondorStrategy — 铁鹰策略
# =====================================================================

class IronCondorStrategy(OptionsStrategy):
    """铁鹰策略 — 同时卖出OTM Put价差 + OTM Call价差

    适用: 低波动横盘市场 (低vol-of-vol, 窄布林带)
    结构: 卖OTM Put + 买更OTM Put + 卖OTM Call + 买更OTM Call
    宽度: 每边10-15% OTM
    预期: 横盘市场高胜率
    """

    name = "铁鹰策略"
    description = "双向价差, 赚取横盘市场权利金"

    put_short_moneyness: Tuple[float, float] = (0.88, 0.92)
    put_long_offset: float = 0.03    # 长腿比短腿低3%
    call_short_moneyness: Tuple[float, float] = (1.08, 1.12)
    call_long_offset: float = 0.03   # 长腿比短腿高3%
    dte_min: int = 30
    dte_max: int = 45

    def scan(self) -> List[TradeRecommendation]:
        """扫描适合铁鹰的横盘标的"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            if np.isnan(atm_iv):
                continue

            # Put短腿
            short_put = _select_contract(
                chain, 'put',
                moneyness_range=self.put_short_moneyness,
                dte_range=(self.dte_min, self.dte_max),
            )
            # Call短腿
            short_call = _select_contract(
                chain, 'call',
                moneyness_range=self.call_short_moneyness,
                dte_range=(self.dte_min, self.dte_max),
            )
            if short_put is None or short_call is None:
                continue

            # Put长腿 (更低strike)
            put_long_target = short_put['strike'] * (1 - self.put_long_offset)
            long_put = _select_contract(
                chain, 'put',
                moneyness_range=(
                    put_long_target / price - 0.02,
                    put_long_target / price + 0.02,
                ),
                dte_range=(self.dte_min, self.dte_max),
            )
            # Call长腿 (更高strike)
            call_long_target = short_call['strike'] * (1 + self.call_long_offset)
            long_call = _select_contract(
                chain, 'call',
                moneyness_range=(
                    call_long_target / price - 0.02,
                    call_long_target / price + 0.02,
                ),
                dte_range=(self.dte_min, self.dte_max),
            )

            if long_put is None or long_call is None:
                continue

            # 净权利金
            sp_prem = short_put.get('mid', 0)
            lp_prem = long_put.get('mid', 0)
            sc_prem = short_call.get('mid', 0)
            lc_prem = long_call.get('mid', 0)
            net_prem = (sp_prem - lp_prem) + (sc_prem - lc_prem)

            if net_prem <= 0:
                continue

            dte = int(short_put['dte'])
            expiry = str(short_put['expiry'])[:10]
            T = dte / 365.0

            # 构建4腿合约符号
            sp_sym = short_put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', short_put['strike'])
            lp_sym = long_put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', long_put['strike'])
            sc_sym = short_call.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', short_call['strike'])
            lc_sym = long_call.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', long_call['strike'])

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='iron_condor',
                contract_symbol=f"{sp_sym}|{lp_sym}|{sc_sym}|{lc_sym}",
                strike=short_put['strike'],  # 下边界
                expiry=expiry,
                option_type='iron_condor',
                quantity=-1,
                premium=net_prem,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': 0.0,  # 铁鹰近似delta中性
                    'gamma': -(
                        _bs_gamma(price, short_put['strike'], T, atm_iv)
                        - _bs_gamma(price, long_put['strike'], T, atm_iv)
                        + _bs_gamma(price, short_call['strike'], T, atm_iv)
                        - _bs_gamma(price, long_call['strike'], T, atm_iv)
                    ),
                    'theta': -(
                        _bs_theta(price, short_put['strike'], T, atm_iv, opt_type='put')
                        - _bs_theta(price, long_put['strike'], T, atm_iv, opt_type='put')
                        + _bs_theta(price, short_call['strike'], T, atm_iv, opt_type='call')
                        - _bs_theta(price, long_call['strike'], T, atm_iv, opt_type='call')
                    ),
                    'vega': -(
                        _bs_vega(price, short_put['strike'], T, atm_iv)
                        - _bs_vega(price, long_put['strike'], T, atm_iv)
                        + _bs_vega(price, short_call['strike'], T, atm_iv)
                        - _bs_vega(price, long_call['strike'], T, atm_iv)
                    ),
                    'short_put_strike': short_put['strike'],
                    'long_put_strike': long_put['strike'],
                    'short_call_strike': short_call['strike'],
                    'long_call_strike': long_call['strike'],
                    'put_spread_prem': sp_prem - lp_prem,
                    'call_spread_prem': sc_prem - lc_prem,
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算铁鹰策略指标"""
        net_prem = trade.premium
        sp_K = trade.greeks.get('short_put_strike', 0)
        lp_K = trade.greeks.get('long_put_strike', 0)
        sc_K = trade.greeks.get('short_call_strike', 0)
        lc_K = trade.greeks.get('long_call_strike', 0)
        T = trade.dte / 365.0

        # 最大亏损 = max(put价差, call价差) - 净权利金
        put_width = sp_K - lp_K
        call_width = lc_K - sc_K
        max_spread = max(put_width, call_width)
        trade.max_loss = (max_spread - net_prem) * 100

        # 盈利概率 ≈ 1 - (call_delta + |put_delta|) — 通常 60-75%
        trade.prob_profit = 0.68

        # 年化收益
        max_profit = net_prem * 100
        if T > 0 and trade.max_loss > 0:
            expected_pnl = max_profit * trade.prob_profit - trade.max_loss * (1 - trade.prob_profit)
            trade.expected_return = (expected_pnl / trade.max_loss) / T

        if trade.expected_return > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / 0.25

        return trade

    def generate_orders(self) -> List[dict]:
        """生成铁鹰4腿订单"""
        trades = self.scan()
        orders = []
        for t in trades:
            t = self.calc_metrics(t)
            syms = t.contract_symbol.split('|')
            if len(syms) == 4:
                sp_sym, lp_sym, sc_sym, lc_sym = syms
                prem = t.greeks
                # 卖Put短腿
                orders.append(OrderDict(
                    symbol=sp_sym, qty=1, side='sell',
                    type='limit', time_in_force='day',
                    limit_price=prem.get('put_spread_prem', t.premium * 0.5) * 0.6,
                ).to_dict())
                # 买Put长腿
                orders.append(OrderDict(
                    symbol=lp_sym, qty=1, side='buy',
                    type='limit', time_in_force='day',
                    limit_price=prem.get('put_spread_prem', t.premium * 0.5) * 0.4,
                ).to_dict())
                # 卖Call短腿
                orders.append(OrderDict(
                    symbol=sc_sym, qty=1, side='sell',
                    type='limit', time_in_force='day',
                    limit_price=prem.get('call_spread_prem', t.premium * 0.5) * 0.6,
                ).to_dict())
                # 买Call长腿
                orders.append(OrderDict(
                    symbol=lc_sym, qty=1, side='buy',
                    type='limit', time_in_force='day',
                    limit_price=prem.get('call_spread_prem', t.premium * 0.5) * 0.4,
                ).to_dict())
        return orders


# =====================================================================
# 7. ProtectivePutStrategy — 保护性看跌 (尾部对冲)
# =====================================================================

class ProtectivePutStrategy(OptionsStrategy):
    """保护性看跌策略 — 买入深度OTM Put作为组合保险

    目的: 防范尾部风险 (-10%以上回撤)
    合约: 15-20% OTM, 60-90 DTE
    成本: 约0.5-1%组合价值/季度
    选择: SPY Put 或 高beta个股Put
    对冲比率: delta加权覆盖-10%回撤
    """

    name = "保护性看跌策略"
    description = "买入深度OTM Put, 组合尾部风险对冲"

    otm_min: float = 0.80
    otm_max: float = 0.85
    dte_min: int = 60
    dte_max: int = 90
    portfolio_value: float = 100_000   # 默认组合价值
    cost_budget_pct: float = 0.01      # 季度成本预算 1%
    target_drawdown: float = -0.10     # 目标对冲 -10% 回撤

    def scan(self) -> List[TradeRecommendation]:
        """扫描保护性Put机会"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            if np.isnan(atm_iv):
                atm_iv = 0.30  # 默认

            # 选深度OTM Put
            contract = _select_contract(
                chain, 'put',
                moneyness_range=(self.otm_min, self.otm_max),
                dte_range=(self.dte_min, self.dte_max),
            )
            if contract is None:
                continue

            strike = contract['strike']
            premium = contract.get('mid', contract.get('ask', 0))
            dte = int(contract['dte'])
            expiry = str(contract['expiry'])[:10]

            if premium <= 0:
                continue

            T = dte / 365.0
            delta = _bs_delta(price, strike, T, atm_iv, opt_type='put')

            # 计算需要多少手才能覆盖-10%回撤
            # 当标的跌10%, OTM Put的delta约变为-0.5~-0.8
            # 对冲比率: 组合价值 * |回撤| / (100 * 预期delta变化 * 股价)
            crash_delta = min(delta * 3, -0.50)  # 跌10%后delta近似
            if abs(crash_delta) < 0.1:
                crash_delta = -0.50
            contracts_needed = max(1, int(
                self.portfolio_value * abs(self.target_drawdown) /
                (100 * abs(crash_delta) * price)
            ))

            cost = premium * 100 * contracts_needed
            cost_pct = cost / self.portfolio_value

            contract_sym = contract.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', strike)

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='buy_put',
                contract_symbol=contract_sym,
                strike=strike,
                expiry=expiry,
                option_type='put',
                quantity=contracts_needed,
                premium=premium,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': delta * contracts_needed,
                    'gamma': _bs_gamma(price, strike, T, atm_iv) * contracts_needed,
                    'theta': _bs_theta(price, strike, T, atm_iv, opt_type='put') * contracts_needed,
                    'vega': _bs_vega(price, strike, T, atm_iv) * contracts_needed,
                    'cost_pct': cost_pct,
                    'contracts_needed': contracts_needed,
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算保护性Put指标"""
        prem = trade.premium
        qty = trade.quantity
        S = trade.underlying_price
        K = trade.strike

        # 最大亏损 = 全部权利金 (Put过期无价值)
        trade.max_loss = prem * 100 * qty
        # 盈亏平衡 = K - 权利金 (不太相关, 目的是保险)
        # 盈利概率 (保险: 不盈利是好事)
        trade.prob_profit = 0.15  # ~15%概率触发保护
        # 预期收益 — 保险性质, 用EV计算
        # 如果发生-20%崩盘: Put内在价值 = (K - S*0.8) * 100 * qty
        crash_value = max(0, K - S * 0.80) * 100 * qty
        # EV = 概率加权
        trade.expected_return = (0.10 * crash_value - trade.max_loss) / trade.max_loss
        # Sharpe不适用于保险策略
        trade.sharpe_est = 0.0

        return trade


# =====================================================================
# 8. GammaScalpingStrategy — Gamma剥头皮
# =====================================================================

class GammaScalpingStrategy(OptionsStrategy):
    """Gamma剥头皮策略 — 买入ATM跨式 + 频繁delta对冲收割gamma

    原理: 买ATM跨式获得正gamma, 股价每次波动时delta对冲锁定利润
    适用: 高gamma标的 + 预期已实现波动率 > 隐含波动率
    对冲频率: 每日或更频繁 (当delta偏移超过阈值)
    """

    name = "Gamma剥头皮策略"
    description = "买ATM跨式 + delta对冲, 收割gamma利润"

    dte_min: int = 20
    dte_max: int = 40
    delta_hedge_threshold: float = 0.10  # delta偏移10%时对冲
    min_gamma: float = 0.02  # 最小gamma要求

    def scan(self) -> List[TradeRecommendation]:
        """扫描高gamma机会"""
        self._load_data()
        trades = []

        for sym in self.symbols:
            price = self._prices.get(sym)
            chain = self._chains.get(sym)
            if price is None or chain is None:
                continue

            atm_iv = self.sig_gen.atm_iv(chain, price)
            rv = _get_historical_vol(self.loader, sym, window=20)

            if np.isnan(atm_iv) or np.isnan(rv):
                continue

            # Gamma剥头皮要求: RV > IV (实际波动超过隐含)
            if rv < atm_iv * 0.90:
                continue  # RV不够高, gamma收益可能不够覆盖theta

            # ATM Call
            call = _select_contract(
                chain, 'call',
                moneyness_range=(0.98, 1.02),
                dte_range=(self.dte_min, self.dte_max),
            )
            # ATM Put
            put = _select_contract(
                chain, 'put',
                moneyness_range=(0.98, 1.02),
                dte_range=(self.dte_min, self.dte_max),
            )

            if call is None or put is None:
                continue

            call_prem = call.get('mid', 0)
            put_prem = put.get('mid', 0)
            total_prem = call_prem + put_prem

            if total_prem <= 0:
                continue

            dte = int(call['dte'])
            expiry = str(call['expiry'])[:10]
            T = dte / 365.0

            gamma = (_bs_gamma(price, call['strike'], T, atm_iv)
                     + _bs_gamma(price, put['strike'], T, atm_iv))

            if gamma < self.min_gamma:
                continue

            call_sym = call.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'call', call['strike'])
            put_sym = put.get('symbol', '') or _build_contract_symbol(
                sym, expiry, 'put', put['strike'])

            # 初始delta (应接近0, ATM跨式)
            net_delta = (_bs_delta(price, call['strike'], T, atm_iv, opt_type='call')
                         + _bs_delta(price, put['strike'], T, atm_iv, opt_type='put'))

            theta = (_bs_theta(price, call['strike'], T, atm_iv, opt_type='call')
                     + _bs_theta(price, put['strike'], T, atm_iv, opt_type='put'))

            trade = TradeRecommendation(
                strategy=self.name,
                underlying=sym,
                action='buy_straddle',
                contract_symbol=f"{call_sym}|{put_sym}",
                strike=call['strike'],
                expiry=expiry,
                option_type='straddle',
                quantity=1,
                premium=total_prem,
                underlying_price=price,
                dte=dte,
                greeks={
                    'delta': net_delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': (_bs_vega(price, call['strike'], T, atm_iv)
                             + _bs_vega(price, put['strike'], T, atm_iv)),
                    'iv': atm_iv,
                    'rv': rv,
                    'gamma_theta_ratio': abs(gamma / theta) if theta != 0 else 0,
                    'hedge_shares': -int(round(net_delta * 100)),
                    'put_strike': put['strike'],
                },
            )
            trades.append(trade)

        return trades

    def calc_metrics(self, trade: TradeRecommendation) -> TradeRecommendation:
        """计算gamma剥头皮指标"""
        prem = trade.premium
        S = trade.underlying_price
        gamma = trade.greeks.get('gamma', 0)
        theta = trade.greeks.get('theta', 0)
        rv = trade.greeks.get('rv', 0.30)
        iv = trade.greeks.get('iv', 0.30)
        T = trade.dte / 365.0

        # 最大亏损 = 全部权利金
        trade.max_loss = prem * 100

        # 每日gamma P&L 近似: 0.5 * gamma * S^2 * (rv^2 - iv^2) / 365
        daily_gamma_pnl = 0.5 * gamma * S ** 2 * (rv ** 2 - iv ** 2) / 365.0 * 100
        # 每日theta损耗
        daily_theta = theta * 100  # 已经是每天
        # 每日净P&L
        daily_net = daily_gamma_pnl + daily_theta

        total_expected = daily_net * trade.dte
        if trade.max_loss > 0:
            trade.expected_return = total_expected / trade.max_loss
            if T > 0:
                trade.expected_return /= T  # 年化

        # 盈利概率: RV > IV 时约55-60%
        trade.prob_profit = 0.55 if rv > iv else 0.40

        if trade.expected_return > 0:
            trade.sharpe_est = (trade.expected_return - 0.05) / 0.45

        return trade


# =====================================================================
# OptionsPortfolioManager — 多策略组合管理
# =====================================================================

class OptionsPortfolioManager:
    """期权组合管理器 — 汇总多策略的Greeks/仓位/P&L

    功能:
        - 组合多策略交易建议
        - 跟踪组合总Greeks (delta/gamma/theta/vega)
        - 仓位限制 (单一标的 <= 5%, 期权总仓位 <= 20%)
        - P&L场景分析 (标的价格 +/-5%, +/-10%, IV +/-10pp)
    """

    def __init__(self, portfolio_value: float = 100_000):
        """初始化组合管理器

        Parameters
        ----------
        portfolio_value : float
            组合总价值 (用于仓位限制计算)
        """
        self.portfolio_value = portfolio_value
        self.strategies: List[OptionsStrategy] = []
        self.trades: List[TradeRecommendation] = []
        self.max_position_pct: float = 0.05    # 单一标的最大5%
        self.max_options_pct: float = 0.20      # 期权总仓位最大20%

    def add_strategy(self, strategy: OptionsStrategy):
        """添加策略"""
        self.strategies.append(strategy)

    def scan_all(self) -> List[TradeRecommendation]:
        """扫描所有策略并合并结果"""
        all_trades = []
        for strat in self.strategies:
            try:
                trades = strat.scan()
                for t in trades:
                    t = strat.calc_metrics(t)
                all_trades.extend(trades)
            except Exception as e:
                logger.error("策略 %s 扫描失败: %s", strat.name, e)

        # 仓位限制过滤
        all_trades = self._apply_position_limits(all_trades)

        self.trades = all_trades
        return all_trades

    def _apply_position_limits(self, trades: List[TradeRecommendation]) -> List[TradeRecommendation]:
        """应用仓位限制"""
        # 按标的汇总仓位
        position_by_sym: Dict[str, float] = {}
        total_options_notional = 0.0
        filtered = []

        # 按Sharpe排序, 优先保留最好的
        trades_sorted = sorted(trades, key=lambda t: t.sharpe_est, reverse=True)

        for t in trades_sorted:
            # 计算该笔交易的名义金额
            notional = abs(t.quantity) * t.underlying_price * 100
            sym_total = position_by_sym.get(t.underlying, 0) + notional

            # 检查单一标的限制
            if sym_total / self.portfolio_value > self.max_position_pct:
                logger.info("跳过 %s %s: 超过单一标的限制 %.1f%%",
                            t.strategy, t.underlying,
                            sym_total / self.portfolio_value * 100)
                continue

            # 检查期权总仓位限制
            if (total_options_notional + notional) / self.portfolio_value > self.max_options_pct:
                logger.info("跳过 %s %s: 超过期权总仓位限制 %.1f%%",
                            t.strategy, t.underlying,
                            (total_options_notional + notional) / self.portfolio_value * 100)
                continue

            position_by_sym[t.underlying] = sym_total
            total_options_notional += notional
            filtered.append(t)

        return filtered

    def portfolio_greeks(self) -> Dict[str, float]:
        """汇总组合总Greeks"""
        total = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        for t in self.trades:
            g = t.greeks
            for key in total:
                total[key] += g.get(key, 0) * abs(t.quantity)
        return total

    def pnl_scenarios(self) -> pd.DataFrame:
        """P&L场景分析 — 标的价格变动 +/-5%, +/-10% 以及 IV变动 +/-10pp

        Returns
        -------
        pd.DataFrame
            场景矩阵, 行=场景, 列=P&L
        """
        scenarios = []
        price_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
        iv_shocks = [-0.10, 0.0, 0.10]  # 绝对pp

        for ps in price_shocks:
            for ivs in iv_shocks:
                total_pnl = 0.0
                for t in self.trades:
                    g = t.greeks
                    S = t.underlying_price
                    dS = S * ps
                    qty = abs(t.quantity)

                    # 近似P&L: delta*dS + 0.5*gamma*dS^2 + theta*1 + vega*dIV
                    delta = g.get('delta', 0)
                    gamma = g.get('gamma', 0)
                    theta = g.get('theta', 0)
                    vega = g.get('vega', 0)

                    pnl = (delta * dS
                           + 0.5 * gamma * dS ** 2
                           + theta  # 1天theta
                           + vega * ivs * 100  # vega per 1%, ivs in decimal
                           ) * 100 * qty

                    total_pnl += pnl

                scenarios.append({
                    'price_shock': f"{ps:+.0%}",
                    'iv_shock': f"{ivs:+.0%}pp",
                    'portfolio_pnl': round(total_pnl, 2),
                    'pnl_pct': round(total_pnl / self.portfolio_value * 100, 3),
                })

        return pd.DataFrame(scenarios)

    def summary(self) -> str:
        """生成组合摘要文本"""
        if not self.trades:
            return "无交易建议"

        greeks = self.portfolio_greeks()
        lines = [
            "=" * 70,
            "期权组合管理器 — 摘要",
            "=" * 70,
            f"组合价值: ${self.portfolio_value:,.0f}",
            f"交易数量: {len(self.trades)}",
            f"涉及标的: {len(set(t.underlying for t in self.trades))}",
            "",
            "── 组合总Greeks ──",
            f"  Delta: {greeks['delta']:+.4f}",
            f"  Gamma: {greeks['gamma']:+.6f}",
            f"  Theta: ${greeks['theta']:+.2f}/天",
            f"  Vega:  ${greeks['vega']:+.2f}/1%IV",
            "",
            "── 策略分布 ──",
        ]

        strat_counts: Dict[str, int] = {}
        for t in self.trades:
            strat_counts[t.strategy] = strat_counts.get(t.strategy, 0) + 1
        for s, c in sorted(strat_counts.items()):
            lines.append(f"  {s}: {c} 笔")

        lines.extend(["", "── 交易明细 ──"])
        for t in self.trades:
            lines.append(
                f"  {t.underlying:6s} | {t.action:20s} | K={t.strike:.1f} | "
                f"DTE={t.dte:3d} | prem={t.premium:.2f} | "
                f"E[r]={t.expected_return:.1%} | P(win)={t.prob_profit:.0%} | "
                f"Sharpe={t.sharpe_est:.2f}"
            )

        return "\n".join(lines)


# =====================================================================
# run_options_scan() — 全策略扫描入口
# =====================================================================

def run_options_scan(loader, symbols: Optional[List[str]] = None,
                     portfolio_value: float = 100_000) -> pd.DataFrame:
    """扫描所有期权策略, 按预估Sharpe排序并打印摘要

    Parameters
    ----------
    loader : AlpacaDataLoader
        Alpaca数据加载器
    symbols : list, optional
        标的列表, 默认主要科技股+指数ETF
    portfolio_value : float
        组合价值, 默认$100,000

    Returns
    -------
    pd.DataFrame
        所有交易建议, 按Sharpe降序排列
    """
    if symbols is None:
        symbols = ['AAPL', 'NVDA', 'TSLA', 'SPY', 'MSFT', 'AMZN',
                    'GOOGL', 'META', 'QQQ', 'IWM']

    print("=" * 70)
    print("期权策略全面扫描")
    print(f"标的: {', '.join(symbols)}")
    print(f"组合价值: ${portfolio_value:,.0f}")
    print("=" * 70)

    # 初始化所有策略
    strategies = [
        CoveredCallStrategy(loader, symbols),
        CashSecuredPutStrategy(loader, symbols),
        IVCrushStrategy(loader, symbols),
        VolatilityArbitrageStrategy(loader, symbols),
        PutSpreadIncomeStrategy(loader, symbols),
        IronCondorStrategy(loader, symbols),
        ProtectivePutStrategy(loader, symbols),
        GammaScalpingStrategy(loader, symbols),
    ]

    # 组合管理器
    pm = OptionsPortfolioManager(portfolio_value=portfolio_value)
    for s in strategies:
        pm.add_strategy(s)

    # 扫描
    trades = pm.scan_all()

    if not trades:
        print("\n未发现交易机会")
        return pd.DataFrame()

    # 构建结果表
    rows = []
    for t in trades:
        rows.append({
            '策略': t.strategy,
            '标的': t.underlying,
            '操作': t.action,
            '行权价': t.strike,
            'DTE': t.dte,
            '权利金': round(t.premium, 2),
            '预期年化': f"{t.expected_return:.1%}",
            '盈利概率': f"{t.prob_profit:.0%}",
            '最大亏损': f"${t.max_loss:,.0f}",
            'Sharpe': round(t.sharpe_est, 2),
            'Delta': round(t.greeks.get('delta', 0), 4),
            'Theta': round(t.greeks.get('theta', 0), 4),
        })

    df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)

    # 打印摘要
    print(pm.summary())
    print("\n── P&L场景分析 ──")
    print(pm.pnl_scenarios().to_string(index=False))
    print("\n── 排序表 (按Sharpe) ──")
    print(df.to_string(index=False))

    return df
