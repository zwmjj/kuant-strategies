"""Cross-asset options strategies — equity/commodity/bond/volatility/crypto/international/leveraged ETFs
Simulates option P&L with a Black-Scholes model, backtested on 8 years of daily data

10 strategies:
    1. VolatilityTermStructureStrategy  — VIX term structure (VRP variance risk premium)
    2. GoldVolatilityStrategy           — gold volatility selling (GLD strangles)
    3. BondVolatilityStrategy           — bond volatility (TLT FOMC calendar effect)
    4. SectorDispersionTrade            — sector dispersion trade (SPY vs sector ETFs)
    5. CryptoVolPremiumStrategy         — crypto volatility premium (BITO/COIN)
    6. CrossAssetStranglePortfolio      — cross-asset strangle portfolio (multi-asset diversification)
    7. CalendarSpreadStrategy           — calendar spread (short-dated vs long-dated theta)
    8. LeveragedETFDecayStrategy        — leveraged ETF decay (volatility drag)
    9. SkewArbitrageStrategy            — skew arbitrage (put/call skew differential)
   10. MacroOptionsOverlay              — macro options overlay (trend-driven option direction)

Entry point: run_cross_asset_options_backtests()
"""

import sys
import warnings
import math
import logging

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
import yfinance as yf

# =====================================================================
# 标的列表
# =====================================================================
EQUITY_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA']
SECTOR_ETFS = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLC', 'XLRE', 'XLB']
COMMODITY_ETFS = ['GLD', 'SLV', 'USO', 'GDX', 'UNG']
BOND_ETFS = ['TLT', 'IEF', 'HYG', 'LQD', 'JNK']
VOL_ETFS = ['VXX', 'UVXY', 'SVXY']
LEVERAGED_ETFS = ['TQQQ', 'SQQQ', 'SPXL', 'SPXS']
INTERNATIONAL_ETFS = ['EEM', 'EFA', 'FXI', 'EWZ']
CRYPTO_ETFS = ['BITO', 'COIN']
REAL_ESTATE_ETFS = ['VNQ', 'IYR']

ALL_SYMS = sorted(set(
    EQUITY_ETFS + SECTOR_ETFS + COMMODITY_ETFS + BOND_ETFS +
    VOL_ETFS + LEVERAGED_ETFS + INTERNATIONAL_ETFS + CRYPTO_ETFS +
    REAL_ESTATE_ETFS
))


# =====================================================================
# BS定价工具
# =====================================================================
def bs(S, K, T, sigma, r=0.05, opt='call'):
    """Black-Scholes option pricing"""
    S, K, T, sigma = float(S), float(K), float(T), float(sigma)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == 'call':
        return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def bs_delta(S, K, T, sigma, r=0.05, opt='call'):
    """Black-Scholes Delta"""
    S, K, T, sigma = float(S), float(K), float(T), float(sigma)
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5 if opt == 'call' else -0.5
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return stats.norm.cdf(d1) if opt == 'call' else stats.norm.cdf(d1) - 1


# =====================================================================
# 数据加载
# =====================================================================
def _load_data(start='2018-01-01', end='2026-03-28'):
    """下载所有标的8年日线数据"""
    print('[数据] 下载跨资产期权回测数据 (8年)...')
    data = yf.download(ALL_SYMS, start=start, end=end,
                       auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')
    ret = close.pct_change()
    # 已实现波动率
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    rv60 = ret.rolling(60).std() * np.sqrt(252)
    # IV代理: 通常IV高于RV 20%
    iv_proxy = rv60 * 1.2
    print(f'  {len(close)}天, {len(close.columns)}个标的: {list(close.columns)}')
    return close, ret, rv20, rv60, iv_proxy


# =====================================================================
# 月度回测框架
# =====================================================================
def monthly_loop(strategy_func, close, label='', period=21):
    """Monthly rolling backtest framework

    Parameters
    ----------
    strategy_func : callable(i) -> float or None
        Takes the day index i and returns that period's return
    close : DataFrame
        Close price data
    label : str
        Strategy label
    period : int
        Holding period (days)
    """
    n = len(close)
    port_ret = []
    for i in range(60, n, period):
        if i + period > n:
            break
        try:
            r = strategy_func(i)
            if r is not None and not np.isnan(r):
                port_ret.append(r)
        except Exception:
            pass
    if len(port_ret) < 12:
        return None
    r = pd.Series(port_ret)
    periods_per_year = 252 / period
    sh = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    md = (cum / cum.cummax() - 1).min()
    tot = cum.iloc[-1] - 1
    cg = (1 + tot) ** (periods_per_year / len(r)) - 1
    wr = (r > 0).mean()
    return {'sharpe': sh, 'cagr': cg, 'mdd': md, 'wr': wr, 'n': len(r), 'returns': r}


# =====================================================================
# 1. VIX期限结构策略 (方差风险溢价VRP)
# =====================================================================
class VolatilityTermStructureStrategy:
    """VIX term structure strategy

    Core logic:
    - The VXX/SVXY ratio captures the VIX term structure
    - Contango (normal): VXX decays naturally -> sell VXX puts (they expire worthless)
    - Backwardation (crisis): VXX spikes -> buy VXX calls
    - Signal: VXX 5-day return. Negative (contango) = sell puts, positive (backwardation) = buy calls
    - The most profitable systematic volatility strategy (VRP = variance risk premium)
    """
    name = 'S1_vol_term_structure'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the VIX term structure strategy"""
        if 'VXX' not in close.columns:
            print('  [跳过] VXX数据不可用')
            return None

        vxx = close['VXX']
        n = len(close)

        def strategy(i):
            if i < 5:
                return None
            S_vxx = float(vxx.iloc[i])
            if pd.isna(S_vxx) or S_vxx <= 0:
                return None

            # VXX 5日收益率作为期限结构信号
            vxx_ret5 = float(vxx.iloc[i] / vxx.iloc[i - 5] - 1) if vxx.iloc[i - 5] > 0 else 0
            sigma = float(iv_proxy['VXX'].iloc[i]) if 'VXX' in iv_proxy.columns and pd.notna(iv_proxy['VXX'].iloc[i]) else 0.8
            T = 21 / 252

            S_end = float(vxx.iloc[min(i + 21, n - 1)])

            if vxx_ret5 < -0.02:
                # Contango: 卖VXX看跌期权 (VXX衰减, puts到期无价值)
                K = S_vxx * 0.90  # 10% OTM put
                prem = bs(S_vxx, K, T, sigma, opt='put')
                pnl = (prem - max(K - S_end, 0)) / S_vxx
            elif vxx_ret5 > 0.05:
                # Backwardation: 买VXX看涨期权 (VXX飙升)
                K = S_vxx * 1.10  # 10% OTM call
                cost = bs(S_vxx, K, T, sigma, opt='call')
                pnl = (max(S_end - K, 0) - cost) / S_vxx
            else:
                # 中性: 不交易
                return None

            return pnl

        return monthly_loop(strategy, close, label='vol_term_structure')


# =====================================================================
# 2. 黄金波动率策略
# =====================================================================
class GoldVolatilityStrategy:
    """Gold volatility strategy

    Core logic:
    - GLD options: IV is typically 15-20%, while realized volatility is often lower
    - Sell GLD strangles (8% OTM each side, 30 DTE)
    - Gold moves slowly most of the time -> high premium capture rate
    - Hedge: buy GLD puts when the gold trend breaks below its 50-day moving average
    """
    name = 'S2_gold_volatility'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the gold volatility strategy"""
        if 'GLD' not in close.columns:
            print('  [跳过] GLD数据不可用')
            return None

        gld = close['GLD']
        gld_sma50 = gld.rolling(50).mean()
        n = len(close)

        def strategy(i):
            S = float(gld.iloc[i])
            if pd.isna(S) or S <= 0:
                return None
            sigma = float(iv_proxy['GLD'].iloc[i]) if pd.notna(iv_proxy['GLD'].iloc[i]) else 0.18
            T = 21 / 252
            S_end = float(gld.iloc[min(i + 21, n - 1)])

            # 卖宽跨式: 8% OTM 每侧
            K_call = S * 1.08
            K_put = S * 0.92
            prem_call = bs(S, K_call, T, sigma, opt='call')
            prem_put = bs(S, K_put, T, sigma, opt='put')
            total_prem = prem_call + prem_put

            strangle_pnl = total_prem - max(S_end - K_call, 0) - max(K_put - S_end, 0)

            # 趋势对冲: 跌破50日均线时买保护性put
            sma = float(gld_sma50.iloc[i]) if pd.notna(gld_sma50.iloc[i]) else S
            hedge_cost = 0
            hedge_payoff = 0
            if S < sma:
                K_hedge = S * 0.95
                hedge_cost = bs(S, K_hedge, T, sigma, opt='put')
                hedge_payoff = max(K_hedge - S_end, 0)

            pnl = (strangle_pnl - hedge_cost + hedge_payoff) / S
            return pnl

        return monthly_loop(strategy, close, label='gold_vol')


# =====================================================================
# 3. 债券波动率策略
# =====================================================================
class BondVolatilityStrategy:
    """Bond volatility strategy

    Core logic:
    - TLT has a distinctive property: IV spikes ahead of FOMC meetings and compresses afterwards
    - Sell TLT straddles 3 days before FOMC (IV inflation), close after the meeting
    - TLT IV is usually > RV -> sell iron condors for income
    - Low correlation with equity option strategies
    """
    name = 'S3_bond_volatility'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the bond volatility strategy"""
        if 'TLT' not in close.columns:
            print('  [跳过] TLT数据不可用')
            return None

        tlt = close['TLT']
        n = len(close)

        def strategy(i):
            S = float(tlt.iloc[i])
            if pd.isna(S) or S <= 0:
                return None
            sigma = float(iv_proxy['TLT'].iloc[i]) if pd.notna(iv_proxy['TLT'].iloc[i]) else 0.18
            T = 21 / 252
            S_end = float(tlt.iloc[min(i + 21, n - 1)])

            # 铁鹰策略: TLT IV通常高于RV
            K_sc = S * 1.05
            K_lc = S * 1.10
            K_sp = S * 0.95
            K_lp = S * 0.90

            credit = (bs(S, K_sc, T, sigma, 'call') - bs(S, K_lc, T, sigma, 'call') +
                      bs(S, K_sp, T, sigma, 'put') - bs(S, K_lp, T, sigma, 'put'))
            max_loss = 0.05 * S - credit
            if max_loss <= 0:
                return None

            if K_sp <= S_end <= K_sc:
                pnl = credit
            elif S_end > K_sc:
                pnl = credit - min(S_end - K_sc, K_lc - K_sc)
            else:
                pnl = credit - min(K_sp - S_end, K_sp - K_lp)

            # FOMC日历效应叠加: 模拟IV在周期性事件前膨胀
            # 每隔约42天(6周~FOMC频率), 额外卖straddle捕捉IV压缩
            day_of_cycle = i % 42
            if day_of_cycle >= 39:  # FOMC前3天
                straddle_prem = bs(S, S, 3 / 252, sigma * 1.3, opt='call') + bs(S, S, 3 / 252, sigma * 1.3, opt='put')
                S_3d = float(tlt.iloc[min(i + 3, n - 1)])
                straddle_pnl = (straddle_prem - abs(S_3d - S)) / S
                pnl = pnl / max_loss + 0.3 * straddle_pnl
            else:
                pnl = pnl / max_loss

            return pnl

        return monthly_loop(strategy, close, label='bond_vol')


# =====================================================================
# 4. 行业分散度交易
# =====================================================================
class SectorDispersionTrade:
    """Sector dispersion trade

    Core logic:
    - Sell SPY straddles + buy straddles on individual sector ETFs
    - If sectors move in different directions (high dispersion), the sector straddles pay more
    - If every sector moves together (correlation spike), the SPY straddle costs less
    - Net edge: dispersion > correlation. Historical win rate around 60%
    """
    name = 'S4_sector_dispersion'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the sector dispersion trade"""
        avail_sectors = [s for s in SECTOR_ETFS if s in close.columns]
        if 'SPY' not in close.columns or len(avail_sectors) < 5:
            print('  [跳过] SPY或行业ETF数据不足')
            return None

        spy = close['SPY']
        n = len(close)

        def strategy(i):
            S_spy = float(spy.iloc[i])
            if pd.isna(S_spy) or S_spy <= 0:
                return None
            sigma_spy = float(iv_proxy['SPY'].iloc[i]) if pd.notna(iv_proxy['SPY'].iloc[i]) else 0.2
            T = 21 / 252

            # 卖SPY跨式
            spy_prem = bs(S_spy, S_spy, T, sigma_spy, opt='call') + bs(S_spy, S_spy, T, sigma_spy, opt='put')
            S_spy_end = float(spy.iloc[min(i + 21, n - 1)])
            spy_pnl = spy_prem - abs(S_spy_end - S_spy)

            # 买行业ETF跨式
            sector_pnls = []
            sector_costs = []
            for sec in avail_sectors:
                S_sec = float(close[sec].iloc[i])
                if pd.isna(S_sec) or S_sec <= 0:
                    continue
                sigma_sec = float(iv_proxy[sec].iloc[i]) if pd.notna(iv_proxy[sec].iloc[i]) else 0.25
                sec_cost = bs(S_sec, S_sec, T, sigma_sec, opt='call') + bs(S_sec, S_sec, T, sigma_sec, opt='put')
                S_sec_end = float(close[sec].iloc[min(i + 21, n - 1)])
                sec_payoff = abs(S_sec_end - S_sec) - sec_cost
                sector_pnls.append(sec_payoff / S_sec)
                sector_costs.append(sec_cost / S_sec)

            if len(sector_pnls) < 4:
                return None

            # 净PnL: SPY卖出收益 + 行业买入收益 (按名义值标准化)
            net_pnl = spy_pnl / S_spy + np.mean(sector_pnls) * 0.6  # 行业头寸较小
            return net_pnl

        return monthly_loop(strategy, close, label='sector_dispersion')


# =====================================================================
# 5. 加密货币波动率溢价策略
# =====================================================================
class CryptoVolPremiumStrategy:
    """Crypto volatility premium strategy

    Core logic:
    - Crypto options (BITO, COIN) carry a large IV premium
    - BITO IV is frequently 80-100% vs RV of 50-60%
    - Sell BITO/COIN strangles: rich premium but also substantial risk
    - Risk management: 5% max position, stop loss set at 2x the premium collected
    """
    name = 'S5_crypto_vol_premium'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the crypto volatility premium strategy"""
        avail = [s for s in CRYPTO_ETFS if s in close.columns]
        if len(avail) == 0:
            print('  [跳过] 加密ETF数据不可用')
            return None

        n = len(close)

        def strategy(i):
            pnls = []
            for sym in avail:
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    continue
                # 加密IV通常很高
                base_sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.8
                sigma = max(base_sigma, 0.5)  # 加密最低50% vol
                T = 21 / 252

                # 卖宽跨式: 15% OTM 每侧 (加密波动大需要更宽)
                K_call = S * 1.15
                K_put = S * 0.85
                prem_call = bs(S, K_call, T, sigma, opt='call')
                prem_put = bs(S, K_put, T, sigma, opt='put')
                total_prem = prem_call + prem_put

                S_end = float(close[sym].iloc[min(i + 21, n - 1)])
                payoff = max(S_end - K_call, 0) + max(K_put - S_end, 0)

                # 止损: 亏损超过2倍权利金则截断
                raw_pnl = total_prem - payoff
                max_loss = -2 * total_prem
                pnl = max(raw_pnl, max_loss) / S
                pnls.append(pnl)

            if not pnls:
                return None
            # 最大5%仓位权重
            return np.mean(pnls) * 0.05

        return monthly_loop(strategy, close, label='crypto_vol')


# =====================================================================
# 6. 跨资产宽跨式组合
# =====================================================================
class CrossAssetStranglePortfolio:
    """Cross-asset strangle portfolio

    Core logic:
    - Diversified strangle selling:
      - 20% SPY/QQQ strangles (equity volatility)
      - 20% GLD/SLV strangles (commodity volatility)
      - 20% TLT strangles (bond volatility)
      - 20% BITO/COIN strangles (crypto volatility)
      - 20% IWM/EEM strangles (small-cap / emerging-market volatility)
    - Key insight: selling volatility across uncorrelated assets gives a far higher Sharpe than a single asset
    """
    name = 'S6_cross_asset_strangle'

    # 资产分组及OTM幅度
    BUCKETS = {
        'equity': (['SPY', 'QQQ'], 0.07, 0.20),        # 符号, OTM%, 权重*5
        'commodity': (['GLD', 'SLV'], 0.08, 0.20),
        'bond': (['TLT'], 0.06, 0.20),
        'crypto': (['BITO', 'COIN'], 0.15, 0.20),
        'em_small': (['IWM', 'EEM'], 0.08, 0.20),
    }

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the cross-asset strangle portfolio"""
        n = len(close)

        def _strangle_pnl(sym, i, otm_pct):
            """单个标的宽跨式PnL"""
            if sym not in close.columns:
                return None
            S = float(close[sym].iloc[i])
            if pd.isna(S) or S <= 0:
                return None
            sigma = float(iv_proxy[sym].iloc[i]) if sym in iv_proxy.columns and pd.notna(iv_proxy[sym].iloc[i]) else 0.3
            T = 21 / 252
            K_call = S * (1 + otm_pct)
            K_put = S * (1 - otm_pct)
            prem = bs(S, K_call, T, sigma, opt='call') + bs(S, K_put, T, sigma, opt='put')
            S_end = float(close[sym].iloc[min(i + 21, n - 1)])
            payoff = max(S_end - K_call, 0) + max(K_put - S_end, 0)
            return (prem - payoff) / S

        def strategy(i):
            total_pnl = 0
            total_weight = 0

            for bucket_name, (syms, otm, weight) in CrossAssetStranglePortfolio.BUCKETS.items():
                bucket_pnls = []
                for sym in syms:
                    p = _strangle_pnl(sym, i, otm)
                    if p is not None:
                        bucket_pnls.append(p)
                if bucket_pnls:
                    total_pnl += weight * np.mean(bucket_pnls)
                    total_weight += weight

            if total_weight < 0.4:  # 至少两个分组
                return None
            return total_pnl / total_weight * total_weight  # 按实际权重缩放

        return monthly_loop(strategy, close, label='cross_asset_strangle')


# =====================================================================
# 7. 日历价差策略
# =====================================================================
class CalendarSpreadStrategy:
    """Calendar spread strategy

    Core logic:
    - Sell short-dated (weekly) ATM options on SPY/QQQ
    - Buy long-dated (monthly) ATM options as a hedge
    - Profit comes from the faster theta decay of the short-dated leg
    - Works best in range-bound markets
    """
    name = 'S7_calendar_spread'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the calendar spread strategy"""
        if 'SPY' not in close.columns:
            print('  [跳过] SPY数据不可用')
            return None

        targets = [s for s in ['SPY', 'QQQ'] if s in close.columns]
        n = len(close)

        def strategy(i):
            pnls = []
            for sym in targets:
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    continue
                sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.2
                K = S  # ATM

                # 短期: 5天 (周度)
                T_short = 5 / 252
                # 长期: 30天 (月度)
                T_long = 30 / 252

                # 卖短期call + 买长期call (call日历价差)
                short_prem = bs(S, K, T_short, sigma, opt='call')
                long_cost = bs(S, K, T_long, sigma, opt='call')
                net_debit = long_cost - short_prem

                # 5天后: 短期到期, 长期还剩25天
                S_5d = float(close[sym].iloc[min(i + 5, n - 1)])
                short_payoff = max(S_5d - K, 0)
                long_value = bs(S_5d, K, 25 / 252, sigma, opt='call')

                # PnL = 长期期权市值 - 短期期权结算 - 初始净支出
                cal_pnl = (long_value - short_payoff - net_debit) / S

                # put日历价差 (类似)
                short_prem_p = bs(S, K, T_short, sigma, opt='put')
                long_cost_p = bs(S, K, T_long, sigma, opt='put')
                net_debit_p = long_cost_p - short_prem_p
                short_payoff_p = max(K - S_5d, 0)
                long_value_p = bs(S_5d, K, 25 / 252, sigma, opt='put')
                cal_pnl_p = (long_value_p - short_payoff_p - net_debit_p) / S

                pnls.append((cal_pnl + cal_pnl_p) / 2)

            if not pnls:
                return None
            return np.mean(pnls)

        # 周度循环 (5天持仓)
        return monthly_loop(strategy, close, label='calendar_spread', period=5)


# =====================================================================
# 8. 杠杆ETF衰减策略
# =====================================================================
class LeveragedETFDecayStrategy:
    """Leveraged ETF decay strategy

    Core logic:
    - Leveraged ETFs (TQQQ, SQQQ, etc.) suffer volatility drag
    - Sell OTM puts on inverse leveraged ETFs (they decay toward zero)
    - Sell OTM calls on 3x long ETFs (they rarely hold their highs)
    - Very high win rate but catastrophic tail risk
    """
    name = 'S8_leveraged_etf_decay'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the leveraged ETF decay strategy"""
        # 反向ETF: 卖puts (它们衰减)
        inverse_etfs = [s for s in ['SQQQ', 'SPXS'] if s in close.columns]
        # 正向杠杆ETF: 卖calls (波动率拖累)
        bull_etfs = [s for s in ['TQQQ', 'SPXL'] if s in close.columns]

        if len(inverse_etfs) == 0 and len(bull_etfs) == 0:
            print('  [跳过] 杠杆ETF数据不可用')
            return None

        n = len(close)

        def strategy(i):
            pnls = []

            # 反向ETF: 卖OTM puts (它们长期下跌)
            for sym in inverse_etfs:
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    continue
                sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.6
                sigma = max(sigma, 0.4)  # 杠杆ETF vol很高
                T = 21 / 252
                K = S * 0.85  # 15% OTM put
                prem = bs(S, K, T, sigma, opt='put')
                S_end = float(close[sym].iloc[min(i + 21, n - 1)])
                pnl = (prem - max(K - S_end, 0)) / S
                pnls.append(pnl)

            # 正向杠杆ETF: 卖OTM calls (波动率拖累使涨幅受限)
            for sym in bull_etfs:
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    continue
                sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.6
                sigma = max(sigma, 0.4)
                T = 21 / 252
                K = S * 1.15  # 15% OTM call
                prem = bs(S, K, T, sigma, opt='call')
                S_end = float(close[sym].iloc[min(i + 21, n - 1)])
                pnl = (prem - max(S_end - K, 0)) / S
                pnls.append(pnl)

            if not pnls:
                return None
            return np.mean(pnls)

        return monthly_loop(strategy, close, label='lev_etf_decay')


# =====================================================================
# 9. 偏度套利策略
# =====================================================================
class SkewArbitrageStrategy:
    """Skew arbitrage strategy

    Core logic:
    - Compare put skew against call skew across assets
    - When an asset's put skew is extreme, sell puts and buy calls as a hedge
    - Cross-asset: if gold put skew is cheap while equity put skew is expensive, trade the spread
    - Market-neutral volatility surface arbitrage
    """
    name = 'S9_skew_arbitrage'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the skew arbitrage strategy"""
        # 多资产偏度比较
        skew_assets = [s for s in ['SPY', 'GLD', 'TLT', 'IWM', 'EEM', 'QQQ'] if s in close.columns]
        if len(skew_assets) < 3:
            print('  [跳过] 偏度套利资产不足')
            return None

        n = len(close)

        # 用20日收益偏度作为隐含偏度代理
        ret_skew = {}
        for sym in skew_assets:
            ret_skew[sym] = ret[sym].rolling(60).skew()

        def strategy(i):
            # 计算各资产偏度
            skews = {}
            for sym in skew_assets:
                sk = float(ret_skew[sym].iloc[i]) if pd.notna(ret_skew[sym].iloc[i]) else 0
                skews[sym] = sk

            if len(skews) < 3:
                return None

            skew_s = pd.Series(skews)
            # 最正偏度 (put偏度便宜/call偏度贵): 卖calls
            most_positive = skew_s.idxmax()
            # 最负偏度 (put偏度贵/call偏度便宜): 卖puts
            most_negative = skew_s.idxmin()

            if most_positive == most_negative:
                return None

            pnls = []

            # 资产1: 偏度最正 -> 卖OTM calls (市场过度悲观右尾)
            sym1 = most_positive
            S1 = float(close[sym1].iloc[i])
            sigma1 = float(iv_proxy[sym1].iloc[i]) if pd.notna(iv_proxy[sym1].iloc[i]) else 0.25
            T = 21 / 252
            K1 = S1 * 1.05
            prem1 = bs(S1, K1, T, sigma1, opt='call')
            S1_end = float(close[sym1].iloc[min(i + 21, n - 1)])
            pnl1 = (prem1 - max(S1_end - K1, 0)) / S1
            pnls.append(pnl1)

            # 资产2: 偏度最负 -> 卖OTM puts (市场过度悲观左尾)
            sym2 = most_negative
            S2 = float(close[sym2].iloc[i])
            sigma2 = float(iv_proxy[sym2].iloc[i]) if pd.notna(iv_proxy[sym2].iloc[i]) else 0.25
            K2 = S2 * 0.95
            prem2 = bs(S2, K2, T, sigma2, opt='put')
            S2_end = float(close[sym2].iloc[min(i + 21, n - 1)])
            pnl2 = (prem2 - max(K2 - S2_end, 0)) / S2
            pnls.append(pnl2)

            return np.mean(pnls)

        return monthly_loop(strategy, close, label='skew_arb')


# =====================================================================
# 10. 宏观期权叠加策略
# =====================================================================
class MacroOptionsOverlay:
    """Macro options overlay strategy

    Core logic:
    - Use the macro regime (SPY/GLD/TLT trends) to decide which options to sell:
      - Risk-on: sell SPY puts + sell GLD calls (long equities, short gold)
      - Risk-off: sell GLD puts + sell SPY calls (long gold, short equities)
      - Inflation: sell TLT calls + sell USO puts (rates up, oil up)
      - Deflation: sell SPY calls + buy TLT calls
    """
    name = 'S10_macro_options_overlay'

    @staticmethod
    def backtest(close, ret, rv20, rv60, iv_proxy):
        """Backtest the macro options overlay strategy"""
        needed = ['SPY', 'GLD', 'TLT']
        if not all(s in close.columns for s in needed):
            print('  [跳过] 宏观标的数据不足')
            return None

        spy_sma50 = close['SPY'].rolling(50).mean()
        gld_sma50 = close['GLD'].rolling(50).mean()
        tlt_sma50 = close['TLT'].rolling(50).mean()
        has_uso = 'USO' in close.columns
        n = len(close)

        def strategy(i):
            spy_above = float(close['SPY'].iloc[i]) > float(spy_sma50.iloc[i]) if pd.notna(spy_sma50.iloc[i]) else True
            gld_above = float(close['GLD'].iloc[i]) > float(gld_sma50.iloc[i]) if pd.notna(gld_sma50.iloc[i]) else True
            tlt_above = float(close['TLT'].iloc[i]) > float(tlt_sma50.iloc[i]) if pd.notna(tlt_sma50.iloc[i]) else True

            T = 21 / 252
            pnls = []

            def _sell_option(sym, otm_pct, opt_type):
                """卖出OTM期权并计算PnL"""
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    return None
                sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.25
                if opt_type == 'call':
                    K = S * (1 + otm_pct)
                else:
                    K = S * (1 - otm_pct)
                prem = bs(S, K, T, sigma, opt=opt_type)
                S_end = float(close[sym].iloc[min(i + 21, n - 1)])
                if opt_type == 'call':
                    return (prem - max(S_end - K, 0)) / S
                return (prem - max(K - S_end, 0)) / S

            def _buy_option(sym, otm_pct, opt_type):
                """买入OTM期权并计算PnL"""
                S = float(close[sym].iloc[i])
                if pd.isna(S) or S <= 0:
                    return None
                sigma = float(iv_proxy[sym].iloc[i]) if pd.notna(iv_proxy[sym].iloc[i]) else 0.25
                if opt_type == 'call':
                    K = S * (1 + otm_pct)
                else:
                    K = S * (1 - otm_pct)
                cost = bs(S, K, T, sigma, opt=opt_type)
                S_end = float(close[sym].iloc[min(i + 21, n - 1)])
                if opt_type == 'call':
                    return (max(S_end - K, 0) - cost) / S
                return (max(K - S_end, 0) - cost) / S

            if spy_above and not gld_above:
                # Risk-on: 卖SPY puts + 卖GLD calls
                p1 = _sell_option('SPY', 0.05, 'put')
                p2 = _sell_option('GLD', 0.05, 'call')
                if p1 is not None:
                    pnls.append(p1)
                if p2 is not None:
                    pnls.append(p2)

            elif not spy_above and gld_above:
                # Risk-off: 卖GLD puts + 卖SPY calls
                p1 = _sell_option('GLD', 0.05, 'put')
                p2 = _sell_option('SPY', 0.05, 'call')
                if p1 is not None:
                    pnls.append(p1)
                if p2 is not None:
                    pnls.append(p2)

            elif not tlt_above and has_uso:
                # 通胀: 卖TLT calls + 卖USO puts
                p1 = _sell_option('TLT', 0.05, 'call')
                p2 = _sell_option('USO', 0.08, 'put')
                if p1 is not None:
                    pnls.append(p1)
                if p2 is not None:
                    pnls.append(p2)

            else:
                # 通缩/不确定: 卖SPY calls + 买TLT calls
                p1 = _sell_option('SPY', 0.05, 'call')
                p2 = _buy_option('TLT', 0.03, 'call')
                if p1 is not None:
                    pnls.append(p1)
                if p2 is not None:
                    pnls.append(p2)

            if not pnls:
                return None
            return np.mean(pnls)

        return monthly_loop(strategy, close, label='macro_overlay')


# =====================================================================
# 主回测入口
# =====================================================================
STRATEGIES = [
    VolatilityTermStructureStrategy,
    GoldVolatilityStrategy,
    BondVolatilityStrategy,
    SectorDispersionTrade,
    CryptoVolPremiumStrategy,
    CrossAssetStranglePortfolio,
    CalendarSpreadStrategy,
    LeveragedETFDecayStrategy,
    SkewArbitrageStrategy,
    MacroOptionsOverlay,
]


def run_cross_asset_options_backtests():
    """Run backtests for all 10 cross-asset options strategies and print a summary

    Downloads 8 years of history, backtests each strategy, and reports:
    - Sharpe/CAGR/MDD/win rate per strategy
    - IS/OOS comparison (50/50 split)
    - Sub-strategy correlation matrix
    - Equal-weight portfolio performance
    """
    close, ret, rv20, rv60, iv_proxy = _load_data()

    results = {}
    print(f'\n[回测] 运行 {len(STRATEGIES)} 种跨资产期权策略...\n')

    for strat_cls in STRATEGIES:
        print(f'  {strat_cls.name}...', end=' ')
        res = strat_cls.backtest(close, ret, rv20, rv60, iv_proxy)
        if res is not None:
            results[strat_cls.name] = res
            print(f'Sharpe={res["sharpe"]:.3f}, CAGR={res["cagr"]:.1%}, WR={res["wr"]:.0%}')
        else:
            print('无结果')

    if not results:
        print('\n[错误] 没有策略产生有效结果')
        return results

    # ═══════════════════════════════════════════
    # 汇总表
    # ═══════════════════════════════════════════
    print(f'\n{"=" * 80}')
    print(f'  跨资产期权策略回测汇总 (8年)')
    print(f'{"=" * 80}')
    print(f'\n{"策略":<35} {"Sharpe":>8} {"CAGR":>8} {"MDD":>8} {"胜率":>6} {"期数":>5}')
    print('-' * 72)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]['sharpe'])
    for name, m in sorted_results:
        flag = ' ***' if m['sharpe'] >= 1.5 else (' **' if m['sharpe'] >= 1.0 else '')
        print(f'{name:<35} {m["sharpe"]:>8.3f} {m["cagr"]:>7.1%} {m["mdd"]:>7.1%} '
              f'{m["wr"]:>5.0%} {m["n"]:>5}{flag}')

    # ═══════════════════════════════════════════
    # IS/OOS 对比
    # ═══════════════════════════════════════════
    print(f'\n=== 样本内/样本外 (50/50分割) ===')
    for name, m in sorted_results:
        r = m['returns']
        sp = len(r) // 2
        if sp < 6:
            continue
        is_r = r.iloc[:sp]
        oos_r = r.iloc[sp:]
        is_sh = is_r.mean() / is_r.std() * np.sqrt(12) if is_r.std() > 0 else 0
        oos_sh = oos_r.mean() / oos_r.std() * np.sqrt(12) if oos_r.std() > 0 else 0
        decay = 1 - oos_sh / is_sh if is_sh > 0 else float('nan')
        print(f'  {name:<33} IS={is_sh:.3f}  OOS={oos_sh:.3f}  衰减={decay:.1%}')

    # ═══════════════════════════════════════════
    # 子策略相关性
    # ═══════════════════════════════════════════
    print(f'\n=== 子策略相关性矩阵 ===')
    if len(results) >= 2:
        min_len = min(len(v['returns']) for v in results.values())
        corr_df = pd.DataFrame({
            k.replace('S', '').replace('_', ' ')[:15]: v['returns'].iloc[:min_len].reset_index(drop=True)
            for k, v in results.items()
        })
        print(corr_df.corr().round(3).to_string())
    else:
        print('  策略不足, 无法计算相关性')

    # ═══════════════════════════════════════════
    # 等权组合
    # ═══════════════════════════════════════════
    if len(results) >= 2:
        min_len = min(len(v['returns']) for v in results.values())
        combined = sum(
            v['returns'].iloc[:min_len].reset_index(drop=True) / len(results)
            for v in results.values()
        )
        sh = combined.mean() / combined.std() * np.sqrt(12) if combined.std() > 0 else 0
        cum = (1 + combined).cumprod()
        md = (cum / cum.cummax() - 1).min()
        tot = cum.iloc[-1] - 1
        cg = (1 + tot) ** (12 / min_len) - 1
        wr = (combined > 0).mean()

        print(f'\n=== 等权组合 ({len(results)}策略) ===')
        print(f'  Sharpe: {sh:.3f}')
        print(f'  CAGR:   {cg:.1%}')
        print(f'  MDD:    {md:.1%}')
        print(f'  胜率:   {wr:.0%}')
        print(f'  期数:   {min_len}')

    print(f'\n{"=" * 80}')
    return results


# =====================================================================
# 直接运行
# =====================================================================
if __name__ == '__main__':
    run_cross_asset_options_backtests()
