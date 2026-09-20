"""Options-enhanced strategies — built on Alpaca options data (Greeks, IV, option flow)

Builds stock selection signals from factors such as implied volatility rank, skew, and gamma exposure.
All strategies inherit from BaseStrategy and take an instrument list plus an AlpacaDataLoader instance.
"""
import logging
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from qf.strategy import BaseStrategy
from qf.signals_options import OptionsSignalGenerator

logger = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

def _zscore(s: pd.Series) -> pd.Series:
    """截面 z-score，NaN 安全"""
    mu = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def _rank_to_signal(s: pd.Series) -> pd.Series:
    """截面百分位排名映射到 [-1, 1]"""
    valid = s.dropna()
    if len(valid) <= 1:
        return s.apply(lambda x: 0.0 if not np.isnan(x) else np.nan)
    ranked = s.rank(pct=True)
    return ranked * 2.0 - 1.0


def _fetch_chain_and_price(loader, symbol: str, sig_gen: OptionsSignalGenerator):
    """获取单个标的的期权链 DataFrame 和当前价格

    Returns
    -------
    (chain_df, underlying_price) 或 (None, None)
    """
    try:
        chain_df = loader.get_option_chain(symbol)
        if chain_df is None or (isinstance(chain_df, pd.DataFrame) and chain_df.empty):
            return None, None
    except Exception as e:
        logger.warning("获取 %s 期权链失败: %s", symbol, e)
        return None, None

    # 获取标的价格
    try:
        snaps = loader.get_snapshots([symbol])
        price = snaps.get(symbol, {}).get('last', None)
        if price is None or price <= 0:
            return None, None
    except Exception as e:
        logger.warning("获取 %s 快照失败: %s", symbol, e)
        return None, None

    # 通过 process_chain 标准化
    chain_df = sig_gen.process_chain(chain_df, price)
    if chain_df.empty:
        return None, None

    return chain_df, price


# =====================================================================
# 1. IV 排名策略
# =====================================================================

class IVRankStrategy(BaseStrategy):
    """Implied volatility rank strategy

    Fetches ATM implied volatility for each instrument and ranks it as a rolling percentile of its own history.
    Low IV rank = cheap options = bullish (volatility mean reversion);
    high IV rank = expensive options = bearish or uncertain.

    Signal = -iv_rank: buy low IV rank names, sell high IV rank names.
    """

    name = "IV Rank (期权)"
    description = "隐含波动率百分位排名 — 低IV看多、高IV看空"

    def __init__(
        self,
        symbols: List[str],
        loader,
        iv_lookback_days: int = 252,
        days_range: tuple = (20, 60),
    ):
        """
        Parameters
        ----------
        symbols : list[str]
            标的列表, 如 ['AAPL', 'MSFT', 'GOOG']
        loader : AlpacaDataLoader
            Alpaca 数据加载器实例
        iv_lookback_days : int
            历史 IV 回溯天数，用于计算百分位排名，默认 252 (约一年)
        days_range : tuple
            ATM IV 筛选的 DTE 范围，默认 (20, 60) 天
        """
        self.symbols = symbols
        self.loader = loader
        self.iv_lookback_days = iv_lookback_days
        self.days_range = days_range
        self.sig_gen = OptionsSignalGenerator()

    def _build_iv_history(self, symbol: str, underlying_price: float) -> pd.Series:
        """构建历史 IV 序列 — 利用历史期权链快照或日K线波动率代理

        优先使用期权链数据；若不可用则用已实现波动率作为代理。

        Returns
        -------
        pd.Series
            历史 IV 时间序列
        """
        # 尝试用股票历史日K线计算已实现波动率作为 IV 代理
        try:
            end = date.today()
            start = end - timedelta(days=int(self.iv_lookback_days * 1.5))
            bars = self.loader.get_daily_bars([symbol], str(start), str(end))
            if bars is not None and not bars.empty:
                if isinstance(bars.columns, pd.MultiIndex):
                    close = bars.xs('close', axis=1, level=1)[symbol]
                elif 'close' in bars.columns:
                    close = bars['close']
                else:
                    close = bars.iloc[:, 0]

                # 20日已实现波动率 * 缩放因子作为 IV 代理
                log_ret = np.log(close / close.shift(1))
                rv = log_ret.rolling(20).std() * np.sqrt(252)
                return rv.dropna()
        except Exception as e:
            logger.debug("构建 %s 历史IV失败: %s", symbol, e)

        return pd.Series(dtype=float)

    def generate_signal(self, data: dict = None) -> pd.Series:
        """Generate the IV rank signal

        Parameters
        ----------
        data : dict, optional
            Kept for BaseStrategy interface compatibility; this strategy fetches its own data via the loader

        Returns
        -------
        pd.Series
            index=symbol, higher values are more bullish (low IV rank)
        """
        records = {}

        for symbol in self.symbols:
            chain_df, price = _fetch_chain_and_price(
                self.loader, symbol, self.sig_gen
            )
            if chain_df is None:
                continue

            # 当前 ATM IV
            current_iv = self.sig_gen.atm_iv(chain_df, price, self.days_range)
            if np.isnan(current_iv):
                continue

            # 历史 IV
            iv_history = self._build_iv_history(symbol, price)
            if iv_history.empty:
                # 没有历史数据时用 0.5 作为默认排名
                records[symbol] = -0.5
                continue

            # 百分位排名
            rank = self.sig_gen.iv_rank(current_iv, iv_history)
            if np.isnan(rank):
                continue

            # 信号: 负的 IV 排名 (低排名 => 正信号 => 看多)
            records[symbol] = -rank

        if not records:
            return pd.Series(dtype=float)

        signal = pd.Series(records)
        return _rank_to_signal(signal)


# =====================================================================
# 2. 偏度 Alpha 策略
# =====================================================================

class SkewAlphaStrategy(BaseStrategy):
    """Implied volatility skew strategy

    Skew is put IV relative to call IV.
    High put skew = the market is pricing downside risk = a contrarian bullish signal.

    Uses OptionsSignalGenerator.iv_skew() to compute 25-delta skew.
    iv_skew() already returns -skew (positive skew is bearish), so this strategy negates it into a contrarian factor:
    positive skew (expensive puts) => contrarian bullish.
    """

    name = "Skew Alpha (期权偏度)"
    description = "IV偏度逆向因子 — 看跌期权贵时逆向看多"

    def __init__(
        self,
        symbols: List[str],
        loader,
        days_range: tuple = (20, 60),
    ):
        """
        Parameters
        ----------
        symbols : list[str]
            标的列表
        loader : AlpacaDataLoader
            数据加载器
        days_range : tuple
            DTE 筛选范围，默认 (20, 60)
        """
        self.symbols = symbols
        self.loader = loader
        self.days_range = days_range
        self.sig_gen = OptionsSignalGenerator()

    def generate_signal(self, data: dict = None) -> pd.Series:
        """Generate the skew alpha signal

        Returns
        -------
        pd.Series
            index=symbol, positive = bullish (puts relatively expensive, a contrarian signal)
        """
        records = {}

        for symbol in self.symbols:
            chain_df, price = _fetch_chain_and_price(
                self.loader, symbol, self.sig_gen
            )
            if chain_df is None:
                continue

            # iv_skew 返回 -skew: 高偏度 => 负值 (看空标的)
            # 我们取负做逆向: 高 put skew => 正信号 (逆向看多)
            skew_signal = self.sig_gen.iv_skew(chain_df, price, self.days_range)
            if np.isnan(skew_signal):
                continue

            # 逆向: iv_skew 返回 -skew，再取负 => 原始 skew
            # 高原始 skew = put 贵 = 逆向看多
            records[symbol] = -skew_signal

        if not records:
            return pd.Series(dtype=float)

        signal = pd.Series(records)
        return _rank_to_signal(signal)


# =====================================================================
# 3. Gamma 暴露策略
# =====================================================================

class GammaExposureStrategy(BaseStrategy):
    """Gamma exposure strategy

    Market-maker gamma exposure tends to "pin" the underlying near high-gamma strikes.
    Closer to the max-gamma strike => lower expected move => more stable (good to hold).
    Further away => higher expected move => greater uncertainty.

    Signal: the negated distance to the max-gamma strike (close = high signal = bullish/stable).
    """

    name = "Gamma Exposure (Gamma暴露)"
    description = "Gamma钉扎效应 — 标的近高Gamma行权价时更稳定"

    def __init__(
        self,
        symbols: List[str],
        loader,
        max_dte: int = 30,
    ):
        """
        Parameters
        ----------
        symbols : list[str]
            标的列表
        loader : AlpacaDataLoader
            数据加载器
        max_dte : int
            仅考虑近期到期 (DTE <= max_dte) 的合约，默认 30 天
        """
        self.symbols = symbols
        self.loader = loader
        self.max_dte = max_dte
        self.sig_gen = OptionsSignalGenerator()

    def _find_max_gamma_strike(self, symbol: str, price: float) -> Optional[float]:
        """找到最大 Gamma 集中的行权价

        优先使用合约元数据中的 open_interest 加权；
        若无 OI 数据则按近 ATM 的合约数量密度估算。

        Returns
        -------
        float or None
            最大 Gamma 集中的行权价
        """
        # 尝试获取合约元数据 (含 open_interest)
        try:
            exp_gte = date.today()
            exp_lte = date.today() + timedelta(days=self.max_dte)
            contracts = self.loader.get_option_contracts(
                symbol,
                expiration_gte=exp_gte,
                expiration_lte=exp_lte,
            )
            if contracts is not None and not contracts.empty:
                # 过滤有 OI 的合约
                if 'open_interest' in contracts.columns:
                    oi = contracts[['strike_price', 'open_interest']].copy()
                    oi['open_interest'] = pd.to_numeric(
                        oi['open_interest'], errors='coerce'
                    ).fillna(0)

                    if oi['open_interest'].sum() > 0:
                        # Gamma 近似: OI 在近 ATM 的行权价贡献最大 Gamma
                        # 用 OI * exp(-moneyness^2) 做 Gamma 权重
                        oi['moneyness'] = oi['strike_price'] / price
                        oi['gamma_weight'] = (
                            oi['open_interest']
                            * np.exp(-10 * (oi['moneyness'] - 1.0) ** 2)
                        )
                        idx = oi['gamma_weight'].idxmax()
                        return float(oi.loc[idx, 'strike_price'])
        except Exception as e:
            logger.debug("获取 %s 合约元数据失败: %s", symbol, e)

        # 回退: 用期权链中合约密度最高的行权价
        try:
            chain_df = self.loader.get_option_chain(symbol)
            if chain_df is not None and not chain_df.empty:
                chain_df = self.sig_gen.process_chain(chain_df, price)
                near = chain_df[chain_df['dte'] <= self.max_dte]
                if not near.empty:
                    # 按行权价计数作为活跃度代理
                    counts = near.groupby('strike').size()
                    return float(counts.idxmax())
        except Exception:
            pass

        return None

    def generate_signal(self, data: dict = None) -> pd.Series:
        """Generate the gamma exposure signal

        Returns
        -------
        pd.Series
            index=symbol, positive = price is close to a high-gamma strike (more stable)
        """
        records = {}

        for symbol in self.symbols:
            # 获取当前价格
            try:
                snaps = self.loader.get_snapshots([symbol])
                price = snaps.get(symbol, {}).get('last', None)
                if price is None or price <= 0:
                    continue
            except Exception:
                continue

            max_gamma_strike = self._find_max_gamma_strike(symbol, price)
            if max_gamma_strike is None or max_gamma_strike <= 0:
                continue

            # 距离最大 Gamma 行权价的百分比距离
            distance_pct = abs(price - max_gamma_strike) / price

            # 信号: 距离越近 (越小) => 越稳定 => 信号越高
            # 取负使得近距离 = 高信号
            records[symbol] = -distance_pct

        if not records:
            return pd.Series(dtype=float)

        signal = pd.Series(records)
        return _rank_to_signal(signal)


# =====================================================================
# 4. 复合期权策略
# =====================================================================

class CompositeOptionsStrategy(BaseStrategy):
    """Composite options strategy

    Blends three sub-factors:
      - 40% IV rank (low rank is bullish)
      - 30% skew (high put skew is contrarian bullish)
      - 30% put/call ratio (high P/C ratio is contrarian bullish)

    Underlying factors are computed with OptionsSignalGenerator.
    """

    name = "Composite Options (复合期权)"
    description = "IV排名40% + 偏度30% + Put/Call比率30% 复合信号"

    def __init__(
        self,
        symbols: List[str],
        loader,
        w_iv_rank: float = 0.40,
        w_skew: float = 0.30,
        w_pcr: float = 0.30,
        iv_lookback_days: int = 252,
        days_range: tuple = (20, 60),
    ):
        """
        Parameters
        ----------
        symbols : list[str]
            标的列表
        loader : AlpacaDataLoader
            数据加载器
        w_iv_rank : float
            IV 排名权重，默认 0.40
        w_skew : float
            偏度权重，默认 0.30
        w_pcr : float
            Put/Call 比率权重，默认 0.30
        iv_lookback_days : int
            IV 历史回溯天数，默认 252
        days_range : tuple
            ATM IV / Skew 的 DTE 筛选范围
        """
        self.symbols = symbols
        self.loader = loader
        self.w_iv_rank = w_iv_rank
        self.w_skew = w_skew
        self.w_pcr = w_pcr
        self.iv_lookback_days = iv_lookback_days
        self.days_range = days_range
        self.sig_gen = OptionsSignalGenerator()

        # 子策略 (用于获取 IV 历史)
        self._iv_rank_strat = IVRankStrategy(
            symbols, loader, iv_lookback_days, days_range
        )

    def generate_signal(self, data: dict = None) -> pd.Series:
        """Generate the composite options signal

        Returns
        -------
        pd.Series
            index=symbol, values in [-1, 1], positive is bullish
        """
        raw_records = {}

        for symbol in self.symbols:
            chain_df, price = _fetch_chain_and_price(
                self.loader, symbol, self.sig_gen
            )
            if chain_df is None:
                continue

            entry = {}

            # (a) IV 排名因子
            current_iv = self.sig_gen.atm_iv(chain_df, price, self.days_range)
            if not np.isnan(current_iv):
                iv_history = self._iv_rank_strat._build_iv_history(symbol, price)
                if not iv_history.empty:
                    rank = self.sig_gen.iv_rank(current_iv, iv_history)
                    entry['iv_rank'] = -rank  # 低排名 = 看多
                else:
                    entry['iv_rank'] = -0.5
            else:
                entry['iv_rank'] = np.nan

            # (b) 偏度因子 (逆向)
            skew_val = self.sig_gen.iv_skew(chain_df, price, self.days_range)
            entry['skew'] = -skew_val if not np.isnan(skew_val) else np.nan

            # (c) Put/Call 比率因子 (逆向)
            pcr = self.sig_gen.put_call_ratio(chain_df)
            # 高 PCR = 看空情绪 => 逆向看多
            entry['pcr'] = pcr if not np.isnan(pcr) else np.nan

            raw_records[symbol] = entry

        if not raw_records:
            return pd.Series(dtype=float)

        raw = pd.DataFrame(raw_records).T

        # 截面 z-score 标准化
        z_iv = _zscore(raw['iv_rank'])        # 已取负: 低排名 => 正值
        z_skew = _zscore(raw['skew'])         # 已取负逆向: 高put skew => 正值
        z_pcr = -_zscore(raw['pcr'])          # 高PCR = 看空 => 取负做逆向看多

        # 加权合成
        composite = (
            self.w_iv_rank * z_iv.fillna(0)
            + self.w_skew * z_skew.fillna(0)
            + self.w_pcr * z_pcr.fillna(0)
        )

        return _rank_to_signal(composite)

    def get_params(self) -> dict:
        """Return the strategy parameters"""
        params = super().get_params()
        params.update({
            'w_iv_rank': self.w_iv_rank,
            'w_skew': self.w_skew,
            'w_pcr': self.w_pcr,
            'iv_lookback_days': self.iv_lookback_days,
            'days_range': self.days_range,
            'symbols': self.symbols,
        })
        return params
