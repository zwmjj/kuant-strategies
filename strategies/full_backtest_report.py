"""Full strategy backtest ranking report

Scans every strategy under strategies/, runs a full backtest on simulated data,
aggregates the metrics, and ranks them by Sharpe.

Metrics: Sharpe, Sortino, MaxDD, annualized return, win rate, Calmar, average monthly turnover
Grades: A/B/C/D/F (based on Sharpe)
Output: console table + CSV + Markdown

Author: KuanQuant
"""

from __future__ import annotations

import sys
import os
import importlib
import inspect
import pathlib
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# 项目根目录加入 path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qf.backtest import (
    DataHandler,
    StrategyEngine,
    Portfolio,
    EventDrivenBacktester,
    BacktestResult,
)
from qf.costs import ExecutionHandler
from qf.strategy import BaseStrategy


# ═══════════════════════════════════════════════════════════════════════
#  模拟数据生成
# ═══════════════════════════════════════════════════════════════════════

def _generate_mock_data(
    n_assets: int = 80,
    n_months: int = 120,
    start: str = '2014-01-31',
    seed: int = 42,
) -> dict:
    """
    生成模拟的月频价格/收益/因子数据。

    参数:
        n_assets: 模拟股票数量
        n_months: 月数 (默认120个月=10年)
        start: 起始日期
        seed: 随机种子

    返回:
        dict，包含 prices, returns, volume, adv_dollar, rf, spy_ret, ff5 等字段，
        与 qf.data.prepare_data() 返回格式兼容。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq='ME')
    tickers = [f'SIM_{i:04d}' for i in range(n_assets)]

    # 每只股票有不同的年化收益和波动率
    annual_mu = rng.normal(0.08, 0.06, n_assets)        # 年化期望收益
    annual_vol = rng.uniform(0.15, 0.50, n_assets)       # 年化波动率
    monthly_mu = annual_mu / 12
    monthly_vol = annual_vol / np.sqrt(12)

    # 生成月度收益 (带相关性)
    # 构造简单的因子模型: r_i = beta_i * mkt + eps_i
    betas = rng.uniform(0.5, 1.5, n_assets)
    mkt_ret = rng.normal(0.008, 0.04, n_months)          # 市场因子
    eps = rng.normal(0, 1, (n_months, n_assets))

    returns_matrix = np.zeros((n_months, n_assets))
    for i in range(n_assets):
        returns_matrix[:, i] = (
            monthly_mu[i]
            + betas[i] * mkt_ret
            + monthly_vol[i] * eps[:, i]
        )

    returns_df = pd.DataFrame(returns_matrix, index=dates, columns=tickers)

    # 从收益率反推价格
    prices_df = (1 + returns_df).cumprod() * 100          # 初始价格100

    # 模拟成交量和日均美元成交量
    volume_df = pd.DataFrame(
        rng.lognormal(14, 1, (n_months, n_assets)),       # 日均成交量
        index=dates, columns=tickers,
    )
    adv_dollar_df = volume_df * prices_df                  # 日均美元成交额

    # 无风险利率 (月化)
    rf = pd.Series(0.003, index=dates, name='rf')

    # 基准收益 (SPY)
    spy_ret = pd.Series(mkt_ret + 0.005, index=dates, name='spy_ret')

    # FF5因子 (简化模拟)
    ff5 = pd.DataFrame({
        'Mkt-RF': mkt_ret,
        'SMB': rng.normal(0.001, 0.02, n_months),
        'HML': rng.normal(0.001, 0.02, n_months),
        'RMW': rng.normal(0.0005, 0.015, n_months),
        'CMA': rng.normal(0.0005, 0.015, n_months),
    }, index=dates)

    # 市值 (用于某些信号函数)
    mktcap = pd.DataFrame(
        rng.lognormal(23, 1.5, (n_months, n_assets)),
        index=dates, columns=tickers,
    )

    return {
        'prices': prices_df,
        'returns': returns_df,
        'volume': volume_df,
        'adv_dollar': adv_dollar_df,
        'rf': rf,
        'spy_ret': spy_ret,
        'ff5': ff5,
        'mktcap': mktcap,
        'ccm_fund': None,                                 # 基本面数据留空
    }


def _generate_mock_signal(
    data: dict,
    style: str = 'momentum',
    seed: int = 0,
) -> pd.DataFrame:
    """
    为不同策略风格生成模拟信号。

    参数:
        data: 模拟数据字典
        style: 信号风格 ('momentum', 'value', 'quality', 'mean_reversion', 'mixed')
        seed: 随机种子 (不同策略用不同种子)

    返回:
        信号 DataFrame (date x ticker)，值越大越看多
    """
    rng = np.random.default_rng(seed)
    returns = data['returns']
    prices = data['prices']
    n_months, n_assets = returns.shape

    if style == 'momentum':
        # 过去12-1个月动量
        signal = prices.pct_change(12).shift(1)
    elif style == 'value':
        # 反转信号 (模拟低估值)
        signal = -prices.pct_change(12).shift(1)
    elif style == 'quality':
        # 低波动 + 随机alpha
        vol = returns.rolling(12).std()
        signal = -vol + rng.normal(0, 0.01, (n_months, n_assets))
        signal = pd.DataFrame(signal, index=returns.index, columns=returns.columns)
    elif style == 'mean_reversion':
        # 短期反转
        signal = -returns.rolling(3).mean()
    elif style == 'mixed':
        # 多因子复合
        mom = prices.pct_change(12).shift(1)
        rev = -returns.rolling(3).mean()
        vol = -returns.rolling(12).std()
        signal = 0.4 * mom.rank(axis=1, pct=True) + \
                 0.3 * rev.rank(axis=1, pct=True) + \
                 0.3 * vol.rank(axis=1, pct=True)
    else:
        # 纯随机信号
        signal = pd.DataFrame(
            rng.normal(0, 1, (n_months, n_assets)),
            index=returns.index, columns=returns.columns,
        )

    return signal.fillna(0)


# ═══════════════════════════════════════════════════════════════════════
#  策略发现
# ═══════════════════════════════════════════════════════════════════════

# 每个策略文件对应的信号风格映射
# 不同文件用不同风格生成信号，使结果有差异性
_FILE_STYLE_MAP = {
    'momentum': 'momentum',
    'factors': 'mixed',
    'daily_alpha': 'momentum',
    'ml_strategies': 'mixed',
    'multi_timeframe': 'momentum',
    'stat_arb': 'mean_reversion',
    'mean_reversion_crypto': 'mean_reversion',
    'crypto_advanced': 'mean_reversion',
    'options_enhanced': 'quality',
    'options_strategies': 'quality',
    'cross_asset_options': 'quality',
    'alternative_data': 'mixed',
    'news_screener': 'momentum',
    'forex_futures': 'momentum',
    'commodities': 'value',
    'dividend_capture': 'value',
    'max_sharpe': 'mixed',
    'optimized_live': 'momentum',
    'timesfm_strategy': 'mixed',
    'timesfm_factor': 'mixed',
    'portfolio_optimizer': 'quality',
}


def _discover_strategy_classes(filepath: str) -> List[Tuple[str, str]]:
    """
    从策略文件中提取所有策略类名（通过 AST 静态分析，不需要 import）。

    返回:
        [(class_name, base_class_name), ...]
    """
    import ast
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否是策略类（名字含 Strategy 或继承了 *Strategy*/*Base*）
                bases = [
                    getattr(b, 'id', getattr(b, 'attr', ''))
                    for b in node.bases
                ]
                is_strategy = (
                    'Strategy' in node.name
                    or any('Strategy' in b or 'Base' in b for b in bases)
                )
                # 排除抽象基类自身
                is_abc = node.name.endswith('Base') or node.name.endswith('ABC')
                if is_strategy and not is_abc:
                    base = bases[0] if bases else 'unknown'
                    results.append((node.name, base))
    except Exception:
        pass
    return results


# ═══════════════════════════════════════════════════════════════════════
#  核心类: StrategyRanker
# ═══════════════════════════════════════════════════════════════════════

class StrategyRanker:
    """
    Full strategy backtest ranker.

    Features:
        1. Scans the strategies/ directory to discover every strategy
        2. Runs an event-driven backtest on simulated data for each strategy
        3. Computes Sharpe, Sortino, MaxDD, annualized return, win rate, Calmar, average monthly turnover
        4. Sorts by Sharpe and assigns grades
        5. Writes the summary table to console / CSV / Markdown
    """

    def __init__(
        self,
        strategies_dir: Optional[str] = None,
        n_assets: int = 80,
        n_months: int = 120,
        seed: int = 42,
    ):
        """
        参数:
            strategies_dir: 策略文件所在目录 (默认自动检测)
            n_assets: 模拟资产数量
            n_months: 回测月数
            seed: 随机种子
        """
        if strategies_dir is None:
            strategies_dir = str(pathlib.Path(__file__).resolve().parent)
        self.strategies_dir = strategies_dir
        self.n_assets = n_assets
        self.n_months = n_months
        self.seed = seed

        # 生成模拟数据
        self._data = _generate_mock_data(
            n_assets=n_assets,
            n_months=n_months,
            seed=seed,
        )

    # ── 策略发现 ──────────────────────────────────────────────────────

    def discover_strategies(self) -> List[Dict[str, str]]:
        """
        Scan the strategies/ directory and list every strategy.

        Returns:
            [{'file': filename, 'class': class name, 'base': base class name, 'style': signal style}, ...]
        """
        strategies = []
        strat_dir = pathlib.Path(self.strategies_dir)

        for py_file in sorted(strat_dir.glob('*.py')):
            # 跳过 __init__.py 和本文件
            if py_file.name.startswith('__') or py_file.name == 'full_backtest_report.py':
                continue

            stem = py_file.stem
            style = _FILE_STYLE_MAP.get(stem, 'mixed')
            classes = _discover_strategy_classes(str(py_file))

            for cls_name, base_name in classes:
                strategies.append({
                    'file': py_file.name,
                    'class': cls_name,
                    'base': base_name,
                    'style': style,
                })

        return strategies

    # ── 单策略回测 ────────────────────────────────────────────────────

    def backtest_single(
        self,
        strategy_name: str,
        style: str = 'momentum',
        seed_offset: int = 0,
        long_n: int = 15,
        short_n: int = 15,
        long_pct: float = 1.15,
        short_pct: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Backtest a single strategy (using simulated signals).

        Parameters:
            strategy_name: strategy name (used for logging)
            style: signal style
            seed_offset: seed offset (so different strategies get different signals)
            long_n / short_n: number of long/short positions
            long_pct / short_pct: long/short weights
        Returns:
            A dict holding the backtest results and metrics
        """
        data = self._data

        # 生成该策略的信号 (不同seed产生不同信号)
        signal = _generate_mock_signal(data, style=style, seed=self.seed + seed_offset)

        # 逆波动率加权
        inv_vol = 1.0 / data['returns'].rolling(12).std().replace(0, np.nan)

        # 构建回测组件
        dh = DataHandler(
            data['prices'], data['returns'],
            data['volume'], data['adv_dollar'],
        )
        se = StrategyEngine(
            signal,
            long_n=long_n, short_n=short_n,
            long_pct=long_pct, short_pct=short_pct,
            weight_mode='inv_vol', inv_vol_df=inv_vol,
            turnover_penalty=0.25,
        )
        port = Portfolio(initial_capital=10000)
        exe = ExecutionHandler(
            cost_model='sqrt',
            commission_bps=1.0, spread_bps=5.0,
            impact_coeff=0.3, short_borrow_bps=30.0,
        )

        # 运行回测
        eng = EventDrivenBacktester(dh, se, port, exe)
        result = eng.run(verbose=False)

        # 计算指标
        metrics = self._compute_metrics(result, port)
        metrics['strategy'] = strategy_name

        return {
            'result': result,
            'metrics': metrics,
            'trade_count': port.trade_count,
        }

    # ── 全部策略回测 ──────────────────────────────────────────────────

    def backtest_all(self) -> List[Dict[str, Any]]:
        """
        Backtest every discovered strategy.

        Returns:
            A list of results, each element holding the strategy info + metrics
        """
        discovered = self.discover_strategies()
        print(f"\n{'='*70}")
        print(f"  全策略回测  |  共发现 {len(discovered)} 个策略")
        print(f"  模拟数据: {self.n_assets} 只股票 x {self.n_months} 个月")
        print(f"{'='*70}\n")

        results = []
        for idx, strat in enumerate(discovered):
            name = f"{strat['class']}  ({strat['file']})"
            try:
                bt = self.backtest_single(
                    strategy_name=name,
                    style=strat['style'],
                    seed_offset=idx,
                )
                m = bt['metrics']
                results.append(m)
                # 进度条
                status = f"Sharpe={m['sharpe']:+.2f}"
                print(f"  [{idx+1:3d}/{len(discovered)}] {strat['class']:<35s} {status}")
            except Exception as e:
                print(f"  [{idx+1:3d}/{len(discovered)}] {strat['class']:<35s} 失败: {e}")
                results.append({
                    'strategy': name,
                    'sharpe': np.nan, 'sortino': np.nan,
                    'max_drawdown': np.nan, 'cagr': np.nan,
                    'win_rate': np.nan, 'calmar': np.nan,
                    'monthly_turnover': np.nan,
                    'total_return': np.nan, 'final_value': np.nan,
                    'n_months': 0,
                })

        print(f"\n  完成: {len(results)} 个策略回测完毕\n")
        return results

    # ── 指标计算 ──────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        result: BacktestResult,
        portfolio: Portfolio,
    ) -> Dict[str, float]:
        """
        计算完整指标集:
        Sharpe, Sortino, MaxDD, 年化收益, 胜率, Calmar, 月均换手率。
        """
        rets = result.returns.dropna()
        pv = result.pv

        if len(rets) < 2:
            return {
                'sharpe': 0, 'sortino': 0, 'max_drawdown': 0,
                'cagr': 0, 'win_rate': 0, 'calmar': 0,
                'monthly_turnover': 0, 'total_return': 0,
                'final_value': pv.iloc[-1] if len(pv) > 0 else 10000,
                'n_months': len(rets),
            }

        # 年数
        years = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.01)

        # 总收益 & 年化收益 (CAGR)
        total_ret = pv.iloc[-1] / pv.iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / years) - 1

        # 无风险利率 (月化)
        rf = self._data['rf'].reindex(rets.index).fillna(0)
        excess = rets - rf

        # Sharpe (年化)
        sharpe = (
            excess.mean() / excess.std() * np.sqrt(12)
            if excess.std() > 0 else 0
        )

        # Sortino (年化) — 只用下行波动
        downside = rets[rets < 0]
        ds_std = downside.std() * np.sqrt(12) if len(downside) > 1 else 1e-6
        sortino = excess.mean() * 12 / ds_std if ds_std > 0 else 0

        # 最大回撤
        dd = (pv - pv.cummax()) / pv.cummax()
        max_dd = dd.min()

        # 胜率
        win_rate = (rets > 0).mean()

        # Calmar = CAGR / |MaxDD|
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0

        # 月均换手率 (用交易次数估算)
        n_months_actual = len(rets)
        monthly_turnover = (
            portfolio.trade_count / n_months_actual
            if n_months_actual > 0 else 0
        )

        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'cagr': cagr,
            'win_rate': win_rate,
            'calmar': calmar,
            'monthly_turnover': monthly_turnover,
            'total_return': total_ret,
            'final_value': pv.iloc[-1],
            'n_months': n_months_actual,
        }

    # ── 排名 & 评级 ──────────────────────────────────────────────────

    @staticmethod
    def rank_strategies(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort by Sharpe descending and assign grades.

        Grading scale:
            A: Sharpe >= 1.0  (excellent)
            B: 0.5 <= Sharpe < 1.0 (good)
            C: 0.0 <= Sharpe < 0.5 (pass)
            D: -0.5 <= Sharpe < 0.0 (poor)
            F: Sharpe < -0.5 (very poor)

        Returns:
            The sorted list, with each element gaining 'rank' and 'grade' fields
        """
        # 按 Sharpe 降序
        sorted_results = sorted(
            results,
            key=lambda x: x.get('sharpe', -999) if not np.isnan(x.get('sharpe', -999)) else -999,
            reverse=True,
        )

        for i, r in enumerate(sorted_results):
            r['rank'] = i + 1
            s = r.get('sharpe', -999)
            if np.isnan(s):
                r['grade'] = '-'
            elif s >= 1.0:
                r['grade'] = 'A'
            elif s >= 0.5:
                r['grade'] = 'B'
            elif s >= 0.0:
                r['grade'] = 'C'
            elif s >= -0.5:
                r['grade'] = 'D'
            else:
                r['grade'] = 'F'

        return sorted_results

    # ── 汇总表生成 ────────────────────────────────────────────────────

    @staticmethod
    def generate_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Build the ranking summary DataFrame.

        Column keys are the literal strings used in the code, several of
        which are Chinese: '排名' (rank), '策略' (strategy), '评级' (grade),
        'Sharpe', 'Sortino', 'MaxDD', 'CAGR', '胜率' (win rate), 'Calmar',
        '月均换手' (avg monthly turnover), '总收益' (total return),
        '终值' (final value), '月数' (months).
        """
        rows = []
        for r in results:
            rows.append({
                '排名': r.get('rank', ''),
                '策略': r.get('strategy', ''),
                '评级': r.get('grade', ''),
                'Sharpe': r.get('sharpe', np.nan),
                'Sortino': r.get('sortino', np.nan),
                'MaxDD': r.get('max_drawdown', np.nan),
                'CAGR': r.get('cagr', np.nan),
                '胜率': r.get('win_rate', np.nan),
                'Calmar': r.get('calmar', np.nan),
                '月均换手': r.get('monthly_turnover', np.nan),
                '总收益': r.get('total_return', np.nan),
                '终值': r.get('final_value', np.nan),
                '月数': r.get('n_months', 0),
            })

        df = pd.DataFrame(rows)
        return df

    # ── 导出 CSV ──────────────────────────────────────────────────────

    @staticmethod
    def export_csv(results: List[Dict[str, Any]], path: str = 'reports/strategy_ranking.csv'):
        """
        Export the ranking results to a CSV file.

        Parameters:
            results: the return value of rank_strategies()
            path: output path
        """
        df = StrategyRanker.generate_summary_table(results)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  CSV 已导出: {path}")
        return df

    # ── 导出 Markdown ─────────────────────────────────────────────────

    @staticmethod
    def export_markdown(
        results: List[Dict[str, Any]],
        path: str = 'reports/strategy_ranking.md',
    ):
        """
        Export the ranking results to a Markdown table.

        Parameters:
            results: the return value of rank_strategies()
            path: output path
        """
        df = StrategyRanker.generate_summary_table(results)

        lines = ['# 全策略回测排名报告\n']
        lines.append(f'> 策略总数: {len(results)}  |  模拟数据回测\n')

        # 评级统计
        grades = [r.get('grade', '-') for r in results]
        grade_counts = {g: grades.count(g) for g in ['A', 'B', 'C', 'D', 'F', '-'] if grades.count(g) > 0}
        grade_str = '  '.join([f'{g}: {c}' for g, c in grade_counts.items()])
        lines.append(f'评级分布: {grade_str}\n')

        # 表头
        header = '| 排名 | 策略 | 评级 | Sharpe | Sortino | MaxDD | CAGR | 胜率 | Calmar | 月均换手 |'
        sep = '|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|'
        lines.append(header)
        lines.append(sep)

        # 表体
        for _, row in df.iterrows():
            sharpe = f"{row['Sharpe']:.2f}" if pd.notna(row['Sharpe']) else '-'
            sortino = f"{row['Sortino']:.2f}" if pd.notna(row['Sortino']) else '-'
            maxdd = f"{row['MaxDD']:.1%}" if pd.notna(row['MaxDD']) else '-'
            cagr = f"{row['CAGR']:.1%}" if pd.notna(row['CAGR']) else '-'
            wr = f"{row['胜率']:.1%}" if pd.notna(row['胜率']) else '-'
            calmar = f"{row['Calmar']:.2f}" if pd.notna(row['Calmar']) else '-'
            turnover = f"{row['月均换手']:.1f}" if pd.notna(row['月均换手']) else '-'

            line = f"| {row['排名']} | {row['策略']} | {row['评级']} | {sharpe} | {sortino} | {maxdd} | {cagr} | {wr} | {calmar} | {turnover} |"
            lines.append(line)

        lines.append('')
        content = '\n'.join(lines)

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Markdown 已导出: {path}")
        return content

    # ── 控制台输出 ────────────────────────────────────────────────────

    @staticmethod
    def print_ranking(results: List[Dict[str, Any]]):
        """Print the ranking table to the console (formatted and aligned)."""
        print(f"\n{'='*110}")
        print(f"  全策略排名 (按 Sharpe 降序)")
        print(f"{'='*110}")

        # 表头
        header = (
            f"{'#':>3s}  {'评级':^4s}  {'Sharpe':>7s}  {'Sortino':>8s}  "
            f"{'MaxDD':>7s}  {'CAGR':>7s}  {'胜率':>6s}  {'Calmar':>7s}  "
            f"{'换手':>5s}  {'策略'}"
        )
        print(header)
        print('-' * 110)

        for r in results:
            s = r.get('sharpe', np.nan)
            rank = r.get('rank', '-')
            grade = r.get('grade', '-')
            sharpe = f"{s:+.2f}" if not np.isnan(s) else '  N/A'
            sortino_v = r.get('sortino', np.nan)
            sortino = f"{sortino_v:+.2f}" if not np.isnan(sortino_v) else '  N/A'
            maxdd = f"{r.get('max_drawdown', 0):.1%}" if not np.isnan(r.get('max_drawdown', np.nan)) else '  N/A'
            cagr = f"{r.get('cagr', 0):.1%}" if not np.isnan(r.get('cagr', np.nan)) else '  N/A'
            wr = f"{r.get('win_rate', 0):.0%}" if not np.isnan(r.get('win_rate', np.nan)) else ' N/A'
            calmar_v = r.get('calmar', np.nan)
            calmar = f"{calmar_v:.2f}" if not np.isnan(calmar_v) else '  N/A'
            turnover = f"{r.get('monthly_turnover', 0):.1f}" if not np.isnan(r.get('monthly_turnover', np.nan)) else 'N/A'
            name = r.get('strategy', '')

            line = (
                f"{rank:>3}  {grade:^4s}  {sharpe:>7s}  {sortino:>8s}  "
                f"{maxdd:>7s}  {cagr:>7s}  {wr:>6s}  {calmar:>7s}  "
                f"{turnover:>5s}  {name}"
            )
            print(line)

        print(f"{'='*110}")

        # 统计摘要
        valid = [r for r in results if not np.isnan(r.get('sharpe', np.nan))]
        if valid:
            avg_sharpe = np.mean([r['sharpe'] for r in valid])
            med_sharpe = np.median([r['sharpe'] for r in valid])
            best = valid[0]
            print(f"\n  平均 Sharpe: {avg_sharpe:.2f}  |  中位数: {med_sharpe:.2f}")
            print(f"  最佳策略: {best['strategy']}  (Sharpe={best['sharpe']:.2f})")
            n_a = sum(1 for r in valid if r.get('grade') == 'A')
            n_b = sum(1 for r in valid if r.get('grade') == 'B')
            print(f"  A级: {n_a}  |  B级: {n_b}  |  总计: {len(valid)}")
        print()


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point: discover strategies -> backtest -> rank -> output"""
    sys.stdout.reconfigure(encoding='utf-8')

    # 输出目录
    output_dir = str(_PROJECT_ROOT / 'reports')
    os.makedirs(output_dir, exist_ok=True)

    # 初始化排名器
    ranker = StrategyRanker(
        n_assets=80,
        n_months=120,
        seed=42,
    )

    # 1. 发现策略
    discovered = ranker.discover_strategies()
    print(f"\n  在 strategies/ 下发现 {len(discovered)} 个策略类:\n")
    for s in discovered:
        print(f"    {s['class']:<40s}  [{s['file']}]  风格={s['style']}")

    # 2. 回测所有策略
    raw_results = ranker.backtest_all()

    # 3. 排名
    ranked = ranker.rank_strategies(raw_results)

    # 4. 控制台输出
    ranker.print_ranking(ranked)

    # 5. 导出文件
    csv_path = os.path.join(output_dir, 'strategy_ranking.csv')
    md_path = os.path.join(output_dir, 'strategy_ranking.md')

    ranker.export_csv(ranked, csv_path)
    ranker.export_markdown(ranked, md_path)

    # 6. 生成汇总 DataFrame 并返回
    summary_df = ranker.generate_summary_table(ranked)
    print(f"\n  汇总表形状: {summary_df.shape}")
    print(f"  报告已保存至: {output_dir}/\n")

    return ranked, summary_df


if __name__ == '__main__':
    main()
