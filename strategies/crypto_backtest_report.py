"""Full crypto strategy backtest report — aggregates every crypto strategy and produces a ranking report

Features:
    1. Aggregates the 5 strategies in strategy_crypto.py + the Bollinger band strategy from mean_reversion_crypto.py
    2. Simulates 180 days of BTC/ETH/SOL daily data with an Ornstein-Uhlenbeck process
    3. Backtests each strategy, computing Sharpe / MaxDD / annualized return / win rate
    4. Produces a summary ranking table (sorted by Sharpe)
    5. Supports quantstats HTML report output

Usage:
    python strategies/crypto_backtest_report.py
    python strategies/crypto_backtest_report.py --html reports/crypto_report.html

Author: KuanQuant
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── 确保项目根目录在 sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf.strategy_crypto import (
    CryptoBaseStrategy,
    CryptoMomentumStrategy,
    CryptoMeanReversionStrategy,
    CryptoTrendFollowStrategy,
    CryptoBTCBetaStrategy,
    CryptoVolTargetStrategy,
    run_crypto_backtest,
    TRADING_DAYS_YEAR,
)
from strategies.mean_reversion_crypto import BollingerMeanReversionCrypto

# ── 尝试导入 quantstats 适配器 (可选) ─────────────────────────────────────
try:
    from qf.integrations.quantstats_adapter import (
        generate_report,
        compute_metrics,
        compare_strategies,
    )
    _HAS_QS_ADAPTER = True
except ImportError:
    _HAS_QS_ADAPTER = False


# ═══════════════════════════════════════════════════════════════════════════
#  模拟数据生成
# ═══════════════════════════════════════════════════════════════════════════

def _ou_process(
    mean_price: float,
    volatility: float,
    mean_reversion_speed: float = 0.05,
    n: int = 180,
    seed: int | None = None,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck 过程 — 模拟均值回归价格序列。

    参数:
        mean_price: 长期均值价格
        volatility: 日波动率 (占均值百分比, 如 0.03 = 3%)
        mean_reversion_speed: 均值回归速度, 越大回归越快
        n: 序列长度 (天数)
        seed: 随机种子

    返回:
        np.ndarray — 模拟价格序列
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    prices = [mean_price]
    for _ in range(n - 1):
        # dP = theta * (mu - P) * dt + sigma * mu * dW
        dp = mean_reversion_speed * (mean_price - prices[-1])
        dp += volatility * mean_price * rng.randn()
        new_price = max(prices[-1] + dp, mean_price * 0.05)  # 防止负价格
        prices.append(new_price)
    return np.array(prices)


def generate_simulated_data(n_days: int = 180, seed: int = 42) -> dict:
    """
    Generate simulated BTC/ETH/SOL daily data.

    Uses an Ornstein-Uhlenbeck process to simulate mean-reverting price series,
    overlaid with a weak trend component so the data more closely resembles real markets.

    Parameters:
        n_days: number of days to simulate
        seed: random seed (for reproducibility)

    Returns:
        dict — {symbol: DataFrame(open, high, low, close, volume)}
    """
    # 结束日期为今天, 起始日期往前推 n_days
    end_date = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=end_date, periods=n_days, freq='D')

    # 各币种配置: (均价, 日波动率, 均值回归速度)
    configs = {
        'BTC/USD': (85_000, 0.025, 0.03),   # BTC 均价 85000, 波动 2.5%
        'ETH/USD': (3_200, 0.035, 0.04),    # ETH 均价 3200, 波动 3.5%
        'SOL/USD': (140, 0.050, 0.05),      # SOL 均价 140, 波动 5%
    }

    rng = np.random.RandomState(seed)
    data_dict = {}

    for symbol, (mean_p, vol, speed) in configs.items():
        # 生成收盘价序列
        close_prices = _ou_process(mean_p, vol, speed, n_days, seed=seed + hash(symbol) % 10000)

        # 叠加微弱漂移 (模拟轻微趋势)
        drift = np.linspace(0, rng.uniform(-0.05, 0.10), n_days)
        close_prices = close_prices * (1 + drift)

        # 从收盘价派生 OHLCV
        noise_scale = vol * 0.3
        open_prices = close_prices * (1 + rng.randn(n_days) * noise_scale)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.randn(n_days)) * noise_scale)
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.randn(n_days)) * noise_scale)

        # 成交量: 对数正态分布
        volume = rng.lognormal(mean=20, sigma=1.0, size=n_days)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
        }, index=dates)

        data_dict[symbol] = df

    return data_dict


# ═══════════════════════════════════════════════════════════════════════════
#  策略注册表
# ═══════════════════════════════════════════════════════════════════════════

def get_all_strategies(symbols: list[str] | None = None) -> list[CryptoBaseStrategy]:
    """
    Aggregate every crypto strategy instance.

    Includes:
        1. Crypto momentum strategy (CryptoMomentumStrategy)
        2. Crypto mean reversion strategy (CryptoMeanReversionStrategy)
        3. Crypto trend following strategy (CryptoTrendFollowStrategy)
        4. Crypto BTC beta strategy (CryptoBTCBetaStrategy)
        5. Crypto volatility targeting strategy (CryptoVolTargetStrategy)
        6. Bollinger band mean reversion strategy (BollingerMeanReversionCrypto)

    Parameters:
        symbols: list of trading pairs, defaults to ['BTC/USD', 'ETH/USD', 'SOL/USD']

    Returns:
        list[CryptoBaseStrategy]
    """
    syms = symbols or ['BTC/USD', 'ETH/USD', 'SOL/USD']

    strategies = [
        # ── strategy_crypto.py 中的 5 个策略 ──
        CryptoMomentumStrategy(symbols=syms, lookback=20, long_n=2, short_n=1),
        CryptoMeanReversionStrategy(symbols=syms, window=20, buy_threshold=-2.0, sell_threshold=2.0),
        CryptoTrendFollowStrategy(symbols=syms, fast_period=10, slow_period=50, long_only=False),
        CryptoBTCBetaStrategy(symbols=syms, beta_window=60, top_n=2),
        CryptoVolTargetStrategy(symbols=syms, target_vol=0.15, max_leverage=2.0),

        # ── mean_reversion_crypto.py 中的布林带策略 ──
        BollingerMeanReversionCrypto(
            symbols=syms, bb_window=20, bb_std=2.0,
            rsi_period=14, rsi_oversold=30, rsi_overbought=70,
            target_vol=0.50,
        ),
    ]

    return strategies


# ═══════════════════════════════════════════════════════════════════════════
#  回测执行 & 指标计算
# ═══════════════════════════════════════════════════════════════════════════

def _compute_win_rate(daily_returns: pd.Series) -> float:
    """
    计算胜率 (正收益天数 / 总交易天数)。

    参数:
        daily_returns: 日收益率序列

    返回:
        float — 胜率, [0, 1]
    """
    valid = daily_returns[daily_returns != 0]
    if len(valid) == 0:
        return 0.0
    return float((valid > 0).sum() / len(valid))


def run_all_backtests(
    data_dict: dict,
    strategies: list[CryptoBaseStrategy] | None = None,
    initial_capital: float = 10_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run backtests for every strategy and return a summary DataFrame.

    Parameters:
        data_dict: simulated or real market data
        strategies: strategy list, defaults to get_all_strategies()
        initial_capital: starting capital
        verbose: whether to print progress information

    Returns:
        pd.DataFrame — sorted by Sharpe descending. Column keys are the
                        literal strings used in the code, several of which are
                        Chinese: '策略名称' (strategy name), 'Sharpe',
                        '年化收益' (annualized return), '总收益' (total return),
                        '最大回撤' (max drawdown), '胜率' (win rate),
                        '回测天数' (backtest days). The index is named
                        '排名' (rank).
    """
    if strategies is None:
        strategies = get_all_strategies()

    results = []

    for i, strat in enumerate(strategies, 1):
        if verbose:
            print(f"  [{i}/{len(strategies)}] 回测: {strat.name} ...")

        try:
            bt_result = run_crypto_backtest(strat, data_dict, initial_capital)

            # 计算胜率
            win_rate = _compute_win_rate(bt_result['daily_returns'])

            results.append({
                '策略名称': strat.name,
                'Sharpe': round(bt_result['sharpe'], 3),
                '年化收益': round(bt_result['cagr'], 4),
                '总收益': round(bt_result['total_return'], 4),
                '最大回撤': round(bt_result['max_drawdown'], 4),
                '胜率': round(win_rate, 4),
                '回测天数': bt_result['n_days'],
                # 保留日收益率序列, 用于后续 quantstats 报告
                '_daily_returns': bt_result['daily_returns'],
                '_equity_curve': bt_result['equity_curve'],
            })

            if verbose:
                print(f"        Sharpe={bt_result['sharpe']:.3f}  "
                      f"年化={bt_result['cagr']:+.2%}  "
                      f"MaxDD={bt_result['max_drawdown']:.2%}  "
                      f"胜率={win_rate:.2%}")

        except Exception as e:
            if verbose:
                print(f"        [错误] {e}")
            results.append({
                '策略名称': strat.name,
                'Sharpe': np.nan,
                '年化收益': np.nan,
                '总收益': np.nan,
                '最大回撤': np.nan,
                '胜率': np.nan,
                '回测天数': 0,
                '_daily_returns': pd.Series(dtype=float),
                '_equity_curve': pd.Series(dtype=float),
            })

    # 构建排名表
    df = pd.DataFrame(results)
    df = df.sort_values('Sharpe', ascending=False, na_position='last').reset_index(drop=True)
    df.index += 1  # 排名从1开始
    df.index.name = '排名'

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  报告输出
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print the summary ranking table to the terminal.

    Parameters:
        df: the DataFrame returned by run_all_backtests()
    """
    # 显示列 (排除内部数据列)
    display_cols = ['策略名称', 'Sharpe', '年化收益', '总收益', '最大回撤', '胜率', '回测天数']
    display_df = df[display_cols].copy()

    # 格式化百分比列
    for col in ['年化收益', '总收益', '最大回撤', '胜率']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    print("\n" + "=" * 90)
    print("  加密货币策略回测汇总排名 (按 Sharpe 降序)")
    print("=" * 90)
    print(display_df.to_string())
    print("=" * 90)


def generate_quantstats_reports(
    df: pd.DataFrame,
    output_dir: str | None = None,
    combined_html: str | None = None,
) -> None:
    """
    Generate a quantstats HTML report for each strategy.

    Parameters:
        df: the DataFrame returned by run_all_backtests()
        output_dir: output directory for per-strategy reports (one HTML each), skipped if None
        combined_html: output path for the combined report, skipped if None
    """
    if not _HAS_QS_ADAPTER:
        print("\n[警告] quantstats 未安装, 跳过 HTML 报告生成。")
        print("       安装: pip install quantstats-lumi")
        return

    # ── 为排名第一的策略生成详细报告 ──
    if combined_html:
        best_row = df.iloc[0]
        best_name = best_row['策略名称']
        best_returns = best_row['_daily_returns']

        if len(best_returns) > 0:
            os.makedirs(os.path.dirname(combined_html) or '.', exist_ok=True)
            try:
                generate_report(
                    best_returns,
                    output_file=combined_html,
                    title=f"Kuant 最佳策略报告 - {best_name}",
                )
                print(f"\n[报告] 最佳策略 HTML 报告已保存: {combined_html}")
            except Exception as e:
                print(f"\n[警告] 最佳策略报告生成失败: {e}")

    # ── 为每个策略生成单独报告 ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for _, row in df.iterrows():
            name = row['策略名称']
            returns = row['_daily_returns']

            if len(returns) == 0:
                continue

            # 文件名: 移除特殊字符
            safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            filepath = os.path.join(output_dir, f"{safe_name}.html")

            try:
                generate_report(
                    returns,
                    output_file=filepath,
                    title=f"Kuant - {name}",
                )
                print(f"  [报告] {name} → {filepath}")
            except Exception as e:
                print(f"  [警告] {name} 报告失败: {e}")

    # ── 策略对比 (使用 quantstats adapter 的 compare_strategies) ──
    try:
        strat_returns = {}
        for _, row in df.iterrows():
            if len(row['_daily_returns']) > 0:
                strat_returns[row['策略名称']] = row['_daily_returns']

        if strat_returns:
            comparison = compare_strategies(strat_returns)
            print("\n  [对比] quantstats 策略对比指标:")
            comp_df = pd.DataFrame(comparison).T
            # 选取关键列
            key_cols = [c for c in ['sharpe', 'sortino', 'max_drawdown', 'cagr', 'volatility', 'win_rate']
                        if c in comp_df.columns]
            if key_cols:
                print(comp_df[key_cols].round(4).to_string())
    except Exception as e:
        print(f"  [警告] 策略对比失败: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point: parse arguments, run the backtests, and write the reports."""
    parser = argparse.ArgumentParser(
        description="加密货币全策略回测报告生成器"
    )
    parser.add_argument(
        '--days', type=int, default=180,
        help='回测天数 (默认 180)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认 42)'
    )
    parser.add_argument(
        '--capital', type=float, default=10_000,
        help='初始资金 (默认 10000)'
    )
    parser.add_argument(
        '--html', type=str, default=None,
        help='最佳策略 HTML 报告输出路径 (如 reports/crypto_best.html)'
    )
    parser.add_argument(
        '--html-dir', type=str, default=None,
        help='各策略单独 HTML 报告输出目录 (如 reports/crypto/)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='静默模式, 仅输出排名表'
    )
    args = parser.parse_args()

    verbose = not args.quiet

    print("=" * 60)
    print("  KuanQuant 加密货币全策略回测报告")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  回测天数: {args.days}   初始资金: ${args.capital:,.0f}")
    print(f"  数据来源: Ornstein-Uhlenbeck 模拟 (seed={args.seed})")
    print("=" * 60)

    # ── 第1步: 生成模拟数据 ──
    if verbose:
        print("\n[1/3] 生成模拟价格数据 (BTC/ETH/SOL) ...")
    data_dict = generate_simulated_data(n_days=args.days, seed=args.seed)

    if verbose:
        for sym, df in data_dict.items():
            print(f"  {sym}: {len(df)}天, "
                  f"起始价=${df['close'].iloc[0]:,.2f}, "
                  f"最终价=${df['close'].iloc[-1]:,.2f}, "
                  f"日均波动={df['close'].pct_change().std():.2%}")

    # ── 第2步: 运行全策略回测 ──
    if verbose:
        print(f"\n[2/3] 运行 6 个策略回测 ...")
    result_df = run_all_backtests(
        data_dict,
        initial_capital=args.capital,
        verbose=verbose,
    )

    # ── 第3步: 输出报告 ──
    if verbose:
        print(f"\n[3/3] 生成报告 ...")
    print_summary_table(result_df)

    # quantstats HTML 报告 (如果指定了输出路径)
    if args.html or args.html_dir:
        generate_quantstats_reports(
            result_df,
            output_dir=args.html_dir,
            combined_html=args.html,
        )

    # 打印最佳策略信息
    best = result_df.iloc[0]
    print(f"\n  最佳策略: {best['策略名称']}")
    print(f"  Sharpe: {best['Sharpe']:.3f}  "
          f"年化: {best['年化收益']:.2%}  "
          f"MaxDD: {best['最大回撤']:.2%}  "
          f"胜率: {best['胜率']:.2%}")

    return result_df


if __name__ == '__main__':
    main()
