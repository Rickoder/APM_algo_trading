import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from useful_function import get_sr        

# Backtester

def compute_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown (worst peak-to-trough drop) of an equity curve.
    """
    peak = equity.cummax()
    return ((equity - peak) / peak).min()

def run_backtest(prices: pd.Series,
                 signals: pd.Series,
                 cost_per_trade: float = 0.001) -> tuple:
    """
    Vectorized daily backtest.

    Arguments:
      prices         : pd.Series of close prices, indexed by Date
      signals        : pd.Series of -1/0/+1 positions, same index
      cost_per_trade : round-turn transaction cost (e.g. 0.001 = 0.1%)

    Returns:
      equity   : pd.Series of compounded equity curve starting at 1.0
      strat_ret: pd.Series of daily strategy returns
      metrics  : dict of total_return, sharpe, max_drawdown
    """
    # 1) Align positions: you only earn returns AFTER you take the pos
    pos = signals.shift(1).fillna(0)

    # 2) Market returns
    ret = prices.pct_change().fillna(0)

    # 3) Strategy returns before cost
    strat_ret = pos * ret

    # 4) Subtract costs on every change of position
    trades    = pos.diff().abs()
    strat_ret -= trades * cost_per_trade

    # 5) Equity curve
    equity = (1 + strat_ret).cumprod()

    # 6) Performance metrics
    metrics = {
        'total_return' : equity.iloc[-1] - 1,
        'sharpe'       : get_sr(strat_ret),
        'max_drawdown' : compute_drawdown(equity)
    }

    return equity, strat_ret, metrics

# Strategy imports

# Each of these modules must define:
#   def generate_signals(prices: pd.Series) -> pd.Series:
#       # returns a Series of -1, 0, +1 signals

from strategies.Strategy_MA         import ma_signals as ma_signals
from strategies.Strategy_MA         import mom_signals as mom_signals
from strategies.Valuation_strategy import pe_signals as pe_signals
from strategies.Valuation_strategy import pb_signals as pb_signals

# Main runner

def main():
    # 1) Load your cleaned price data
    df     = pd.read_csv('data/data_cleaned.csv', parse_dates=['Date'], index_col='Date')
    results = []

    # 2) If needed, load extra inputs for some strategies:
    # pe_df = pd.read_excel('data/PE RATIO.xlsx', sheet_name='PE_ratio_hist')
    # pb_df = pd.read_csv('data/price_to_book_ratio.csv')
    # news_df = pd.read_csv('data/your_sentiment_data.csv')

    # 3) Map strategy names to their signal generators

    strategies = {
        'MA': lambda: ma_signals(prices),     # ensure args are loaded
        'Momentum':    lambda: mom_signals(prices),
        'Value (P/E)':  lambda: pe_signals(pe_df),             
        'Value (P/B)': lambda: pb_signals(pb_df),
    }

    # 4) Loop and backtest
    for asset in df.columns:
        prices = df[asset]
        for name, gen in strategies.items():
            try:
                signals = gen().astype(int)
                equity, strat_ret, m = run_backtest(prices, signals)
                results.append({'Strategy': name, **m})
            except Exception as e:
                print(f"Error running strategy {name}: {e}")

            equity.plot(title=name)
            plt.savefig(f'plots/{name}_equity_curve.png')

    # 5) Summarize all results
    summary = pd.DataFrame(results)
    print(summary.to_markdown(index=False))
    summary.to_csv('backtest_summary.csv', index=False)

if __name__ == '__main__':
    main()
