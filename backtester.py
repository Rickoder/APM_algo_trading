import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from useful_function import get_sr        

# Define drawdown function
def compute_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown (worst peak-to-trough drop) of an equity curve.
    """
    peak = equity.cummax()
    return ((equity - peak) / peak).min()

# Define backtest function
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
    # 0) Pre-process raw signals so:
    #    • +1→+1 (and –1→–1) maintains the existing unit position
    #    • 0 only flattens you after five consecutive zeros
    sig = signals.copy().astype(int)
    smoothed = []
    prev_sig = 0
    zero_run = 0
    for s in sig:
        if s == 0 and prev_sig != 0:
            zero_run += 1
            if zero_run < 5:
                # keep the old position
                smoothed.append(prev_sig)
            else:
                # after 5 zeros, flatten
                prev_sig = 0
                zero_run = 0
                smoothed.append(prev_sig)
        else:
            prev_sig = s
            zero_run = 0
            smoothed.append(prev_sig)
    signals = pd.Series(smoothed, index=signals.index)

    # 1) Align positions: you only earn returns AFTER you take the pos
    pos = signals.shift(1).fillna(0)

    # 2) Market returns
    ret = prices.pct_change().fillna(0)

    # 3) Strategy returns before cost
    strat_ret = pos * ret

    # 4) Subtract costs on every change of position
    trades = pos.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * cost_per_trade

    # stop-loss cap at 15%, if breached, close the position
    stop_loss = 0.15   # 15% stop-loss threshold
    equity_temp = (1 + strat_ret).cumprod()
    dd = (equity_temp.cummax() - equity_temp) / equity_temp.cummax()
    breaches = np.where(dd > stop_loss)[0]
    if breaches.size:
        breach_idx  = breaches[0]
        breach_date = equity_temp.index[breach_idx]
        # cap that day's return
        strat_ret.loc[breach_date] = -stop_loss
        # (do NOT zero out pos: future signals still apply)

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
#       returns a Series of -1, 0, +1 signals

# from Strategy_MA         import ma_signals as ma_signals
# from Strategy_MA         import mom_signals as mom_signals
# from Valuation_strategy import pe_signals as pe_signals
# from Valuation_strategy import pb_signals as pb_signals
from strategy_random import random_signal as random_signal

# Main runner

def main():
    # 1) Load your cleaned price data
    df = (
        pd.read_csv('DF_data_cleaned.csv',
                    parse_dates=['Date'],
                    index_col='Date')
          .iloc[1:100, 1:10]
    )

    results     = []
    equity_dict = {}

    # 2) If needed, load extra inputs for some strategies:
    # pe_df   = pd.read_excel('data/PE RATIO.xlsx', sheet_name='PE_ratio_hist')
    # pb_df   = pd.read_csv('data/price_to_book_ratio.csv')
    # news_df = pd.read_csv('data/your_sentiment_data.csv')

    # 3) Map strategy names to their signal generators
    strategies = {
        'Random': random_signal,
        # 'MA':       ma_signals,
        # 'Momentum': mom_signals,
        # 'Value (P/E)': lambda pr: pe_signals(pe_df),
        # 'Value (P/B)': lambda pr: pb_signals(pb_df),
    }

    # 4) Loop and backtest
    for asset in df.columns:
        prices = df[asset]
        for name, gen in strategies.items():
            try:
                # 1) generate
                raw = gen(prices)
                # 2) array → trim to length
                raw = np.asarray(raw)[:len(prices)]
                # 3) sanitize NaN/inf → 0
                raw = np.nan_to_num(raw, nan=0., posinf=0., neginf=0.)
                # 4) build an integer Series aligned to dates
                signals = pd.Series(raw, index=prices.index).astype(int)

                #  Plot price + buy/sell markers
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(prices.index, prices, label='Price', lw=1)
                buys  = signals[signals == 1].index
                sells = signals[signals == -1].index
                ax.scatter(buys,  prices.loc[buys],  marker='^', s=100,
                           label='Buy',  edgecolors='green', facecolors='none')
                ax.scatter(sells, prices.loc[sells], marker='v', s=100,
                           label='Sell', edgecolors='red',   facecolors='none')
                ax.set_title(f'{asset} - {name} signals')
                ax.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f'plots/{asset}_{name}_signals.png')
                plt.close(fig)

                #  Run the backtest
                equity, strat_ret, m = run_backtest(prices, signals)
                results.append({'Asset': asset, 'Strategy': name, **m})
                equity_dict[f'{asset}|{name}'] = equity

                # Plot equity curve
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(equity.index, equity, label='Strategy Equity', lw=1.5)
                ax.set_title(f'{asset} — {name} Equity Curve')
                ax.set_ylabel('Equity (× initial)')
                ax.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f'plots/{asset}_{name}_equity.png')
                plt.close(fig)

            except Exception as e:
                print(f"Error running strategy {name} on {asset}: {e}")

    # 5) Summarize all results
    summary = pd.DataFrame(results)
    print(summary.to_markdown(index=False))
    summary.to_csv('backtest_summary.csv', index=False)

    # 6) Overall equal-weight portfolio per strategy
    eq_df = pd.DataFrame(equity_dict)
    port_results = []
    for strat in strategies:
        cols = [c for c in eq_df.columns if c.endswith(f'|{strat}')]
        if not cols:
            continue
        port_eq = eq_df[cols].mean(axis=1)

        # plot overall curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(port_eq.index, port_eq, lw=2, label=f'{strat} Portfolio')
        ax.set_title(f'Overall Equal-Weight — {strat}')
        ax.set_ylabel('Equity (× initial)')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'plots/Overall_{strat}_equity.png')
        plt.close(fig)

        # metrics
        port_ret = port_eq.pct_change().fillna(0)
        port_results.append({
            'Strategy'     : strat,
            'total_return' : port_eq.iloc[-1] - 1,
            'sharpe'       : get_sr(port_ret),
            'max_drawdown' : compute_drawdown(port_eq)
        })

    port_summary = pd.DataFrame(port_results)
    print('\n**Overall Portfolio Results**\n')
    print(port_summary.to_markdown(index=False))
    port_summary.to_csv('backtest_portfolio_summary.csv', index=False)


main()