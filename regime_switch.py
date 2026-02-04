import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download USD/CAD data
def get_usdcad_data(start_date='2025-10-31', end_date='2026-02-01'):
    """
    Download USD/CAD forex data from Yahoo Finance
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    ticker = "USDCAD=X"  # Yahoo Finance ticker for USD/CAD
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def simple_regime_detection(data, fast_period=20, slow_period=50, 
                            trending_threshold=1.0, mean_rev_threshold=0.3):
    """
    Simple regime detection using ONLY MA spread
    
    Parameters:
    -----------
    fast_period : int
        Fast moving average period (default 20)
    slow_period : int
        Slow moving average period (default 50)
    trending_threshold : float
        Spread % above which market is trending (default 1.0%)
    mean_rev_threshold : float
        Spread % below which market is mean-reverting (default 0.3%)
    
    Returns:
    --------
    DataFrame with regime classification
    """
    df = data.copy()
    
    # Calculate moving averages
    df['MA_Fast'] = df['Close'].rolling(window=fast_period).mean()
    df['MA_Slow'] = df['Close'].rolling(window=slow_period).mean()
    
    # Calculate MA spread (percentage)
    df['MA_Spread'] = ((df['MA_Fast'] - df['MA_Slow']) / df['MA_Slow']) * 100
    
    # Absolute spread for regime detection
    df['Abs_Spread'] = abs(df['MA_Spread'])
    
    # Simple regime classification based ONLY on spread magnitude
    df['Regime'] = 'Transitional'
    df.loc[df['Abs_Spread'] >= trending_threshold, 'Regime'] = 'Trending'
    df.loc[df['Abs_Spread'] <= mean_rev_threshold, 'Regime'] = 'Mean-Reverting'
    
    return df

def calculate_regime_stats(df):
    """
    Calculate statistics for each regime
    """
    df['Returns'] = df['Close'].pct_change()
    
    stats = {}
    for regime in ['Trending', 'Mean-Reverting', 'Transitional']:
        regime_data = df[df['Regime'] == regime]
        
        if len(regime_data) > 0:
            stats[regime] = {
                'Days': len(regime_data),
                'Percentage': len(regime_data) / len(df[df['Regime'].notna()]) * 100,
                'Avg_Return_bps': regime_data['Returns'].mean() * 10000,
                'Volatility_bps': regime_data['Returns'].std() * 10000,
                'Avg_Abs_Spread': regime_data['Abs_Spread'].mean(),
            }
    
    return pd.DataFrame(stats).T

def plot_simple_regime(df, figsize=(15, 10)):
    """
    Create visualization focused on MA spread
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Price with MAs
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='USD/CAD', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(df.index, df['MA_Fast'], label='Fast MA (20)', color='blue', linewidth=1.5, alpha=0.7)
    ax1.plot(df.index, df['MA_Slow'], label='Slow MA (50)', color='red', linewidth=1.5, alpha=0.7)
    
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('USD/CAD with Moving Averages', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MA Spread with regime zones
    ax2 = axes[1]
    
    # Color background by regime
    trending = df[df['Regime'] == 'Trending']
    mean_rev = df[df['Regime'] == 'Mean-Reverting']
    
    for idx in trending.index:
        ax2.axvspan(idx, idx + timedelta(days=1), alpha=0.15, color='green')
    for idx in mean_rev.index:
        ax2.axvspan(idx, idx + timedelta(days=1), alpha=0.15, color='red')
    
    # Plot spread
    ax2.plot(df.index, df['MA_Spread'], label='MA Spread', color='purple', linewidth=2)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Trending Threshold (+1%)')
    ax2.axhline(y=-1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0.3, color='orange', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='Mean-Rev Threshold (0.3%)')
    ax2.axhline(y=-0.3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Fill areas
    ax2.fill_between(df.index, 0, df['MA_Spread'], 
                      where=df['MA_Spread'] > 0, alpha=0.3, color='blue')
    ax2.fill_between(df.index, 0, df['MA_Spread'], 
                      where=df['MA_Spread'] < 0, alpha=0.3, color='red')
    
    ax2.set_ylabel('Spread (%)', fontsize=11)
    ax2.set_title('MA Spread (Fast - Slow) as % of Slow MA', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime classification
    ax3 = axes[2]
    regime_numeric = df['Regime'].map({'Trending': 1, 'Transitional': 0, 'Mean-Reverting': -1})
    
    colors_map = {'Trending': 'green', 'Transitional': 'gray', 'Mean-Reverting': 'red'}
    for regime, color in colors_map.items():
        mask = df['Regime'] == regime
        ax3.scatter(df.index[mask], regime_numeric[mask], 
                   label=regime, color=color, alpha=0.7, s=20, edgecolors='none')
    
    ax3.set_ylabel('Regime', fontsize=11)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Mean-Reverting', 'Transitional', 'Trending'], fontsize=10)
    ax3.set_title('Regime Classification (Based on MA Spread Only)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10, ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main execution function
    """
    print("="*70)
    print("SIMPLIFIED REGIME DETECTION - MA SPREAD ONLY")
    print("="*70)
    print("\nDetects regimes using ONLY the spread between two moving averages")
    print("\nRegime Rules:")
    print("  • TRENDING: Absolute spread >= 1.0%")
    print("  • MEAN-REVERTING: Absolute spread <= 0.3%")
    print("  • TRANSITIONAL: 0.3% < Absolute spread < 1.0%")
    print("="*70)
    
    # Parameters
    FAST_MA = 20
    SLOW_MA = 50
    TRENDING_THRESHOLD = 1.0  # percentage
    MEAN_REV_THRESHOLD = 0.3  # percentage
    
    # Get data
    data = get_usdcad_data(start_date='2025-01-01')
    
    # Detect regimes
    df = simple_regime_detection(
        data, 
        fast_period=FAST_MA, 
        slow_period=SLOW_MA,
        trending_threshold=TRENDING_THRESHOLD,
        mean_rev_threshold=MEAN_REV_THRESHOLD
    )
    
    # Calculate statistics
    stats = calculate_regime_stats(df)
    
    # Print results
    # print(f"\nData period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(df)}")
    
    print("\n" + "="*70)
    print("REGIME STATISTICS")
    print("="*70)
    print(stats.round(2).to_string())
    
    # Current status
    print("\n" + "="*70)
    print("CURRENT STATUS")
    print("="*70)
    current_regime = df['Regime'].iloc[-1]
    current_spread = df['MA_Spread'].iloc[-1]
    current_abs_spread = df['Abs_Spread'].iloc[-1]
    
    # print(f"Current Price: {df['Close'].iloc[-1]:.4f}")
    # print(f"Fast MA (20): {df['MA_Fast'].iloc[-1]:.4f}")
    # print(f"Slow MA (50): {df['MA_Slow'].iloc[-1]:.4f}")
    print(f"MA Spread: {current_spread:+.3f}%")
    print(f"Absolute Spread: {current_abs_spread:.3f}%")
    print(f"\nCurrent Regime: {current_regime}")
    
    # Recommendation
    print("\n" + "-"*70)
    if current_regime == 'Trending':
        print("→ STRATEGY: Use TREND-FOLLOWING")
        print("  Examples: Moving average crossover, momentum, breakout strategies")
        if current_spread > 0:
            print("  Direction: MAs suggest UPTREND (Fast > Slow)")
        else:
            print("  Direction: MAs suggest DOWNTREND (Fast < Slow)")
    elif current_regime == 'Mean-Reverting':
        print("→ STRATEGY: Use MEAN-REVERSION")
        print("  Examples: Bollinger Bands, RSI, support/resistance fading")
        print("  Note: Market oscillating around MAs - fade extremes")
    else:
        print("→ STRATEGY: CAUTIOUS / REDUCE SIZE")
        print("  Market in transition - wait for clearer regime signal")
        print("  Consider staying flat or using smaller positions")
    
    # Regime history
    print("\n" + "="*70)
    print("RECENT REGIME CHANGES")
    print("="*70)
    regime_changes = df[df['Regime'] != df['Regime'].shift(1)].tail(10)
    # if len(regime_changes) > 0:
    #     for idx, row in regime_changes.iterrows():
    #         # print(f"{idx.strftime('%Y-%m-%d')}: {row['Regime']:15s} (Spread: {row['MA_Spread']:+.2f}%)")
    # else:
    #     print("No recent regime changes")
    
    # Create visualization
    fig = plot_simple_regime(df)
    plt.savefig('/Users/daiphylee/Downloads/F25 FARMSA/Quant/simple_regime_spread.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: simple_regime_spread.png")
    
    # Save data
    output_cols = ['Close', 'MA_Fast', 'MA_Slow', 'MA_Spread', 'Abs_Spread', 'Regime']
    df[output_cols].to_csv('/Users/daiphylee/Downloads/F25 FARMSA/Quant/simple_regime_data.csv')
    print(f"✓ Data saved to: simple_regime_data.csv")
    
    # Quick reference
    print("\n" + "="*70)
    print("QUICK INTERPRETATION GUIDE")
    print("="*70)
    print("MA Spread = (Fast MA - Slow MA) / Slow MA × 100")
    print()
    print("Large spread (>1%)    → TRENDING market")
    print("Small spread (<0.3%)  → MEAN-REVERTING market") 
    print("Medium spread         → TRANSITIONAL (mixed signals)")
    print()
    print("Positive spread       → Fast MA above Slow MA (potential uptrend)")
    print("Negative spread       → Fast MA below Slow MA (potential downtrend)")
    print("="*70)
    
    return df, stats

if __name__ == "__main__":
    df, stats = main()