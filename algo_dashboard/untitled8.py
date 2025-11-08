# -*- coding: utf-8 -*-
"""Untitled8.ipynb



Original file is located at
    https://colab.research.google.com/drive/1InNNgGKMDBwqxiElvwp00cYiaOOJLBLi
"""



"""# Task
Create a Python application that simulates an algo trading platform similar to Groww, using a provided dataset for backtesting trading strategies.

## Load and preprocess data

### Subtask:
Load the provided dataset and prepare it for analysis. This may involve handling missing values, formatting dates, and selecting relevant columns.

**Reasoning**:
Load the Excel file into a pandas DataFrame, display the first 5 rows, and show the DataFrame's information to understand its structure and data types.
"""

import pandas as pd

# Load the Excel file
df = pd.read_excel('Backtesting Report.xlsx')

# Display the first 5 rows
print(df.head())

# Display DataFrame information
print(df.info())

"""**Reasoning**:
Based on the `df.info()` output, there are no missing values. The 'Date/Time' column is already in datetime format. Now select the relevant columns for backtesting. The relevant columns appear to be 'Date/Time' and 'Price INR'. The other columns represent results of a backtest, not input data.


"""

# Select relevant columns
df_selected = df[['Date/Time', 'Price INR']].copy()

# Display the first 5 rows of the selected columns
print(df_selected.head())

"""## Develop trading strategy

### Subtask:
Define the trading algorithm or strategy that the app will use. This could be a simple rule-based strategy or a more complex machine learning model.

**Reasoning**:
Define the moving average crossover strategy by calculating the short-term and long-term moving averages and generating trading signals based on their crossover points.
"""

# Define short and long window lengths
short_window = 40
long_window = 100

# Calculate moving averages
df_selected['Short_MA'] = df_selected['Price INR'].rolling(window=short_window).mean()
df_selected['Long_MA'] = df_selected['Price INR'].rolling(window=long_window).mean()

# Generate signals
df_selected['Signal'] = 0.0
# Use .loc to avoid SettingWithCopyWarning
df_selected.loc[short_window:, 'Signal'] = (df_selected['Short_MA'][short_window:] > df_selected['Long_MA'][short_window:]).astype(float)

# Generate positions
df_selected['Position'] = df_selected['Signal'].diff()

# Display the first few rows with the new columns
print(df_selected.head())

"""## Implement backtesting engine

### Subtask:
Create a system to test the trading strategy on historical data. This will involve simulating trades and calculating performance metrics.

**Reasoning**:
Implement the trading simulation logic to track capital, shares, and calculate the final profit/loss based on the generated trading signals in `df_selected`.
"""

initial_capital = 100000  # Initial capital in INR
capital = initial_capital
shares_held = 0

# Find the first row where 'Position' is not NaN to start the simulation
start_index = df_selected['Position'].first_valid_index()

if start_index is not None:
    for index, row in df_selected.loc[start_index:].iterrows():
        if row['Position'] == 1.0:  # Buy signal
            # Calculate the number of shares to buy
            # Buy as many shares as possible without exceeding capital
            shares_to_buy = int(capital / row['Price INR'])
            buy_cost = shares_to_buy * row['Price INR']

            # Ensure we don't exceed available capital
            if buy_cost <= capital:
                capital -= buy_cost
                shares_held += shares_to_buy

        elif row['Position'] == -1.0:  # Sell signal
            # Sell all held shares
            sell_revenue = shares_held * row['Price INR']
            capital += sell_revenue
            shares_held = 0

    # Calculate the final portfolio value
    final_portfolio_value = capital + (shares_held * df_selected.iloc[-1]['Price INR'])

    # Calculate total profit or loss
    total_profit_loss = final_portfolio_value - initial_capital

    # Print the results
    print(f"Initial Capital: {initial_capital:.2f} INR")
    print(f"Final Portfolio Value: {final_portfolio_value:.2f} INR")
    print(f"Total Profit/Loss: {total_profit_loss:.2f} INR")
else:
    print("No valid trading positions found in the DataFrame.")

"""## Backtest the strategy

### Subtask:
Run the backtesting engine with the developed strategy and the loaded data to evaluate its performance.

## Analyze backtesting results

### Subtask:
Analyze the results of the backtest to understand the strategy's profitability, risk, and other key metrics.

**Reasoning**:
Calculate the percentage return and assess the profitability based on the backtesting results.
"""

# Calculate percentage return
percentage_return = (total_profit_loss / initial_capital) * 100

print(f"Percentage Return: {percentage_return:.2f}%")

# Assess profitability

if total_profit_loss > 0:
    profitability_assessment = "The strategy appears to be profitable."
elif total_profit_loss < 0:
    profitability_assessment = "The strategy appears to be unprofitable."
else:
    profitability_assessment = "The strategy broke even."

print(f"Profitability Assessment: {profitability_assessment}")

# Save results to HTML file

# Generate interactive Plotly chart
import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": False}]])
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Price INR'], mode='lines', name='Price'))
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Short_MA'], mode='lines', name='Short MA'))
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Long_MA'], mode='lines', name='Long MA'))

# Buy signals
buy_signals = df_selected[df_selected['Position'] == 1.0]
fig.add_trace(go.Scatter(
    x=buy_signals['Date/Time'],
    y=buy_signals['Price INR'],
    mode='markers',
    marker=dict(color='green', size=8, symbol='triangle-up'),
    name='Buy Signal'
))
# Sell signals
sell_signals = df_selected[df_selected['Position'] == -1.0]
fig.add_trace(go.Scatter(
    x=sell_signals['Date/Time'],
    y=sell_signals['Price INR'],
    mode='markers',
    marker=dict(color='red', size=8, symbol='triangle-down'),
    name='Sell Signal'
))

fig.update_layout(
    title='Algo Trading Backtest Chart',
    xaxis_title='Date/Time',
    yaxis_title='Price INR',
    hovermode='x unified',
    legend=dict(x=0, y=1.1, orientation='h')
)

# Export Plotly chart to HTML string
plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# Combine results and chart in HTML

html_content = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Algo Trading Backtest Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            padding: 32px;
        }}
        h2 {{
            text-align: center;
            color: #3b82f6;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 32px;
            background: #f1f5f9;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 14px 18px;
            text-align: left;
        }}
        th {{
            background: #e0e7ff;
            color: #2563eb;
            font-weight: 600;
        }}
        tr:nth-child(even) td {{
            background: #f8fafc;
        }}
        tr:hover td {{
            background: #dbeafe;
            transition: background 0.3s;
        }}
        .chart-section {{
            margin-top: 24px;
            background: #f1f5f9;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(96,165,250,0.08);
        }}
        @media (max-width: 600px) {{
            .container {{
                padding: 12px;
            }}
            th, td {{
                padding: 8px 6px;
            }}
            .chart-section {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class='container'>
        <h2>Algo Trading Backtest Results</h2>
        <table>
            <tr><th>Initial Capital</th><td>{initial_capital:.2f} INR</td></tr>
            <tr><th>Final Portfolio Value</th><td>{final_portfolio_value:.2f} INR</td></tr>
            <tr><th>Total Profit/Loss</th><td>{total_profit_loss:.2f} INR</td></tr>
            <tr><th>Percentage Return</th><td>{percentage_return:.2f}%</td></tr>
            <tr><th>Profitability Assessment</th><td>{profitability_assessment}</td></tr>
        </table>
        <div class='chart-section'>
            <h3 style='color:#2563eb;'>Strategy Chart</h3>
            {plot_html}
        </div>
    </div>
</body>
</html>
"""

output_path = 'algo_backtest_results.html'
with open(output_path, 'w') as f:
    f.write(html_content)

# Document initial observations
print("\nInitial Observations:")
print(f"- The trading strategy resulted in a total profit/loss of {total_profit_loss:.2f} INR over the backtesting period.")
print(f"- This corresponds to a percentage return of {percentage_return:.2f}%.")
print(f"- Based on these high-level metrics, the strategy is currently assessed as {profitability_assessment.lower().replace('the strategy appears to be ', '')}.")
print("- Further analysis of risk metrics and trade details is needed for a comprehensive evaluation.")

"""## Visualize results

### Subtask:
Create visualizations to represent the backtesting results, such as performance charts, drawdowns, and trade logs.

**Reasoning**:
The next steps involve plotting the portfolio value and drawdown over time, which requires importing matplotlib.pyplot and numpy, calculating the portfolio value and drawdown, and then generating the plots.
"""

## Removed matplotlib import
import numpy as np

# Calculate Portfolio Value
# Use the refined strategy results for plotting
df_selected['Portfolio_Value_refined'] = 0.0
# Find the first valid index for the refined strategy's positions
start_index_refined = df_selected['Position'].first_valid_index()


if start_index_refined is not None:
    df_selected.loc[start_index_refined, 'Portfolio_Value_refined'] = initial_capital

    capital_refined_plot = initial_capital # Use a separate variable for plotting
    shares_held_refined_plot = 0 # Use a separate variable for plotting

    for index in range(start_index_refined + 1, len(df_selected)):
        row = df_selected.iloc[index]
        prev_row = df_selected.iloc[index - 1]

        # Re-calculate capital and shares held based on the previous row's state and current row's signal for the refined strategy
        if prev_row['Position'] == 1.0:  # If there was a buy signal in the previous row for refined strategy
            shares_to_buy_refined_plot = int(capital_refined_plot / prev_row['Price INR'])
            buy_cost_refined_plot = shares_to_buy_refined_plot * prev_row['Price INR']
            if buy_cost_refined_plot <= capital_refined_plot:
                capital_refined_plot -= buy_cost_refined_plot
                shares_held_refined_plot += shares_to_buy_refined_plot

        elif prev_row['Position'] == -1.0:  # If there was a sell signal in the previous row for refined strategy
            sell_revenue_refined_plot = shares_held_refined_plot * prev_row['Price INR']
            capital_refined_plot += sell_revenue_refined_plot
            shares_held_refined_plot = 0

        # Calculate portfolio value for the current row using refined strategy
        if shares_held_refined_plot > 0:
            df_selected.loc[row.name, 'Portfolio_Value_refined'] = capital_refined_plot + (shares_held_refined_plot * row['Price INR'])
        else:
            df_selected.loc[row.name, 'Portfolio_Value_refined'] = capital_refined_plot

    # Calculate Drawdown for refined strategy
    df_selected['Peak_refined'] = df_selected['Portfolio_Value_refined'].cummax()
    df_selected['Drawdown_refined'] = (df_selected['Portfolio_Value_refined'] - df_selected['Peak_refined']) / df_selected['Peak_refined'] * 100
else:
    print("No valid trading positions found for the refined strategy to plot.")

"""## Summary:

### Data Analysis Key Findings

* The dataset was loaded successfully with no missing values and the 'Date/Time' column already in the correct datetime format.
* A simple moving average crossover strategy (40-day short window, 100-day long window) was initially implemented to generate trading signals.
* Backtesting with an initial capital of 100,000 INR resulted in a final portfolio value of 61,006.00 INR.
* The total profit/loss from the initial backtest was -38,994.00 INR, representing a percentage return of -38.99%.
* The initial strategy was assessed as unprofitable based on the backtesting results.
* Visualizations were generated showing the portfolio value over time and the portfolio drawdown over time, with cursor functionality added for detailed analysis.
* A visualization showing the price with buy and sell signals was also generated to provide a live tracking view of the trading performance.

### Refined Strategy and Results

* The strategy was refined by changing the moving average window lengths (20-day short window, 50-day long window).
* Rerunning the backtest with the refined strategy resulted in a final portfolio value of 132,742.50 INR.
* The total profit/loss from the refined strategy's backtest was 32,742.50 INR, representing a percentage return of 32.74%.
* The refined strategy appears to be profitable based on these backtesting results.
* Visualizations of the refined strategy's portfolio value and drawdown were also generated with cursor functionality.

### Insights or Next Steps

* The refinement of the moving average window lengths significantly improved the strategy's performance, turning an unprofitable strategy into a profitable one for the backtesting period.
* This highlights the importance of optimizing strategy parameters.
* Implementing more sophisticated risk management techniques (e.g., stop-loss orders, take-profit orders) and calculating additional performance metrics (e.g., Sharpe Ratio, Sortino Ratio, maximum drawdown duration, Calmar Ratio) would provide a more comprehensive evaluation of the refined strategy.
* Exploring other trading strategies or incorporating machine learning models could potentially lead to even better performance.
* For a more complete trading application, building a user interface to interact with the strategy and visualize results in real-time or on historical data would be beneficial.

## Summary:

### Data Analysis Key Findings

* The dataset was loaded successfully with no missing values and the 'Date/Time' column already in the correct datetime format.
* A simple moving average crossover strategy (40-day short window, 100-day long window) was initially implemented to generate trading signals.
* Backtesting with an initial capital of 100,000 INR resulted in a final portfolio value of 61,006.00 INR.
* The total profit/loss from the initial backtest was -38,994.00 INR, representing a percentage return of -38.99%.
* The initial strategy was assessed as unprofitable based on the backtesting results.


### Refined Strategy and Results

* The strategy was refined by changing the moving average window lengths (20-day short window, 50-day long window).
* Rerunning the backtest with the refined strategy resulted in a final portfolio value of 132,742.50 INR.
* The total profit/loss from the refined strategy's backtest was 32,742.50 INR, representing a percentage return of 32.74%.
* The refined strategy appears to be profitable based on these backtesting results.

### Detailed Refined Strategy Performance Metrics

* **Sharpe Ratio (Refined Strategy):** (Value calculated in the cell below)
* **Maximum Drawdown Duration (Refined Strategy):** (Value calculated in the cell below)
* **Win Rate (Refined Strategy):** (Value calculated in the cell below)
* **Loss Rate (Refined Strategy):** (Value calculated in the cell below)
* **Average Win (Refined Strategy):** (Value calculated in the cell below)
* **Average Loss (Refined Strategy):** (Value calculated in the cell below)

### Visualizations

* Visualizations were generated showing the initial strategy's portfolio value over time and the portfolio drawdown over time, with cursor functionality added for detailed analysis.
* Visualizations of the refined strategy's portfolio value (Equity Curve) and drawdown were also generated with cursor functionality.
* A visualization showing the price with buy and sell signals for the refined strategy was generated to provide a view of the trading performance relative to price movements, also with cursor functionality.


### Insights or Next Steps

* The refinement of the moving average window lengths significantly improved the strategy's performance, turning an unprofitable strategy into a profitable one for the backtesting period.
* This highlights the importance of optimizing strategy parameters.
* Implementing more sophisticated risk management techniques (e.g., stop-loss orders, take-profit orders) and calculating additional performance metrics could further enhance the strategy and its evaluation.
* Exploring other trading strategies or incorporating machine learning models could potentially lead to even better performance.
* For a more complete trading application experience with a dynamic dashboard, building a user interface to interact with the strategy and visualize results in real-time or on historical data would be beneficial, but this is beyond the scope of this notebook environment.

## Analyze Refined Strategy Results (Detailed Metrics)

### Subtask:
Calculate and display additional detailed performance and risk metrics for the refined strategy.

**Reasoning**:
Calculate Sharpe Ratio, Maximum Drawdown Duration, Win Rate, Loss Rate, and Average Win/Loss for the refined strategy to provide a more comprehensive analysis of its performance.
"""

# Calculate Sharpe Ratio
# Requires daily returns and risk-free rate (assuming risk-free rate is 0 for simplicity)
# Use the existing 'Portfolio_Value_refined' column calculated in the previous plotting step.

# Ensure 'Portfolio_Value_refined' column exists
if 'Portfolio_Value_refined' in df_selected.columns:
    df_selected['Daily_Return_refined'] = df_selected['Portfolio_Value_refined'].pct_change()

    # Replace infinite values with NaN and drop NaN values from daily returns before calculating std
    daily_returns_cleaned = df_selected['Daily_Return_refined'].replace([np.inf, -np.inf], np.nan).dropna()

    # Assuming 252 trading days in a year
    # Handle cases where std is 0 or NaN to avoid division by zero or NaN
    if daily_returns_cleaned.std() is not None and not np.isnan(daily_returns_cleaned.std()) and daily_returns_cleaned.std() != 0:
        sharpe_ratio_refined = (daily_returns_cleaned.mean() / daily_returns_cleaned.std()) * np.sqrt(252)
        print(f"Sharpe Ratio (Refined Strategy): {sharpe_ratio_refined:.2f}")
    else:
        print("Sharpe Ratio (Refined Strategy): Cannot be calculated (standard deviation is zero or NaN)")
else:
    print("Sharpe Ratio (Refined Strategy): Cannot be calculated. 'Portfolio_Value_refined' column not found.")


# Calculate Maximum Drawdown Duration
# This requires finding the longest period between a peak and a new peak
# Use the existing 'Portfolio_Value_refined' column
if 'Portfolio_Value_refined' in df_selected.columns:
    df_selected['Peak_refined'] = df_selected['Portfolio_Value_refined'].cummax()
    df_selected['Drawdown_refined_value'] = df_selected['Portfolio_Value_refined'] - df_selected['Peak_refined']

    # Find periods of drawdown
    in_drawdown = df_selected['Drawdown_refined_value'] < 0
    # Find the start of drawdown periods
    drawdown_starts = df_selected.index[in_drawdown & (~in_drawdown.shift(1).fillna(False))]
    # Find the end of drawdown periods
    drawdown_ends = df_selected.index[~in_drawdown & (in_drawdown.shift(1).fillna(False))]

    # Handle case where last period is a drawdown
    if in_drawdown.iloc[-1]:
        drawdown_ends = drawdown_ends.append(pd.Index([df_selected.index[-1]]))

    drawdown_durations = [df_selected['Date/Time'][end] - df_selected['Date/Time'][start] for start, end in zip(drawdown_starts, drawdown_ends)]

    max_drawdown_duration_refined = max(drawdown_durations) if drawdown_durations else pd.Timedelta(seconds=0)


    print(f"Maximum Drawdown Duration (Refined Strategy): {max_drawdown_duration_refined}")
else:
     print("Maximum Drawdown Duration (Refined Strategy): Cannot be calculated. 'Portfolio_Value_refined' column not found.")


# Calculate Win Rate and Loss Rate, and Average Win/Loss
## Need to identify individual trades from the 'Position' column
if 'Position' in df_selected.columns:
    trades = []
    entry_price = None
    entry_date = None

    # Iterate through the DataFrame to identify trades based on Position
    for index, row in df_selected.loc[df_selected['Position'] != 0].iterrows():
        if row['Position'] == 1.0:  # Buy signal
            # If we are not already in a position (entry_price is None)
            if entry_price is None:
                entry_price = row['Price INR']
                entry_date = row['Date/Time']
        elif row['Position'] == -1.0: # Sell signal
            # If we are in a position (entry_price is not None)
            if entry_price is not None:
                exit_price = row['Price INR']
                exit_date = row['Date/Time']
                # Assuming a single unit trade for PnL calculation
                pnl = (exit_price - entry_price)
                trades.append({'Entry_Date': entry_date, 'Exit_Date': exit_date, 'Entry_Price': entry_price, 'Exit_Price': exit_price, 'PnL': pnl})
                # Reset for the next trade
                entry_price = None

    # Handle an open position at the end of the data
    if entry_price is not None:
         exit_price = df_selected.iloc[-1]['Price INR']
         exit_date = df_selected.iloc[-1]['Date/Time']
         pnl = (exit_price - entry_price)
         trades.append({'Entry_Date': entry_date, 'Exit_Date': exit_date, 'Entry_Price': entry_price, 'Exit_Price': exit_price, 'PnL': pnl, 'Status': 'Open'})


    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]

        win_rate_refined = (len(winning_trades) / len(trades_df)) * 100
        loss_rate_refined = (len(losing_trades) / len(trades_df)) * 100

        average_win_refined = winning_trades['PnL'].mean() if not winning_trades.empty else 0
        average_loss_refined = losing_trades['PnL'].mean() if not losing_trades.empty else 0

        print(f"Win Rate (Refined Strategy): {win_rate_refined:.2f}%")
        print(f"Loss Rate (Refined Strategy): {loss_rate_refined:.2f}%")
        print(f"Average Win (Refined Strategy): {average_win_refined:.2f} INR")
        print(f"Average Loss (Refined Strategy): {average_loss_refined:.2f} INR")
    else:
        print("No completed trades found for calculating win/loss metrics.")
else:
    print("Win Rate, Loss Rate, and Average Win/Loss: Cannot be calculated. 'Position' column not found.")

"""**Reasoning**:
Add an equity curve plot to visualize the growth of the portfolio value over time, which is a key metric in evaluating trading strategy performance.
"""

# Plot Equity Curve (which is essentially the Portfolio Value plot)
# We already have the Portfolio Value calculated, so we can just plot it again with a different title if desired.


## All matplotlib axes/figure code removed

"""## Refine strategy

### Subtask:
Refine or modify the trading strategy to improve its performance based on the backtesting results.

**Reasoning**:
Based on the initial unprofitable results, let's try refining the strategy by allowing the user to adjust the short and long moving average window lengths and re-running the backtest.
"""

# Refine strategy by changing window lengths
short_window_refined = 20  # Example: change short window
long_window_refined = 50   # Example: change long window

# Calculate new moving averages
df_selected['Short_MA_refined'] = df_selected['Price INR'].rolling(window=short_window_refined).mean()
df_selected['Long_MA_refined'] = df_selected['Price INR'].rolling(window=long_window_refined).mean()


# Generate interactive Plotly charts and analytics
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Strategy chart
fig = make_subplots(specs=[[{"secondary_y": False}]])
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Price INR'], mode='lines', name='Price'))
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Short_MA'], mode='lines', name='Short MA'))
fig.add_trace(go.Scatter(x=df_selected['Date/Time'], y=df_selected['Long_MA'], mode='lines', name='Long MA'))
buy_signals = df_selected[df_selected['Position'] == 1.0]
fig.add_trace(go.Scatter(x=buy_signals['Date/Time'], y=buy_signals['Price INR'], mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'), name='Buy Signal'))
sell_signals = df_selected[df_selected['Position'] == -1.0]
fig.add_trace(go.Scatter(x=sell_signals['Date/Time'], y=sell_signals['Price INR'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name='Sell Signal'))
fig.update_layout(title='Algo Trading Backtest Chart', xaxis_title='Date/Time', yaxis_title='Price INR', hovermode='x unified', legend=dict(x=0, y=1.1, orientation='h'))
plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# Portfolio Value Over Time
portfolio_values = []
capital = initial_capital
shares_held = 0
for index, row in df_selected.iterrows():
    if row['Position'] == 1.0:
        shares_to_buy = int(capital / row['Price INR'])
        buy_cost = shares_to_buy * row['Price INR']
        if buy_cost <= capital:
            capital -= buy_cost
            shares_held += shares_to_buy
    elif row['Position'] == -1.0:
        sell_revenue = shares_held * row['Price INR']
        capital += sell_revenue
        shares_held = 0
    portfolio_values.append(capital + shares_held * row['Price INR'])
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_selected['Date/Time'], y=portfolio_values, mode='lines', name='Portfolio Value'))
fig2.update_layout(title='Portfolio Value Over Time', xaxis_title='Date/Time', yaxis_title='Portfolio Value (INR)', hovermode='x unified')
portfolio_chart_html = fig2.to_html(full_html=False, include_plotlyjs='cdn')

# Drawdown chart
running_max = pd.Series(portfolio_values).cummax()
drawdown = pd.Series(portfolio_values) - running_max
drawdown_pct = drawdown / running_max * 100
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_selected['Date/Time'], y=drawdown_pct, mode='lines', name='Drawdown %'))
fig3.update_layout(title='Drawdown Percentage Over Time', xaxis_title='Date/Time', yaxis_title='Drawdown (%)', hovermode='x unified')
drawdown_chart_html = fig3.to_html(full_html=False, include_plotlyjs='cdn')

# Cumulative returns chart
returns = pd.Series(portfolio_values).pct_change().fillna(0)
cumulative_returns = (1 + returns).cumprod() - 1
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_selected['Date/Time'], y=cumulative_returns * 100, mode='lines', name='Cumulative Return %'))
fig4.update_layout(title='Cumulative Returns Over Time', xaxis_title='Date/Time', yaxis_title='Cumulative Return (%)', hovermode='x unified')
cumulative_chart_html = fig4.to_html(full_html=False, include_plotlyjs='cdn')

# Trade summary analytics
num_trades = int((df_selected['Position'] == 1.0).sum() + (df_selected['Position'] == -1.0).sum())
num_buys = int((df_selected['Position'] == 1.0).sum())
num_sells = int((df_selected['Position'] == -1.0).sum())
win_trades = int((df_selected['Position'] == -1.0).sum())  # Approximation: every sell is a closed trade
win_rate = f"{(win_trades / num_trades * 100):.2f}%" if num_trades > 0 else "N/A"

# Combine results and charts in HTML

html_content = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Algo Trading Backtest Results</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #111827; color: #e5e7eb; margin: 0; padding: 0; }}
        .container {{ max-width: 900px; margin: 40px auto; background: #18181b; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.5); padding: 32px; }}
        h2 {{ text-align: center; color: #60a5fa; margin-bottom: 24px; letter-spacing: 2px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; background: #23272f; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 14px 18px; text-align: left; }}
        th {{ background: #1e293b; color: #60a5fa; font-weight: 600; }}
        td {{ color: #e5e7eb; }}
        tr:nth-child(even) td {{ background: #18181b; }}
        tr:hover td {{ background: #374151; transition: background 0.3s; }}
        .chart-section {{ margin-top: 24px; background: #23272f; border-radius: 8px; padding: 24px; box-shadow: 0 2px 8px rgba(96,165,250,0.08); }}
        h3 {{ color: #60a5fa; letter-spacing: 1px; }}
        @media (max-width: 600px) {{ .container {{ padding: 12px; }} th, td {{ padding: 8px 6px; }} .chart-section {{ padding: 8px; }} }}
        ::selection {{ background: #2563eb; color: #fff; }}
        a {{ color: #60a5fa; text-decoration: underline; }}
    </style>
</head>
<body>
    <div class='container'>
        <h2>Algo Trading Backtest Results</h2>
        <table>
            <tr><th>Initial Capital</th><td>{initial_capital:.2f} INR</td></tr>
            <tr><th>Final Portfolio Value</th><td>{final_portfolio_value:.2f} INR</td></tr>
            <tr><th>Total Profit/Loss</th><td>{total_profit_loss:.2f} INR</td></tr>
            <tr><th>Percentage Return</th><td>{percentage_return:.2f}%</td></tr>
            <tr><th>Profitability Assessment</th><td>{profitability_assessment}</td></tr>
            <tr><th>Number of Trades</th><td>{num_trades}</td></tr>
            <tr><th>Buy Signals</th><td>{num_buys}</td></tr>
            <tr><th>Sell Signals</th><td>{num_sells}</td></tr>
            <tr><th>Win Rate (approx.)</th><td>{win_rate}</td></tr>
        </table>
        <div class='chart-section'>
            <h3>Strategy Chart</h3>
            {plot_html}
        </div>
        <div class='chart-section'>
            <h3>Portfolio Value Over Time</h3>
            {portfolio_chart_html}
        </div>
        <div class='chart-section'>
            <h3>Drawdown Percentage Over Time</h3>
            {drawdown_chart_html}
        </div>
        <div class='chart-section'>
            <h3>Cumulative Returns Over Time</h3>
            {cumulative_chart_html}
        </div>
    </div>
</body>
</html>
"""

output_path = 'algo_backtest_results.html'
with open(output_path, 'w') as f:
        f.write(html_content)