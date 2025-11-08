from flask import Flask, render_template, render_template_string, request, send_from_directory, jsonify
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

# Create the static folder if it doesn't exist
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configure Flask app
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Set data file path
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'Backtesting Report.xlsx')

def load_data():
    """Load and preprocess trading data from Excel file."""
    try:
        # Create sample data if file doesn't exist
        if not os.path.exists(DATA_PATH):
            sample_data = pd.DataFrame({
                'Date/Time': pd.date_range(start='2023-01-01', periods=100, freq='H'),
                'Price INR': np.random.normal(loc=100, scale=10, size=100),
                'Volume': np.random.randint(1000, 10000, size=100)
            })
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            sample_data.to_excel(DATA_PATH, index=False)
            app.logger.info(f"Created sample data file at {DATA_PATH}")
            return sample_data.copy()
            
        # Configure pandas settings
        pd.set_option('mode.copy_on_write', True)
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('future.no_silent_downcasting', True)
            
        # Read the Excel file
        df = pd.read_excel(DATA_PATH).copy()  # Create copy to avoid chained assignment warnings
        
        # Validate required columns
        required_cols = ['Date/Time', 'Price INR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date/Time']):
            df.loc[:, 'Date/Time'] = pd.to_datetime(df['Date/Time'])
            
        # Sort by date
        df.sort_values('Date/Time', inplace=True)
        
        # Add Volume if not present
        if 'Volume' not in df.columns:
            random_volumes = np.random.randint(100, 1000, size=len(df))
            df.loc[:, 'Volume'] = df['Price INR'].values * random_volumes
        
        # Handle missing values with modern methods
        price_temp = df['Price INR'].copy()
        volume_temp = df['Volume'].copy()
        
        price_filled = price_temp.fillna(method='ffill').fillna(method='bfill')
        volume_filled = volume_temp.fillna(0)
        
        df.loc[:, 'Price INR'] = price_filled
        df.loc[:, 'Volume'] = volume_filled
        
        return df
        
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        # Return sample data if there's an error
        sample_data = pd.DataFrame({
            'Date/Time': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'Price INR': np.random.normal(loc=100, scale=10, size=100),
            'Volume': np.random.randint(1000, 10000, size=100)
        })
        return sample_data.copy()

# Enhanced chart configuration
def create_enhanced_chart_layout(title='', height=400):  # Reduced height from 800 to 400
    return {
        'title': {
            'text': title,
            'font': {'size': 18, 'color': '#1e293b', 'family': 'Inter, system-ui, sans-serif'},
            'y': 0.95
        },
        'height': height,
        'template': 'plotly_white',
        'paper_bgcolor': 'rgba(248, 250, 252, 0.95)',
        'plot_bgcolor': 'rgba(248, 250, 252, 0.5)',
        'font': {'family': 'Inter, system-ui, sans-serif', 'color': '#1e293b'},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#e2e8f0',
            'borderwidth': 1,
            'font': {'size': 10}
        },
        'xaxis': {
            'showgrid': False,
            'showline': False,
            'linecolor': 'rgba(0,0,0,0)',
            'showspikes': False,
            'rangeslider': {'visible': False},
            'title': {'font': {'size': 11}},
            'rangeselector': None,
            'rangeslider': None
        },
        'yaxis': {
            'showgrid': False,  # Removed grid
            'showline': False,  # Removed axis line
            'linecolor': 'rgba(0,0,0,0)',  # Made axis line transparent
            'showspikes': False,  # Removed spikes
            'title': {'font': {'size': 11}}  # Reduced font size
        },
        'hovermode': 'x unified',
        'hoverdistance': 100,
        'spikedistance': 1000,
        'hoverlabel': {
            'bgcolor': '#1e293b',
            'font': {'color': '#ffffff', 'size': 12},
            'bordercolor': '#1e293b'
        },
        'modebar': {
            'bgcolor': 'rgba(255, 255, 255, 0.95)',
            'color': '#475569',
            'activecolor': '#2563eb',
            'remove': ['toImage', 'zoom2d', 'pan2d', 'lasso2d', 'resetScale2d', 
                      'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian']  # Remove unnecessary buttons
        },
        'margin': {'t': 100, 'b': 50, 'l': 50, 'r': 50}
    }

def get_chart_config():
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawopenpath',
            'drawclosedpath',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': [
            'lasso2d',
            'select2d'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart_export',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }

# Enhanced chart configuration with detailed tracking
def create_detailed_chart_config():
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'scrollZoom': True,
        'modeBarButtonsToAdd': [],  # Remove additional buttons
        'modeBarButtonsToRemove': [
            'lasso2d', 
            'select2d',
            'zoom2d',
            'pan2d',
            'resetScale2d',
            'toggleSpikelines',
            'hoverClosestCartesian',
            'hoverCompareCartesian',
            'toImage'
        ],
        'showLink': False,  # Remove link to plotly
        'responsive': True
    }

def create_detailed_chart_layout(title='', height=800):
    return {
        'title': {
            'text': title,
            'font': {'size': 24, 'color': '#1e293b', 'family': 'Inter, system-ui, sans-serif'},
            'y': 0.98
        },
        'height': height,
        'template': 'plotly_white',
        'paper_bgcolor': 'rgba(248, 250, 252, 0.95)',
        'plot_bgcolor': 'rgba(248, 250, 252, 0.5)',
        'font': {'family': 'Inter, system-ui, sans-serif', 'color': '#1e293b'},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#e2e8f0',
            'borderwidth': 1,
            'font': {'size': 11}
        },
        'xaxis': {
            'showgrid': True,
            'gridcolor': '#e2e8f0',
            'gridwidth': 1,
            'linecolor': '#cbd5e1',
            'showspikes': True,
            'spikemode': 'across+marker',
            'spikethickness': 1,
            'spikecolor': '#94a3b8',
            'spikesnap': 'cursor',
            'showline': True,
            'showticklabels': True,
            'tickformat': '%Y-%m-%d %H:%M',
            'rangeslider': {'visible': False},  # Remove range slider
            'rangeselector': {'visible': False}  # Remove time range selector
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': '#e2e8f0',
            'gridwidth': 1,
            'linecolor': '#cbd5e1',
            'showspikes': True,
            'spikemode': 'across+marker',
            'spikethickness': 1,
            'spikecolor': '#94a3b8',
            'showline': True,
            'zeroline': True,
            'zerolinecolor': '#cbd5e1',
            'tickprefix': '₹ ',
            'tickformat': ',.2f',
            'tickmode': 'auto',
            'nticks': 15
        },
        'hovermode': 'x unified',
        'hoverdistance': 50,
        'spikedistance': 1000,
        'hoverlabel': {
            'bgcolor': '#1e293b',
            'font': {'color': '#ffffff', 'size': 12},
            'bordercolor': '#1e293b'
        },
        'annotations': [],
        'shapes': []
    }

def format_hover_data(x, y, additional_data=None):
    hover_text = f"""
    <b>Date:</b> {x}<br>
    <b>Price:</b> ₹{y:,.2f}<br>
    """
    if additional_data:
        for key, value in additional_data.items():
            if isinstance(value, float):
                hover_text += f"<b>{key}:</b> {value:.2f}<br>"
            else:
                hover_text += f"<b>{key}:</b> {value}<br>"
    return hover_text

# Generate chart and results
def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss):
    """Calculate optimal position size based on risk"""
    # Ensure we have valid inputs
    if not all(isinstance(x, (int, float)) for x in [capital, risk_per_trade, entry_price, stop_loss]):
        return 1  # Default to 1 unit if invalid inputs
        
    # Calculate risk amount (minimum 0.5% of capital)
    risk_per_trade = max(risk_per_trade, 0.5)
    risk_amount = capital * (risk_per_trade / 100)
    
    # Calculate price risk (minimum 0.5% of entry price)
    price_risk = max(abs(entry_price - stop_loss), entry_price * 0.005)
    
    # Calculate position size with minimum and maximum constraints
    position_size = risk_amount / price_risk if price_risk > 0 else 1
    
    # Apply position size limits
    max_position = capital / entry_price  # Cannot exceed available capital
    min_position = 1  # Minimum 1 unit
    
    return max(min(int(position_size), int(max_position)), min_position)

def calculate_stop_loss(df_selected, atr_periods=14, atr_multiplier=2):
    """Calculate optimal stop loss using ATR and volatility-based adjustments"""
    if len(df_selected) < atr_periods + 1:
        current_price = df_selected['Price INR'].iloc[-1]
        return {
            'Stop_Loss_Long': current_price * 0.98,  # Default 2% below current price
            'Stop_Loss_Short': current_price * 1.02   # Default 2% above current price
        }

    # Create a copy to avoid SettingWithCopy warning
    df = df_selected.copy()
    
    # Calculate ATR components
    price = df['Price INR']
    
    hl = price.diff().abs()
    hc = (price - price.shift()).abs()
    lc = (price.shift() - price.shift()).abs()
    
    # Stack components for max calculation
    components = pd.concat([hl, hc, lc], axis=1)
    tr = components.max(axis=1)
    tr.fillna(0, inplace=True)
    
    # Calculate ATR with Exponential Moving Average
    atr = tr.ewm(span=atr_periods, adjust=False).mean()
    atr = atr.fillna(method='bfill')
    
    # Calculate historical volatility
    returns = price.pct_change()
    returns = returns.fillna(0)
    volatility = returns.rolling(window=min(20, len(df))).std().fillna(0.02) * np.sqrt(252)
    
    # Calculate volatility scalar
    vol_last = volatility.iloc[-1]
    vol_mean = volatility.mean()
    volatility_scalar = vol_last / vol_mean if vol_last > 0 and vol_mean > 0 else 1.0

    # Adjust ATR multiplier based on volatility
    dynamic_multiplier = atr_multiplier * np.clip(volatility_scalar, 0.8, 1.2)
    
    # Get current price and ATR
    current_price = price.iloc[-1]
    current_atr = atr.iloc[-1]
    
    # Handle zero/NaN ATR
    if pd.isna(current_atr) or current_atr == 0:
        current_atr = current_price * 0.02
    
    # Calculate stop losses
    stop_loss_long = current_price - (current_atr * dynamic_multiplier)
    stop_loss_short = current_price + (current_atr * dynamic_multiplier)
    
    # Apply min/max constraints
    stop_loss_long = max(stop_loss_long, current_price * 0.9)
    stop_loss_short = min(stop_loss_short, current_price * 1.1)
    
    return {
        'Stop_Loss_Long': stop_loss_long,  # No more than 10% below price
        'Stop_Loss_Short': stop_loss_short  # No more than 10% above price
    }
    

def calculate_portfolio_risk(df_selected, initial_capital):
    """Calculate portfolio risk metrics"""
    # Create a copy to avoid SettingWithCopy warning
    df = df_selected.copy()
    
    # Calculate returns safely
    price = df['Price INR']
    returns = price.pct_change()
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {
            'Value_at_Risk': 0.0,
            'Expected_Shortfall': 0.0,
            'Risk_Reward_Ratio': 0.0,
            'Recommended_Risk_Per_Trade': 1.0
        }
    
    # Calculate Value at Risk (95% confidence)
    var_95 = abs(np.percentile(returns, 5)) * initial_capital
    
    # Calculate Expected Shortfall (CVaR)
    threshold = np.percentile(returns, 5)
    negative_returns = returns[returns <= threshold]
    cvar_95 = (abs(negative_returns.mean()) * initial_capital 
               if len(negative_returns) > 0 
               else var_95)
    
    # Risk per trade analysis
    non_zero_returns = returns[returns.abs() > 1e-10]  # Better than returns != 0
    
    if len(non_zero_returns) > 0:
        avg_trade_return = non_zero_returns.mean()
        trade_volatility = non_zero_returns.std()
        
        # Calculate risk/reward ratio safely
        risk_reward_ratio = (abs(avg_trade_return / trade_volatility) 
                           if trade_volatility > 1e-10 
                           else 1.0)
    else:
        risk_reward_ratio = 1.0
        trade_volatility = 0.01
    
    # Ensure we don't return NaN or zero values
    min_var = 0.01 * initial_capital  # Minimum VaR is 1% of capital
    min_es = 0.015 * initial_capital  # Minimum ES is 1.5% of capital
    min_risk_reward = 1.0  # Minimum risk/reward ratio
    min_risk_per_trade = 0.5  # Minimum risk per trade percentage
    max_risk_per_trade = 2.0  # Maximum risk per trade percentage
    
    return {
        'Value_at_Risk': max(var_95, min_var),
        'Expected_Shortfall': max(cvar_95, min_es),
        'Risk_Reward_Ratio': max(risk_reward_ratio, min_risk_reward),
        'Recommended_Risk_Per_Trade': np.clip(100 * trade_volatility, min_risk_per_trade, max_risk_per_trade)
    }

def calculate_metrics(df_selected, initial_capital=100000):
    """Calculate trading performance metrics with error handling"""
    # Default metrics in case of errors
    default_metrics = {
        'Sharpe_Ratio': 0.0,
        'Sortino_Ratio': 0.0,
        'Max_Drawdown': 0.0,
        'CAGR': 0.0,
        'Recommended_Position_Size': round(initial_capital * 0.02),
        'Stop_Loss_Long': 0.0,
        'Stop_Loss_Short': 0.0,
        'Value_at_Risk': initial_capital * 0.01,
        'Expected_Shortfall': initial_capital * 0.015,
        'Risk_Reward_Ratio': 1.0,
        'Recommended_Risk_Per_Trade': 1.0
    }
    
    try:
        # Convert initial_capital to float if it's not already
        initial_capital = float(initial_capital)
    except (TypeError, ValueError):
        initial_capital = 100000

    # Input validation
    if df_selected is None or len(df_selected) < 2:
        return default_metrics

    try:
        # Create a copy to avoid SettingWithCopy warning
        df = df_selected.copy()
        
        # Calculate price returns safely
        price = df['Price INR']
        returns = price.pct_change()
        returns_clean = returns.fillna(0)  # Handle NaN returns
        
        # Calculate portfolio values
        portfolio_values = initial_capital * (1 + returns_clean).cumprod()
        
        # Constants
        risk_free_rate = 0.05  # 5% annual
        trading_days = 252
        rf_daily = (1 + risk_free_rate) ** (1/trading_days) - 1
        
        # Calculate excess returns
        excess_returns = returns_clean - rf_daily
        
        # Annualized metrics
        returns_annualized = returns_clean.mean() * trading_days
        volatility_annualized = returns_clean.std() * np.sqrt(trading_days)
        
        # Sharpe Ratio calculation with safety checks
        sharpe_ratio = ((returns_annualized - risk_free_rate) / volatility_annualized 
                       if volatility_annualized > 1e-10 
                       else 0.0)
        
        # Sortino Ratio calculation with safety checks
        negative_returns = returns_clean[returns_clean < 0]
        downside_std = (negative_returns.std() * np.sqrt(trading_days) 
                       if len(negative_returns) > 0 
                       else float('inf'))
        sortino_ratio = ((returns_annualized - risk_free_rate) / downside_std 
                        if downside_std != float('inf') 
                        else 0.0)
        
        # Maximum Drawdown calculation using expanding window
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min() * 100)
        
        # CAGR calculation with safety checks
        start_date = df['Date/Time'].iloc[0]
        end_date = df['Date/Time'].iloc[-1]
        total_days = (end_date - start_date).days
        total_years = max(total_days / 365.0, 1/365.0)  # Ensure non-zero denominator
        final_value = portfolio_values.iloc[-1]
        
        if final_value > 0 and initial_capital > 0:
            cagr = ((final_value / initial_capital) ** (1/total_years) - 1) * 100
        else:
            cagr = 0.0
        
        # Calculate risk management suggestions
        stop_losses = calculate_stop_loss(df)
        risk_metrics = calculate_portfolio_risk(df, initial_capital)
        
        # Position sizing based on volatility with constraints
        volatility = returns_clean.std()
        max_pos_pct = 0.02  # Maximum 2% of capital
        min_pos_pct = 0.01  # Minimum 1% of capital
        position_pct = np.clip(max_pos_pct * (1 - volatility/0.1), min_pos_pct, max_pos_pct)
        position_size = round(initial_capital * position_pct)
        
        # Construct final metrics dictionary with type safety
        final_metrics = {
            'Sharpe_Ratio': float(np.clip(sharpe_ratio, -100, 100)),  # Limit extreme values
            'Sortino_Ratio': float(np.clip(sortino_ratio, -100, 100)),  # Limit extreme values
            'Max_Drawdown': float(np.clip(max_drawdown, 0, 100)),  # Must be between 0-100%
            'CAGR': float(np.clip(cagr, -100, 1000)),  # Reasonable CAGR range
            'Recommended_Position_Size': int(position_size),  # Must be integer
            'Stop_Loss_Long': float(stop_losses.get('Stop_Loss_Long', price.iloc[-1] * 0.98)),
            'Stop_Loss_Short': float(stop_losses.get('Stop_Loss_Short', price.iloc[-1] * 1.02)),
            'Value_at_Risk': float(risk_metrics.get('Value_at_Risk', initial_capital * 0.01)),
            'Expected_Shortfall': float(risk_metrics.get('Expected_Shortfall', initial_capital * 0.015)),
            'Risk_Reward_Ratio': float(risk_metrics.get('Risk_Reward_Ratio', 1.0)),
            'Recommended_Risk_Per_Trade': float(risk_metrics.get('Recommended_Risk_Per_Trade', 1.0))
        }
        
        return final_metrics
        
    except Exception as e:
        app.logger.error(f"Error in calculate_metrics: {str(e)}")
        return default_metrics

def generate_analysis(df_selected, initial_capital=100000, short_window=20, long_window=50):
    """Generate comprehensive technical analysis with detailed breakdowns"""
    df = df_selected.copy()
    
    # Calculate Simple Moving Averages - smooth them for cleaner lines
    df['Short_MA'] = df['Price INR'].rolling(window=short_window, min_periods=1).mean()
    df['Short_MA'] = df['Short_MA'].fillna(method='ffill').fillna(method='bfill')
    df['Long_MA'] = df['Price INR'].rolling(window=long_window, min_periods=1).mean()
    df['Long_MA'] = df['Long_MA'].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate Exponential Moving Average with more smoothing
    df['EMA_20'] = df['Price INR'].ewm(span=20, adjust=False, min_periods=1).mean()
    
    # Calculate Bollinger Bands with smoother transitions
    df['BB_middle'] = df['Price INR'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['Price INR'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + 1.5 * rolling_std  # Reduced multiplier from 2 to 1.5
    df['BB_lower'] = df['BB_middle'] - 1.5 * rolling_std
    
    # Calculate RSI (14-day)
    delta = df['Price INR'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Price INR'].ewm(span=12, adjust=False).mean()
    exp2 = df['Price INR'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Add volume (dummy data if not present)
    if 'Volume' not in df.columns:
        df['Volume'] = df['Price INR'] * np.random.randint(100, 1000, size=len(df))
    
    # Initialize the position column
    df['Position'] = 0.0
    
    # Generate trading signals based on multiple indicators
    df['Signal'] = 0.0
    
    # Conditions for buy signals (1.0)
    buy_conditions = (
        (df['Short_MA'] > df['Long_MA']) &  # Moving Average Crossover
        (df['RSI'] < 70) &  # RSI not overbought
        (df['MACD'] > df['Signal_Line'])  # MACD above signal line
    )
    
    # Conditions for sell signals (-1.0)
    sell_conditions = (
        (df['Short_MA'] < df['Long_MA']) &  # Moving Average Crossover
        (df['RSI'] > 30) &  # RSI not oversold
        (df['MACD'] < df['Signal_Line'])  # MACD below signal line
    )
    
    df.loc[buy_conditions, 'Signal'] = 1.0
    df.loc[sell_conditions, 'Signal'] = -1.0
    
    # Trading orders
    df['Position'] = df['Signal'].diff()
    
    # Create subplots for price, indicators, and volume
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.2, 0.15, 0.15],
                       subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'))
    
    # Remove range selector buttons
    rangeselector_config = dict(
        visible=False  # This will hide the range selector buttons
    )

    # Enhanced main price chart with simplified indicators
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['Price INR'],
            name='Price',
            line=dict(
                color='#2563eb',
                width=1.5,
                shape='spline'  # Smooth line
            ),
            mode='lines'
        ), 
        row=1, col=1
    )
    
    # Enhanced Moving Averages with better visibility
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['Short_MA'],
            mode='lines',
            name=f'SMA {short_window}',
            line=dict(color='#3b82f6', width=1.5, dash='solid'),
            opacity=0.8
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['Long_MA'],
            mode='lines',
            name=f'SMA {long_window}',
            line=dict(color='#f59e0b', width=1.5, dash='solid'),
            opacity=0.8
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['EMA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='#8b5cf6', width=1.5, dash='dot'),
            opacity=0.8
        ), row=1, col=1
    )

    # Enhanced Bollinger Bands with fill
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(147, 197, 253, 0.8)', width=1, dash='dash'),
            fill=None,
            opacity=0.3
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(147, 197, 253, 0.8)', width=1, dash='dash'),
            fill='tonexty',  # Fill area between upper and lower bands
            fillcolor='rgba(147, 197, 253, 0.1)',
            opacity=0.3
        ), row=1, col=1
    )

    # Enhanced Buy/Sell signals with animations and better visibility
    buy_signals = df[df['Position'] == 1.0]
    fig.add_trace(
        go.Scatter(
            x=buy_signals['Date/Time'],
            y=buy_signals['Price INR'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='#10b981',
                size=12,
                symbol='triangle-up',
                line=dict(color='#064e3b', width=1),
                opacity=0.8
            ),
            hovertemplate='<b>Buy Signal</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: ₹%{y:.2f}<br>' +
                         '<extra></extra>'
        ), row=1, col=1
    )
    
    sell_signals = df[df['Position'] == -1.0]
    fig.add_trace(
        go.Scatter(
            x=sell_signals['Date/Time'],
            y=sell_signals['Price INR'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                color='#ef4444',
                size=12,
                symbol='triangle-down',
                line=dict(color='#7f1d1d', width=1),
                opacity=0.8
            ),
            hovertemplate='<b>Sell Signal</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: ₹%{y:.2f}<br>' +
                         '<extra></extra>'
        ), row=1, col=1
    )

    # Enhanced Volume chart with color coding based on price movement
    df['VolumeColor'] = ['rgba(239, 68, 68, 0.7)' if close < open else 'rgba(16, 185, 129, 0.7)' 
                         for close, open in zip(df['Price INR'].shift(-1), df['Price INR'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date/Time'],
            y=df['Volume'],
            name='Volume',
            marker_color=df['VolumeColor'],
            marker_line_color=df['VolumeColor'].map(
                lambda x: x.replace('0.7', '1')
            ),
            marker_line_width=1,
            opacity=0.8,
            hovertemplate='<b>Volume</b><br>' +
                         'Date: %{x}<br>' +
                         'Volume: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ), row=2, col=1
    )

    # Enhanced RSI with color zones and better visualization
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#f59e0b', width=1.5),
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.1)',
            hovertemplate='<b>RSI</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.1f}<br>' +
                         '<extra></extra>'
        ), row=3, col=1
    )
    
    # Add colored RSI zones
    fig.add_shape(
        type="rect",
        x0=df['Date/Time'].iloc[0],
        x1=df['Date/Time'].iloc[-1],
        y0=70,
        y1=100,
        fillcolor="rgba(239, 68, 68, 0.1)",
        line_width=0,
        row=3, col=1
    )
    fig.add_shape(
        type="rect",
        x0=df['Date/Time'].iloc[0],
        x1=df['Date/Time'].iloc[-1],
        y0=0,
        y1=30,
        fillcolor="rgba(16, 185, 129, 0.1)",
        line_width=0,
        row=3, col=1
    )
    
    # Add RSI levels with enhanced styling
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(239, 68, 68, 0.5)",
        annotation_text="Overbought",
        annotation_position="right",
        annotation=dict(font_size=10, font_color="rgba(239, 68, 68, 0.8)"),
        row=3, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="rgba(16, 185, 129, 0.5)",
        annotation_text="Oversold",
        annotation_position="right",
        annotation=dict(font_size=10, font_color="rgba(16, 185, 129, 0.8)"),
        row=3, col=1
    )

    # Enhanced MACD visualization
    # Calculate MACD histogram
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['MACD_Color'] = ['rgba(239, 68, 68, 0.7)' if hist < 0 else 'rgba(16, 185, 129, 0.7)' 
                        for hist in df['MACD_Hist']]
    
    # Add MACD histogram
    fig.add_trace(
        go.Bar(
            x=df['Date/Time'],
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker_color=df['MACD_Color'],
            marker_line_color=df['MACD_Color'].map(lambda x: x.replace('0.7', '1')),
            marker_line_width=1,
            opacity=0.8,
            hovertemplate='<b>MACD Histogram</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ), row=4, col=1
    )
    
    # Add MACD and Signal lines with enhanced styling
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#3b82f6', width=1.5),
            hovertemplate='<b>MACD</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ), row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date/Time'],
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='#f59e0b', width=1.5, dash='dot'),
            hovertemplate='<b>Signal Line</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ), row=4, col=1
    )

    # Update layout for better visualization
    fig.update_layout(
        title='',  # Removed the title
        xaxis_title='Date/Time',
        yaxis_title='Price INR',
        height=1000,  # Increased height for better visibility
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Price INR", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    # Calculate portfolio performance
    portfolio_values = []
    capital = initial_capital
    shares_held = 0
    
    for index, row in df.iterrows():
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
    
    final_portfolio_value = portfolio_values[-1]
    total_profit_loss = final_portfolio_value - initial_capital
    percentage_return = (total_profit_loss / initial_capital) * 100
    
    if percentage_return > 0:
        profitability_assessment = "The strategy was profitable"
    else:
        profitability_assessment = "The strategy was not profitable"
    
    # Portfolio value chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date/Time'], y=portfolio_values, mode='lines', name='Portfolio Value'))
    fig2.update_layout(title='Portfolio Value Over Time', xaxis_title='Date/Time', yaxis_title='Portfolio Value (INR)', hovermode='x unified')
    
    # Drawdown calculation
    running_max = pd.Series(portfolio_values).cummax()
    drawdown = pd.Series(portfolio_values) - running_max
    drawdown_pct = drawdown / running_max * 100
    
    # Drawdown chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['Date/Time'], y=drawdown_pct, mode='lines', name='Drawdown %'))
    fig3.update_layout(title='Drawdown Percentage Over Time', xaxis_title='Date/Time', yaxis_title='Drawdown (%)', hovermode='x unified')
    
    # Cumulative returns calculation and chart
    returns = pd.Series(portfolio_values).pct_change()
    returns = returns.replace(np.nan, 0)
    cumulative_returns = (1 + returns).cumprod() - 1
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['Date/Time'], y=cumulative_returns * 100, mode='lines', name='Cumulative Return %'))
    fig4.update_layout(title='Cumulative Returns Over Time', xaxis_title='Date/Time', yaxis_title='Cumulative Return (%)', hovermode='x unified')
    
    # Trade summary
    num_trades = int((df['Position'] == 1.0).sum() + (df['Position'] == -1.0).sum())
    num_buys = int((df['Position'] == 1.0).sum())
    num_sells = int((df['Position'] == -1.0).sum())
    win_trades = int((df['Position'] == -1.0).sum())  # Approximation: every sell is a closed trade
    win_rate = f"{(win_trades / num_trades * 100):.2f}%" if num_trades > 0 else "N/A"

    return {
        'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
        'portfolio_chart_html': fig2.to_html(full_html=False, include_plotlyjs='cdn'),
        'drawdown_chart_html': fig3.to_html(full_html=False, include_plotlyjs='cdn'),
        'cumulative_chart_html': fig4.to_html(full_html=False, include_plotlyjs='cdn'),
        'final_portfolio_value': final_portfolio_value,
        'total_profit_loss': total_profit_loss,
        'percentage_return': percentage_return,
        'profitability_assessment': profitability_assessment,
        'num_trades': num_trades,
        'num_buys': num_buys,
        'num_sells': num_sells,
        'win_rate': win_rate
    }

    return {
        'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
        'portfolio_chart_html': fig2.to_html(full_html=False, include_plotlyjs='cdn'),
        'drawdown_chart_html': fig3.to_html(full_html=False, include_plotlyjs='cdn'),
        'cumulative_chart_html': fig4.to_html(full_html=False, include_plotlyjs='cdn'),
        'final_portfolio_value': final_portfolio_value,
        'total_profit_loss': total_profit_loss,
        'percentage_return': percentage_return,
        'profitability_assessment': profitability_assessment,
        'num_trades': num_trades,
        'num_buys': num_buys,
        'num_sells': num_sells,
        'win_rate': win_rate
    }

# --- Navigation base template and helper ---
base_template = '''
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Algo Trading Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%); }
            .container-main { max-width: 1100px; margin-top: 24px; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">Algo Dashboard</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="#navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                        <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
                        <li class="nav-item"><a class="nav-link" href="/charts">Charts</a></li>
                        <li class="nav-item"><a class="nav-link" href="/metrics">Metrics</a></li>
                        <li class="nav-item"><a class="nav-link" href="/risk">Risk</a></li>
                        <li class="nav-item"><a class="nav-link" href="/about">About</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        <div class="container container-main">{{ content|safe }}</div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
</html>
'''

def render_with_nav(content_html: str):
        """Wrap content HTML into the base template"""
        return render_template_string(base_template, content=content_html)


# --- New page routes ---
@app.route('/charts')
def charts():
        try:
                df = load_data()
                # Calculate technical indicators
                # RSI
                delta = df['Price INR'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                df['BB_middle'] = df['Price INR'].rolling(window=20).mean()
                bb_std = df['Price INR'].rolling(window=20).std()
                df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
                df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
                
                # MACD
                exp1 = df['Price INR'].ewm(span=12, adjust=False).mean()
                exp2 = df['Price INR'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
                
                # Create separate charts for each indicator
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='#2563eb', width=1.5)
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                rsi_fig.update_layout(
                    title='Relative Strength Index (RSI)',
                    height=400,
                    yaxis_title='RSI Value'
                )
                
                bb_fig = go.Figure()
                bb_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['Price INR'],
                    name='Price',
                    line=dict(color='#2563eb')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['BB_upper'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['BB_middle'],
                    name='Middle Band',
                    line=dict(color='orange')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['BB_lower'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))
                bb_fig.update_layout(
                    title='Bollinger Bands',
                    height=400,
                    yaxis_title='Price'
                )
                
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color='#2563eb')
                ))
                macd_fig.add_trace(go.Scatter(
                    x=df['Date/Time'],
                    y=df['Signal_Line'],
                    name='Signal Line',
                    line=dict(color='orange')
                ))
                macd_fig.add_trace(go.Bar(
                    x=df['Date/Time'],
                    y=df['MACD_Hist'],
                    name='MACD Histogram',
                    marker_color=np.where(df['MACD_Hist'] >= 0, '#10b981', '#ef4444')
                ))
                macd_fig.update_layout(
                    title='MACD',
                    height=400,
                    yaxis_title='Value'
                )

                return render_template(
                    'charts.html',
                    rsi_chart=rsi_fig.to_html(full_html=False),
                    bb_chart=bb_fig.to_html(full_html=False),
                    macd_chart=macd_fig.to_html(full_html=False)
                )
        except Exception as e:
                return render_template('error.html', error=str(e))


@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    try:
        data = request.get_json()
        
        # Parse and validate initial capital with better error handling
        try:
            initial_capital = float(data.get('initial_capital', 100000))
            if initial_capital <= 0:
                return jsonify({'error': 'Initial capital must be greater than 0'}), 400
            if initial_capital > 1000000000:  # Add upper limit validation
                return jsonify({'error': 'Initial capital is too large'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Please enter a valid number for initial capital'}), 400
            
        time_period = data.get('time_period', 'ALL')
        
        # Load and validate data
        df = load_data()
        if df is None or len(df) == 0:
            return jsonify({'error': 'No trading data available'}), 400
            
        # Ensure proper data format
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['Date/Time']):
                df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            if not pd.api.types.is_numeric_dtype(df['Price INR']):
                return jsonify({'error': 'Invalid price data format'}), 400
                
            df = df.sort_values('Date/Time')
            df = df.dropna(subset=['Date/Time', 'Price INR'])  # Remove rows with NaN values
            
            if len(df) < 2:
                return jsonify({'error': 'Insufficient data points for analysis'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing data: {str(e)}'}), 400
        
        # Filter data based on time period
        try:
            if time_period != 'ALL':
                end_date = df['Date/Time'].max()
                if time_period == '1D':
                    start_date = end_date - pd.Timedelta(days=1)
                elif time_period == '1W':
                    start_date = end_date - pd.Timedelta(weeks=1)
                elif time_period == '1Y':
                    start_date = end_date - pd.Timedelta(days=365)
                df = df[df['Date/Time'] >= start_date]
                
                if len(df) == 0:
                    return jsonify({'error': f'No data available for the selected time period: {time_period}'}), 400
                if len(df) < 2:
                    return jsonify({'error': f'Insufficient data points for the selected time period: {time_period}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error filtering data: {str(e)}'}), 400
        
        try:
            # Calculate metrics
            metrics = calculate_metrics(df, initial_capital=initial_capital)
            baseline_metrics = calculate_metrics(df, initial_capital=100000)
            
            # Ensure all metric values are valid numbers
            for metric in ['Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown', 'CAGR']:
                if not isinstance(metrics.get(metric), (int, float)) or pd.isna(metrics.get(metric)):
                    metrics[metric] = 0.0
                if not isinstance(baseline_metrics.get(metric), (int, float)) or pd.isna(baseline_metrics.get(metric)):
                    baseline_metrics[metric] = 0.0
            
            # Create chart data
            chart_data = [{
                'type': 'bar',
                'name': f'Current (₹{initial_capital:,.0f})',
                'x': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'CAGR (%)'],
                'y': [
                    float(metrics['Sharpe_Ratio']),
                    float(metrics['Sortino_Ratio']),
                    float(metrics['Max_Drawdown']),
                    float(metrics['CAGR'])
                ],
                'text': [f'{metrics[k]:.2f}' for k in ['Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown', 'CAGR']],
                'textposition': 'auto',
                'marker': {
                    'color': '#2563eb',
                    'opacity': 0.8
                }
            }]
            
            # Enhanced layout
            layout = {
                'title': {
                    'text': f'Performance Metrics (Capital: ₹{initial_capital:,.0f})',
                    'font': {'size': 20, 'color': '#1e293b'}
                },
                'height': 400,
                'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50},
                'plot_bgcolor': 'rgba(248, 250, 252, 0.95)',
                'paper_bgcolor': 'rgba(248, 250, 252, 0.95)',
                'xaxis': {
                    'gridcolor': '#e2e8f0',
                    'showgrid': False,
                    'tickangle': -45
                },
                'yaxis': {
                    'gridcolor': '#e2e8f0',
                    'showgrid': True,
                    'zeroline': True,
                    'zerolinecolor': '#94a3b8',
                    'zerolinewidth': 1
                }
            }
            
            return jsonify({
                'metrics': metrics,
                'chart_data': chart_data,
                'layout': layout
            })
            
        except Exception as e:
            app.logger.error(f"Error calculating metrics: {str(e)}")
            return jsonify({'error': f'Error calculating metrics: {str(e)}'}), 400
            
    except Exception as e:
        app.logger.error(f"Error updating metrics: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 400

@app.route('/metrics', methods=['GET', 'POST'])
def metrics_page():
    """Display and calculate performance metrics."""
    try:
        # Get parameters from form or use defaults
        initial_capital = float(request.form.get('initial_capital', 100000))
        time_period = request.form.get('time_period', 'ALL')
        
        # Load and validate data
        df = load_data()
        if df is None or len(df) == 0:
            raise ValueError("No data available for analysis")
            
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date/Time']):
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            
        # Filter data based on time period
        if time_period != 'ALL':
            end_date = df['Date/Time'].max()
            if time_period == '1D':
                start_date = end_date - pd.Timedelta(days=1)
            elif time_period == '1W':
                start_date = end_date - pd.Timedelta(weeks=1)
            elif time_period == '1Y':
                start_date = end_date - pd.Timedelta(days=365)
            df = df[df['Date/Time'] >= start_date]
        
        # Calculate metrics
        metrics = calculate_metrics(df, initial_capital=initial_capital)
        
        # Create visualization data
        chart_data = {
            'metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'CAGR (%)'],
            'values': [
                metrics['Sharpe_Ratio'],
                metrics['Sortino_Ratio'],
                metrics['Max_Drawdown'],
                metrics['CAGR']
            ]
        }
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=chart_data['metrics'],
                y=chart_data['values'],
                marker_color=['#2563eb', '#2563eb', 
                            '#ef4444' if metrics['Max_Drawdown'] > 20 else '#2563eb',
                            '#10b981' if metrics['CAGR'] > 0 else '#ef4444']
            )
        ])
        
        fig.update_layout(
            title='Key Performance Metrics',
            yaxis_title='Value',
            template='plotly_white',
            height=400,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Create metrics table
        rows = ""
        rows += f"<tr><th>Sharpe Ratio</th><td>{metrics.get('Sharpe_Ratio', 'N/A'):.2f}</td></tr>"
        rows += f"<tr><th>Sortino Ratio</th><td>{metrics.get('Sortino_Ratio', 'N/A'):.2f}</td></tr>"
        rows += f"<tr><th>Max Drawdown</th><td>{metrics.get('Max_Drawdown', 'N/A'):.2f}%</td></tr>"
        rows += f"<tr><th>CAGR</th><td>{metrics.get('CAGR', 'N/A'):.2f}%</td></tr>"
        table = f"<h3>Performance Metrics</h3><table class='table table-striped'>{rows}</table>"
        
        return render_template(
            'metrics.html',
            metrics=metrics,
            table=table,
            chart_html=chart_html,
            initial_capital=initial_capital,
            time_period=time_period
        )
        
    except Exception as e:
        app.logger.error(f"Error in metrics page: {str(e)}")
        return render_template(
            'metrics.html',
            error=str(e),
            table="<div class='alert alert-danger'>Error calculating metrics</div>",
            chart_html="",
            initial_capital=100000,
            time_period='ALL'
        )


@app.route('/risk', methods=['GET', 'POST'])
def risk_page():
        try:
                df = load_data()
                # Get user inputs or use defaults
                initial_capital = float(request.form.get('initial_capital', 100000))
                risk_per_trade = float(request.form.get('risk_per_trade', 1.0))
                stop_loss_pct = float(request.form.get('stop_loss_pct', 2.0))
                var_confidence = float(request.form.get('var_confidence', 95.0))
                
                # Calculate custom stop losses based on user input
                current_price = df['Price INR'].iloc[-1]
                custom_stop_loss = current_price * (1 - stop_loss_pct/100)
                
                # Use existing helpers with custom inputs
                stop_losses = calculate_stop_loss(df)
                portfolio_risk = calculate_portfolio_risk(df, initial_capital)
                position_size = calculate_position_size(initial_capital, 
                                                      risk_per_trade, 
                                                      current_price, 
                                                      custom_stop_loss)
                
                # Create risk metrics table
                rows = ''
                rows += f"<tr><th>Recommended Position Size</th><td>{position_size}</td></tr>"
                rows += f"<tr><th>Stop Loss (Long)</th><td>{stop_losses.get('Stop_Loss_Long', 'N/A'):.2f} INR</td></tr>"
                rows += f"<tr><th>Stop Loss (Short)</th><td>{stop_losses.get('Stop_Loss_Short', 'N/A'):.2f} INR</td></tr>"
                rows += f"<tr><th>VaR (95%)</th><td>{portfolio_risk.get('Value_at_Risk', 'N/A'):.2f} INR</td></tr>"
                rows += f"<tr><th>Expected Shortfall</th><td>{portfolio_risk.get('Expected_Shortfall', 'N/A'):.2f} INR</td></tr>"
                table = f"<h3>Risk Management</h3><table class='table table-striped'>{rows}</table>"
                
                # Create pie chart of risk allocation
                risk_values = [
                    portfolio_risk.get('Value_at_Risk', 0),
                    portfolio_risk.get('Expected_Shortfall', 0),
                    position_size * df['Price INR'].iloc[-1],  # Total position value
                    initial_capital - (position_size * df['Price INR'].iloc[-1])  # Remaining capital
                ]
                
                risk_labels = [
                    'Value at Risk (95%)',
                    'Expected Shortfall',
                    'Allocated Capital',
                    'Available Capital'
                ]
                
                # Calculate risk values based on user inputs
                var_amount = portfolio_risk.get('Value_at_Risk', 0) * (var_confidence / 95.0)  # Adjust VaR for confidence level
                es_amount = portfolio_risk.get('Expected_Shortfall', 0) * (var_confidence / 95.0)
                position_value = position_size * current_price
                available_capital = initial_capital - position_value
                
                # Create pie chart with updated values
                fig = go.Figure(data=[go.Pie(
                    labels=risk_labels,
                    values=[var_amount, es_amount, position_value, available_capital],
                    hole=.3,
                    textinfo='label+percent',
                    marker=dict(colors=['#ef4444', '#f59e0b', '#2563eb', '#10b981']),
                    textfont=dict(color='#ffffff'),
                    hovertemplate="<b>%{label}</b><br>" +
                                "Amount: ₹%{value:,.2f}<br>" +
                                "Percentage: %{percent}<br>" +
                                "<extra></extra>"
                )])
                
                # Update layout with custom title including input parameters
                title_text = (f'Risk and Capital Allocation<br>' +
                            f'<span style="font-size: 14px">Capital: ₹{initial_capital:,.0f} | ' +
                            f'Risk/Trade: {risk_per_trade:.1f}% | ' +
                            f'Stop Loss: {stop_loss_pct:.1f}% | ' +
                            f'VaR Confidence: {var_confidence:.1f}%</span>')
                
                fig.update_layout(
                    title={
                        'text': title_text,
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20, color='#1e293b')
                    },
                    paper_bgcolor='rgba(248, 250, 252, 0.95)',
                    plot_bgcolor='rgba(248, 250, 252, 0.95)',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                
        except Exception as e:
                table = f"<div class='alert alert-danger'>Error preparing risk metrics: {e}</div>"
                chart_html = ""
                
        return render_template('risk.html', table=table, chart_html=chart_html)


@app.route('/about')
def about_page():
        content = '''
        <h3>About this Dashboard</h3>
        <p>This dashboard displays backtest charts, performance metrics and risk metrics for your strategy.</p>
        <ul>
            <li>Charts: interactive Plotly charts</li>
            <li>Metrics: Sharpe, Sortino, CAGR, drawdown</li>
            <li>Risk: ATR-based stops, VaR and position sizing</li>
        </ul>
        '''
        return render_template('about.html')

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algo Trading Dashboard</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%); margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 32px; }
    h2 { text-align: center; color: #3b82f6; margin-bottom: 24px; }
    form { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; margin-bottom: 32px; }
    label { font-weight: 500; color: #2563eb; }
    input { padding: 8px 12px; border-radius: 6px; border: 1px solid #e0e7ff; font-size: 1rem; }
    button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 10px 24px; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #1d4ed8; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 32px; background: #f1f5f9; border-radius: 8px; overflow: hidden; }
    th, td { padding: 14px 18px; text-align: left; }
    th { background: #e0e7ff; color: #2563eb; font-weight: 600; }
    tr:nth-child(even) td { background: #f8fafc; }
    tr:hover td { background: #dbeafe; transition: background 0.3s; }
    .chart-section { margin-top: 24px; background: #f1f5f9; border-radius: 8px; padding: 24px; box-shadow: 0 2px 8px rgba(59,130,246,0.08); }
    @media (max-width: 600px) { .container { padding: 12px; } th, td { padding: 8px 6px; } .chart-section { padding: 8px; } }
    </style>
</head>
<body>
    <div class="container">
        <h2>Algo Trading Dashboard</h2>
        <form method="post">
            <label>Initial Capital (INR): <input type="number" name="initial_capital" value="{{ initial_capital }}" min="1000" step="1000" required></label>
            <label>Short MA Window: <input type="number" name="short_window" value="{{ short_window }}" min="1" max="100" required></label>
            <label>Long MA Window: <input type="number" name="long_window" value="{{ long_window }}" min="1" max="300" required></label>
            <button type="submit" name="analyze">Analyze</button>
        </form>
        <table>
            <tr><th>Final Portfolio Value</th><td>{{ final_portfolio_value }} INR</td></tr>
            <tr><th>Total Profit/Loss</th><td>{{ total_profit_loss }} INR</td></tr>
            <tr><th>Percentage Return</th><td>{{ percentage_return }}%</td></tr>
            <tr><th>Profitability Assessment</th><td>{{ profitability_assessment }}</td></tr>
            <tr><th>Number of Trades</th><td>{{ num_trades }}</td></tr>
            <tr><th>Buy Signals</th><td>{{ num_buys }}</td></tr>
            <tr><th>Sell Signals</th><td>{{ num_sells }}</td></tr>
            <tr><th>Win Rate (approx.)</th><td>{{ win_rate }}</td></tr>
            <tr><th>Sharpe Ratio</th><td>{{ "%.2f"|format(sharpe_ratio) }}</td></tr>
            <tr><th>Sortino Ratio</th><td>{{ "%.2f"|format(sortino_ratio) }}</td></tr>
            <tr><th>Maximum Drawdown</th><td>{{ "%.2f"|format(max_drawdown) }}%</td></tr>
            <tr><th>CAGR</th><td>{{ "%.2f"|format(cagr) }}%</td></tr>
        </table>
        
        <h3 style="color:#2563eb; margin-top: 20px;">Risk Management Metrics</h3>
        <table>
            <tr><th>Recommended Position Size</th><td>{{ "%.0f"|format(metrics['Recommended_Position_Size']) }} units</td></tr>
            <tr><th>Suggested Stop Loss (Long)</th><td>{{ "%.2f"|format(metrics['Stop_Loss_Long']) }} INR</td></tr>
            <tr><th>Suggested Stop Loss (Short)</th><td>{{ "%.2f"|format(metrics['Stop_Loss_Short']) }} INR</td></tr>
            <tr><th>Value at Risk (95%)</th><td>{{ "%.2f"|format(metrics['Value_at_Risk']) }} INR</td></tr>
            <tr><th>Expected Shortfall</th><td>{{ "%.2f"|format(metrics['Expected_Shortfall']) }} INR</td></tr>
            <tr><th>Risk/Reward Ratio</th><td>{{ "%.2f"|format(metrics['Risk_Reward_Ratio']) }}</td></tr>
            <tr><th>Recommended Risk per Trade</th><td>{{ "%.2f"|format(metrics['Recommended_Risk_Per_Trade']) }}%</td></tr>
        </table>
        <div class="chart-section">
            <h3 style="color:#2563eb;">Strategy Chart</h3>
            {{ chart_html|safe }}
        </div>
        <div class="chart-section">
            <h3 style="color:#2563eb;">Portfolio Value Over Time</h3>
            {{ portfolio_chart_html|safe }}
        </div>
        <div class="chart-section">
            <h3 style="color:#2563eb;">Drawdown Percentage Over Time</h3>
            {{ drawdown_chart_html|safe }}
        </div>
        <div class="chart-section">
            <h3 style="color:#2563eb;">Cumulative Returns Over Time</h3>
            {{ cumulative_chart_html|safe }}
        </div>
    </div>
</body>
</html>
'''

@app.errorhandler(500)
def handle_500(e):
    # Render the professional one-page dashboard with safe defaults on server error
    initial_capital = 100000
    short_window = 20
    long_window = 50
    return render_template('home.html',
        initial_capital=initial_capital,
        short_window=short_window,
        long_window=long_window,
        final_portfolio_value='N/A',
        total_profit_loss='N/A',
        percentage_return='N/A',
        profitability_assessment='An error occurred',
        num_trades=0,
        num_buys=0,
        num_sells=0,
        win_rate='N/A',
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        cagr=0.0,
        chart_html='',
        portfolio_chart_html='',
        drawdown_chart_html='',
        cumulative_chart_html='',
        recommended_position_size=0,
        stop_loss_long=0.0,
        stop_loss_short=0.0,
        value_at_risk=0.0,
        expected_shortfall=0.0,
        risk_reward_ratio=0.0,
        recommended_risk_per_trade=0.0)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Securely serve static files from the static directory"""
    try:
        if '..' in filename or filename.startswith('/'):
            raise ValueError("Invalid file path")
        return send_from_directory('static', filename)
    except Exception as e:
        app.logger.error(f"Error serving static file {filename}: {str(e)}")
        return f"Error serving file: {filename}", 404

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main dashboard with detailed analysis and enhanced interactivity"""
    
    # Default values
    initial_capital = 100000
    short_window = 20
    long_window = 50
    context = {
        'initial_capital': initial_capital,
        'short_window': short_window,
        'long_window': long_window
    }
    
    try:
        # Update parameters from form if POST request
        if request.method == 'POST':
            initial_capital = int(request.form.get('initial_capital', initial_capital))
            short_window = int(request.form.get('short_window', short_window))
            long_window = int(request.form.get('long_window', long_window))

        # Load and validate data
        df = load_data()
        if df is None or len(df) == 0:
            raise ValueError('No data available for analysis')

        # Generate analyses
        results = generate_analysis(df, initial_capital, short_window, long_window)
        metrics = calculate_metrics(df)
        stop_losses = calculate_stop_loss(df)
        portfolio_risk = calculate_portfolio_risk(df, initial_capital)

        # Calculate recommended position size
        if isinstance(metrics, dict) and metrics.get('Recommended_Position_Size') is not None:
            rec_pos_size = metrics.get('Recommended_Position_Size')
        else:
            rec_pos_size = calculate_position_size(
                initial_capital,
                portfolio_risk.get('Recommended_Risk_Per_Trade', 1.0),
                df['Price INR'].iloc[-1],
                stop_losses.get('Stop_Loss_Long', df['Price INR'].iloc[-1]*0.98)
            )

        # Prepare context with all computed values
        context = {
            'initial_capital': initial_capital,
            'short_window': short_window,
            'long_window': long_window,
            'final_portfolio_value': results.get('final_portfolio_value', initial_capital),
            'total_profit_loss': results.get('total_profit_loss', 0),
            'percentage_return': results.get('percentage_return', 0),
            'profitability_assessment': results.get('profitability_assessment', 'N/A'),
            'num_trades': results.get('num_trades', 0),
            'num_buys': results.get('num_buys', 0),
            'num_sells': results.get('num_sells', 0),
            'win_rate': results.get('win_rate', 'N/A'),
            'sharpe_ratio': metrics.get('Sharpe_Ratio', 0.0),
            'sortino_ratio': metrics.get('Sortino_Ratio', 0.0),
            'max_drawdown': metrics.get('Max_Drawdown', 0.0),
            'cagr': metrics.get('CAGR', 0.0),
            'chart_html': results.get('chart_html', ''),
            'portfolio_chart_html': results.get('portfolio_chart_html', ''),
            'drawdown_chart_html': results.get('drawdown_chart_html', ''),
            'cumulative_chart_html': results.get('cumulative_chart_html', ''),
            'recommended_position_size': rec_pos_size,
            'stop_loss_long': stop_losses.get('Stop_Loss_Long', 0.0),
            'stop_loss_short': stop_losses.get('Stop_Loss_Short', 0.0),
            'value_at_risk': portfolio_risk.get('Value_at_Risk', 0.0),
            'expected_shortfall': portfolio_risk.get('Expected_Shortfall', 0.0),
            'risk_reward_ratio': portfolio_risk.get('Risk_Reward_Ratio', 0.0),
            'recommended_risk_per_trade': portfolio_risk.get('Recommended_Risk_Per_Trade', 1.0)
        }
        
        return render_template('home.html', **context)
        
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error in index route: {str(e)}")
        
        # Add error message to context
        context['error_message'] = f"An error occurred: {str(e)}"
        
        # Return error page with context
        return render_template('error.html', **context), 500


if __name__ == '__main__':
    print("\nStarting Algo Trading Dashboard...")
    print("----------------------------------------")
    port = 5000
    url = f"http://localhost:{port}"
    print(f"Dashboard URL: {url}")
    print("----------------------------------------")
    print("Press Ctrl+C to stop the server")
    app.run(port=port)
