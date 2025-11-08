"""
Configuration settings for the Algo Trading Dashboard
"""

# Flask app settings
SECRET_KEY = 'your-secret-key-here'  # Change this in production
DEBUG = True
TEMPLATES_AUTO_RELOAD = True
SEND_FILE_MAX_AGE_DEFAULT = 0  # Disable caching for development

# Data settings
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_RISK_FREE_RATE = 0.05  # 5% annual
TRADING_DAYS_PER_YEAR = 252

# Technical analysis settings
DEFAULT_SHORT_WINDOW = 20
DEFAULT_LONG_WINDOW = 50
ATR_PERIODS = 14
ATR_MULTIPLIER = 2

# Risk management settings
MAX_POSITION_SIZE_PCT = 0.02  # 2% of capital
MIN_POSITION_SIZE_PCT = 0.01  # 1% of capital
MAX_RISK_PER_TRADE = 2.0  # 2%
MIN_RISK_PER_TRADE = 0.5  # 0.5%