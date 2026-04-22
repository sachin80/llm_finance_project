from pathlib import Path
# Absolute path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input is the Excel file from class to create the universe of assets 
PATH_IN = PROJECT_ROOT/"data"/"SP500.xlsx"
SHEET = "Sheet1"

# Output directory for saved files and results
OUT_DIR = PROJECT_ROOT/ "outputs"
OUT_DIR.mkdir(exist_ok=True)

# OpenAI embedding model
EMBED_MODEL = "text-embedding-3-large"

# Shared AI-theme statement for both TF-IDF and embedding methods.
THEME_TEXT = '''
Companies that benefit from AI infrastructure growth, including GPUs,
AI compute hardware, cloud platforms, semiconductor equipment, networking,
data-center power and cooling, and software infrastructure that enables
large-scale AI deployment.
'''

# Price window:
# Start date chosen to be before AI hype due to ChatGPT release.
# End date is the most recent trading day.
START_DATE = "2022-01-01"
END_DATE = "2026-04-20"

# Risk-model settings:
# Look back days is the standard number of trading days for 1 year. 
# EWMA halflife controls importance of past observations, so after 60 days
# observations have half the weight of newer ones. 
LOOKBACK_DAYS = 252
EWMA_HALFLIFE = 60

# Optimizer settings: 
# Risk aversion parameter chosen to penalize portfolio variance relative to alpha.
# Turnover penalty parameter chosen to lightly regularize large updates during rebalance.
# Max weight parameter chosen to reduce one-stock domination. 
# RISK_AVERSION = 10.0
RISK_AVERSION = 1.0
TURNOVER_PENALTY = 0.01
MAX_WEIGHT = 0.20

# Backtest settings:
# Initial cash chosen to get because easier to analyze. 
# Commission parameter chosen to apply a simple 0.1% trading cost. 
INITIAL_CASH = 100000.0
COMMISSION = 0.001
