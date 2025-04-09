Financial Asset Allocation Framework using ML, Optimization, and LSTM
This repository presents an integrated asset allocation pipeline combining traditional portfolio optimization, feature engineering, machine learning for ranking, and LSTM-based forecasting. It's a modular, extensible framework built for backtesting and experimentation in computational finance.

Designed for researchers, quants, and ML-finance enthusiasts seeking to blend quantitative strategies with modern AI.

# Key Features
Synthetic Data Generation: Simulate multi-asset historical price series using Gaussian returns.

Financial Feature Engineering: Momentum, rolling mean/volatility features extracted from historical returns.

Feature Selection: Leverages RandomForestRegressor for importance-based filtering.

#Optimization Techniques:

Sharpe Ratio Maximization using convex programming (CVXPY).

Minimum Risk Portfolio construction using quadratic optimization.

LSTM Model for Time Series Forecasting:

Predicts future returns based on historical windows.

# Learning to Rank:

Implements a ML-based return ranking via Random Forest regression.

🧪 Modules Overview
🔧 generate_fake_data()
Simulates n_assets worth of historical price data over n_days using log-normal returns.

Used for testing and prototyping strategies without real market data.

# generate_features()
Constructs:

Rolling mean

Rolling standard deviation

Momentum (current - past value)

Generates interpretable features aligned with market behavior.

#select_features()
Uses RandomForestRegressor to compute feature importances.

Selects top 10 most relevant features for prediction.

# Optimization Classes
LinearProgram and ConvexProgram
Wrappers around CVXPY's constraint-based modeling.

Enable flexibility for adding new optimization formulations (LP, QP, SOCP).

sharpe_ratio_objective()
Maximizes return-to-risk ratio using mean-variance optimization.

Enforces fully invested portfolio (sum(weights) = 1).

min_risk_objective()
Minimizes total portfolio variance.

Ideal for risk-averse investors.

# LSTMModel
A PyTorch-based LSTM with:

2 layers

32 hidden units

Final linear layer for regression output

prepare_lstm_data()
Prepares rolling time-windowed sequences for LSTM training.

train_lstm_model()
Trains the LSTM model on a single asset's returns.

# learning_to_rank()
Applies RandomForestRegressor to predict and rank future returns.

Emulates "learning-to-rank" strategies in quant models.

# Execution Flow
bash
Copy
Edit
python main.py
Main script executes:

Data generation

Feature engineering and selection

Optimization for asset allocation

LSTM training and forecasting

Learning-to-rank for asset scoring

Sample Output
yaml
Copy
Edit
Optimized Weights (Sharpe Ratio): [0.12, 0.20, ...]
Optimized Weights (Min Risk): [0.25, 0.18, ...]
Epoch 0, Loss: 0.003520
Epoch 5, Loss: 0.002103
...
Sample Predictions: [0.0013, 0.0017, ...]
Ranked Returns:
2021-01-25    0.00180
2021-01-17    0.00145
...
# Technologies Used
Python

PyTorch — For LSTM-based forecasting

CVXPY — For convex optimization of portfolio weights

Scikit-learn — Feature selection and return prediction

Pandas/Numpy — Data manipulation and numerical operations

# Project Structure
graphql
Copy
Edit
.
├── main.py                # Main driver script
├── README.md              # You're reading it!
├── lstm_model.py          # LSTM definition and training
├── optimizer.py           # Linear/Convex optimization classes
├── features.py            # Feature engineering and selection
└── utils.py               # Data generation and helpers
(Note: You can refactor into modules above for production use)

# Future Work
✅ Add backtesting framework with daily/weekly rebalancing

✅ Use real-world financial data via yfinance

🔄 Include transaction cost modeling

🔮 Extend LSTM to multi-asset prediction

📈 Implement a full backtest with Sharpe, Sortino, and drawdown metrics

🧑‍💻 Author
Sukanya Singh
B.Tech in ECE | DSA + ML + Computational Finance Enthusiast
Feel free to connect or suggest improvements.

📜 License
MIT License.
Free to use, adapt, and contribute.
