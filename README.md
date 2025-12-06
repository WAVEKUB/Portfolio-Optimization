# Portfolio Optimization with Metaheuristic Algorithms

This repository contains an experimental framework for **portfolio optimization** using multiple **metaheuristic algorithms** on **real stock market data**.  
The goal is to find portfolio weights that **maximize the Sharpe Ratio** under long-only and turnover constraints.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run the Experiment](#run-the-experiment)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Interpreting the Results](#interpreting-the-results)
- [Possible Extensions](#possible-extensions)
- [Disclaimer](#disclaimer)

---

## Overview

Modern portfolio theory suggests that we can construct an “optimal” portfolio by balancing **expected return** and **risk**.

In this project:

- Historical **Close prices** are downloaded via `yfinance` for a basket of large-cap US stocks.
- **Annualized returns** and **covariance** are estimated directly from the data.
- Several **metaheuristic optimization algorithms** search for portfolio weights.
- Each algorithm is evaluated using the **Sharpe Ratio**.
- The framework runs **multiple independent runs per algorithm** (default: `n_runs = 5`) and compares their **average performance**.

This code is intended as a **research / portfolio piece** to demonstrate:

- Numerical optimization on real financial data  
- Clean experiment structure in Python  
- Comparison of multiple global optimization methods

---

## Key Features

- 🔁 **Multiple runs per algorithm** (`n_runs = 5` by default) for more robust comparison  
- 📈 **Annualization** of return and risk using the estimated number of trading days from the dataset  
- 📊 **Automatic visualizations**:
  - `results/sharpe_comparison.png`: Average Sharpe Ratio per algorithm
  - `results/allocation_<ALGO>.png`: Pie chart for the best algorithm’s average allocation
- 🔒 **Realistic constraints**:
  - Long-only portfolios (weights ≥ 0, sum to 1)
  - **Turnover constraint** relative to a current (equal-weight) portfolio
- 🧾 **Console summary**:
  - Baseline equal-weight portfolio performance
  - Average performance table (Return, Volatility, Sharpe, Objective) per algorithm
  - Average weights per asset per algorithm

---

## Algorithms Implemented

All algorithms minimize the **negative Sharpe Ratio** (`objective_neg_sharpe`), so minimizing the objective is equivalent to maximizing the Sharpe Ratio.

Implemented algorithms (see `src/optimizers/` and `src/optimizers/__init__.py`):

- **ACOR** – Ant Colony Optimization for continuous domains (`acor.py`)
- **BAT** – Bat Algorithm (`bat.py`)
- **HS** – Harmony Search (`harmony.py`)
- **GA** – Genetic Algorithm (`ga.py`)
- **PSO** – Particle Swarm Optimization (`pso.py`)
- **DE** – Differential Evolution (`de.py`)
- **SA** – Simulated Annealing (`sa.py`)
- **RRHC** – Random Restart Hill Climbing (`rrhc.py`)
- **PS** – Pattern Search (`pattern.py`)

These correspond to the algorithm names in `names_list` in `main.py`:

```python
names_list = ["ACOR", "BAT", "HS", "GA", "PSO", "DE", "SA", "RRHC", "PS"]
```

Each optimizer respects:

- The **simplex constraint** (valid portfolio weights)
- A configurable **L1-turnover limit** vs the current portfolio via `project_to_simplex_with_turnover`.

---

## Project Structure

```text
.
├── .gitignore
├── config.py          # Configuration (assets universe, risk-free rate, date range)
├── main.py            # Entry point for running experiments and plotting results
├── README.md          # Project documentation (this file)
├── requirements.txt   # Python dependencies
└── src/
    ├── data.py        # Data download and preprocessing (yfinance)
    ├── utils.py       # Projection, portfolio statistics, negative Sharpe objective
    └── optimizers/    # Metaheuristic optimization algorithms
        ├── __init__.py
        ├── acor.py    # Ant Colony Optimization
        ├── bat.py     # Bat Algorithm
        ├── de.py      # Differential Evolution
        ├── ga.py      # Genetic Algorithm
        ├── harmony.py # Harmony Search
        ├── pattern.py # Pattern Search
        ├── pso.py     # Particle Swarm Optimization
        ├── rrhc.py    # Random Restart Hill Climbing
        └── sa.py      # Simulated Annealing
```

> Note: The `results/` folder is **created at runtime** by `main.py` when saving plots.

---

## Getting Started

### Prerequisites

- Python **3.x**
- Git
- (Optional) A virtual environment (`venv` or similar)

### Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/WAVEKUB/Portfolio-Optimization.git
   cd Portfolio-Optimization
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv

   # On Linux / macOS:
   source .venv/bin/activate

   # On Windows:
   .venv\Scriptsctivate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` includes (at minimum):

   - `numpy`
   - `pandas`
   - `yfinance`
   - `matplotlib`
   - `seaborn`

### Run the Experiment

From the project root:

```bash
python main.py
```

This will:

1. Download historical price data using `yfinance` for the configured date range
2. Prepare annualized return & covariance matrices
3. Run **all optimization algorithms**, each with `n_runs = 5` random seeds
4. Print summary tables in the console
5. Save comparison plots into the `results/` directory

---

## Configuration

Global settings are defined in `config.py`:

```python
ASSETS = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com Inc.
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms (Facebook)
    'NVDA',  # NVIDIA Corporation
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'WMT'    # Walmart Inc.
]

RISK_FREE_RATE = 0.02  # annual risk-free rate (e.g. 0.02 == 2%)

#PERIOD = "2y"         # optional alternative if you want to use a relative period instead of fixed dates

START = "2023-01-01"   # start date (YYYY-MM-DD) for historical data
END   = "2024-12-31"   # end date (YYYY-MM-DD)   for historical data
```

You can change:

- `ASSETS`: the universe of tickers  
- `RISK_FREE_RATE`: annual risk-free rate used in the Sharpe Ratio  
- `START`, `END`: date range for historical data (if you prefer a fixed window)  
- Or uncomment and use `PERIOD` if you want a rolling window like `"2y"` instead of fixed dates.

Algorithm hyperparameters (iterations, population sizes, etc.) are set directly in `main.py` inside the `algs_run` dictionary.

---

## How It Works

### 1. Data Download & Preprocessing (`src/data.py`)

- `get_data(assets, start=None, end=None, period=None)`  

  - In this project, `main.py` calls `get_data` using `config.START` and `config.END` so that the historical window is explicitly controlled by dates.
  - Internally it uses `yfinance.download` to fetch **Close** prices for all assets over that range.
  - Handles the single-asset case by reshaping to a DataFrame and cleaning NaNs.

- `prepare_data(prices_df)`:
  - Computes daily returns using `pct_change().dropna()`
  - Estimates **trading days per year** from the index:
    - Counts returns per calendar year and averages → `trading_days`
    - If this fails, falls back to 252
  - Annualizes:
    - Expected returns: `mu_annual = returns_daily.mean() * trading_days`
    - Covariance: `Sigma_annual = returns_daily.cov() * trading_days`
  - Returns `(mu_annual, Sigma_annual, trading_days)`

### 2. Portfolio Statistics & Objective (`src/utils.py`)

Core functions:

- `portfolio_stats(w, mu, Sigma, rf)`:

  - `ret = w @ mu` (expected annual return)  
  - `vol = sqrt(w^T Sigma w)` (annual volatility)  
  - Sharpe Ratio:  

    \[
    	ext{Sharpe} = rac{	ext{ret} - r_f}{	ext{vol} + 1	ext{e-}12}
    \]

- `objective_neg_sharpe(w, mu, Sigma, rf)`:
  - Calls `portfolio_stats` and returns **`-Sharpe`** (used as the objective to minimize).

- `project_to_simplex(w)`:
  - Projects a vector onto the **probability simplex**:  
    - \( w_i \ge 0 \) and \( \sum_i w_i = 1 \)
  - If projection collapses to zero, it falls back to equal weights.

- `project_to_simplex_with_turnover(w, current=None, max_turnover=1.0)`:
  - First projects `w` to the simplex.
  - If `current` and `max_turnover` are provided:
    - Computes L1 turnover: `|w_proj - current|_1`
    - If turnover exceeds `max_turnover`, scales the change and re-projects.
  - Special cases:
    - `max_turnover <= 0` → returns `current` unchanged  
    - `max_turnover >= 2` → effectively only simplex projection

This models a **trading constraint** limiting how far the new portfolio can move away from the **current equal-weight portfolio** in a single optimization step.

### 3. Optimization Loop (`main.py`)

- Computes baseline **equal-weight** portfolio:

  ```python
  current_weight = np.ones(n_assets) / n_assets
  r_ann, vol_ann, sharpe_ann = portfolio_stats(
      current_weight,
      mu_annual_vec,
      Sigma_annual_mat,
      rf=config.RISK_FREE_RATE
  )
  ```

- Sets global seed and number of runs:

  ```python
  seed = 2025
  rs_all = np.random.RandomState(seed)
  n_runs = 5  # can be increased for more robust statistics
  ```

- For each run:
  - Creates `rs_run` and builds `algs_run` dict where each entry calls a specific optimizer with:
    - `mu_annual_vec`, `Sigma_annual_mat`
    - `rf=config.RISK_FREE_RATE`
    - `current=current_weight`
    - `max_turnover=1.0`
    - Algorithm-specific hyperparameters (`iters`, `pop`, `swarm`, etc.)

- For each algorithm & run:
  - Calls the optimizer: `w, f, history = algs_run[name]()`
  - Evaluates `(ret, vol, sharpe)` via `portfolio_stats`
  - Stores weights and metrics in `acc[name]`

### 4. Aggregation & Reporting

After all runs:

- Computes mean metrics:

  ```python
  avg_comp = pd.DataFrame(
      rows_avg,
      columns=["Algo", "Return_mean", "Volatility_mean", "Sharpe_mean", "NegObj_mean"]
  ).sort_values("Sharpe_mean", ascending=False)
  ```

- Computes average weights:

  ```python
  avg_weights_df = pd.DataFrame(avg_weights, index=config.ASSETS)
  ```

- Prints to console:
  - Detected `trading_days`
  - Baseline equal-weight performance
  - Average performance per algorithm
  - Average weights per algorithm
  - The **best algorithm** based on `Sharpe_mean`

### 5. Visualization & Results

If `results/` does not exist, it is created. Then:

- **Sharpe comparison plot** (`results/sharpe_comparison.png`):

  - Uses `seaborn.lineplot` to show `Sharpe_mean` for each algorithm.
  - Highlights the best-performing algorithm with a different marker and label.

- **Allocation pie chart** (`results/allocation_<ALGO>.png`):

  - Uses the average weights of the **best** algorithm.
  - Groups very small weights (`< 0.01`) into an “Others” slice for clarity.

---

## Interpreting the Results

After `python main.py`, you will see:

1. **Console output**
   - Estimated trading days per year from the dataset
   - Baseline equal-weight portfolio: annual return, volatility, Sharpe
   - A table of **average performance** per algorithm:
     - `Algo`, `Return_mean`, `Volatility_mean`, `Sharpe_mean`, `NegObj_mean`
   - Average weights per asset for each algorithm
   - The name of the best algorithm based on `Sharpe_mean`

2. **Plots in `results/`**
   - `sharpe_comparison.png`:  
     Visual comparison of average Sharpe Ratios for all algorithms.
   - `allocation_<ALGO>.png`:  
     Asset allocation of the best algorithm, with small weights grouped into `Others`.

These outputs let you compare:

- Which algorithms tend to find **better risk-adjusted portfolios**
- How the resulting **allocations differ** between algorithms and relative to equal weight

---

## Possible Extensions

Ideas for future work:

- Add **transaction cost modeling** (explicit cost term or penalty in the objective).
- Allow **short selling** with bounds (e.g. \(-0.1 \le w_i \le 0.3\)).
- Add **position / sector limits** (max weight per asset, sector, etc.).
- Compare with **classical convex optimizers** (e.g., quadratic programming / CVXPY).
- Perform **statistical tests** on Sharpe differences between algorithms.
- Wrap this optimizer in a **REST API** or build a small **dashboard** for interactive experiments.

---

## Disclaimer

This project is for **educational and research purposes only**.  
It is **not** financial advice, and the author is **not** responsible for any investment decisions made using this code or its outputs.
