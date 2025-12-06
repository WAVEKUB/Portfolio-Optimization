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
In this project, we:

- Download **historical prices** for a basket of large-cap US stocks.
- Estimate **annualized returns** and **covariance**.
- Use several **metaheuristic optimization algorithms** to search for portfolio weights.
- Evaluate each algorithm based on the **Sharpe Ratio**.
- Compare their **average performance** across multiple random runs.

This project is designed as a **research / portfolio piece** to demonstrate:

- Numerical optimization on real financial data  
- Clean experiment structure in Python  
- Comparison of multiple global optimization methods

---

## Key Features

- 🔁 **Multiple independent runs** per algorithm for fair comparison  
- 📈 **Annualization** of return and risk based on actual trading days in the data  
- 📊 **Automatic visualizations**:
  - Average Sharpe Ratio per algorithm
  - Pie chart of the best algorithm’s average allocation
- 🔒 **Realistic constraints**:
  - Long-only portfolio (weights ≥ 0, sum to 1)
  - Turnover constraint relative to a current portfolio

---

## Algorithms Implemented

All algorithms optimize the **negative Sharpe Ratio** (so minimizing objective = maximizing Sharpe):

- **ACOR** – Ant Colony Optimization for continuous domains  
- **BAT** – Bat Algorithm  
- **HS** – Harmony Search  
- **GA** – Genetic Algorithm  
- **PSO** – Particle Swarm Optimization  
- **DE** – Differential Evolution  
- **SA** – Simulated Annealing  
- **RRHC** – Random Restart Hill Climbing  
- **PS** – Pattern Search  

Each algorithm respects the **simplex constraint** (weights form a valid portfolio) and a configurable **turnover limit**.

---

## Project Structure

```text
.
├── config.py          # Configuration (assets universe, risk-free rate, period)
├── main.py            # Entry point for running all experiments
├── requirements.txt   # Python dependencies
├── src/
│   ├── data.py        # Data download and preprocessing (yfinance)
│   ├── utils.py       # Projection and portfolio utility functions
│   └── optimizers/    # Metaheuristic optimization algorithms
│       ├── __init__.py
│       ├── acor.py    # Ant Colony Optimization
│       ├── bat.py     # Bat Algorithm
│       ├── de.py      # Differential Evolution
│       ├── ga.py      # Genetic Algorithm
│       ├── harmony.py # Harmony Search
│       ├── pattern.py # Pattern Search
│       ├── pso.py     # Particle Swarm Optimization
│       ├── rrhc.py    # Random Restart Hill Climbing
│       └── sa.py      # Simulated Annealing
└── results/           # Generated plots (created at runtime)
```

> Note: The `results/` folder is created automatically when you run `main.py`.

---

## Getting Started

### Prerequisites

- Python **3.x**
- Git
- (Optional but recommended) A virtual environment (e.g. `venv`)

### Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/WAVEKUB/Portfolio-Optimization.git
   cd Portfolio-Optimization
   ```

2. **Create and activate a virtual environment** (optional but recommended)

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

### Run the Experiment

From the project root:

```bash
python main.py
```

This will:

1. Download price data using `yfinance`
2. Run **multiple optimization algorithms**, each with several random seeds
3. Print summary statistics in the console
4. Save comparison plots into the `results/` directory

---

## Configuration

All high-level settings live in `config.py`:

```python
# config.py

ASSETS = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc.
    'AMZN',  # Amazon.com Inc.
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms
    'NVDA',  # NVIDIA Corporation
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'WMT'    # Walmart Inc.
]

RISK_FREE_RATE = 0.2   # annual risk-free rate (e.g., 0.1 == 10%)
PERIOD = "2y"          # length of historical data to download
```

You can customize:

- `ASSETS`: universe of tickers for the portfolio  
- `RISK_FREE_RATE`: annual risk-free rate used in Sharpe Ratio  
- `PERIOD`: window of historical data (e.g. `"1y"`, `"5y"`)

Algorithm-specific hyperparameters (population sizes, iterations, etc.) are configured inside `main.py` and individual files in `src/optimizers/`.

---

## How It Works

1. **Data Download & Preprocessing**  
   - `src/data.py` uses `yfinance` to download **adjusted close** prices for the specified tickers and period.  
   - Daily returns are computed and then **annualized**:
     - Expected returns: mean daily return × trading days per year  
     - Covariance: daily covariance × trading days per year  

2. **Objective Function**  
   In `src/utils.py`, the portfolio statistics are computed as:

   - \( r = w^\top \mu \) (expected return)  
   - \( \sigma = \sqrt{w^\top \Sigma w} \) (volatility)  
   - Sharpe Ratio: \( \text{Sharpe} = \dfrac{r - r_f}{\sigma} \)

   The **objective** each algorithm minimizes is:

   ```python
   objective_neg_sharpe(w, mu, Sigma, rf)
   ```

3. **Constraints**  
   - **Simplex constraint**: weights are projected so that  
     - \( w_i \ge 0 \)  
     - \( \sum_i w_i = 1 \)  
   - **Turnover constraint**:  
     - The helper `project_to_simplex_with_turnover` limits the L1 distance between the new weights and a given **current portfolio**, modeling a cap on trading volume.

4. **Multiple Runs Per Algorithm**  
   - `main.py` sets a global seed and generates multiple random seeds.
   - Each algorithm is run `n_runs` times with different random states.
   - For each run, the optimized weights are evaluated and stored.

5. **Aggregation & Reporting**  
   - For each algorithm, the code aggregates:
     - Mean return
     - Mean volatility
     - Mean Sharpe Ratio
     - Mean objective value
     - Average weights across runs
   - A summary `DataFrame` is printed, and the **best algorithm** (by average Sharpe) is highlighted.

---

## Interpreting the Results

After running `python main.py`, you will get:

1. **Console Output**
   - Detected number of trading days per year
   - Baseline performance of the **equal-weight** portfolio
   - A table of average performance per algorithm (Return, Volatility, Sharpe, Objective)
   - A table of **average weights** per asset for each algorithm

2. **Plots in `results/`**
   - `sharpe_comparison.png`  
     Line plot showing the **average Sharpe Ratio** for each algorithm, with the best one highlighted.
   - `allocation_<ALGO>.png`  
     Pie chart showing the **average asset allocation** of the best-performing algorithm (small weights grouped into an “Others” category).

These outputs make it easy to compare:

- Which algorithm tends to find better risk-adjusted portfolios  
- How the allocation differs between algorithms and relative to equal weight

---

## Possible Extensions

Ideas for future work / improvements:

- Add **transaction cost modeling** (e.g., penalty on turnover in the objective).
- Include **short selling** with bounds (e.g. \(-0.1 \le w_i \le 0.3\)).
- Introduce **sector or position limits** (max weight per asset or sector).
- Compare against **classical optimizers** (e.g. quadratic programming / convex solvers).
- Add **statistical tests** to compare Sharpe Ratios across algorithms.
- Wrap the optimization into a **REST API** or **web dashboard**.

These extensions are good directions if you want to turn this into a more complete research project or a production-like system.

---

## Disclaimer

This project is for **educational and research purposes only**.  
It is **not** financial advice, and the author is **not** responsible for any investment decisions made using this code or its outputs.
