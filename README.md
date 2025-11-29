# Portfolio Optimization

This project implements various metaheuristic algorithms to optimize a stock portfolio. It refactors a Jupyter Notebook into a modular Python project structure.

## Project Structure

```
portfolio/
├── config.py          # Configuration (assets, risk-free rate, period)
├── main.py            # Entry point for execution
├── requirements.txt   # Project dependencies
├── results/           # Output directory for plots
└── src/
    ├── data.py        # Data fetching and preparation
    ├── utils.py       # Helper functions
    └── optimizers/    # Optimization algorithms
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

## Setup

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to execute the optimizations:

```bash
python main.py
```

## Output

The script will:
1.  Download historical stock data.
2.  Run multiple optimization algorithms.
3.  Display the average performance of each algorithm in the console.
4.  Generate visualization plots in the `results/` directory:
    -   `sharpe_comparison.png`: Comparison of average Sharpe Ratios.
    -   `allocation_<ALGO>.png`: Asset allocation pie chart for the best-performing algorithm.
