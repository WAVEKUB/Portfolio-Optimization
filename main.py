# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src.data import get_data, prepare_data
from src.utils import portfolio_stats
from src.optimizers import (
    bat_optimize, harmony_search, ga_optimize, pso_optimize,
    de_optimize, sa_optimize, rrhc_optimize, pattern_search_optimize,
    acor_optimize
)

def main():
    # 1. Get Data
    prices_df = get_data(config.ASSETS, period=config.PERIOD)
    
    # 2. Prepare Data
    mu_annual, Sigma_annual, trading_days = prepare_data(prices_df)
    
    mu_annual_vec = mu_annual.values
    Sigma_annual_mat = Sigma_annual.values
    n_assets = len(config.ASSETS)
    
    # Current weight (equal weights)
    current_weight = np.ones(n_assets) / n_assets
    
    r_ann, vol_ann, sharpe_ann = portfolio_stats(current_weight, mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE)
    print(f"Detected trading_days={trading_days}; Current portfolio annual ret={r_ann:.4f}, vol={vol_ann:.4f}, sharpe={sharpe_ann:.4f}")

    # 3. Run Optimizers
    seed = 2025
    rs_all = np.random.RandomState(seed)
    n_runs = 5 # Reduced for quicker verification, can be increased
    names_list = ["ACOR", "BAT", "HS", "GA", "PSO", "DE", "SA", "RRHC", "PS"]
    
    acc = {name: {"rets": [], "vols": [], "sharpes": [], "negobjs": [], "weights": []} for name in names_list}
    seeds = rs_all.randint(0, 2**31 - 1, size=n_runs)
    
    print(f"Running optimizations ({n_runs} runs per algorithm)...")
    
    for i, sd in enumerate(seeds):
        print(f"Run {i+1}/{n_runs}...")
        rs_run = np.random.RandomState(int(sd))
        
        algs_run = {
            "ACOR": lambda rs=rs_run: acor_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=600, archive_size=50, n_ants=40, current=current_weight, max_turnover=1.0),
            "BAT":  lambda rs=rs_run: bat_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=700, n_bats=60, current=current_weight, max_turnover=1.0),
            "HS":   lambda rs=rs_run: harmony_search(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=700, HMS=60, current=current_weight, max_turnover=1.0),
            "GA":   lambda rs=rs_run: ga_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=500, pop=100, current=current_weight, max_turnover=1.0),
            "PSO":  lambda rs=rs_run: pso_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=700, swarm=80, current=current_weight, max_turnover=1.0),
            "DE":   lambda rs=rs_run: de_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=700, pop=80, current=current_weight, max_turnover=1.0),
            "SA":   lambda rs=rs_run: sa_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=3000, current=current_weight, max_turnover=1.0),
            "RRHC": lambda rs=rs_run: rrhc_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, restarts=25, iters=300, current=current_weight, max_turnover=1.0),
            "PS":   lambda rs=rs_run: pattern_search_optimize(mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE, rs=rs_run, iters=250, current=current_weight, max_turnover=1.0)
        }

        for name in names_list:
            try:
                w, f, h = algs_run[name]()
                r, v, s = portfolio_stats(w, mu_annual_vec, Sigma_annual_mat, rf=config.RISK_FREE_RATE)
                acc[name]["weights"].append(w)
                acc[name]["rets"].append(r)
                acc[name]["vols"].append(v)
                acc[name]["sharpes"].append(s)
                acc[name]["negobjs"].append(f)
            except Exception as e:
                print(f"Run {i} failed for {name}: {e}")

    # 4. Results
    rows_avg = []
    avg_weights = {}
    for name in names_list:
        if len(acc[name]["rets"]) == 0:
            continue
        rows_avg.append([
            name,
            float(np.mean(acc[name]["rets"])),
            float(np.mean(acc[name]["vols"])),
            float(np.mean(acc[name]["sharpes"])),
            float(np.mean(acc[name]["negobjs"]))
        ])
        avg_weights[name] = np.mean(np.vstack(acc[name]["weights"]), axis=0)

    avg_comp = pd.DataFrame(rows_avg, columns=["Algo", "Return_mean", "Volatility_mean", "Sharpe_mean", "NegObj_mean"]).sort_values("Sharpe_mean", ascending=False)
    print("\nAverage performance across runs:")
    print(avg_comp.round(6))

    avg_weights_df = pd.DataFrame(avg_weights, index=config.ASSETS)
    print("\nAverage weights across runs:")
    print(avg_weights_df.applymap(lambda x: round(float(x), 4)))

    print(f"\nBest algorithm based on average Sharpe Ratio: {avg_comp.iloc[0]['Algo']}")
    # 5. Plot
    if not os.path.exists("results"):
        os.makedirs("results")

    avg_comp_sorted = avg_comp.sort_values(by="Sharpe_mean", ascending=False)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Algo", y="Sharpe_mean", data=avg_comp_sorted, marker='o', color='skyblue')
    
    best_algo_mean = avg_comp.iloc[0]["Algo"]
    best_algo_data = avg_comp_sorted[avg_comp_sorted["Algo"] == best_algo_mean]
    plt.plot(best_algo_data["Algo"], best_algo_data["Sharpe_mean"], marker='o', markersize=10, color='orange', label=f'Best: {best_algo_mean}')
    
    plt.xlabel("Algorithm")
    plt.ylabel("Average Sharpe Ratio")
    plt.title("Average Sharpe Ratio of Portfolio Optimization Algorithms")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/sharpe_comparison.png")
    print("\nPlot saved to results/sharpe_comparison.png")

    # 6. Pie Chart for Best Algorithm
    best_weights = avg_weights[best_algo_mean]
    
    # Filter out very small weights for cleaner chart
    threshold = 0.01
    labels = []
    sizes = []
    other_weight = 0
    
    for asset, weight in zip(config.ASSETS, best_weights):
        if weight >= threshold:
            labels.append(asset)
            sizes.append(weight)
        else:
            other_weight += weight
            
    if other_weight > 0:
        labels.append("Others")
        sizes.append(other_weight)
        
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Asset Allocation - Best Algorithm ({best_algo_mean})")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig(f"results/allocation_{best_algo_mean}.png")
    print(f"Pie chart saved to results/allocation_{best_algo_mean}.png")

if __name__ == "__main__":
    main()
