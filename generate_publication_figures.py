import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart."""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.2f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va,
            fontsize=10,
            fontweight='bold'
        )

def main():
    # Set seaborn theme for professional aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    out_dir = "results/greedy_vs_aco"
    metrics_file = os.path.join(out_dir, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"File {metrics_file} not found!")
        return
        
    df = pd.read_csv(metrics_file)
    
    report_data = []
    
    # Pre-process travel time and CO2 for better readability if needed
    if "greedy_duration_sec" in df.columns:
        df["greedy_duration_min"] = df["greedy_duration_sec"] / 60.0
        df["aco_duration_min"] = df["aco_duration_sec"] / 60.0
    if "greedy_co2_g" in df.columns:
        df["greedy_co2_kg"] = df["greedy_co2_g"] / 1000.0
        df["aco_co2_kg"] = df["aco_co2_g"] / 1000.0
    
    metrics_map = {
        "distance_km": ("Distance (km)", "Lower", "greedy_distance_km", "aco_distance_km"),
        "duration_min": ("Travel Time (min)", "Lower", "greedy_duration_min", "aco_duration_min"),
        "avg_congestion": ("Avg Congestion", "Lower", "greedy_avg_congestion", "aco_avg_congestion"),
        "fuel_l": ("Fuel (L)", "Lower", "greedy_fuel_l", "aco_fuel_l"),
        "co2_kg": ("CO2 (kg)", "Lower", "greedy_co2_kg", "aco_co2_kg"),
        "visited_cells": ("Visited Grid Cells", "Lower", "greedy_visited_cells", "aco_visited_cells"),
        "runtime_sec": ("Runtime (s)", "Lower", "greedy_runtime_sec", "aco_runtime_sec")
    }
    
    for key, (name, objective, g_col, a_col) in metrics_map.items():
        if g_col not in df.columns or a_col not in df.columns:
            continue
            
        g_val = df[g_col].mean()
        a_val = df[a_col].mean()
        diff = a_val - g_val
        
        # Improvement: if lower is better, improvement is positive when diff is negative
        if objective == "Lower":
            imp_pct = - (diff / g_val) * 100 if g_val != 0 else 0
            better = "ACO" if a_val < g_val else "Greedy" if g_val < a_val else "Tie"
        else:
            imp_pct = (diff / g_val) * 100 if g_val != 0 else 0
            better = "ACO" if a_val > g_val else "Greedy" if g_val > a_val else "Tie"
            
        report_data.append({
            "Metric": name,
            "Greedy": round(g_val, 4),
            "ACO": round(a_val, 4),
            "Difference": round(diff, 4),
            "Improvement (%)": round(imp_pct, 2),
            "Better Algorithm": better
        })
        
    report_df = pd.DataFrame(report_data)
    
    # Save reports
    report_df.to_csv(os.path.join(out_dir, "percentage_difference_table.csv"), index=False)
    report_df.to_excel(os.path.join(out_dir, "percentage_difference_table.xlsx"), index=False)
    
    with open(os.path.join(out_dir, "percentage_difference_report.md"), "w", encoding="utf-8") as f:
        f.write("# Routing Algorithm Comparison Report\n\n")
        f.write(report_df.to_markdown(index=False))
        
    # Generate side-by-side bar charts for every metric
    colors = ['#1f77b4', '#ff7f0e']
    
    for key, (name, objective, g_col, a_col) in metrics_map.items():
        if g_col not in df.columns or a_col not in df.columns: continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        g_val = df[g_col].mean()
        a_val = df[a_col].mean()
        
        bars = ax.bar(['Greedy', 'ACO'], [g_val, a_val], color=colors, width=0.5, edgecolor='black')
        ax.set_ylabel(name, fontweight='bold')
        ax.set_title(f"Average {name} Comparison", fontweight='bold')
        add_value_labels(ax)
        
        # Add legend for clarity
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[0], edgecolor='black', label='Greedy'),
                           Patch(facecolor=colors[1], edgecolor='black', label='ACO')]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_comparison_{key}.png"), dpi=300)
        plt.close()
        
    # Percentage Improvement Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = report_df['Metric'].tolist()
    imp_vals = report_df['Improvement (%)'].tolist()
    
    bars = ax.bar(metrics_names, imp_vals, color=['#2ca02c' if x >= 0 else '#d62728' for x in imp_vals], edgecolor='black')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_ylabel("Improvement (%)", fontweight='bold')
    ax.set_title("ACO Percentage Improvement over Greedy", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    add_value_labels(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_percentage_improvement.png"), dpi=300)
    plt.close()
    
    # Final Summary Dashboard (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Greedy vs ACO Routing Performance Dashboard', fontsize=18, fontweight='bold')
    
    dash_metrics = [
        ("distance_km", "Distance (km)"),
        ("duration_min", "Travel Time (min)"),
        ("fuel_l", "Fuel (L)"),
        ("co2_kg", "CO2 (kg)")
    ]
    
    for i, (key, title) in enumerate(dash_metrics):
        ax = axes[i//2, i%2]
        if key not in metrics_map:
            continue
            
        _, _, g_col, a_col = metrics_map[key]
        if g_col not in df.columns or a_col not in df.columns:
            continue
            
        g_val = df[g_col].mean()
        a_val = df[a_col].mean()
        
        bars = ax.bar(['Greedy', 'ACO'], [g_val, a_val], color=colors, width=0.5, edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Value")
        add_value_labels(ax)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "dashboard_summary.png"), dpi=300)
    plt.close()
    
    print("All publication-quality figures and reports generated successfully.")

if __name__ == "__main__":
    main()
