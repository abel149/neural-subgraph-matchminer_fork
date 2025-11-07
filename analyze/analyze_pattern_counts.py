import networkx as nx
import argparse
import json
import numpy as np
import pickle
from scipy.stats import ttest_rel, ttest_ind
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

def format_log_ticks(x, pos):
    if x >= 1e9:
        return f'{x/1e9:.0f}B'
    elif x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    elif x >= 1:
        return f'{int(x)}'
    else:
        return f'{x:.1f}'

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--counts_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.set_defaults(counts_path="results/counts.json")
    parser.set_defaults(out_path="results/analysis.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    all_counts = {}
    for fn in os.listdir(args.counts_path):
        if not fn.endswith(".json"): 
            continue

        with open(os.path.join(args.counts_path, fn), "r") as f:
            graphlet_lens, n_matches, n_matches_bl = json.load(f)
            name = fn[:-5]
            all_counts[name] = graphlet_lens, n_matches

    all_labels, all_xs, all_ys, all_ub_ys, all_lb_ys = [], [], [], [], []
    all_raw_ys, all_raw_ub_ys, all_raw_lb_ys = [], [], []  
    
    for name, (sizes, counts) in all_counts.items():
        all_labels.append(name)

        matches_by_size = defaultdict(list)
        for i in range(len(sizes)):
            matches_by_size[sizes[i]].append(counts[i])

        ys_raw = []
        ub_ys_raw, lb_ys_raw = [], []
        
        for size in sorted(matches_by_size.keys()):
            values = matches_by_size[size]
            
            a, b = np.percentile(values, [25, 75])
            median_val = np.median(values)
            
            ys_raw.append(median_val)
            ub_ys_raw.append(b)
            lb_ys_raw.append(a)

        all_raw_ys.append(ys_raw)
        all_raw_ub_ys.append(ub_ys_raw)
        all_raw_lb_ys.append(lb_ys_raw)
        
        all_xs.append(list(sorted(matches_by_size.keys())))
        all_ys.append(ys_raw)  
        all_ub_ys.append(ub_ys_raw)
        all_lb_ys.append(lb_ys_raw)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_labels)))
    
    for i in range(len(all_xs)):
        line = plt.plot(all_xs[i], all_ys[i], label=all_labels[i], 
                       marker="o", linewidth=2.5, markersize=8, 
                       color=colors[i])
        
        plt.fill_between(all_xs[i], all_lb_ys[i], all_ub_ys[i], 
                        alpha=0.2, color=colors[i])
        
        for j, (x, y_raw) in enumerate(zip(all_xs[i], all_raw_ys[i])):
            if y_raw >= 1e6:
                annotation = f'{y_raw/1e6:.1f}M'
            elif y_raw >= 1e3:
                annotation = f'{y_raw/1e3:.1f}K'
            elif y_raw >= 1:
                annotation = f'{int(y_raw)}'
            else:
                annotation = f'{y_raw:.2f}'
                
            plt.annotate(annotation, (x, y_raw), 
                        textcoords="offset points", xytext=(0, 15), 
                        ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 edgecolor=colors[i], alpha=0.8))

    plt.xlabel("Graph Size", fontsize=14, fontweight='bold')
    plt.ylabel("Frequency", fontsize=14, fontweight='bold')
    plt.title("Pattern Counts by Graph Size (Log Scale)", fontsize=16, fontweight='bold')
    
    plt.yscale("log")
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_log_ticks))
    
    plt.grid(True, which="major", alpha=0.6, linestyle='-', linewidth=0.8)
    plt.grid(True, which="minor", alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.legend(fontsize=12, frameon=True, shadow=True, fancybox=True)
    
    all_values = [val for ys in all_ys for val in ys if val > 0]
    if all_values:
        min_val, max_val = min(all_values), max(all_values)
        plt.ylim(min_val * 0.5, max_val * 2)
    
    plt.tight_layout()
    plt.savefig("plots/pattern-counts.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

    print("\n" + "="*70)
    print("EXACT COUNT SUMMARY")
    print("="*70)
    
    for i, label in enumerate(all_labels):
        print(f"\n{label.upper()}:")
        print("-" * 40)
        for j, size in enumerate(all_xs[i]):
            median_val = all_raw_ys[i][j]
            q25, q75 = all_raw_lb_ys[i][j], all_raw_ub_ys[i][j]
            
            print(f"  Size {size:2d}: {median_val:>10,.0f} "
                  f"(IQR: {q25:>8,.0f} - {q75:>8,.0f})")
    
    all_medians = [val for ys in all_raw_ys for val in ys]
    if all_medians:
        print(f"\nDATA CHARACTERISTICS:")
        print(f"  Range: {min(all_medians):,.0f} to {max(all_medians):,.0f}")
        print(f"  Ratio (max/min): {max(all_medians)/min(all_medians):,.1f}")
        print(f"  Orders of magnitude: {np.log10(max(all_medians)/min(all_medians)):.1f}")
        print(f"  → Log scale is {'ESSENTIAL' if max(all_medians)/min(all_medians) > 100 else 'RECOMMENDED'}")