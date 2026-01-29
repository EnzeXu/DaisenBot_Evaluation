#!/usr/bin/env python3
"""
Box-and-Whisker Plot Visualization (Method-Averaged Design)

For each (benchmark, question) cluster:
- Shows 4 boxes (one per score type: SBERT, BERTScore, ROUGE-L, BLEU)
- Each box contains 3 data points (one per method, averaged across n_times_idx runs)
- This shows the distribution/spread of methods for each score type

This answers: "For this question and score type, how do the different methods compare?"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the CSV file
csv_path = Path(__file__).parent / "question_open_ended.csv"
df = pd.read_csv(csv_path)

# Display basic information
print("="*80)
print("BOX PLOT DESIGN: Methods Averaged to Show Score Type Distribution")
print("="*80)
print(f"Total rows: {len(df)}")
print(f"\nUnique benchmarks: {df['benchmark'].nunique()}")
print(f"Unique questions: {df['question_id'].nunique()}")
print(f"Methods: {df['method'].nunique()} → {sorted(df['method'].unique())}")
print(f"n_times_idx range: {df['n_times_idx'].min()} to {df['n_times_idx'].max()}")

# Average scores across n_times_idx for each method/benchmark/question
print("\n" + "="*80)
print("AVERAGING ACROSS n_times_idx REPETITIONS")
print("="*80)
grouped = df.groupby(['method', 'benchmark', 'question_id'])[
    ['score_sbert', 'score_bertscore', 'score_rougel', 'score_bleu']
].mean().reset_index()

print(f"After averaging: {len(grouped)} unique (method, benchmark, question) combinations")

# Create clusters: benchmark + question_id
grouped['cluster'] = grouped['benchmark'] + ' - ' + grouped['question_id']

# Get unique clusters in order
clusters = sorted(grouped['cluster'].unique())
print(f"\nNumber of clusters: {len(clusters)}")
print(f"First 5 clusters: {list(clusters[:5])}")

# Show what each box will contain
print(f"\n{'='*80}")
print("BOX PLOT STRUCTURE")
print(f"{'='*80}")
print(f"Each cluster = 1 benchmark-question pair")
print(f"Each cluster shows 4 boxes (one per score type)")
print(f"Each box contains {len(grouped['method'].unique())} data points (one per method, averaged)")
print(f"\nExample for first cluster:")
example_cluster = clusters[0]
example_data = grouped[grouped['cluster'] == example_cluster]
print(f"\n{example_cluster}:")
print(f"  Methods in this cluster: {len(example_data)}")
for _, row in example_data.iterrows():
    print(f"    {row['method']:30s} → SBERT:{row['score_sbert']:.4f} BERT:{row['score_bertscore']:.4f} "
          f"ROUGE:{row['score_rougel']:.4f} BLEU:{row['score_bleu']:.4f}")
print(f"\n  So the SBERT box will contain these {len(example_data)} values:")
for _, row in example_data.iterrows():
    print(f"    {row['method']:30s} → {row['score_sbert']:.4f}")

# Prepare data for plotting
score_columns = ['score_sbert', 'score_bertscore', 'score_rougel', 'score_bleu']
score_labels = ['SBERT', 'BERTScore', 'ROUGE-L', 'BLEU']

# Create a figure with box plots
fig, ax = plt.subplots(figsize=(max(20, len(clusters) * 2), 8))

# Number of clusters and score types
n_clusters = len(clusters)
n_scores = len(score_columns)

# Position for each box
positions = []
tick_positions = []
tick_labels = []

current_pos = 0
cluster_spacing = 1.2  # Space between clusters
box_spacing = 0.25     # Space between boxes within a cluster

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Blue, Red, Green, Orange

print(f"\n{'='*80}")
print("GENERATING BOX PLOTS")
print(f"{'='*80}")

# Collect all box plot data
for i, cluster in enumerate(clusters):
    cluster_data = grouped[grouped['cluster'] == cluster]
    
    # Start position for this cluster
    cluster_start = current_pos
    
    for j, (score_col, color, score_label) in enumerate(zip(score_columns, colors, score_labels)):
        # Get all method scores for this score type
        # This creates the distribution showing how methods differ
        scores = cluster_data[score_col].values
        pos = current_pos + j * box_spacing
        positions.append(pos)
        
        if i == 0 and j == 0:
            print(f"\nFirst box ({cluster} - {score_label}):")
            print(f"  Contains {len(scores)} method scores: {scores}")
            print(f"  Methods: {cluster_data['method'].tolist()}")
        
        # Create box plot for this score type
        bp = ax.boxplot([scores], positions=[pos], widths=0.18,
                        patch_artist=True,
                        boxprops={'facecolor': color, 'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 1},
                        medianprops={'color': 'black', 'linewidth': 2},
                        whiskerprops={'color': 'black', 'linewidth': 1},
                        capprops={'color': 'black', 'linewidth': 1},
                        showfliers=True,
                        flierprops={'marker': 'o', 'markersize': 6, 
                                   'markerfacecolor': color, 'markeredgecolor': 'black',
                                   'alpha': 0.6})
        
        # Overlay individual method points with labels
        for idx, (score, method) in enumerate(zip(scores, cluster_data['method'].values)):
            # Add jitter to x position for visibility
            x_jitter = pos + (idx - len(scores)/2 + 0.5) * 0.03
            ax.scatter(x_jitter, score, alpha=0.8, s=80, c='white', 
                      edgecolors='black', linewidths=2, zorder=3)
            # Add method initial as label
            method_initial = method[0].upper()  # First letter of method name
            ax.text(x_jitter, score, method_initial, ha='center', va='center',
                   fontsize=7, fontweight='bold', zorder=4)
    
    # Mark the center of this cluster for labeling
    cluster_center = cluster_start + (n_scores - 1) * box_spacing / 2
    tick_positions.append(cluster_center)
    tick_labels.append(cluster.replace(' - ', '\n'))
    
    # Move to next cluster
    current_pos += (n_scores - 1) * box_spacing + cluster_spacing

# Set labels and title
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Benchmark - Question ID', fontsize=14, fontweight='bold')
ax.set_title('Method Performance Distribution by Score Type\n' +
             'Each cluster = 1 question | Each box = 1 score type showing distribution across 3 methods\n' +
             '(Methods averaged across n_times_idx runs | Points labeled: D=Daisen, G=GPT)',
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, alpha=0.7, edgecolor='black', label=label) 
                   for color, label in zip(colors, score_labels)]
legend_elements.append(Patch(facecolor='white', edgecolor='black', 
                             label=f'Each box: {len(grouped["method"].unique())} methods'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
          title='Score Types', title_fontsize=12, framealpha=0.95)

# Add grid for readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = Path(__file__).parent / "boxplot_method_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"✓ Visualization saved to: {output_path}")
print(f"{'='*80}")

# Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
HOW TO READ THIS PLOT:

1. STRUCTURE:
   - X-axis: Benchmark-Question pairs (clusters)
   - Y-axis: Score values (0-1)
   - Each cluster has 4 boxes (blue, red, green, orange)

2. EACH BOX SHOWS:
   - Median (thick black line)
   - Quartiles (box boundaries) 
   - Range (whiskers)
   - Individual method scores (labeled points: D=Daisen, G=GPT)

3. WHAT IT TELLS YOU:
   - Wide box → methods disagree on this score type
   - Narrow box → methods agree (similar performance)
   - High median → all methods do well on this score type
   - Low median → all methods struggle
   - Outliers → one method performs very differently

4. COMPARE:
   - Across boxes in same cluster → which score type is harder?
   - Same color boxes across clusters → consistency of that score
   - Point positions within boxes → which method is best?

EXAMPLE:
  If SBERT box is high and narrow → all methods score well and similarly
  If BLEU box is low and wide → methods disagree, some do poorly
""")

# Print statistics per cluster
print("\n" + "="*80)
print("PER-CLUSTER STATISTICS (Score Type Comparison)")
print("="*80)

for cluster in clusters:
    cluster_data = grouped[grouped['cluster'] == cluster]
    print(f"\n{cluster}:")
    
    for score_col, score_label in zip(score_columns, score_labels):
        scores = cluster_data[score_col].values
        print(f"  {score_label:12s}: median={np.median(scores):.4f}, "
              f"mean={np.mean(scores):.4f}, "
              f"std={np.std(scores):.4f}, "
              f"range=[{scores.min():.4f}, {scores.max():.4f}]")

# Print which score types have highest variance
print("\n" + "="*80)
print("SCORE TYPE VARIANCE ANALYSIS")
print("="*80)
print("(High variance = methods disagree more on this score type)\n")

variance_by_score = {}
for score_col, score_label in zip(score_columns, score_labels):
    all_variances = []
    for cluster in clusters:
        cluster_data = grouped[grouped['cluster'] == cluster]
        scores = cluster_data[score_col].values
        all_variances.append(np.var(scores))
    avg_variance = np.mean(all_variances)
    variance_by_score[score_label] = avg_variance

# Sort by variance
sorted_variance = sorted(variance_by_score.items(), key=lambda x: x[1], reverse=True)
for rank, (score_label, variance) in enumerate(sorted_variance, 1):
    print(f"  {rank}. {score_label:12s}: avg variance = {variance:.6f} "
          f"(methods {'disagree more' if rank <= 2 else 'agree more'})")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
