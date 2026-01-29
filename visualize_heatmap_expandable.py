#!/usr/bin/env python3
"""
Expandable Heatmap Visualization for Multi-Benchmark Evaluation

This design scales to handle:
- Multiple benchmarks (easily add new ones)
- Many questions per benchmark
- Multiple methods
- Multiple score types

The heatmap groups questions by benchmark and uses visual separators
for easy navigation across different benchmark types.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle

# Read data
csv_path = Path(__file__).parent / "question_open_ended.csv"
df = pd.read_csv(csv_path)

# Average across n_times_idx runs
df_avg = df.groupby(['method', 'benchmark', 'question_id'])[
    ['score_sbert', 'score_bertscore', 'score_rougel', 'score_bleu']
].mean().reset_index()

score_info = [
    ('score_sbert', 'SBERT'),
    ('score_bertscore', 'BERTScore'),
    ('score_rougel', 'ROUGE-L'),
    ('score_bleu', 'BLEU')
]

methods = sorted(df_avg['method'].unique())
benchmarks = sorted(df_avg['benchmark'].unique())

print("="*80)
print("EXPANDABLE HEATMAP VISUALIZATION")
print("="*80)
print(f"Benchmarks: {len(benchmarks)} → {benchmarks}")
print(f"Total questions: {df_avg['question_id'].nunique()}")
print(f"Methods: {len(methods)}")
print(f"Score types: {len(score_info)}")
print("\nThis design scales automatically to any number of benchmarks!")

# ============================================================================
# VISUALIZATION 1: One heatmap per score type (grouped by benchmark)
# ============================================================================
print("\n" + "="*80)
print("OPTION 1: Separate Heatmap per Score Type (with benchmark grouping)")
print("="*80)

# Create row labels with benchmark grouping
df_avg['row_label'] = df_avg['benchmark'] + ' | ' + df_avg['question_id']

# Sort by benchmark then question_id for grouping
df_avg_sorted = df_avg.sort_values(['benchmark', 'question_id'])
row_order = df_avg_sorted['row_label'].unique()

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, max(12, len(row_order) * 0.4)))
axes = axes.flatten()

for idx, (score_col, score_label) in enumerate(score_info):
    ax = axes[idx]
    
    # Pivot data
    heatmap_data = df_avg.pivot(index='row_label', columns='method', values=score_col)
    heatmap_data = heatmap_data.reindex(row_order)  # Maintain benchmark grouping
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', 
                cmap='RdYlGn', vmin=0, vmax=1, 
                ax=ax, cbar_kws={'label': 'Score'},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(f'{score_label} Scores', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Benchmark | Question', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Add horizontal lines to separate benchmarks
    current_benchmark = None
    y_pos = 0
    for label in row_order:
        benchmark = label.split(' | ')[0]
        if current_benchmark is not None and benchmark != current_benchmark:
            # Draw separator line
            ax.axhline(y=y_pos, color='black', linewidth=3, zorder=10)
        current_benchmark = benchmark
        y_pos += 1

plt.suptitle('Method Performance by Score Type (Grouped by Benchmark)\n' +
             'Darker green = better, Dark red = worse | Black lines separate benchmarks',
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path = Path(__file__).parent / "heatmap_by_score.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# VISUALIZATION 2: One large comprehensive heatmap (all scores combined)
# ============================================================================
print("\n" + "="*80)
print("OPTION 2: Single Comprehensive Heatmap (all data at once)")
print("="*80)

# Reshape data to have method+score as columns
df_melted = df_avg.melt(
    id_vars=['method', 'benchmark', 'question_id', 'row_label'],
    value_vars=['score_sbert', 'score_bertscore', 'score_rougel', 'score_bleu'],
    var_name='score_type',
    value_name='score'
)

# Create column labels
score_name_map = {
    'score_sbert': 'SBERT',
    'score_bertscore': 'BERT',
    'score_rougel': 'ROUGE',
    'score_bleu': 'BLEU'
}
df_melted['col_label'] = df_melted['method'].str.replace('gpt-5.2_', 'GPT-5.2_').str.replace('daisenbot_', 'Daisen_') + '\n' + df_melted['score_type'].map(score_name_map)

# Pivot for comprehensive heatmap
df_melted_sorted = df_melted.sort_values(['benchmark', 'question_id'])
comprehensive_data = df_melted.pivot(index='row_label', columns='col_label', values='score')
comprehensive_data = comprehensive_data.reindex(row_order)

# Sort columns to group by method
col_order = sorted(comprehensive_data.columns, key=lambda x: (x.split('\n')[0], x.split('\n')[1]))
comprehensive_data = comprehensive_data[col_order]

# Create figure
fig, ax = plt.subplots(figsize=(18, max(14, len(row_order) * 0.5)))

sns.heatmap(comprehensive_data, annot=True, fmt='.3f',
            cmap='RdYlGn', vmin=0, vmax=1,
            ax=ax, cbar_kws={'label': 'Score', 'shrink': 0.8},
            linewidths=0.5, linecolor='lightgray')

ax.set_title('Comprehensive Performance Heatmap: All Methods × All Score Types\n' +
             '(Grouped by benchmark | Black lines separate benchmarks)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Method & Score Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Benchmark | Question ID', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=8)

# Add benchmark separators
current_benchmark = None
y_pos = 0
for label in row_order:
    benchmark = label.split(' | ')[0]
    if current_benchmark is not None and benchmark != current_benchmark:
        ax.axhline(y=y_pos, color='black', linewidth=4, zorder=10)
    current_benchmark = benchmark
    y_pos += 1

# Add vertical lines to separate methods
n_scores = 4
for i in range(1, len(methods)):
    x_pos = i * n_scores
    ax.axvline(x=x_pos, color='navy', linewidth=2, linestyle='--', alpha=0.7, zorder=10)

plt.tight_layout()
output_path = Path(__file__).parent / "heatmap_comprehensive.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# VISUALIZATION 3: Benchmark-focused view (one heatmap per benchmark)
# ============================================================================
print("\n" + "="*80)
print("OPTION 3: One Heatmap per Benchmark (best for many benchmarks)")
print("="*80)

for benchmark in benchmarks:
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    
    # Create 2x2 grid for this benchmark
    fig, axes = plt.subplots(2, 2, figsize=(14, max(8, len(benchmark_data['question_id'].unique()) * 0.6)))
    axes = axes.flatten()
    
    for idx, (score_col, score_label) in enumerate(score_info):
        ax = axes[idx]
        
        # Pivot
        heatmap_data = benchmark_data.pivot(index='question_id', columns='method', values=score_col)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f',
                    cmap='RdYlGn', vmin=0, vmax=1,
                    ax=ax, cbar_kws={'label': 'Score'},
                    linewidths=1, linecolor='white')
        
        ax.set_title(f'{score_label}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Method', fontsize=10)
        ax.set_ylabel('Question ID', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.suptitle(f'Benchmark: {benchmark.upper()}\n' +
                 f'Performance across all methods and score types ({len(heatmap_data)} questions)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / f"heatmap_benchmark_{benchmark}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS BY BENCHMARK
# ============================================================================
print("\n" + "="*80)
print("BENCHMARK COMPARISON SUMMARY")
print("="*80)

for benchmark in benchmarks:
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark.upper()}")
    print(f"{'='*80}")
    print(f"Questions: {benchmark_data['question_id'].nunique()}")
    
    # Average performance per method across all scores
    for method in methods:
        method_data = benchmark_data[benchmark_data['method'] == method]
        if len(method_data) > 0:
            avg_scores = method_data[['score_sbert', 'score_bertscore', 'score_rougel', 'score_bleu']].mean()
            overall_avg = avg_scores.mean()
            print(f"\n  {method}:")
            print(f"    Overall Average: {overall_avg:.4f}")
            for score_col, score_label in score_info:
                print(f"    {score_label:12s}: {method_data[score_col].mean():.4f}")

# Overall winner by benchmark and score type
print("\n" + "="*80)
print("WINNERS BY BENCHMARK & SCORE TYPE")
print("="*80)

for benchmark in benchmarks:
    print(f"\n{benchmark.upper()}:")
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    
    for score_col, score_label in score_info:
        # Average across all questions for this benchmark
        method_avg = benchmark_data.groupby('method')[score_col].mean().sort_values(ascending=False)
        winner = method_avg.index[0]
        winner_score = method_avg.iloc[0]
        print(f"  {score_label:12s}: {winner:30s} ({winner_score:.4f})")

# ============================================================================
# EXPANDABILITY GUIDE
# ============================================================================
print("\n" + "="*80)
print("EXPANDABILITY GUIDE")
print("="*80)
print("""
✅ TO ADD NEW BENCHMARKS:
   Just add rows to your CSV with new benchmark names.
   The heatmaps will automatically scale!

✅ TO ADD NEW METHODS:
   Add new method column entries in CSV.
   Heatmaps will add new columns automatically.

✅ TO ADD NEW SCORE TYPES:
   1. Add new score_* column to CSV
   2. Update score_info list in this script
   3. Heatmaps will automatically include new scores

✅ RECOMMENDED LAYOUTS:
   - Few benchmarks (1-3): Use OPTION 1 or 2 (combined view)
   - Many benchmarks (4+): Use OPTION 3 (separate per benchmark)
   - For presentations: Use OPTION 3 (cleaner, focused)
   - For overview: Use OPTION 2 (everything at once)

✅ HEATMAP ADVANTAGES:
   - Scales to 100+ questions (just scrolls vertically)
   - Easy to spot patterns (color = instant understanding)
   - Publication-ready (clear, professional)
   - Handles missing data gracefully (leaves cells empty/gray)
   - Can add annotations (stars, circles for significance)

📊 FILES GENERATED:
   - heatmap_by_score.png          → 4 separate score heatmaps
   - heatmap_comprehensive.png     → Single unified view
   - heatmap_benchmark_*.png       → One file per benchmark
""")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
