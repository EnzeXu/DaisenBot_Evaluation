#!/usr/bin/env python3
"""
Multiple Choice Evaluation Visualization - Heatmap

For multiple choice questions with binary scores (1=correct, 0=incorrect),
we average across n_times_idx to get accuracy rates (0.0 to 1.0).

This creates expandable heatmaps showing method accuracy across benchmarks and questions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Read data
csv_path = Path(__file__).parent / "question_multiple_choice.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("MULTIPLE CHOICE EVALUATION - HEATMAP VISUALIZATION")
print("="*80)
print(f"Total rows: {len(df)}")
print(f"Unique benchmarks: {df['benchmark'].nunique()} → {sorted(df['benchmark'].unique())}")
print(f"Unique questions: {df['question_id'].nunique()}")
print(f"Methods: {sorted(df['method'].unique())}")
print(f"n_times_idx range: {df['n_times_idx'].min()} to {df['n_times_idx'].max()}")
print(f"\nScore values: {sorted(df['score'].unique())} (1=correct, 0=incorrect)")

# Average scores across n_times_idx runs to get accuracy rate
print("\n" + "="*80)
print("COMPUTING ACCURACY RATES (average across n_times_idx)")
print("="*80)
df_avg = df.groupby(['method', 'benchmark', 'question_id'])['score'].mean().reset_index()
df_avg.rename(columns={'score': 'accuracy'}, inplace=True)

print(f"After averaging: {len(df_avg)} unique (method, benchmark, question) combinations")
print(f"\nAccuracy interpretation:")
print(f"  1.0 = correct on all runs (3/3)")
print(f"  0.67 = correct on 2/3 runs")
print(f"  0.33 = correct on 1/3 runs")
print(f"  0.0 = incorrect on all runs (0/3)")

# Show example
print(f"\nExample accuracy calculation:")
example_q = df_avg['question_id'].iloc[0]
example_method = df_avg['method'].iloc[0]
example_data = df[(df['question_id'] == example_q) & (df['method'] == example_method)]
print(f"\n{example_method} on {example_q}:")
print(f"  Run 0: {example_data[example_data['n_times_idx']==0]['score'].values[0]:.1f}")
print(f"  Run 1: {example_data[example_data['n_times_idx']==1]['score'].values[0]:.1f}")
print(f"  Run 2: {example_data[example_data['n_times_idx']==2]['score'].values[0]:.1f}")
avg_acc = df_avg[(df_avg['question_id']==example_q) & (df_avg['method']==example_method)]['accuracy'].values[0]
print(f"  → Accuracy: {avg_acc:.3f} ({int(avg_acc*3)}/3 correct)")

# Create row labels with benchmark grouping
df_avg['row_label'] = df_avg['benchmark'] + ' | ' + df_avg['question_id']

# Sort by benchmark then question_id
df_avg_sorted = df_avg.sort_values(['benchmark', 'question_id'])
row_order = df_avg_sorted['row_label'].unique()

methods = sorted(df_avg['method'].unique())
benchmarks = sorted(df_avg['benchmark'].unique())

# ============================================================================
# VISUALIZATION 1: Single heatmap (all methods)
# ============================================================================
print("\n" + "="*80)
print("OPTION 1: Single Accuracy Heatmap (all methods)")
print("="*80)

fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2), max(12, len(row_order) * 0.5)))

# Pivot data
heatmap_data = df_avg.pivot(index='row_label', columns='method', values='accuracy')
heatmap_data = heatmap_data.reindex(row_order)

# Create heatmap with custom colormap (red=0%, green=100%)
sns.heatmap(heatmap_data, annot=True, fmt='.2f', 
            cmap='RdYlGn', vmin=0, vmax=1, 
            ax=ax, cbar_kws={'label': 'Accuracy Rate'},
            linewidths=1, linecolor='white')

ax.set_title('Multiple Choice Accuracy by Method\n' +
             '(1.0 = always correct, 0.0 = always incorrect | Black lines separate benchmarks)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Benchmark | Question', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=9)

# Add horizontal lines to separate benchmarks
current_benchmark = None
y_pos = 0
for label in row_order:
    benchmark = label.split(' | ')[0]
    if current_benchmark is not None and benchmark != current_benchmark:
        ax.axhline(y=y_pos, color='black', linewidth=3, zorder=10)
    current_benchmark = benchmark
    y_pos += 1

plt.tight_layout()
output_path = Path(__file__).parent / "multiple_choice_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# VISUALIZATION 2: One heatmap per benchmark
# ============================================================================
print("\n" + "="*80)
print("OPTION 2: One Heatmap per Benchmark")
print("="*80)

for benchmark in benchmarks:
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 
                                     max(6, len(benchmark_data['question_id'].unique()) * 0.8)))
    
    # Pivot
    heatmap_data = benchmark_data.pivot(index='question_id', columns='method', values='accuracy')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Accuracy'},
                linewidths=1, linecolor='white')
    
    ax.set_title(f'Benchmark: {benchmark.upper()}\n' +
                 f'Multiple Choice Accuracy ({len(heatmap_data)} questions)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Question ID', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f"multiple_choice_heatmap_{benchmark}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("OVERALL ACCURACY BY METHOD")
print("="*80)

for method in sorted(methods):
    method_data = df_avg[df_avg['method'] == method]
    avg_acc = method_data['accuracy'].mean()
    perfect_count = (method_data['accuracy'] == 1.0).sum()
    zero_count = (method_data['accuracy'] == 0.0).sum()
    total_questions = len(method_data)
    
    print(f"\n{method}:")
    print(f"  Overall Accuracy: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
    print(f"  Perfect (3/3):    {perfect_count}/{total_questions} questions ({perfect_count/total_questions*100:.1f}%)")
    print(f"  Failed (0/3):     {zero_count}/{total_questions} questions ({zero_count/total_questions*100:.1f}%)")
    print(f"  Partial correct:  {total_questions - perfect_count - zero_count}/{total_questions} questions")

# Accuracy by benchmark
print("\n" + "="*80)
print("ACCURACY BY BENCHMARK")
print("="*80)

for benchmark in sorted(benchmarks):
    print(f"\n{benchmark.upper()}:")
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    
    for method in sorted(methods):
        method_data = benchmark_data[benchmark_data['method'] == method]
        if len(method_data) > 0:
            avg_acc = method_data['accuracy'].mean()
            perfect = (method_data['accuracy'] == 1.0).sum()
            total = len(method_data)
            print(f"  {method:30s}: {avg_acc:.3f} ({avg_acc*100:.1f}%) | {perfect}/{total} perfect")

# Winner by benchmark
print("\n" + "="*80)
print("BEST METHOD PER BENCHMARK")
print("="*80)

for benchmark in sorted(benchmarks):
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    method_avg = benchmark_data.groupby('method')['accuracy'].mean().sort_values(ascending=False)
    winner = method_avg.index[0]
    winner_acc = method_avg.iloc[0]
    print(f"{benchmark:15s}: {winner:30s} ({winner_acc:.3f} = {winner_acc*100:.1f}%)")

# Most difficult questions (lowest average accuracy across methods)
print("\n" + "="*80)
print("MOST DIFFICULT QUESTIONS (lowest average accuracy)")
print("="*80)

question_difficulty = df_avg.groupby(['benchmark', 'question_id'])['accuracy'].mean().sort_values()
print("\nTop 10 hardest questions:")
for i, ((benchmark, q_id), avg_acc) in enumerate(question_difficulty.head(10).items(), 1):
    # Get question text
    q_text = df[(df['benchmark']==benchmark) & (df['question_id']==q_id)]['q_text'].iloc[0]
    q_text_short = q_text.split('\n')[0][:60] + '...'
    print(f"{i:2d}. {benchmark} - {q_id}: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
    print(f"    {q_text_short}")

# Easiest questions
print("\n" + "="*80)
print("EASIEST QUESTIONS (highest average accuracy)")
print("="*80)

print("\nTop 10 easiest questions:")
for i, ((benchmark, q_id), avg_acc) in enumerate(question_difficulty.tail(10).items(), 1):
    q_text = df[(df['benchmark']==benchmark) & (df['question_id']==q_id)]['q_text'].iloc[0]
    q_text_short = q_text.split('\n')[0][:60] + '...'
    print(f"{i:2d}. {benchmark} - {q_id}: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
    print(f"    {q_text_short}")

# Consistency analysis (variance in accuracy)
print("\n" + "="*80)
print("CONSISTENCY ANALYSIS")
print("="*80)
print("(Methods with low variance are more consistent across questions)\n")

for method in sorted(methods):
    method_data = df_avg[df_avg['method'] == method]
    variance = method_data['accuracy'].var()
    std = method_data['accuracy'].std()
    print(f"{method:30s}: std={std:.4f}, var={variance:.4f}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
