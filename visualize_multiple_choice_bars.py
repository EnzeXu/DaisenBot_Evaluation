#!/usr/bin/env python3
"""
Multiple Choice Evaluation Visualization - Bar Charts

Shows accuracy rates as bar charts, making it easy to compare
method performance across benchmarks and questions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read data
csv_path = Path(__file__).parent / "question_multiple_choice.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("MULTIPLE CHOICE EVALUATION - BAR CHART VISUALIZATION")
print("="*80)

# Average scores across n_times_idx runs
df_avg = df.groupby(['method', 'benchmark', 'question_id'])['score'].mean().reset_index()
df_avg.rename(columns={'score': 'accuracy'}, inplace=True)

df_avg['cluster'] = df_avg['benchmark'] + ' - ' + df_avg['question_id']
clusters = sorted(df_avg['cluster'].unique())
methods = sorted(df_avg['method'].unique())
benchmarks = sorted(df_avg['benchmark'].unique())

print(f"Questions: {len(clusters)}")
print(f"Methods: {len(methods)}")
print(f"Benchmarks: {len(benchmarks)}")

# ============================================================================
# VISUALIZATION 1: Grouped bar chart (all questions)
# ============================================================================
print("\n" + "="*80)
print("OPTION 1: Grouped Bar Chart (all questions)")
print("="*80)

fig, ax = plt.subplots(figsize=(max(20, len(clusters) * 1.5), 8))

x = np.arange(len(clusters))
width = 0.25
colors = {'daisenbot_base': '#e74c3c', 
          'gpt-5.2_with_image': '#3498db', 
          'gpt-5.2_without_image': '#2ecc71'}

for m_idx, method in enumerate(methods):
    method_accuracies = []
    for cluster in clusters:
        cluster_data = df_avg[(df_avg['cluster'] == cluster) & (df_avg['method'] == method)]
        if len(cluster_data) > 0:
            method_accuracies.append(cluster_data['accuracy'].values[0])
        else:
            method_accuracies.append(0)
    
    offset = (m_idx - 1) * width
    bars = ax.bar(x + offset, method_accuracies, width,
                  label=method.replace('_', ' ').title(),
                  color=colors.get(method, '#999999'),
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold')
ax.set_xlabel('Benchmark - Question', fontsize=12, fontweight='bold')
ax.set_title('Multiple Choice Accuracy by Method\n' +
             '(1.0 = always correct, 0.0 = always incorrect)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([c.replace(' - ', '\n') for c in clusters], rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / "multiple_choice_bars.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# VISUALIZATION 2: Per-benchmark comparison
# ============================================================================
print("\n" + "="*80)
print("OPTION 2: Per-Benchmark Bar Charts")
print("="*80)

for benchmark in benchmarks:
    benchmark_data = df_avg[df_avg['benchmark'] == benchmark]
    questions = sorted(benchmark_data['question_id'].unique())
    
    fig, ax = plt.subplots(figsize=(max(12, len(questions) * 1.2), 6))
    
    x = np.arange(len(questions))
    width = 0.25
    
    for m_idx, method in enumerate(methods):
        accuracies = []
        for q_id in questions:
            q_data = benchmark_data[(benchmark_data['question_id'] == q_id) & 
                                   (benchmark_data['method'] == method)]
            if len(q_data) > 0:
                accuracies.append(q_data['accuracy'].values[0])
            else:
                accuracies.append(0)
        
        offset = (m_idx - 1) * width
        bars = ax.bar(x + offset, accuracies, width,
                      label=method.replace('_', ' ').title(),
                      color=colors.get(method, '#999999'),
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy Rate', fontsize=11, fontweight='bold')
    ax.set_xlabel('Question ID', fontsize=11, fontweight='bold')
    ax.set_title(f'Benchmark: {benchmark.upper()}\n' +
                 f'Multiple Choice Accuracy ({len(questions)} questions)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f"multiple_choice_bars_{benchmark}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# VISUALIZATION 3: Overall method comparison (single bar chart)
# ============================================================================
print("\n" + "="*80)
print("OPTION 3: Overall Method Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 6))

overall_accuracies = []
for method in methods:
    method_data = df_avg[df_avg['method'] == method]
    overall_accuracies.append(method_data['accuracy'].mean())

x = np.arange(len(methods))
bars = ax.bar(x, overall_accuracies, 
              color=[colors.get(m, '#999999') for m in methods],
              alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, acc in zip(bars, overall_accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{acc:.3f}\n({acc*100:.1f}%)', ha='center', va='bottom', 
           fontsize=12, fontweight='bold')

ax.set_ylabel('Overall Accuracy', fontsize=13, fontweight='bold')
ax.set_xlabel('Method', fontsize=13, fontweight='bold')
ax.set_title('Multiple Choice: Overall Method Performance\n' +
             f'(Averaged across {len(clusters)} questions)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').replace('gpt-5.2', 'GPT-5.2').replace('daisenbot', 'Daisen') 
                    for m in methods], fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / "multiple_choice_overall.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# Print summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for method in methods:
    method_data = df_avg[df_avg['method'] == method]
    avg_acc = method_data['accuracy'].mean()
    print(f"\n{method}:")
    print(f"  Overall Accuracy: {avg_acc:.3f} ({avg_acc*100:.1f}%)")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
