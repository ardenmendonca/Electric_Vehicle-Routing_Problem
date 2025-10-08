#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Read data
df = pd.read_csv('logs/run_instance_1_T2000_A0.980.csv')

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('EVRP SA Analysis - instance Run 1\nT_init=2000, α=0.98', fontsize=16)

# 1. Fitness vs Evaluations
ax1.plot(df['Evaluation'], df['Best_Fitness'], 'b-', linewidth=2, label='Best Fitness', alpha=0.8)
ax1.plot(df['Evaluation'], df['Current_Fitness'], 'r-', linewidth=1, alpha=0.6, label='Current Fitness')
ax1.set_xlabel('Evaluations')
ax1.set_ylabel('Fitness')
ax1.set_title('Fitness Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add slope calculation for best fitness
if len(df) > 100:
    # Calculate slope over different windows
    window_sizes = [100, 500, 1000]
    colors = ['green', 'orange', 'purple']
    
    for i, (window, color) in enumerate(zip(window_sizes, colors)):
        if len(df) > window:
            slopes = []
            evals = []
            for j in range(window, len(df), window//4):
                x = df['Evaluation'][j-window:j].values
                y = df['Best_Fitness'][j-window:j].values
                if len(x) > 10:  # Ensure enough points
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                    evals.append(x[-1])
            
            if slopes:
                ax1_twin = ax1.twinx()
                ax1_twin.plot(evals, slopes, '--', color=color, alpha=0.7, 
                             label=f'Slope (window={window})')
                ax1_twin.set_ylabel('Fitness Slope', color=color)
                ax1_twin.tick_params(axis='y', labelcolor=color)
                break

# 2. Temperature vs Evaluations  
ax2.semilogy(df['Evaluation'], df['Temperature'], 'g-', linewidth=2)
ax2.set_xlabel('Evaluations')
ax2.set_ylabel('Temperature (log scale)')
ax2.set_title('Temperature Schedule')
ax2.grid(True, alpha=0.3)

# 3. Acceptance Rate Analysis
window_size = max(100, len(df) // 50)
acceptance_rate = df['Accepted'].rolling(window=window_size, center=True).mean()
ax3.plot(df['Evaluation'], acceptance_rate, 'purple', linewidth=2)
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% line')
ax3.set_xlabel('Evaluations')
ax3.set_ylabel('Acceptance Rate')
ax3.set_title(f'Acceptance Rate (rolling window = {window_size})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Improvement Distribution
improvements = df[df['Improvement'] != 0]['Improvement']
if len(improvements) > 0:
    ax4.hist(improvements, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax4.set_xlabel('Fitness Improvement')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Fitness Changes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/run_instance_1_T2000_A0.980_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional detailed analysis
print("=== Run Analysis ===")
print(f"Total Evaluations: {len(df)}")
print(f"Initial Fitness: {df['Best_Fitness'].iloc[0]:.6f}")
print(f"Final Fitness: {df['Best_Fitness'].iloc[-1]:.6f}")
print(f"Total Improvement: {df['Best_Fitness'].iloc[0] - df['Best_Fitness'].iloc[-1]:.6f}")
print(f"Average Acceptance Rate: {df['Accepted'].mean():.3f}")
print(f"Final Temperature: {df['Temperature'].iloc[-1]:.6f}")
print(f"Runtime: {df['Time_Elapsed'].iloc[-1]:.2f} seconds")

# Find improvement phases
best_improvements = df[df['Improvement'] > 0]['Improvement']
if len(best_improvements) > 0:
    print(f"Number of improvements: {len(best_improvements)}")
    print(f"Average improvement: {best_improvements.mean():.6f}")
    print(f"Largest improvement: {best_improvements.max():.6f}")
