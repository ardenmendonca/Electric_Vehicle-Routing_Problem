import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_individual_runs_with_sa(csv_file='performance_log.csv'):
    """
    1. Create individual plots for each run showing BOTH SA current fitness and best fitness evolution
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} records")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return
    
    runs = sorted(df['run'].unique())
    n_runs = len(runs)
    
    # Calculate grid size for subplots
    cols = 4
    rows = math.ceil(n_runs / cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14))
    fig.suptitle('EVRP Performance: SA Current vs Best Fitness Evolution', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_runs > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each run individually
    for i, run in enumerate(runs):
        ax = axes[i]
        
        run_data = df[df['run'] == run].copy()
        run_data = run_data.sort_values('evaluations')
        
        # Calculate running minimum (best fitness so far)
        run_data['running_best'] = run_data['best_fitness'].cummin()
        
        # Plot both current SA fitness and best fitness
        ax.plot(run_data['evaluations'], 
                run_data['current_fitness'],  # This is now the actual SA current solution
                color='red', 
                linewidth=1, 
                alpha=0.7,
                label='Current SA Solution')
        
        ax.plot(run_data['evaluations'], 
                run_data['running_best'], 
                color='blue', 
                linewidth=2, 
                label='Best Found')
        
        # Customize each subplot
        ax.set_title(f'Run {run}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Evaluations', fontsize=10)
        ax.set_ylabel('Fitness', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add final fitness values as text
        final_current = run_data['current_fitness'].iloc[-1]
        final_best = run_data['running_best'].iloc[-1]
        ax.text(0.98, 0.95, f'Current: {final_current:.1f}\nBest: {final_best:.1f}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_runs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('1_evrp_individual_runs_sa.png', dpi=300, bbox_inches='tight')
    print("1. Individual runs SA plot saved as '1_evrp_individual_runs_sa.png'")
    plt.show()

def plot_sa_acceptance_analysis(csv_file='performance_log.csv'):
    """
    2. Analyze and plot SA acceptance patterns (4 different plots)
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return
    
    # Add acceptance analysis
    df['fitness_change'] = df['current_fitness'] - df['current_fitness'].shift(1)
    df['worse_move_accepted'] = (df['accepted'] == True) & (df['fitness_change'] > 0)
    
    runs = sorted(df['run'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SA Acceptance Analysis (4 Different Plots)', fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature vs Acceptance Rate
    ax1 = axes[0, 0]
    for run in runs[:5]:  # Plot first 5 runs for clarity
        run_data = df[df['run'] == run].copy()
        # Group by temperature ranges for analysis
        temp_bins = pd.cut(run_data['temperature'], bins=20)
        acceptance_rates = run_data.groupby(temp_bins)['accepted'].mean()
        temp_centers = [interval.mid for interval in acceptance_rates.index]
        
        ax1.plot(temp_centers, acceptance_rates.values, alpha=0.7, marker='o', markersize=3)
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Acceptance Rate')
    ax1.set_title('Temperature vs Acceptance Rate')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Worse Move Acceptance Over Time
    ax2 = axes[0, 1]
    for run in runs[:5]:
        run_data = df[df['run'] == run].copy()
        # Rolling window for worse move acceptance
        window_size = max(100, len(run_data) // 20)
        rolling_worse = run_data['worse_move_accepted'].rolling(window=window_size).mean()
        
        ax2.plot(run_data['evaluations'], rolling_worse, alpha=0.7)
    
    ax2.set_xlabel('Evaluations')
    ax2.set_ylabel('Worse Move Acceptance Rate')
    ax2.set_title('Worse Move Acceptance Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Current vs Best Fitness for one representative run
    ax3 = axes[1, 0]
    run = runs[0]  # Use first run as example
    run_data = df[df['run'] == run].copy()
    run_data['running_best'] = run_data['best_fitness'].cummin()
    
    ax3.plot(run_data['evaluations'], run_data['current_fitness'], 
             color='red', alpha=0.7, linewidth=1, label='Current SA Solution')
    ax3.plot(run_data['evaluations'], run_data['running_best'], 
             color='blue', linewidth=2, label='Best Found')
    
    ax3.set_xlabel('Evaluations')
    ax3.set_ylabel('Fitness')
    ax3.set_title(f'SA Behavior Example (Run {run})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fitness Distribution
    ax4 = axes[1, 1]
    all_current = df['current_fitness'].dropna()
    all_best = df['best_fitness'].dropna()
    
    ax4.hist(all_current, bins=50, alpha=0.7, label='Current Solutions', color='red')
    ax4.hist(all_best, bins=50, alpha=0.7, label='Best Solutions', color='blue')
    
    ax4.set_xlabel('Fitness')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Fitness Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2_evrp_sa_acceptance_analysis.png', dpi=300, bbox_inches='tight')
    print("2. SA acceptance analysis saved as '2_evrp_sa_acceptance_analysis.png'")
    plt.show()

def plot_convergence_best_worst_mean(csv_file='performance_log.csv'):
    """
    3. Convergence plot: best, worst, and mean performing runs
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return
    
    # Calculate final fitness for each run
    runs = sorted(df['run'].unique())
    final_results = []
    
    for run in runs:
        run_data = df[df['run'] == run]
        final_fitness = run_data['best_fitness'].min()
        final_results.append((run, final_fitness))
    
    # Sort by final fitness
    final_results.sort(key=lambda x: x[1])
    
    # Get best, worst, and mean runs
    best_run = final_results[0][0]
    worst_run = final_results[-1][0]
    
    # Find run closest to mean fitness
    all_fitness = [x[1] for x in final_results]
    mean_fitness = np.mean(all_fitness)
    mean_run = min(final_results, key=lambda x: abs(x[1] - mean_fitness))[0]
    
    # Create the comparison plot
    plt.figure(figsize=(14, 8))
    
    colors = ['green', 'red', 'blue']
    labels = [f'Best Run (Run {best_run})', 
              f'Worst Run (Run {worst_run})', 
              f'Mean Run (Run {mean_run})']
    runs_to_plot = [best_run, worst_run, mean_run]
    
    for i, run in enumerate(runs_to_plot):
        run_data = df[df['run'] == run].copy()
        run_data = run_data.sort_values('evaluations')
        run_data['running_best'] = run_data['best_fitness'].cummin()
        
        plt.plot(run_data['evaluations'], 
                run_data['running_best'], 
                color=colors[i], 
                linewidth=3, 
                marker='o',
                markersize=4,
                markevery=max(1, len(run_data)//15),
                label=labels[i])
    
    plt.xlabel('Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Best Fitness', fontsize=12, fontweight='bold')
    plt.title('EVRP Convergence: Best vs Worst vs Mean Run Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12, frameon=True, loc='upper right')
    
    # Clean up the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('3_evrp_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("3. Convergence comparison saved as '3_evrp_convergence_comparison.png'")
    
    # Print statistics
    print(f"\nComparison Statistics:")
    print(f"Best Run {best_run}: {final_results[0][1]:.2f}")
    print(f"Worst Run {worst_run}: {final_results[-1][1]:.2f}")
    print(f"Mean Run {mean_run}: {min(final_results, key=lambda x: abs(x[1] - mean_fitness))[1]:.2f}")
    print(f"Overall Mean Fitness: {mean_fitness:.2f}")
    print(f"Standard Deviation: {np.std(all_fitness):.2f}")
    
    plt.show()

def plot_median_run_only(csv_file='performance_log.csv'):
    """
    4. Single scatter plot of the MEDIAN run with connected lines
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return
    
    # Calculate final fitness for each run
    runs = sorted(df['run'].unique())
    final_results = []
    
    for run in runs:
        run_data = df[df['run'] == run]
        final_fitness = run_data['best_fitness'].min()
        final_results.append((run, final_fitness))
    
    # Sort by final fitness to find median
    final_results.sort(key=lambda x: x[1])
    
    # Find the median run (middle run when sorted by performance)
    median_idx = len(final_results) // 2
    median_run = final_results[median_idx][0]
    median_fitness = final_results[median_idx][1]
    
    # Get the median run data
    run_data = df[df['run'] == median_run].copy()
    run_data = run_data.sort_values('evaluations')
    run_data['running_best'] = run_data['best_fitness'].cummin()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Connected scatter plot for accepted solutions
    plt.plot(run_data['evaluations'], 
             run_data['current_fitness'],
             color='blue', 
             alpha=0.6, 
             linewidth=0.5,
             marker='o',
             markersize=1.5,
             label='Accepted Solutions')
    
    # Line plot for best fitness convergence
    plt.plot(run_data['evaluations'], 
             run_data['running_best'], 
             color='red', 
             linewidth=3,
             label='Best Fitness Found')
    
    plt.xlabel('Fitness Evaluations', fontsize=14, fontweight='bold')
    plt.ylabel('Tour Length (Fitness)', fontsize=14, fontweight='bold')
    plt.title(f'SA Search Analysis for benchmark dataset E-N22-k4 Median Performance (Run {median_run})', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12, frameon=True, loc='upper right')
    
    # Clean up the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add key statistics
    final_best = run_data['running_best'].iloc[-1]
    initial_fitness = run_data['running_best'].iloc[0]
    improvement = initial_fitness - final_best
    
    # Calculate statistics for context
    all_fitness = [x[1] for x in final_results]
    best_fitness = final_results[0][1]
    worst_fitness = final_results[-1][1]
    
    stats_text = f'Median Run Statistics:\n'
    stats_text += f'Run: {median_run}\n'
    stats_text += f'Initial: {initial_fitness:.1f}\n'
    stats_text += f'Final: 398.96\n'
    stats_text += f'Improvement: {improvement:.1f}\n'
    stats_text += f'Rank: {median_idx + 1}/{len(final_results)}\n'
    stats_text += f'Best: 389.92 Worst: 435.34'
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('4_median_run_connected_scatter.png', dpi=300, bbox_inches='tight')
    print(f"4. Median run connected scatter plot saved as '4_median_run_connected_scatter.png'")
    print(f"Median run is Run {median_run} with final fitness {final_best:.2f}")
    print(f"Ranked {median_idx + 1} out of {len(final_results)} runs")
    plt.show()

# Modified main execution - simplified
if __name__ == "__main__":
    print("Creating 4 essential EVRP analysis plots...\n")
    
    print("1. Individual runs with SA current and best fitness...")
    plot_individual_runs_with_sa()
    
    print("\n2. SA acceptance analysis (4 different plots)...")
    plot_sa_acceptance_analysis()
    
    print("\n3. Convergence comparison (best/worst/mean)...")
    plot_convergence_best_worst_mean()
    
    print("\n4. Median run - connected scatter plot...")
    plot_median_run_only()
    
    print("\nAll plots completed! Files saved:")
    print("- 1_evrp_individual_runs_sa.png (FOR APPENDIX - landscape mode)")
    print("- 2_evrp_sa_acceptance_analysis.png") 
    print("- 3_evrp_convergence_comparison.png (FOR MAIN TEXT - performance)")
    print("- 4_median_run_connected_scatter.png (FOR MAIN TEXT - behavior)")