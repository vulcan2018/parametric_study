import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

def recover_and_analyze_partial_results():
    """Recover and analyze results from a crashed parametric run"""
    
    # Find the most recent results directory
    result_dirs = glob.glob("parametric_snr_results_*")
    if not result_dirs:
        print("No parametric SNR results found!")
        return None
    
    latest_dir = max(result_dirs)
    print(f"Analyzing crashed run from: {latest_dir}")
    
    # Check what we have
    print("\nInventory of available files:")
    
    # Look for master CSV (might be incomplete)
    master_csv = glob.glob(f"{latest_dir}/parametric_snr_master_*.csv")
    if master_csv:
        print(f"✓ Master CSV found: {os.path.basename(master_csv[0])}")
        try:
            master_df = pd.read_csv(master_csv[0])
            print(f"  - Contains {len(master_df)} test results")
            print(f"  - Configurations present: {master_df['config_name'].nunique()}")
        except:
            print("  - Master CSV appears corrupted")
            master_df = None
    else:
        print("✗ No master CSV found")
        master_df = None
    
    # Look for individual config results
    config_dirs = glob.glob(f"{latest_dir}/config_*")
    print(f"✓ Found {len(config_dirs)} configuration directories")
    
    completed_configs = []
    partial_data = []
    
    for config_dir in config_dirs:
        config_name = os.path.basename(config_dir).replace('config_', '')
        csv_file = f"{config_dir}/config_results_{config_name}.csv"
        
        if os.path.exists(csv_file):
            try:
                config_df = pd.read_csv(csv_file)
                completed_configs.append(config_name)
                partial_data.append(config_df)
                
                # Check completeness (should have 45 rows: 15 SNR × 3 scenarios)
                expected_rows = 15 * 3
                actual_rows = len(config_df)
                completeness = (actual_rows / expected_rows) * 100
                
                print(f"  ✓ {config_name}: {actual_rows}/{expected_rows} tests ({completeness:.1f}% complete)")
                
            except Exception as e:
                print(f"  ✗ {config_name}: Error reading CSV - {e}")
        else:
            print(f"  ✗ {config_name}: No CSV file found")
    
    # Combine all available data
    if partial_data:
        print(f"\nCombining data from {len(partial_data)} completed configurations...")
        combined_df = pd.concat(partial_data, ignore_index=True)
        
        print(f"Total recovered test results: {len(combined_df)}")
        print(f"Configurations with data: {combined_df['config_name'].nunique()}")
        print(f"Unique configurations: {sorted(combined_df['config_name'].unique())}")
        
        # Save recovered data
        recovered_file = f"{latest_dir}/recovered_results.csv"
        combined_df.to_csv(recovered_file, index=False)
        print(f"✓ Saved recovered data to: {recovered_file}")
        
        # Quick analysis
        print("\nQUICK ANALYSIS OF RECOVERED DATA:")
        print("="*50)
        
        threshold_sde = 7
        
        for config in sorted(combined_df['config_name'].unique()):
            config_data = combined_df[combined_df['config_name'] == config]
            
            # Check if we have all scenarios
            scenarios = config_data['scenario'].unique()
            expected_scenarios = ['correct_ettv', 'incorrect_ettv', 'no_ttv']
            missing_scenarios = set(expected_scenarios) - set(scenarios)
            
            print(f"\n{config}:")
            if missing_scenarios:
                print(f"  ⚠ Missing scenarios: {missing_scenarios}")
            
            for scenario in ['correct_ettv', 'no_ttv']:
                if scenario in scenarios:
                    scenario_data = config_data[config_data['scenario'] == scenario]
                    max_sde = scenario_data['sde'].max()
                    above_threshold = scenario_data[scenario_data['sde'] >= threshold_sde]
                    
                    if len(above_threshold) > 0:
                        threshold = above_threshold['count_rate'].min()
                        print(f"  {scenario}: Max SDE={max_sde:.1f}, Threshold={threshold:.0f}")
                    else:
                        print(f"  {scenario}: Max SDE={max_sde:.1f}, No detection")
        
        # Create summary plot of recovered data
        create_recovery_summary_plot(combined_df, latest_dir)
        
        return combined_df, latest_dir
        
    else:
        print("No usable configuration data found!")
        return None, latest_dir

def create_recovery_summary_plot(df, output_dir):
    """Create summary plots from recovered data"""
    
    configs = sorted(df['config_name'].unique())
    
    if len(configs) == 0:
        print("No data to plot")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance curves for each config
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    
    for i, config in enumerate(configs):
        config_data = df[(df['config_name'] == config) & (df['scenario'] == 'correct_ettv')]
        if len(config_data) > 0:
            # Sort by count_rate for proper line plotting
            config_data = config_data.sort_values('count_rate')
            ax1.loglog(config_data['count_rate'], config_data['sde'], 
                      'o-', color=colors[i], alpha=0.8, markersize=4,
                      label=config.replace('_', ' '))
    
    ax1.axhline(y=7, color='k', linestyle='--', alpha=0.7, label='Detection Threshold')
    ax1.set_xlabel('Count Rate')
    ax1.set_ylabel('SDE')
    ax1.set_title('TTV-BLS Performance (Recovered Data)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection success rates
    ax2 = axes[0, 1]
    success_rates = []
    config_labels = []
    
    for config in configs:
        config_data = df[(df['config_name'] == config) & (df['scenario'] == 'correct_ettv')]
        if len(config_data) > 0:
            success_rate = len(config_data[config_data['sde'] >= 7]) / len(config_data)
            success_rates.append(success_rate)
            config_labels.append(config.replace('_', '\n'))
    
    if success_rates:
        bars = ax2.bar(range(len(success_rates)), success_rates, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Detection Success Rate')
        ax2.set_title('Detection Success Rates')
        ax2.set_xticks(range(len(config_labels)))
        ax2.set_xticklabels(config_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Peak SDE comparison
    ax3 = axes[1, 0]
    peak_sdes_ttv = []
    peak_sdes_std = []
    
    for config in configs:
        ttv_data = df[(df['config_name'] == config) & (df['scenario'] == 'correct_ettv')]
        std_data = df[(df['config_name'] == config) & (df['scenario'] == 'no_ttv')]
        
        peak_ttv = ttv_data['sde'].max() if len(ttv_data) > 0 else 0
        peak_std = std_data['sde'].max() if len(std_data) > 0 else 0
        
        peak_sdes_ttv.append(peak_ttv)
        peak_sdes_std.append(peak_std)
    
    if peak_sdes_ttv:
        x = np.arange(len(configs))
        width = 0.35
        
        ax3.bar(x - width/2, peak_sdes_ttv, width, label='TTV-BLS', alpha=0.8)
        ax3.bar(x + width/2, peak_sdes_std, width, label='Standard BLS', alpha=0.8)
        
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Peak SDE')
        ax3.set_title('Peak SDE Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data completeness
    ax4 = axes[1, 1]
    completeness = []
    
    for config in configs:
        config_data = df[df['config_name'] == config]
        expected_tests = 15 * 3  # 15 SNR levels × 3 scenarios
        actual_tests = len(config_data)
        completeness.append((actual_tests / expected_tests) * 100)
    
    bars = ax4.bar(range(len(completeness)), completeness, alpha=0.7, color='orange')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Completeness (%)')
    ax4.set_title('Data Completeness')
    ax4.set_xticks(range(len(config_labels)))
    ax4.set_xticklabels(config_labels, rotation=45, ha='right')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Complete')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, completeness):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recovered_data_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Summary plot saved to: {output_dir}/recovered_data_analysis.png")

def check_progress_log(result_dir):
    """Check the progress log to see where the crash occurred"""
    log_file = f"{result_dir}/parametric_snr_progress.log"
    
    if os.path.exists(log_file):
        print(f"\nChecking progress log: {log_file}")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            print(f"Log contains {len(lines)} entries")
            print("Last few entries:")
            for line in lines[-10:]:
                print(f"  {line.strip()}")
            
            # Look for completion markers
            completed_configs = []
            for line in lines:
                if "Completed configuration" in line:
                    # Extract config name
                    parts = line.split("configuration ")
                    if len(parts) > 1:
                        config_name = parts[1].split(":")[0]
                        completed_configs.append(config_name)
            
            print(f"\nConfigurations that completed successfully: {len(completed_configs)}")
            for config in completed_configs:
                print(f"  ✓ {config}")
        else:
            print("Progress log is empty")
    else:
        print("No progress log found")

# Run the recovery analysis
if __name__ == "__main__":
    print("PARAMETRIC SNR RECOVERY ANALYSIS")
    print("="*50)
    
    df, result_dir = recover_and_analyze_partial_results()
    
    if df is not None:
        print(f"\n✓ Successfully recovered {len(df)} test results")
        print("✓ Summary plot created")
        print("✓ You can now use this data for analysis!")
        
        # Check progress log for crash details
        check_progress_log(result_dir)
        
    else:
        print("✗ No recoverable data found")
