import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import seaborn as sns
from matplotlib.colors import LogNorm

def analyze_parametric_snr_results():
    """Comprehensive analysis of parametric SNR results"""
    
    # Find the most recent results directory
    result_dirs = glob.glob("parametric_snr_results_*")
    if not result_dirs:
        print("No parametric SNR results found!")
        return
    
    latest_dir = max(result_dirs)
    print(f"Analyzing results from: {latest_dir}")
    
    # Load master CSV file
    csv_files = glob.glob(f"{latest_dir}/parametric_snr_master_*.csv")
    if not csv_files:
        print("No master CSV file found!")
        return
    
    df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(df)} test results across {df['config_name'].nunique()} configurations")
    
    # Create comprehensive analysis plots
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # 1. SNR Threshold Heatmap
    plt.subplot(4, 3, 1)
    threshold_data = calculate_thresholds(df)
    create_threshold_heatmap(threshold_data, 'TTV-BLS (Correct E_TTV)', 'correct_ettv')
    
    plt.subplot(4, 3, 2)
    create_threshold_heatmap(threshold_data, 'Standard BLS', 'no_ttv')
    
    plt.subplot(4, 3, 3)
    create_advantage_heatmap(threshold_data)
    
    # 2. Performance curves for each configuration
    plt.subplot(4, 3, 4)
    plot_performance_curves_summary(df)
    
    # 3. Detection success rate matrix
    plt.subplot(4, 3, 5)
    create_detection_success_matrix(df)
    
    # 4. Peak SDE comparison
    plt.subplot(4, 3, 6)
    plot_peak_sde_comparison(df)
    
    # 5. Individual configuration curves (best few)
    plt.subplot(4, 3, 7)
    plot_best_configurations(df)
    
    # 6. Amplitude vs Period analysis
    plt.subplot(4, 3, 8)
    plot_amplitude_period_analysis(df)
    
    # 7. SNR advantage factor
    plt.subplot(4, 3, 9)
    plot_snr_advantage_factors(df)
    
    # 8. Detectability regions
    plt.subplot(4, 3, 10)
    plot_detectability_regions(df)
    
    # 9. E_TTV sensitivity analysis
    plt.subplot(4, 3, 11)
    plot_ettv_sensitivity(df)
    
    # 10. Summary statistics
    plt.subplot(4, 3, 12)
    plot_summary_statistics(df)
    
    plt.tight_layout()
    plt.savefig('parametric_snr_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate detailed report
    generate_parametric_report(df, threshold_data)
    
    return df, threshold_data

def calculate_thresholds(df, threshold_sde=7):
    """Calculate SNR thresholds for each configuration and scenario"""
    thresholds = []
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        a_ttv = config_data['a_ttv'].iloc[0]
        p_ttv = config_data['p_ttv'].iloc[0]
        
        row = {'config_name': config, 'a_ttv': a_ttv, 'p_ttv': p_ttv}
        
        for scenario in ['correct_ettv', 'incorrect_ettv', 'no_ttv']:
            scenario_data = config_data[config_data['scenario'] == scenario]
            above_threshold = scenario_data[scenario_data['sde'] >= threshold_sde]
            
            if len(above_threshold) > 0:
                threshold = above_threshold['count_rate'].min()
                row[f'threshold_{scenario}'] = threshold
                row[f'max_sde_{scenario}'] = scenario_data['sde'].max()
            else:
                row[f'threshold_{scenario}'] = None
                row[f'max_sde_{scenario}'] = scenario_data['sde'].max()
        
        thresholds.append(row)
    
    return pd.DataFrame(thresholds)

def create_threshold_heatmap(threshold_data, title, scenario):
    """Create heatmap of SNR thresholds"""
    # Prepare data for heatmap
    pivot_data = threshold_data.pivot(index='a_ttv', columns='p_ttv', 
                                     values=f'threshold_{scenario}')
    
    # Replace None with high value for visualization
    pivot_data = pivot_data.fillna(1e6)
    
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='viridis_r', 
                norm=LogNorm(vmin=100, vmax=1e6), cbar_kws={'label': 'SNR Threshold'})
    plt.title(f'{title}\nSNR Thresholds')
    plt.xlabel('TTV Period (days)')
    plt.ylabel('TTV Amplitude (days)')

def create_advantage_heatmap(threshold_data):
    """Create heatmap of SNR advantage factors"""
    # Calculate advantage factor
    advantage_data = threshold_data.copy()
    advantage_data['advantage'] = (
        advantage_data['threshold_no_ttv'] / advantage_data['threshold_correct_ettv']
    ).fillna(1.0)
    
    pivot_data = advantage_data.pivot(index='a_ttv', columns='p_ttv', values='advantage')
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0.5, vmax=5.0, cbar_kws={'label': 'SNR Advantage Factor'})
    plt.title('TTV-BLS SNR Advantage\n(Higher = Better)')
    plt.xlabel('TTV Period (days)')
    plt.ylabel('TTV Amplitude (days)')

def plot_performance_curves_summary(df):
    """Plot performance curves for all configurations"""
    configs = df['config_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    
    for i, config in enumerate(configs):
        config_data = df[(df['config_name'] == config) & (df['scenario'] == 'correct_ettv')]
        if len(config_data) > 0:
            plt.loglog(config_data['count_rate'], config_data['sde'], 
                      'o-', color=colors[i], alpha=0.7, markersize=3,
                      label=f"{config.replace('_', ' ')}")
    
    plt.axhline(y=7, color='k', linestyle='--', alpha=0.7, label='Detection Threshold')
    plt.xlabel('Count Rate')
    plt.ylabel('SDE')
    plt.title('Performance Curves (TTV-BLS)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

def create_detection_success_matrix(df, threshold_sde=7):
    """Create matrix showing detection success rates"""
    success_data = []
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        a_ttv = config_data['a_ttv'].iloc[0]
        p_ttv = config_data['p_ttv'].iloc[0]
        
        for scenario in ['correct_ettv', 'no_ttv']:
            scenario_data = config_data[config_data['scenario'] == scenario]
            success_rate = len(scenario_data[scenario_data['sde'] >= threshold_sde]) / len(scenario_data)
            
            success_data.append({
                'config_name': config,
                'a_ttv': a_ttv,
                'p_ttv': p_ttv,
                'scenario': scenario,
                'success_rate': success_rate
            })
    
    success_df = pd.DataFrame(success_data)
    
    # Plot for TTV-BLS
    ttv_success = success_df[success_df['scenario'] == 'correct_ettv']
    pivot_data = ttv_success.pivot(index='a_ttv', columns='p_ttv', values='success_rate')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Detection Success Rate'})
    plt.title('Detection Success Rate\n(TTV-BLS, Correct E_TTV)')
    plt.xlabel('TTV Period (days)')
    plt.ylabel('TTV Amplitude (days)')

def plot_peak_sde_comparison(df):
    """Compare peak SDE values across configurations"""
    peak_sdes = []
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        
        for scenario in ['correct_ettv', 'no_ttv']:
            scenario_data = config_data[config_data['scenario'] == scenario]
            peak_sde = scenario_data['sde'].max()
            
            peak_sdes.append({
                'config': config.replace('_', ' '),
                'scenario': 'TTV-BLS' if scenario == 'correct_ettv' else 'Standard BLS',
                'peak_sde': peak_sde
            })
    
    peak_df = pd.DataFrame(peak_sdes)
    
    # Create grouped bar plot
    configs = peak_df['config'].unique()
    x = np.arange(len(configs))
    width = 0.35
    
    ttv_sdes = peak_df[peak_df['scenario'] == 'TTV-BLS']['peak_sde'].values
    std_sdes = peak_df[peak_df['scenario'] == 'Standard BLS']['peak_sde'].values
    
    plt.bar(x - width/2, ttv_sdes, width, label='TTV-BLS', alpha=0.8)
    plt.bar(x + width/2, std_sdes, width, label='Standard BLS', alpha=0.8)
    
    plt.xlabel('Configuration')
    plt.ylabel('Peak SDE')
    plt.title('Peak SDE Comparison')
    plt.xticks(x, [c.replace('_', '\n') for c in configs], rotation=45, ha='right')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

def plot_best_configurations(df, top_n=3):
    """Plot performance curves for best configurations"""
    # Find configurations with best improvement
    config_improvements = {}
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        ttv_max = config_data[config_data['scenario'] == 'correct_ettv']['sde'].max()
        std_max = config_data[config_data['scenario'] == 'no_ttv']['sde'].max()
        
        if std_max > 0:
            improvement = ttv_max / std_max
            config_improvements[config] = improvement
    
    # Get top configurations
    best_configs = sorted(config_improvements.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    colors = ['red', 'blue', 'green']
    
    for i, (config, improvement) in enumerate(best_configs):
        config_data = df[df['config_name'] == config]
        
        ttv_data = config_data[config_data['scenario'] == 'correct_ettv']
        std_data = config_data[config_data['scenario'] == 'no_ttv']
        
        plt.loglog(ttv_data['count_rate'], ttv_data['sde'], 
                  'o-', color=colors[i], linewidth=2, markersize=4,
                  label=f"{config.replace('_', ' ')} (TTV-BLS)")
        plt.loglog(std_data['count_rate'], std_data['sde'], 
                  's--', color=colors[i], linewidth=2, markersize=4, alpha=0.7,
                  label=f"{config.replace('_', ' ')} (Std BLS)")
    
    plt.axhline(y=7, color='k', linestyle='--', alpha=0.7)
    plt.xlabel('Count Rate')
    plt.ylabel('SDE')
    plt.title(f'Top {top_n} Performing Configurations')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

def plot_amplitude_period_analysis(df):
    """Analyze detectability as function of amplitude and period"""
    threshold_data = calculate_thresholds(df)
    
    # Create scatter plot
    valid_data = threshold_data[threshold_data['threshold_correct_ettv'].notna()]
    
    scatter = plt.scatter(valid_data['p_ttv'], valid_data['a_ttv'], 
                         c=valid_data['threshold_correct_ettv'], 
                         s=100, cmap='viridis_r', norm=LogNorm())
    
    plt.colorbar(scatter, label='SNR Threshold')
    plt.xlabel('TTV Period (days)')
    plt.ylabel('TTV Amplitude (days)')
    plt.title('Detectability Map\n(Lower threshold = Easier detection)')
    plt.yscale('log')
    
    # Add contour lines
    if len(valid_data) > 4:
        from scipy.interpolate import griddata
        
        # Create regular grid
        p_range = np.linspace(valid_data['p_ttv'].min(), valid_data['p_ttv'].max(), 50)
        a_range = np.logspace(np.log10(valid_data['a_ttv'].min()), 
                             np.log10(valid_data['a_ttv'].max()), 50)
        P_grid, A_grid = np.meshgrid(p_range, a_range)
        
        # Interpolate threshold values
        threshold_grid = griddata(
            (valid_data['p_ttv'], valid_data['a_ttv']), 
            valid_data['threshold_correct_ettv'],
            (P_grid, A_grid), method='linear'
        )
        
        # Add contour lines for key thresholds
        contour_levels = [1000, 10000, 100000]
        plt.contour(P_grid, A_grid, threshold_grid, levels=contour_levels, 
                   colors='white', alpha=0.7, linewidths=1)

def plot_snr_advantage_factors(df):
    """Plot SNR advantage factors"""
    threshold_data = calculate_thresholds(df)
    
    # Calculate advantage factors
    valid_data = threshold_data[
        (threshold_data['threshold_correct_ettv'].notna()) & 
        (threshold_data['threshold_no_ttv'].notna())
    ]
    
    if len(valid_data) > 0:
        valid_data = valid_data.copy()
        valid_data['advantage_factor'] = (
            valid_data['threshold_no_ttv'] / valid_data['threshold_correct_ettv']
        )
        
        bars = plt.bar(range(len(valid_data)), valid_data['advantage_factor'], 
                      alpha=0.7, color='skyblue')
        
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No advantage')
        plt.xlabel('Configuration')
        plt.ylabel('SNR Advantage Factor')
        plt.title('SNR Advantage Factors\n(Higher = Better)')
        plt.xticks(range(len(valid_data)), 
                  [c.replace('_', '\n') for c in valid_data['config_name']], 
                  rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, valid_data['advantage_factor']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)

def plot_detectability_regions(df):
    """Plot detectability regions in parameter space"""
    # Define detectability criteria
    criteria = [
        {'threshold': 1000, 'label': 'Easy (SNR < 1k)', 'color': 'green'},
        {'threshold': 10000, 'label': 'Moderate (SNR < 10k)', 'color': 'yellow'},
        {'threshold': 100000, 'label': 'Hard (SNR < 100k)', 'color': 'orange'},
        {'threshold': float('inf'), 'label': 'Very Hard (SNR > 100k)', 'color': 'red'}
    ]
    
    threshold_data = calculate_thresholds(df)
    
    for i, row in threshold_data.iterrows():
        threshold = row['threshold_correct_ettv']
        
        if pd.isna(threshold):
            color = 'red'
            label = 'Very Hard'
        else:
            for criterion in criteria:
                if threshold <= criterion['threshold']:
                    color = criterion['color']
                    label = criterion['label']
                    break
        
        plt.scatter(row['p_ttv'], row['a_ttv'], 
                   c=color, s=200, alpha=0.7, edgecolors='black')
        
        # Add text label
        plt.text(row['p_ttv'], row['a_ttv'], row['config_name'].replace('_', '\n'), 
                ha='center', va='center', fontsize=6, weight='bold')
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=c['color'], markersize=10, 
                                 label=c['label']) for c in criteria]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel('TTV Period (days)')
    plt.ylabel('TTV Amplitude (days)')
    plt.title('Detectability Regions')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

def plot_ettv_sensitivity(df):
    """Analyze E_TTV sensitivity"""
    ettv_comparison = []
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        
        correct_sde = config_data[config_data['scenario'] == 'correct_ettv']['sde'].max()
        incorrect_sde = config_data[config_data['scenario'] == 'incorrect_ettv']['sde'].max()
        
        if correct_sde > 0:
            sensitivity = (correct_sde - incorrect_sde) / correct_sde
        else:
            sensitivity = 0
        
        ettv_comparison.append({
            'config': config.replace('_', ' '),
            'sensitivity': sensitivity,
            'correct_sde': correct_sde,
            'incorrect_sde': incorrect_sde
        })
    
    ettv_df = pd.DataFrame(ettv_comparison)
    
    bars = plt.bar(range(len(ettv_df)), ettv_df['sensitivity'], 
                  alpha=0.7, color='coral')
    
    plt.xlabel('Configuration')
    plt.ylabel('E_TTV Sensitivity')
    plt.title('E_TTV Knowledge Importance\n(Higher = More Sensitive)')
    plt.xticks(range(len(ettv_df)), 
              [c.replace(' ', '\n') for c in ettv_df['config']], 
              rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, ettv_df['sensitivity']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)

def plot_summary_statistics(df):
    """Plot key summary statistics"""
    # Calculate key metrics
    threshold_data = calculate_thresholds(df)
    
    metrics = {
        'Detectable Configs\n(SNR < 10k)': len(threshold_data[
            (threshold_data['threshold_correct_ettv'].notna()) & 
            (threshold_data['threshold_correct_ettv'] <= 10000)
        ]),
        'Easy Configs\n(SNR < 1k)': len(threshold_data[
            (threshold_data['threshold_correct_ettv'].notna()) & 
            (threshold_data['threshold_correct_ettv'] <= 1000)
        ]),
        'Configs with\nAdvantage': len(threshold_data[
            (threshold_data['threshold_correct_ettv'].notna()) & 
            (threshold_data['threshold_no_ttv'].notna()) & 
            (threshold_data['threshold_no_ttv'] > threshold_data['threshold_correct_ettv'])
        ]),
        'Total Configs': len(threshold_data)
    }
    
    bars = plt.bar(metrics.keys(), metrics.values(), 
                  color=['green', 'blue', 'orange', 'gray'], alpha=0.7)
    
    plt.ylabel('Number of Configurations')
    plt.title('Summary Statistics')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontsize=10, weight='bold')

def generate_parametric_report(df, threshold_data):
    """Generate comprehensive text report"""
    
    with open('parametric_snr_analysis_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARAMETRIC SNR ANALYSIS - COMPREHENSIVE REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total configurations: {df['config_name'].nunique()}\n")
        f.write(f"Total tests: {len(df)}\n")
        f.write(f"SNR range: {df['count_rate'].min():.0f} - {df['count_rate'].max():.0f}\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        
        detectable = threshold_data[threshold_data['threshold_correct_ettv'].notna()]
        easy_detection = detectable[detectable['threshold_correct_ettv'] <= 1000]
        with_advantage = threshold_data[
            (threshold_data['threshold_correct_ettv'].notna()) & 
            (threshold_data['threshold_no_ttv'].notna()) & 
            (threshold_data['threshold_no_ttv'] > threshold_data['threshold_correct_ettv'])
        ]
        
        f.write(f"Detectable configurations: {len(detectable)}/{len(threshold_data)}\n")
        f.write(f"Easy detection (SNR < 1000): {len(easy_detection)}/{len(threshold_data)}\n")
        f.write(f"Configurations with SNR advantage: {len(with_advantage)}/{len(threshold_data)}\n\n")
        
        # Best performers
        f.write("BEST PERFORMING CONFIGURATIONS:\n")
        f.write("-" * 40 + "\n")
        
        best_configs = detectable.nsmallest(5, 'threshold_correct_ettv')
        for _, config in best_configs.iterrows():
            f.write(f"{config['config_name']}: ")
            f.write(f"A={config['a_ttv']:.3f}, P={config['p_ttv']:.0f} ")
            f.write(f"→ SNR threshold: {config['threshold_correct_ettv']:.0f}\n")
        
        f.write("\n")
        
        # Detailed results for each configuration
        f.write("DETAILED CONFIGURATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for _, config in threshold_data.iterrows():
            f.write(f"\n{config['config_name']}:\n")
            f.write(f"  A_TTV = {config['a_ttv']:.3f} days\n")
            f.write(f"  P_TTV = {config['p_ttv']:.0f} days\n")
            
            if pd.notna(config['threshold_correct_ettv']):
                f.write(f"  SNR Threshold (TTV-BLS): {config['threshold_correct_ettv']:.0f}\n")
            else:
                f.write(f"  SNR Threshold (TTV-BLS): Not achieved\n")
                
            if pd.notna(config['threshold_no_ttv']):
                f.write(f"  SNR Threshold (Std BLS): {config['threshold_no_ttv']:.0f}\n")
            else:
                f.write(f"  SNR Threshold (Std BLS): Not achieved\n")
            
            if pd.notna(config['threshold_correct_ettv']) and pd.notna(config['threshold_no_ttv']):
                advantage = config['threshold_no_ttv'] / config['threshold_correct_ettv']
                f.write(f"  SNR Advantage Factor: {advantage:.2f}x\n")
            
            f.write(f"  Peak SDE (TTV-BLS): {config['max_sde_correct_ettv']:.1f}\n")
            f.write(f"  Peak SDE (Std BLS): {config['max_sde_no_ttv']:.1f}\n")
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n")
        
        if len(easy_detection) > 0:
            f.write("✓ PROMISING: These configurations are detectable at realistic SNR levels:\n")
            for _, config in easy_detection.iterrows():
                f.write(f"  - {config['config_name']}: requires SNR > {config['threshold_correct_ettv']:.0f}\n")
        
        if len(with_advantage) > 0:
            f.write(f"\n✓ TTV-BLS shows SNR advantage in {len(with_advantage)} configurations\n")
        
        ettv_critical = len(threshold_data[
            (threshold_data['threshold_correct_ettv'].notna()) & 
            (threshold_data['threshold_incorrect_ettv'].isna() | 
             (threshold_data['threshold_incorrect_ettv'] > threshold_data['threshold_correct_ettv'] * 5))
        ])
        
        if ettv_critical > len(threshold_data) * 0.5:
            f.write(f"\n⚠ WARNING: E_TTV knowledge is critical for {ettv_critical} configurations\n")
            f.write("   → Develop robust E_TTV search strategies\n")
        
        min_useful_snr = detectable['threshold_correct_ettv'].min() if len(detectable) > 0 else None
        if min_useful_snr:
            f.write(f"\n📊 SURVEY REQUIREMENTS: Minimum useful SNR ≈ {min_useful_snr:.0f}\n")
            f.write(f"   → Target count rates > {min_useful_snr:.0f} for TTV detection\n")
    
    print("Comprehensive report saved to: parametric_snr_analysis_report.txt")

# Run the analysis
if __name__ == "__main__":
    df, threshold_data = analyze_parametric_snr_results()
