#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from math import sin, pi, sqrt, fabs, trunc, floor, ceil
from numpy.random import default_rng
from astropy.table import Table, Column, vstack
import multiprocessing as mp
from matplotlib import cm
import time
from numba import jit, njit
from itertools import starmap
import numpy.lib.recfunctions as rfn
import batman
import sys
import os
import json
import configparser
import pickle
import pandas as pd
from datetime import datetime
from functools import partial

def read_parameters_from_file():
    """Read simulation parameters from external file"""
    
    if os.path.exists('parameters.json'):
        try:
            with open('parameters.json', 'r') as f:
                params = json.load(f)
            print("Parameters loaded from parameters.json")
            return params
        except Exception as e:
            print(f"Error reading parameters.json: {e}")
    
    elif os.path.exists('parameters.txt'):
        try:
            params = {}
            with open('parameters.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        except ValueError:
                            params[key] = value
            
            print("Parameters loaded from parameters.txt")
            return params
        except Exception as e:
            print(f"Error reading parameters.txt: {e}")
    
    print("ERROR: No parameter file found!")
    print("Please create one of: parameters.json, parameters.txt")
    sys.exit(1)

def save_simulation_metadata(ttv_configs):
    """Save metadata about the simulation setup"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'script_version': '2.0_parametric_parallel',
        'simulation_parameters': {
            'cadence': cadence,
            'duration': duration,
            'period': period,
            'epoch': epoch,
            'r_planet': r_planet,
            'min_period': 0.6,
            'max_period': 75.0
        },
        'ttv_configurations': ttv_configs,
        'snr_levels': snr_levels.tolist(),
        'parallel_cores': num_cores,
        'python_version': sys.version,
        'numpy_version': np.__version__
    }
    
    with open('simulation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

# Load parameters
params = read_parameters_from_file()

# Required parameters
required_params = ['cadence', 'duration', 'period', 'epoch', 'r_planet', 'run_name']

# Check for parametric study configuration
analysis_mode = params.get('analysis_mode', 'parametric_snr')
save_data = params.get('save_data', True)
save_lightcurves = params.get('save_lightcurves', True)
save_bls_details = params.get('save_bls_details', True)
num_cores = params.get('num_cores', 0)  # 0 = serial, >0 = parallel

missing_params = []
for param in required_params:
    if param not in params:
        missing_params.append(param)

if missing_params:
    print("ERROR: Missing required parameters in parameter file:")
    for param in missing_params:
        print(f"  - {param}")
    print("\nExiting...")
    sys.exit(1)

# Extract base parameters
cadence = params['cadence']
duration = params['duration']
period = params['period']
epoch = params['epoch']
r_planet = params['r_planet']
run_name = params['run_name']

# Define TTV configurations to test
if 'ttv_configurations' in params:
    # User-defined configurations
    ttv_configurations = params['ttv_configurations']
else:
    # Default parametric study configurations
    ttv_configurations = [
        {"A_TTV": 0.005, "P_TTV": 30, "E_TTV": -15.0, "name": "very_weak_short"},
        {"A_TTV": 0.01, "P_TTV": 30, "E_TTV": -15.0, "name": "weak_short"},
        {"A_TTV": 0.01, "P_TTV": 60, "E_TTV": -30.0, "name": "weak_medium"},
        {"A_TTV": 0.01, "P_TTV": 120, "E_TTV": -60.0, "name": "weak_long"},
        {"A_TTV": 0.02, "P_TTV": 30, "E_TTV": -15.0, "name": "moderate_short"},
        {"A_TTV": 0.02, "P_TTV": 60, "E_TTV": -30.0, "name": "moderate_medium"},
        {"A_TTV": 0.02, "P_TTV": 120, "E_TTV": -60.0, "name": "moderate_long"},
        {"A_TTV": 0.05, "P_TTV": 60, "E_TTV": -30.0, "name": "strong_medium"},
        {"A_TTV": 0.05, "P_TTV": 120, "E_TTV": -60.0, "name": "strong_long"},
    ]

# Define SNR levels
snr_levels = np.logspace(2, 6, 15)  # 100 to 1,000,000

# Set up parallelization
if num_cores == 0:
    parallel_mode = "Serial"
    actual_cores = 1
elif num_cores > 0:
    parallel_mode = "Parallel"
    actual_cores = min(num_cores, mp.cpu_count())
else:
    parallel_mode = "Auto-detect"
    actual_cores = mp.cpu_count()

print("="*60)
print("PARAMETRIC SNR STUDY")
print("="*60)
print(f"Base parameters:")
print(f"  Cadence: {cadence} seconds")
print(f"  Duration: {duration} days")
print(f"  Period: {period} days")
print(f"  Epoch: {epoch} days")
print(f"  Planet radius: {r_planet}")
print(f"  Run name: {run_name}")
print(f"\nParallelization:")
print(f"  Mode: {parallel_mode}")
print(f"  Cores to use: {actual_cores}")
print(f"  Available cores: {mp.cpu_count()}")
print(f"\nTTV Configurations to test: {len(ttv_configurations)}")
for config in ttv_configurations:
    print(f"  {config['name']}: A={config['A_TTV']}, P={config['P_TTV']}, E={config['E_TTV']}")
print(f"\nSNR levels: {len(snr_levels)} (from {snr_levels[0]:.0f} to {snr_levels[-1]:.0f})")
print(f"Total tests: {len(ttv_configurations) * len(snr_levels) * 3} = {len(ttv_configurations)} configs × {len(snr_levels)} SNR × 3 scenarios")
print("="*60)

rng = default_rng()

# Set plot parameters
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def compute_transit_template(cadence, period, r_planet):
    """Compute transit template using batman"""
    params = batman.TransitParams()
    params.rp = r_planet
    params.a = 15.
    params.inc = 90.
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"
    params.per = period
    params.t0 = 0.0
    
    t = np.arange(-period*0.5, period*0.5, cadence/86400.0)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params) - 1.0
    
    nzidx = np.where(flux != 0.0)[0]
    nz_start = min(nzidx) - 3
    nz_end = max(nzidx) + 3
    
    return t[nz_start:nz_end], flux[nz_start:nz_end], t[nz_end] - t[nz_start]

def create_lightcurve(cadence, duration, period, epoch, a_ttv=0.0, p_ttv=0.0, e_ttv=0.0,
                      count_rate=0.0, r_planet=0.1):
    """Create synthetic lightcurve with TTVs"""
    # Use local RNG to avoid multiprocessing issues
    local_rng = default_rng()
    
    # Compute transit template
    template_t, template_flux, template_duration = compute_transit_template(cadence, period, r_planet)

    # Create time array
    t = np.arange(0.0, duration, cadence/86400.0, dtype=np.float32)
    flux = np.zeros_like(t)
    
    # Add transits with TTVs
    n = 0
    while True:
        tmid = epoch + n*period
        if p_ttv > 0.0 and a_ttv > 0.0:
            tmid += a_ttv*sin(2*pi*(n*period-e_ttv)/p_ttv)
        if tmid > duration + template_duration:
            break
        flux += np.interp(t-tmid, template_t, template_flux, left=0.0, right=0.0)
        n += 1

    # Add noise
    if count_rate > 0.0:
        sigma = 1.0/sqrt(count_rate*cadence)
        flux = local_rng.normal(flux, sigma)
        flux_err = np.full_like(flux, sigma, dtype=np.float32)
    else:
        flux_err = np.zeros_like(flux, dtype=np.float32)

    return t, np.array(flux, dtype=np.float32), flux_err

def distort_timebase(t, epoch, p_ttv, a_ttv, e_ttv=0.0):
    """Apply TTV correction to timebase"""
    return t - a_ttv*np.sin(2*pi*(t-(epoch+e_ttv))/p_ttv) if p_ttv > 0.0 and a_ttv > 0.0 else t

def optimal_sample_periods(min_period, max_period, obs_duration, bin_width):
    """Generate optimal period sampling"""
    sample_periods = []
    period = min_period
    while period < max_period:
        sample_periods.append(period)
        deltaFreq = bin_width / (period * obs_duration)
        period = 1.0 / (-deltaFreq + 1.0 / period)
    return np.array(sample_periods)

def transit_search(tstamp, flux, flux_err, min_period=None, max_period=None, sample_periods=None,
                   bin_width=45.0, min_box_width=2, max_box_width=5):
    """Perform BLS transit search"""
    t0 = np.min(tstamp)
    bin_width /= (24*60)  # Convert to days
    obs_duration = np.max(tstamp) - np.min(tstamp)

    if sample_periods is None:
        sample_periods = optimal_sample_periods(min_period, max_period, obs_duration, bin_width)

    weight = 1.0/(flux_err*flux_err)
    wflux = flux*weight
    wflux2 = (flux*flux)*weight
    chisqr0 = np.sum(wflux2)
    T = np.sum(weight)
    pdgram = []
    
    for period in sample_periods:
        num_bins = int((period / bin_width) + 0.5)
        real_bin_width = period/num_bins
        bin_idx = np.floor(np.mod((tstamp-t0)/period, 1.0)*num_bins).astype(np.int32)

        # Compute partial sums
        pstat = np.zeros((4, num_bins), dtype=np.float32)
        pstat[0, :] = np.bincount(bin_idx, wflux, num_bins)
        pstat[1, :] = np.bincount(bin_idx, wflux2, num_bins)
        pstat[2, :] = np.bincount(bin_idx, weight, num_bins)
        pstat[3, :] = np.bincount(bin_idx, minlength=num_bins)

        # Extend arrays for wrap-around
        pstat = np.concatenate((pstat, pstat[:max_box_width, :]), axis=1)

        # Stack shifted copies
        pstat = np.stack([np.roll(pstat, -shft, axis=1)
                         for shft in range(max_box_width+1)], axis=2)

        # Accumulate sums
        sstat = np.cumsum(pstat, axis=-1)[:, :, min_box_width:]

        # Compute delta-chisq
        S = sstat[0, :, :]
        Q = sstat[1, :, :]
        R = sstat[2, :, :]
        N = sstat[3, :, :]
        denom = (R*(T-R))
        delta_chisq = np.where((N>2) * (denom!=0.0), -(S*S)*T/denom, 0.0)

        # Find best detection
        min_idx = np.unravel_index(np.argmin(delta_chisq), delta_chisq.shape)
        delta_chisq = delta_chisq[min_idx]
        epoch = (min_idx[0]+(min_idx[1]+min_box_width)*0.5)*real_bin_width
        depth = S[min_idx]*T/(R[min_idx]*(T-R[min_idx]))
        depth_err = sqrt(T/(R[min_idx]*(T-R[min_idx])))
        num_pts = N[min_idx]
        width = (min_idx[1]+min_box_width)*real_bin_width
        pdgram.append((period, delta_chisq, epoch+t0, depth, depth_err, num_pts, width))

    # Create results array
    res = np.array(pdgram, dtype=[('period', 'f8'), ('delta_chisq', 'f4'), ('epoch', 'f4'), ('depth', 'f4'),
                                  ('depth_err', 'f4'), ('num_pts_in_transit', 'i4'), ('width', 'f4')])

    # Compute SDE
    median = np.median(res['delta_chisq'])
    rms = np.median(np.abs(res['delta_chisq']-median))*1.48
    res = rfn.append_fields(res, 'sde', (median-res['delta_chisq'])/rms, dtypes=('f4'), usemask=False)

    return res

def ttv_bls_search(t, flux, flux_err, p_ttv, a_ttv, e_ttv, min_period, max_period):
    """Single TTV-BLS search"""
    t_prime = distort_timebase(t, 0.0, p_ttv, a_ttv, e_ttv)
    res = transit_search(t_prime, flux, flux_err, min_period, max_period)
    best_idx = np.argmax(res['sde'])
    return res[best_idx]

def process_single_snr_level(args):
    """Process a single SNR level for a configuration - designed for multiprocessing"""
    (config, i, count_rate, snr_name, save_lc, save_bls, run_id, test_counter) = args
    
    try:
        # Create lightcurve for this configuration and SNR
        start_time = time.time()
        t, flux, flux_err = create_lightcurve(cadence, duration, period, epoch, 
                                            config['A_TTV'], config['P_TTV'], config['E_TTV'],
                                            count_rate=count_rate, r_planet=r_planet)
        lc_creation_time = time.time() - start_time
        
        # Save lightcurve data if requested
        lc_file = None
        if save_lc:
            lc_file = save_lightcurve_data(t, flux, flux_err, config, count_rate, run_id)
        
        # Test scenarios
        scenarios = [
            {'name': 'correct_ettv', 'params': (config['P_TTV'], config['A_TTV'], config['E_TTV'])},
            {'name': 'incorrect_ettv', 'params': (config['P_TTV'], config['A_TTV'], 0.0)},
            {'name': 'no_ttv', 'params': (0.0, 0.0, 0.0)}
        ]
        
        scenario_results = []
        bls_files = []
        
        for j, scenario in enumerate(scenarios):
            current_test = test_counter + j + 1
            
            start_time = time.time()
            try:
                p_ttv_param, a_ttv_param, e_ttv_param = scenario['params']
                bls_result = ttv_bls_search(t, flux, flux_err, p_ttv_param, a_ttv_param, e_ttv_param, 0.6, 75.0)
                
                # Save detailed BLS results if requested
                bls_file = None
                if save_bls:
                    bls_file = save_bls_results(bls_result, config, count_rate, scenario['name'], run_id)
                    bls_files.append(bls_file)
                
                summary_result = {
                    'config_name': config['name'],
                    'a_ttv': config['A_TTV'],
                    'p_ttv': config['P_TTV'],
                    'e_ttv': config['E_TTV'],
                    'count_rate': count_rate,
                    'snr_level': snr_name,
                    'sde': float(bls_result['sde']),
                    'period': float(bls_result['period']),
                    'depth': float(bls_result['depth']),
                    'epoch': float(bls_result['epoch']),
                    'width': float(bls_result['width']),
                    'num_pts_in_transit': int(bls_result['num_pts_in_transit']),
                    'scenario': scenario['name'],
                    'ttv_params': scenario['params'],
                    'success': True
                }
                
            except Exception as e:
                summary_result = {
                    'config_name': config['name'],
                    'a_ttv': config['A_TTV'],
                    'p_ttv': config['P_TTV'],
                    'e_ttv': config['E_TTV'],
                    'count_rate': count_rate,
                    'snr_level': snr_name,
                    'sde': 0, 'period': 0, 'depth': 0, 'epoch': 0, 'width': 0,
                    'num_pts_in_transit': 0, 'scenario': scenario['name'],
                    'ttv_params': scenario['params'], 'success': False,
                    'error': str(e)
                }
            
            computation_time = time.time() - start_time
            summary_result.update({
                'computation_time': computation_time,
                'lightcurve_creation_time': lc_creation_time if j == 0 else None
            })
            
            scenario_results.append(summary_result)
        
        return {
            'snr_index': i,
            'count_rate': count_rate,
            'scenario_results': scenario_results,
            'lightcurve_file': lc_file,
            'bls_files': bls_files,
            'success': True
        }
        
    except Exception as e:
        return {
            'snr_index': i,
            'count_rate': count_rate,
            'scenario_results': [],
            'lightcurve_file': None,
            'bls_files': [],
            'success': False,
            'error': str(e)
        }

def save_lightcurve_data(t, flux, flux_err, config, count_rate, run_id):
    """Save lightcurve data"""
    lc_data = {
        'time': t.tolist(),
        'flux': flux.tolist(),
        'flux_err': flux_err.tolist(),
        'config': config,
        'count_rate': count_rate,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"lightcurve_data_{run_id}_snr_{int(count_rate)}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(lc_data, f)
    
    return filename

def save_bls_results(bls_result, config, count_rate, scenario, run_id):
    """Save BLS results"""
    # Convert numpy structured array to dictionary
    result_data = {}
    for field_name in bls_result.dtype.names:
        value = bls_result[field_name]
        if hasattr(value, 'tolist'):
            result_data[field_name] = value.tolist()
        else:
            result_data[field_name] = float(value) if np.isscalar(value) else value
    
    full_result = {
        'bls_result': result_data,
        'config': config,
        'count_rate': count_rate,
        'scenario': scenario,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"bls_result_{run_id}_snr_{int(count_rate)}_{scenario}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(full_result, f)
    
    return filename

def parametric_snr_analysis():
    """Perform parametric SNR analysis across multiple TTV configurations"""
    
    # Create master run ID
    master_run_id = f"parametric_snr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    output_dir = f"parametric_snr_results_{master_run_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Save metadata
    metadata = save_simulation_metadata(ttv_configurations)
    
    # Initialize master results storage
    all_config_results = {}
    master_results_list = []
    
    print(f"\nStarting parametric SNR analysis")
    print(f"Results will be saved in: {output_dir}")
    
    total_configs = len(ttv_configurations)
    total_tests = total_configs * len(snr_levels) * 3
    current_test = 0
    
    for config_idx, config in enumerate(ttv_configurations):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {config_idx+1}/{total_configs}: {config['name']}")
        print(f"A_TTV={config['A_TTV']}, P_TTV={config['P_TTV']}, E_TTV={config['E_TTV']}")
        print(f"{'='*60}")
        
        # Create sub-directory for this configuration
        config_dir = f"config_{config['name']}"
        os.makedirs(config_dir, exist_ok=True)
        os.chdir(config_dir)
        
        # Results for this configuration
        config_results = {
            'correct_ettv': [],
            'incorrect_ettv': [],
            'no_ttv': [],
            'config': config
        }
        
        lightcurve_files = []
        bls_result_files = []
        
        # Prepare arguments for parallel processing
        snr_args = []
        for i, count_rate in enumerate(snr_levels):
            snr_name = f"{int(count_rate)}"
            run_id = f"{master_run_id}_{config['name']}"
            test_counter = current_test + i * 3
            
            snr_args.append((config, i, count_rate, snr_name, save_lightcurves, save_bls_details, 
                           run_id, test_counter))
        
        # Process SNR levels (parallel or serial)
        if num_cores > 0:
            print(f"  Processing {len(snr_levels)} SNR levels in parallel using {actual_cores} cores...")
            
            with mp.Pool(processes=actual_cores) as pool:
                snr_results = pool.map(process_single_snr_level, snr_args)
        else:
            print(f"  Processing {len(snr_levels)} SNR levels serially...")
            snr_results = []
            for i, args in enumerate(snr_args):
                print(f"    SNR level {i+1}/{len(snr_levels)}: count_rate={args[2]:.0f}")
                result = process_single_snr_level(args)
                snr_results.append(result)
        
        # Process results
        for snr_result in snr_results:
            if snr_result['success']:
                # Collect scenario results
                for scenario_result in snr_result['scenario_results']:
                    scenario_name = scenario_result['scenario']
                    config_results[scenario_name].append(scenario_result)
                    master_results_list.append(scenario_result)
                
                # Collect file names
                if snr_result['lightcurve_file']:
                    lightcurve_files.append(snr_result['lightcurve_file'])
                if snr_result['bls_files']:
                    bls_result_files.extend(snr_result['bls_files'])
                    
                current_test += 3
            else:
                print(f"    ERROR processing SNR level {snr_result['count_rate']}: {snr_result.get('error', 'Unknown error')}")
        
        # Save individual configuration results
        config_results['lightcurve_files'] = lightcurve_files
        config_results['bls_result_files'] = bls_result_files
        
        with open(f'config_results_{config["name"]}.pkl', 'wb') as f:
            pickle.dump(config_results, f)
        
        # Save CSV for this configuration
        config_df_results = []
        for scenario_name in ['correct_ettv', 'incorrect_ettv', 'no_ttv']:
            config_df_results.extend(config_results[scenario_name])
        
        if config_df_results:
            config_df = pd.DataFrame(config_df_results)
            config_df.to_csv(f'config_results_{config["name"]}.csv', index=False)
        
        all_config_results[config['name']] = config_results
        
        print(f"  Configuration {config['name']} complete ({current_test}/{total_tests} tests done)")
        
        os.chdir('..')  # Return to main results directory
    
    # Save master results
    master_results = {
        'master_run_id': master_run_id,
        'configurations': ttv_configurations,
        'metadata': metadata,
        'all_config_results': all_config_results,
        'snr_levels': snr_levels.tolist()
    }
    
    # Save master pickle file
    with open(f'parametric_snr_master_{master_run_id}.pkl', 'wb') as f:
        pickle.dump(master_results, f)
    
    # Save master CSV file
    if master_results_list:
        master_df = pd.DataFrame(master_results_list)
        master_df.to_csv(f'parametric_snr_master_{master_run_id}.csv', index=False)
    
    # Save master JSON file
    json_results = {
        'master_run_id': master_run_id,
        'configurations': ttv_configurations,
        'metadata': metadata,
        'summary_stats': generate_summary_stats(master_df) if master_results_list else {}
    }
    
    with open(f'parametric_snr_master_{master_run_id}.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PARAMETRIC SNR ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in directory: {output_dir}")
    print(f"Master result files:")
    print(f"  - parametric_snr_master_{master_run_id}.pkl (full data)")
    print(f"  - parametric_snr_master_{master_run_id}.csv (spreadsheet format)")
    print(f"  - parametric_snr_master_{master_run_id}.json (summary)")
    print(f"Individual configuration results in subdirectories")
    print(f"Total tests completed: {current_test}/{total_tests}")
    
    if num_cores > 0:
        print(f"Parallelization used {actual_cores} cores")
    
    os.chdir('..')
    return master_results, output_dir

def generate_summary_stats(df):
    """Generate summary statistics from the master dataframe"""
    if df.empty:
        return {}
        
    summary = {}
    
    for config_name in df['config_name'].unique():
        config_data = df[df['config_name'] == config_name]
        
        # Find thresholds for each scenario
        threshold_sde = 7
        config_summary = {'config_name': config_name}
        
        for scenario in ['correct_ettv', 'incorrect_ettv', 'no_ttv']:
            scenario_data = config_data[config_data['scenario'] == scenario]
            above_threshold = scenario_data[scenario_data['sde'] >= threshold_sde]
            
            if len(above_threshold) > 0:
                threshold = above_threshold['count_rate'].min()
                config_summary[f'threshold_{scenario}'] = float(threshold)
            else:
                config_summary[f'threshold_{scenario}'] = None
                
            if len(scenario_data) > 0:
                config_summary[f'max_sde_{scenario}'] = float(scenario_data['sde'].max())
            else:
                config_summary[f'max_sde_{scenario}'] = 0.0
        
        summary[config_name] = config_summary
    
    return summary

# Main execution
if __name__ == "__main__":
    if analysis_mode == 'parametric_snr':
        parametric_snr_analysis()
    else:
        print(f"Analysis mode '{analysis_mode}' not supported")
        print("Use 'parametric_snr' for parametric SNR study")
        sys.exit(1)#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from math import sin, pi, sqrt, fabs, trunc, floor, ceil
from numpy.random import default_rng
from astropy.table import Table, Column, vstack
import multiprocessing as mp
from matplotlib import cm
import time
from numba import jit, njit
from itertools import starmap
import numpy.lib.recfunctions as rfn
import batman
import sys
import os
import json
import configparser
import pickle
import pandas as pd
from datetime import datetime

def read_parameters_from_file():
    """Read simulation parameters from external file"""
    
    if os.path.exists('parameters.json'):
        try:
            with open('parameters.json', 'r') as f:
                params = json.load(f)
            print("Parameters loaded from parameters.json")
            return params
        except Exception as e:
            print(f"Error reading parameters.json: {e}")
    
    elif os.path.exists('parameters.txt'):
        try:
            params = {}
            with open('parameters.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        except ValueError:
                            params[key] = value
            
            print("Parameters loaded from parameters.txt")
            return params
        except Exception as e:
            print(f"Error reading parameters.txt: {e}")
    
    print("ERROR: No parameter file found!")
    print("Please create one of: parameters.json, parameters.txt")
    sys.exit(1)

def save_simulation_metadata(ttv_configs):
    """Save metadata about the simulation setup"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'script_version': '2.0_parametric',
        'simulation_parameters': {
            'cadence': cadence,
            'duration': duration,
            'period': period,
            'epoch': epoch,
            'r_planet': r_planet,
            'min_period': 0.6,
            'max_period': 75.0
        },
        'ttv_configurations': ttv_configs,
        'snr_levels': snr_levels.tolist(),
        'python_version': sys.version,
        'numpy_version': np.__version__
    }
    
    with open('simulation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

# Load parameters
params = read_parameters_from_file()

# Required parameters
required_params = ['cadence', 'duration', 'period', 'epoch', 'r_planet', 'run_name']

# Check for parametric study configuration
analysis_mode = params.get('analysis_mode', 'parametric_snr')
save_data = params.get('save_data', True)
save_lightcurves = params.get('save_lightcurves', True)
save_bls_details = params.get('save_bls_details', True)

missing_params = []
for param in required_params:
    if param not in params:
        missing_params.append(param)

if missing_params:
    print("ERROR: Missing required parameters in parameter file:")
    for param in missing_params:
        print(f"  - {param}")
    print("\nExiting...")
    sys.exit(1)

# Extract base parameters
cadence = params['cadence']
duration = params['duration']
period = params['period']
epoch = params['epoch']
r_planet = params['r_planet']
run_name = params['run_name']

# Define TTV configurations to test
if 'ttv_configurations' in params:
    # User-defined configurations
    ttv_configurations = params['ttv_configurations']
else:
    # Default parametric study configurations
    ttv_configurations = [
        {"A_TTV": 0.005, "P_TTV": 30, "E_TTV": -15.0, "name": "very_weak_short"},
        {"A_TTV": 0.01, "P_TTV": 30, "E_TTV": -15.0, "name": "weak_short"},
        {"A_TTV": 0.01, "P_TTV": 60, "E_TTV": -30.0, "name": "weak_medium"},
        {"A_TTV": 0.01, "P_TTV": 120, "E_TTV": -60.0, "name": "weak_long"},
        {"A_TTV": 0.02, "P_TTV": 30, "E_TTV": -15.0, "name": "moderate_short"},
        {"A_TTV": 0.02, "P_TTV": 60, "E_TTV": -30.0, "name": "moderate_medium"},
        {"A_TTV": 0.02, "P_TTV": 120, "E_TTV": -60.0, "name": "moderate_long"},
        {"A_TTV": 0.05, "P_TTV": 60, "E_TTV": -30.0, "name": "strong_medium"},
        {"A_TTV": 0.05, "P_TTV": 120, "E_TTV": -60.0, "name": "strong_long"},
    ]

# Define SNR levels
snr_levels = np.logspace(2, 6, 15)  # 100 to 1,000,000

print("="*60)
print("PARAMETRIC SNR STUDY")
print("="*60)
print(f"Base parameters:")
print(f"  Cadence: {cadence} seconds")
print(f"  Duration: {duration} days")
print(f"  Period: {period} days")
print(f"  Epoch: {epoch} days")
print(f"  Planet radius: {r_planet}")
print(f"  Run name: {run_name}")
print(f"\nTTV Configurations to test: {len(ttv_configurations)}")
for config in ttv_configurations:
    print(f"  {config['name']}: A={config['A_TTV']}, P={config['P_TTV']}, E={config['E_TTV']}")
print(f"\nSNR levels: {len(snr_levels)} (from {snr_levels[0]:.0f} to {snr_levels[-1]:.0f})")
print(f"Total tests: {len(ttv_configurations) * len(snr_levels) * 3} = {len(ttv_configurations)} configs × {len(snr_levels)} SNR × 3 scenarios")
print("="*60)

rng = default_rng()

# Set plot parameters
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def compute_transit_template(cadence, period, r_planet):
    """Compute transit template using batman"""
    params = batman.TransitParams()
    params.rp = r_planet
    params.a = 15.
    params.inc = 90.
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"
    params.per = period
    params.t0 = 0.0
    
    t = np.arange(-period*0.5, period*0.5, cadence/86400.0)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params) - 1.0
    
    nzidx = np.where(flux != 0.0)[0]
    nz_start = min(nzidx) - 3
    nz_end = max(nzidx) + 3
    
    return t[nz_start:nz_end], flux[nz_start:nz_end], t[nz_end] - t[nz_start]

def create_lightcurve(cadence, duration, period, epoch, a_ttv=0.0, p_ttv=0.0, e_ttv=0.0,
                      count_rate=0.0, r_planet=0.1):
    """Create synthetic lightcurve with TTVs"""
    # Compute transit template
    template_t, template_flux, template_duration = compute_transit_template(cadence, period, r_planet)

    # Create time array
    t = np.arange(0.0, duration, cadence/86400.0, dtype=np.float32)
    flux = np.zeros_like(t)
    
    # Add transits with TTVs
    n = 0
    while True:
        tmid = epoch + n*period
        if p_ttv > 0.0 and a_ttv > 0.0:
            tmid += a_ttv*sin(2*pi*(n*period-e_ttv)/p_ttv)
        if tmid > duration + template_duration:
            break
        flux += np.interp(t-tmid, template_t, template_flux, left=0.0, right=0.0)
        n += 1

    # Add noise
    if count_rate > 0.0:
        sigma = 1.0/sqrt(count_rate*cadence)
        flux = rng.normal(flux, sigma)
        flux_err = np.full_like(flux, sigma, dtype=np.float32)
    else:
        flux_err = np.zeros_like(flux, dtype=np.float32)

    return t, np.array(flux, dtype=np.float32), flux_err

def distort_timebase(t, epoch, p_ttv, a_ttv, e_ttv=0.0):
    """Apply TTV correction to timebase"""
    return t - a_ttv*np.sin(2*pi*(t-(epoch+e_ttv))/p_ttv) if p_ttv > 0.0 and a_ttv > 0.0 else t

def optimal_sample_periods(min_period, max_period, obs_duration, bin_width):
    """Generate optimal period sampling"""
    sample_periods = []
    period = min_period
    while period < max_period:
        sample_periods.append(period)
        deltaFreq = bin_width / (period * obs_duration)
        period = 1.0 / (-deltaFreq + 1.0 / period)
    return np.array(sample_periods)

def transit_search(tstamp, flux, flux_err, min_period=None, max_period=None, sample_periods=None,
                   bin_width=45.0, min_box_width=2, max_box_width=5):
    """Perform BLS transit search"""
    t0 = np.min(tstamp)
    bin_width /= (24*60)  # Convert to days
    obs_duration = np.max(tstamp) - np.min(tstamp)

    if sample_periods is None:
        sample_periods = optimal_sample_periods(min_period, max_period, obs_duration, bin_width)

    weight = 1.0/(flux_err*flux_err)
    wflux = flux*weight
    wflux2 = (flux*flux)*weight
    chisqr0 = np.sum(wflux2)
    T = np.sum(weight)
    pdgram = []
    
    for period in sample_periods:
        num_bins = int((period / bin_width) + 0.5)
        real_bin_width = period/num_bins
        bin_idx = np.floor(np.mod((tstamp-t0)/period, 1.0)*num_bins).astype(np.int32)

        # Compute partial sums
        pstat = np.zeros((4, num_bins), dtype=np.float32)
        pstat[0, :] = np.bincount(bin_idx, wflux, num_bins)
        pstat[1, :] = np.bincount(bin_idx, wflux2, num_bins)
        pstat[2, :] = np.bincount(bin_idx, weight, num_bins)
        pstat[3, :] = np.bincount(bin_idx, minlength=num_bins)

        # Extend arrays for wrap-around
        pstat = np.concatenate((pstat, pstat[:max_box_width, :]), axis=1)

        # Stack shifted copies
        pstat = np.stack([np.roll(pstat, -shft, axis=1)
                         for shft in range(max_box_width+1)], axis=2)

        # Accumulate sums
        sstat = np.cumsum(pstat, axis=-1)[:, :, min_box_width:]

        # Compute delta-chisq
        S = sstat[0, :, :]
        Q = sstat[1, :, :]
        R = sstat[2, :, :]
        N = sstat[3, :, :]
        denom = (R*(T-R))
        delta_chisq = np.where((N>2) * (denom!=0.0), -(S*S)*T/denom, 0.0)

        # Find best detection
        min_idx = np.unravel_index(np.argmin(delta_chisq), delta_chisq.shape)
        delta_chisq = delta_chisq[min_idx]
        epoch = (min_idx[0]+(min_idx[1]+min_box_width)*0.5)*real_bin_width
        depth = S[min_idx]*T/(R[min_idx]*(T-R[min_idx]))
        depth_err = sqrt(T/(R[min_idx]*(T-R[min_idx])))
        num_pts = N[min_idx]
        width = (min_idx[1]+min_box_width)*real_bin_width
        pdgram.append((period, delta_chisq, epoch+t0, depth, depth_err, num_pts, width))

    # Create results array
    res = np.array(pdgram, dtype=[('period', 'f8'), ('delta_chisq', 'f4'), ('epoch', 'f4'), ('depth', 'f4'),
                                  ('depth_err', 'f4'), ('num_pts_in_transit', 'i4'), ('width', 'f4')])

    # Compute SDE
    median = np.median(res['delta_chisq'])
    rms = np.median(np.abs(res['delta_chisq']-median))*1.48
    res = rfn.append_fields(res, 'sde', (median-res['delta_chisq'])/rms, dtypes=('f4'), usemask=False)

    return res

def ttv_bls_search(t, flux, flux_err, p_ttv, a_ttv, e_ttv, min_period, max_period):
    """Single TTV-BLS search"""
    t_prime = distort_timebase(t, 0.0, p_ttv, a_ttv, e_ttv)
    res = transit_search(t_prime, flux, flux_err, min_period, max_period)
    best_idx = np.argmax(res['sde'])
    return res[best_idx]

def save_lightcurve_data(t, flux, flux_err, config, count_rate, run_id):
    """Save lightcurve data"""
    lc_data = {
        'time': t.tolist(),
        'flux': flux.tolist(),
        'flux_err': flux_err.tolist(),
        'config': config,
        'count_rate': count_rate,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"lightcurve_data_{run_id}_snr_{int(count_rate)}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(lc_data, f)
    
    return filename

def save_bls_results(bls_result, config, count_rate, scenario, run_id):
    """Save BLS results"""
    # Convert numpy structured array to dictionary
    result_data = {}
    for field_name in bls_result.dtype.names:
        value = bls_result[field_name]
        if hasattr(value, 'tolist'):
            result_data[field_name] = value.tolist()
        else:
            result_data[field_name] = float(value) if np.isscalar(value) else value
    
    full_result = {
        'bls_result': result_data,
        'config': config,
        'count_rate': count_rate,
        'scenario': scenario,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"bls_result_{run_id}_snr_{int(count_rate)}_{scenario}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(full_result, f)
    
    return filename

def parametric_snr_analysis():
    """Perform parametric SNR analysis across multiple TTV configurations"""
    
    # Create master run ID
    master_run_id = f"parametric_snr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    output_dir = f"parametric_snr_results_{master_run_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Save metadata
    metadata = save_simulation_metadata(ttv_configurations)
    
    # Initialize master results storage
    all_config_results = {}
    master_results_list = []
    
    print(f"\nStarting parametric SNR analysis")
    print(f"Results will be saved in: {output_dir}")
    
    total_configs = len(ttv_configurations)
    total_tests = total_configs * len(snr_levels) * 3
    current_test = 0
    
    for config_idx, config in enumerate(ttv_configurations):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {config_idx+1}/{total_configs}: {config['name']}")
        print(f"A_TTV={config['A_TTV']}, P_TTV={config['P_TTV']}, E_TTV={config['E_TTV']}")
        print(f"{'='*60}")
        
        # Create sub-directory for this configuration
        config_dir = f"config_{config['name']}"
        os.makedirs(config_dir, exist_ok=True)
        os.chdir(config_dir)
        
        # Results for this configuration
        config_results = {
            'correct_ettv': [],
            'incorrect_ettv': [],
            'no_ttv': [],
            'config': config
        }
        
        lightcurve_files = []
        bls_result_files = []
        
        for i, count_rate in enumerate(snr_levels):
            print(f"\n  SNR level {i+1}/{len(snr_levels)}: count_rate={count_rate:.0f}")
            
            # Create lightcurve for this configuration and SNR
            start_time = time.time()
            t, flux, flux_err = create_lightcurve(cadence, duration, period, epoch, 
                                                config['A_TTV'], config['P_TTV'], config['E_TTV'],
                                                count_rate=count_rate, r_planet=r_planet)
            lc_creation_time = time.time() - start_time
            
            # Save lightcurve data
            if save_lightcurves:
                lc_file = save_lightcurve_data(t, flux, flux_err, config, count_rate, 
                                             f"{master_run_id}_{config['name']}")
                lightcurve_files.append(lc_file)
            
            # Test scenarios
            scenarios = [
                {'name': 'correct_ettv', 'params': (config['P_TTV'], config['A_TTV'], config['E_TTV'])},
                {'name': 'incorrect_ettv', 'params': (config['P_TTV'], config['A_TTV'], 0.0)},
                {'name': 'no_ttv', 'params': (0.0, 0.0, 0.0)}
            ]
            
            for j, scenario in enumerate(scenarios):
                current_test += 1
                print(f"    Test {current_test}/{total_tests}: {scenario['name']}")
                
                start_time = time.time()
                try:
                    p_ttv_param, a_ttv_param, e_ttv_param = scenario['params']
                    bls_result = ttv_bls_search(t, flux, flux_err, p_ttv_param, a_ttv_param, e_ttv_param, 0.6, 75.0)
                    
                    # Save detailed BLS results
                    if save_bls_details:
                        bls_file = save_bls_results(bls_result, config, count_rate, scenario['name'], 
                                                  f"{master_run_id}_{config['name']}")
                        bls_result_files.append(bls_file)
                    
                    summary_result = {
                        'config_name': config['name'],
                        'a_ttv': config['A_TTV'],
                        'p_ttv': config['P_TTV'],
                        'e_ttv': config['E_TTV'],
                        'count_rate': count_rate,
                        'snr_level': f"{int(count_rate)}",
                        'sde': float(bls_result['sde']),
                        'period': float(bls_result['period']),
                        'depth': float(bls_result['depth']),
                        'epoch': float(bls_result['epoch']),
                        'width': float(bls_result['width']),
                        'num_pts_in_transit': int(bls_result['num_pts_in_transit']),
                        'scenario': scenario['name'],
                        'ttv_params': scenario['params'],
                        'success': True
                    }
                    
                except Exception as e:
                    print(f"      Error in {scenario['name']} test: {e}")
                    summary_result = {
                        'config_name': config['name'],
                        'a_ttv': config['A_TTV'],
                        'p_ttv': config['P_TTV'],
                        'e_ttv': config['E_TTV'],
                        'count_rate': count_rate,
                        'snr_level': f"{int(count_rate)}",
                        'sde': 0, 'period': 0, 'depth': 0, 'epoch': 0, 'width': 0,
                        'num_pts_in_transit': 0, 'scenario': scenario['name'],
                        'ttv_params': scenario['params'], 'success': False,
                        'error': str(e)
                    }
                
                computation_time = time.time() - start_time
                summary_result.update({
                    'computation_time': computation_time,
                    'lightcurve_creation_time': lc_creation_time if j == 0 else None
                })
                
                config_results[scenario['name']].append(summary_result)
                master_results_list.append(summary_result)
        
        # Save individual configuration results
        config_results['lightcurve_files'] = lightcurve_files
        config_results['bls_result_files'] = bls_result_files
        
        with open(f'config_results_{config["name"]}.pkl', 'wb') as f:
            pickle.dump(config_results, f)
        
        # Save CSV for this configuration
        config_df_results = []
        for scenario_name in ['correct_ettv', 'incorrect_ettv', 'no_ttv']:
            config_df_results.extend(config_results[scenario_name])
        
        config_df = pd.DataFrame(config_df_results)
        config_df.to_csv(f'config_results_{config["name"]}.csv', index=False)
        
        all_config_results[config['name']] = config_results
        
        print(f"  Configuration {config['name']} complete")
        
        os.chdir('..')  # Return to main results directory
    
    # Save master results
    master_results = {
        'master_run_id': master_run_id,
        'configurations': ttv_configurations,
        'metadata': metadata,
        'all_config_results': all_config_results,
        'snr_levels': snr_levels.tolist()
    }
    
    # Save master pickle file
    with open(f'parametric_snr_master_{master_run_id}.pkl', 'wb') as f:
        pickle.dump(master_results, f)
    
    # Save master CSV file
    master_df = pd.DataFrame(master_results_list)
    master_df.to_csv(f'parametric_snr_master_{master_run_id}.csv', index=False)
    
    # Save master JSON file
    json_results = {
        'master_run_id': master_run_id,
        'configurations': ttv_configurations,
        'metadata': metadata,
        'summary_stats': generate_summary_stats(master_df)
    }
    
    with open(f'parametric_snr_master_{master_run_id}.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PARAMETRIC SNR ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in directory: {output_dir}")
    print(f"Master result files:")
    print(f"  - parametric_snr_master_{master_run_id}.pkl (full data)")
    print(f"  - parametric_snr_master_{master_run_id}.csv (spreadsheet format)")
    print(f"  - parametric_snr_master_{master_run_id}.json (summary)")
    print(f"Individual configuration results in subdirectories")
    
    os.chdir('..')
    return master_results, output_dir

def generate_summary_stats(df):
    """Generate summary statistics from the master dataframe"""
    summary = {}
    
    for config_name in df['config_name'].unique():
        config_data = df[df['config_name'] == config_name]
        
        # Find thresholds for each scenario
        threshold_sde = 7
        config_summary = {'config_name': config_name}
        
        for scenario in ['correct_ettv', 'incorrect_ettv', 'no_ttv']:
            scenario_data = config_data[config_data['scenario'] == scenario]
            above_threshold = scenario_data[scenario_data['sde'] >= threshold_sde]
            
            if len(above_threshold) > 0:
                threshold = above_threshold['count_rate'].min()
                config_summary[f'threshold_{scenario}'] = float(threshold)
            else:
                config_summary[f'threshold_{scenario}'] = None
                
            if len(scenario_data) > 0:
                config_summary[f'max_sde_{scenario}'] = float(scenario_data['sde'].max())
            else:
                config_summary[f'max_sde_{scenario}'] = 0.0
        
        summary[config_name] = config_summary
    
    return summary

# Main execution
if __name__ == "__main__":
    if analysis_mode == 'parametric_snr':
        parametric_snr_analysis()
    else:
        print(f"Analysis mode '{analysis_mode}' not supported")
        print("Use 'parametric_snr' for parametric SNR study")
        sys.exit(1)
