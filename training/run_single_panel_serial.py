'''
Worker: runs for a single TEST_PANEL, then exits. (called by run_BO_raw_all_panels.py)

'''

import os
import sys
from pathlib import Path



if len(sys.argv) not in (2, 3):
    raise SystemExit("usage: run_single_panel_serial.py <panel> [start_freq]")
os.environ["SHM_TEST_PANEL"] = sys.argv[1]
_start_freq = int(sys.argv[2]) if len(sys.argv) == 3 else 1

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from BO_AE import main as BO_AE_main
from path_performance import main as pp_main
from features_extractor import pre_compute_features_for_all_panels
from sensitivity_study import run_raw_sweep, run_types_sweep, run_pre_processed_sweep
from Fitness_test_metrics import main as fitness_test_metrics_main
from Compute_WAE import main as WAE_main
from Fitness_summary import main as metrics_summary_main
from extract_shi import pre_compute_AE_output

if __name__ == "__main__":

    #Optimize and train AE for every path and frequency (BO is runned only for the first path in each frequency)
    print("=== Running BO for AE training for all panels ===")
    BO_AE_main()

    #Evaluate performance of AEs
    out_dir = _PROJECT_ROOT / f"test_{sys.argv[1]}_wo123" /"path_performance_results"
    pp_main(out_dir)

    # compute and cache the AE output
    print("=== Pre-computing AE output for all panels ===")
    pre_compute_AE_output() 
  
    #Run BO for CAE-GCN workflow and evaluate perforamnce 
    print("=== Running pre-processed features sweep for GCN workflow ===")
    run_pre_processed_sweep(start_freq=_start_freq)

    #Extract features for GCN dataset
    print("=== Pre-computing features for all panels ===")
    pre_compute_features_for_all_panels()

    #Run BO for GCN workflow with raw features and evaluate performance
    print("=== Running raw features sweep for GCN workflow ===")
    run_raw_sweep(start_freq=_start_freq)

    #Run BO for different adjencency types and evaluate performance (not present in the paper)
    #run_types_sweep(start_freq=_start_freq)

    #Evaluate test fitness metrics for all the models (AE, GCN, CAE-GCN) and for all the frequencies
    print("=== Running test fitness metrics for all models and frequencies ===")
    fitness_test_metrics_main()

    #Compute WAE  metrics for all the models (AE, GCN, CAE-GCN) 
    print("=== Running WAE metrics for all models and frequencies ===")
    WAE_main()

    #Stores metrics for every model type and frequnecy and validation fold in a xlsx file
    print("=== Running metrics summary for all models and frequencies ===")
    metrics_summary_main()


