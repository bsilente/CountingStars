import sys
import os
import argparse
import importlib.util
from datetime import datetime
import pandas as pd

EXPERIMENTS = {

    # Test verison
    "base": "configs.base_config",

    # Experiment reproduction
    "Iridium_load_0_1": "configs.Iridium_0_1",
    "Iridium_load_0_5": "configs.Iridium_0_5",
    "Iridium_load_0_9": "configs.Iridium_0_9",
    "Starlink_load_0_1": "configs.starlink_0_1",
    "Starlink_load_0_5": "configs.starlink_0_5",
    "Starlink_load_0_9": "configs.starlink_0_9",
}

def load_config_module(config_path):
    try:
        module_spec = importlib.util.find_spec(config_path)
        if module_spec is None:
            print(f"Error: Could not find config file module '{config_path}'.")
            print("Please check:")
            print(f"  1. If a file named '{config_path.split('.')[-1]}.py' exists in the configs folder.")
            print(f"  2. If the path '{config_path}' in the EXPERIMENTS dictionary is correct.")
            return None
        
        config_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(config_module)
        sys.modules['config'] = config_module
        return config_module
    except Exception as e:
        print(f"An error occurred while loading the config file '{config_path}': {e}")
        return None

def apply_overrides(config_module, cli_args):
    if cli_args.kb is not None:
        print(f"▶️  Command line override parameter: kb = {cli_args.kb}")
        config_module.kb = cli_args.kb
        print("    -> Recalculating parameters dependent on kb...")
        if hasattr(config_module, 'memory_all'):
             config_module.memory_all = 128 * config_module.kb
             print(f"       -> memory_all updated to: {config_module.memory_all}")
        if hasattr(config_module, 'memory_all'):
            config_module.MEMORY_POOL_SIZE = config_module.memory_all
            config_module.ELASTIC_SKETCH_TOTAL_MEMORY_BYTES = config_module.memory_all * 8
            config_module.CM_SKETCH_MEMORY_BYTES = config_module.memory_all * 8
            config_module.BFCM_SKETCH_TOTAL_MEMORY_BYTES = config_module.memory_all * 8
            print("       -> Memory for MEMORY_POOL_SIZE, ELASTIC_SKETCH, CM_SKETCH, BFCM_SKETCH has been updated.")

def run_single_experiment(experiment_name: str, cli_args: argparse.Namespace):
    """Runs a single specified experiment and applies command-line override parameters."""
    if experiment_name not in EXPERIMENTS:
        print(f"Error: Experiment '{experiment_name}' is not defined in the EXPERIMENTS dictionary.")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return None

    config_path = EXPERIMENTS[experiment_name]
    print(f"\n{'='*60}")
    print(f"▶️  Starting experiment: {experiment_name}")
    print(f"▶️  Loading config file: {config_path}")

    config_module = load_config_module(config_path)
    if not config_module:
        return None

    apply_overrides(config_module, cli_args)

    from main_simulation import main_runner
    
    print(f"{'='*60}")

    results = main_runner(config_module, show_progress=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Experiment {experiment_name} complete.")
    print(f"{'='*60}\n")
    
    if results and "error" not in results:
        final_kb = config_module.kb
        return { 
            "experiment_name": experiment_name,
            "final_kb": final_kb, 
            **results 
        }
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run network simulation experiments. You can specify a base experiment configuration and optionally override parameters via the command line.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        help="Specify the name of the base experiment configuration to run.\n"
             f"Available options: {list(EXPERIMENTS.keys())}"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run all experiments defined in the EXPERIMENTS dictionary in order.\n"
             "Note: Override parameters cannot be specified when using --all."
    )
    
    parser.add_argument(
        "--kb",
        type=int,
        default=None,
        help="Override the kb value in the config file via the command line.\n"
             "This is an integer value."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=f"results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Specify the CSV file path to save the summary of all experiment results."
    )

    args = parser.parse_args()
    
    if args.all and args.kb is not None:
        parser.error("Argument conflict: Cannot specify override parameters (--kb) while running all experiments (--all).")
        return

    if not args.all and not args.experiment:
        parser.print_help()
        print("\nPlease provide an action: Specify an experiment (-e) or run all experiments (-a).")
        return

    all_results = []

    if args.all:
        for name in EXPERIMENTS:
            result = run_single_experiment(name, args)
            if result:
                all_results.append(result)
    elif args.experiment:
        result = run_single_experiment(args.experiment, args)
        if result:
            all_results.append(result)

    if all_results:
        print(f"All experiments have finished. Collected {len(all_results)} results in total.")
        
        df = pd.DataFrame(all_results)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            df.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"Results summary has been saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results to CSV file: {e}")


if __name__ == "__main__":
    main()
