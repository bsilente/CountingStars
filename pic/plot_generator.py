import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import os
import argparse

def plot_metrics_from_file(file_path, config, output_path):
    try:
        data = pd.read_excel(file_path, header=0) 
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'. Please check if the file path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}': {e}")
        return
        
    metrics_to_plot = config.get('metrics', {})
    x_axis_col = config.get('x_axis_col')
    methods = config.get('methods', {})
    figure_size = config.get('figure_size', (18, 5)) 

    if x_axis_col is None:
        print("Error: 'x_axis_col' is not specified in the config.")
        return

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figure_size)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, (metric_name, method_mappings) in zip(axes, metrics_to_plot.items()):
        x_axis_data = data.iloc[:, x_axis_col].rename('memory_KB')
        plot_df = pd.DataFrame({'memory_KB': x_axis_data})
        for method_name, y_axis_col in method_mappings.items():
            if y_axis_col >= len(data.columns):
                print(f"Warning: The configured column index {y_axis_col} for '{method_name}' is out of range, skipping.")
                continue
            plot_df[method_name] = data.iloc[:, y_axis_col]
        plot_df_sorted = plot_df.sort_values(by='memory_KB').reset_index(drop=True)
        for method_name in method_mappings.keys():
            if method_name in plot_df_sorted:
                style = methods.get(method_name, {})
                ax.plot(plot_df_sorted['memory_KB'], plot_df_sorted[method_name],
                        label=method_name,
                        marker=style.get('marker', 'o'),
                        color=style.get('color', 'k'),
                        markersize=8,
                        linestyle='-',
                        linewidth=2)
        ax.set_xlabel('memory usage (KB)', fontsize=26)
        ax.set_ylabel(metric_name, fontsize=26)
        ax.legend(fontsize=14)
        filename_lower = os.path.basename(file_path).lower()
        if 'iridium' in filename_lower:
            major_ticks = np.arange(2, 11, 2)
            ax.set_xticks(major_ticks)
            ax.set_xlim(1.8, 10.2)
        elif 'starlink' in filename_lower:
            major_ticks = np.arange(200, 1001, 200)
            ax.set_xticks(major_ticks)
            ax.set_xlim(180, 1020)
        else:
            major_ticks = np.arange(200, 1001, 200)
            ax.set_xticks(major_ticks)

        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(which='major', linestyle=':', linewidth='1.0', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='1.0', color='gray')
        ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Chart has been saved to: {output_path}")

def main(args):
    CONFIG = {
        'figure_size': (16, 4),
        'x_axis_col': 12,
        'metrics': {
            'ARE': {
                'CountingStars': 0, 'Elastic Sketch': 3, 'Count-Min Sketch': 6, 'FlowLIDAR': 9
            },
            'WMRE': {
                'CountingStars': 1, 'Elastic Sketch': 4, 'Count-Min Sketch': 7, 'FlowLIDAR': 10
            },
            'RE': {
                'CountingStars': 2, 'Elastic Sketch': 5, 'Count-Min Sketch': 8, 'FlowLIDAR': 11
            }
        },
        'methods': {
            'CountingStars':    {'marker': 'o', 'color': '#1f77b4'},
            'Elastic Sketch':   {'marker': 'x', 'color': '#ff7f0e'},
            'Count-Min Sketch': {'marker': '^', 'color': '#2ca02c'},
            'FlowLIDAR':        {'marker': 's', 'color': '#d62728'}
        }
    }    
    input_directory = args.input_dir
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(input_directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(input_directory, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_filename)
            plot_metrics_from_file(file_path, CONFIG, output_path)

    print("\nAll Excel files have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process Excel files and generate a chart for each file.")
    parser.add_argument('-i', '--input-dir', default='.', 
                        help='Path to the input directory containing Excel files.')
    parser.add_argument('-o', '--output-dir', default='.', 
                        help='Path to the output directory for saving the generated charts.')
    args = parser.parse_args()
    main(args)
