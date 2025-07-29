import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import os
import argparse

def plot_metrics_from_file(file_path, config, output_path):
    """
    从指定的 Excel 文件读取数据，根据配置生成图表，并将其保存到指定路径。

    Args:
        file_path (str): 输入的 Excel 文件路径。
        config (dict): 包含绘图设置的配置字典。
        output_path (str): 生成的图表图片的保存路径。
    """
    try:
        # 读取Excel文件，将第一行作为表头
        data = pd.read_excel(file_path, header=0) 
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}")
        return
        
    # 从配置中获取绘图所需的参数
    metrics_to_plot = config.get('metrics', {})
    x_axis_col = config.get('x_axis_col')
    methods = config.get('methods', {})
    figure_size = config.get('figure_size', (18, 5)) 

    if x_axis_col is None:
        print("错误：配置中未指定 'x_axis_col'。")
        return

    # 创建一个图表，包含多个子图
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figure_size)
    # 如果只有一个子图，将其放入列表中以便统一处理
    if len(metrics_to_plot) == 1:
        axes = [axes]

    # 遍历每个评价指标并绘制对应的子图
    for ax, (metric_name, method_mappings) in zip(axes, metrics_to_plot.items()):
        
        # 提取X轴数据
        x_axis_data = data.iloc[:, x_axis_col].rename('memory_KB')
        plot_df = pd.DataFrame({'memory_KB': x_axis_data})

        # 提取Y轴数据
        for method_name, y_axis_col in method_mappings.items():
            if y_axis_col >= len(data.columns):
                print(f"警告：为 '{method_name}' 配置的列索引 {y_axis_col} 超出范围，跳过。")
                continue
            plot_df[method_name] = data.iloc[:, y_axis_col]

        # 按X轴数据排序
        plot_df_sorted = plot_df.sort_values(by='memory_KB').reset_index(drop=True)

        # 绘制每条曲线
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

        # 设置坐标轴标签和图例
        ax.set_xlabel('memory usage (KB)', fontsize=26)
        ax.set_ylabel(metric_name, fontsize=26)
        ax.legend(fontsize=14)

        # 根据文件名动态设置X轴范围和刻度
        filename_lower = os.path.basename(file_path).lower()
        if 'iridium' in filename_lower:
            major_ticks = np.arange(2, 11, 2)
            ax.set_xticks(major_ticks)
            ax.set_xlim(1.8, 10.2) # 设置一个略宽的范围以避免标记被裁剪
        elif 'starlink' in filename_lower:
            major_ticks = np.arange(200, 1001, 200)
            ax.set_xticks(major_ticks)
            ax.set_xlim(180, 1020) # 设置一个略宽的范围
        else:
            # 如果文件名中不包含关键字，则使用默认范围
            major_ticks = np.arange(200, 1001, 200)
            ax.set_xticks(major_ticks)

        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(which='major', linestyle=':', linewidth='1.0', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='1.0', color='gray')
        ax.tick_params(axis='both', which='major', labelsize=20)


    plt.tight_layout()
    # 保存图表到文件，而不是显示它
    plt.savefig(output_path, bbox_inches='tight')
    # 关闭图表以释放内存
    plt.close(fig)
    print(f"图表已保存到: {output_path}")

def main(args):
    """
    主函数，用于处理命令行参数并执行绘图逻辑。
    """
    # 绘图的全局配置
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

    # 如果输出目录不存在，则创建它
    os.makedirs(output_directory, exist_ok=True)
    
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_directory):
        # 检查文件是否为Excel文件
        if filename.endswith('.xlsx'):
            # 构建完整的文件路径
            file_path = os.path.join(input_directory, filename)
            # 构建输出的PNG文件名（将.xlsx替换为.png）
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_filename)
            
            # 调用函数进行绘图和保存
            plot_metrics_from_file(file_path, CONFIG, output_path)

    print("\n所有Excel文件处理完毕。")


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="批量处理Excel文件并为每个文件生成图表。")
    
    # 添加输入目录参数
    parser.add_argument('-i', '--input-dir', default='.', 
                        help='包含Excel文件的输入目录路径。(默认: 当前目录)')
    
    # 添加输出目录参数
    parser.add_argument('-o', '--output-dir', default='.', 
                        help='用于保存生成图表的输出目录路径。(默认: 当前目录)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数
    main(args)
