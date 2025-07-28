# run_experiments.py
# 这是新的仿真项目主入口，支持 "base_config + override" 模式和命令行参数覆盖

import sys
import os
import argparse
import importlib.util
from datetime import datetime
import pandas as pd

# ------------------------------------------------------------------------------
# [BUG FIX] 关键修改:
# 我们不再在文件顶部导入 main_runner。
# 因为 main_simulation 模块会触发对 config 的早期导入。
# 我们将把导入语句移动到 run_single_experiment 函数内部，
# 确保它在我们的动态配置加载之后执行。
# 
# from main_simulation import main_runner  <- 删除这一行
# ------------------------------------------------------------------------------


# --- 定义你的实验 ---
# 在这里列出所有你想要运行的实验。
# 键 (key) 是你将在命令行中使用的实验名称。
# 值 (value) 是对应的特定实验配置文件路径 (使用点号分隔)。
# 这些配置文件应该从 base_config 导入并覆盖部分参数。
EXPERIMENTS = {
    "exp_base": "configs.base_config",
    # 示例: 假设您在 configs/ 文件夹下创建了6个配置文件
    "Iridium_load_0_1": "configs.Iridium_0_1",
    "Iridium_load_0_5": "configs.Iridium_0_5",
    "Iridium_load_0_9": "configs.Iridium_0_9",
    "Starlink_load_0_1": "configs.starlink_0_1",
    "Starlink_load_0_5": "configs.starlink_0_5",
    "Starlink_load_0_9": "configs.starlink_0_9",
}

def load_config_module(config_path):
    """动态加载指定的配置文件作为模块"""
    try:
        # 这一步会自动处理配置文件内部的 "from .base_config import *"
        module_spec = importlib.util.find_spec(config_path)
        if module_spec is None:
            print(f"错误: 找不到配置文件模块 '{config_path}'。")
            print("请检查：")
            print(f"  1. configs 文件夹下是否存在名为 '{config_path.split('.')[-1]}.py' 的文件。")
            print(f"  2. EXPERIMENTS 字典中的路径 '{config_path}' 是否正确。")
            return None
        
        config_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(config_module)
        
        # 关键步骤: 将加载的模块放入 sys.modules 中，并命名为 'config'。
        # 这样，当其他文件 (如 simulation_core) 执行 `import config` 时，
        # 它们会获取到我们刚刚为本次实验加载的特定配置模块。
        sys.modules['config'] = config_module
        return config_module
    except Exception as e:
        print(f"加载配置文件 '{config_path}' 时发生错误: {e}")
        return None

def apply_overrides(config_module, cli_args):
    """
    将命令行传入的参数应用（覆盖）到已加载的配置模块上。
    """
    # 检查用户是否通过命令行传入了 kb 参数
    if cli_args.kb is not None:
        print(f"▶️  命令行覆盖参数: kb = {cli_args.kb}")
        
        # 1. 直接覆盖 kb 的值
        config_module.kb = cli_args.kb
        
        # 2. 重新计算所有依赖于 kb 的参数，确保一致性
        print("    -> 正在重新计算依赖于 kb 的参数...")
        
        # 重新计算 memory_all
        if hasattr(config_module, 'memory_all'):
             config_module.memory_all = 128 * config_module.kb
             print(f"       -> memory_all 更新为: {config_module.memory_all}")
        
        # 重新计算所有依赖于 memory_all 的参数
        if hasattr(config_module, 'memory_all'):
            config_module.MEMORY_POOL_SIZE = config_module.memory_all
            config_module.ELASTIC_SKETCH_TOTAL_MEMORY_BYTES = config_module.memory_all * 8
            config_module.CM_SKETCH_MEMORY_BYTES = config_module.memory_all * 8
            config_module.BFCM_SKETCH_TOTAL_MEMORY_BYTES = config_module.memory_all * 8
            print("       -> MEMORY_POOL_SIZE, ELASTIC_SKETCH, CM_SKETCH, BFCM_SKETCH 的内存已更新。")


def run_single_experiment(experiment_name: str, cli_args: argparse.Namespace):
    """运行单个指定的实验，并应用命令行覆盖参数"""
    if experiment_name not in EXPERIMENTS:
        print(f"错误: 实验 '{experiment_name}' 未在 EXPERIMENTS 字典中定义。")
        print(f"可用实验: {list(EXPERIMENTS.keys())}")
        return None

    config_path = EXPERIMENTS[experiment_name]
    print(f"\n{'='*60}")
    print(f"▶️  开始运行实验: {experiment_name}")
    print(f"▶️  加载配置文件: {config_path}")
    
    # 1. 加载配置并注入到 sys.modules
    config_module = load_config_module(config_path)
    if not config_module:
        return None

    # 2. 应用命令行覆盖
    apply_overrides(config_module, cli_args)
    
    # --------------------------------------------------------------------------
    # [BUG FIX] 关键修改:
    # 只有在配置被完全加载和设置好之后，我们才导入 main_simulation。
    # 这确保了 main_simulation 及其所有子模块在导入时
    # 会看到我们刚刚注入的、正确的 'config' 模块。
    from main_simulation import main_runner
    # --------------------------------------------------------------------------
    
    print(f"{'='*60}")

    # 3. 调用仿真主函数
    results = main_runner(config_module, show_progress=True)
    
    print(f"\n{'='*60}")
    print(f"✅ 实验 {experiment_name} 完成。")
    print(f"{'='*60}\n")
    
    # 在结果中加入实验名称，方便追溯
    if results and "error" not in results:
        final_kb = config_module.kb
        return { 
            "experiment_name": experiment_name,
            "final_kb": final_kb, 
            **results 
        }
    return None


def main():
    """主函数，解析命令行参数并启动实验"""
    parser = argparse.ArgumentParser(
        description="运行网络仿真实验。可以指定一个基础实验配置，并选择性地通过命令行覆盖参数。",
        formatter_class=argparse.RawTextHelpFormatter # 保持换行格式
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        help="指定要运行的基础实验配置的名称。\n"
             f"可用选项: {list(EXPERIMENTS.keys())}"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="按顺序运行所有在 EXPERIMENTS 字典中定义的实验。\n"
             "注意: 当使用 --all 时，无法同时指定覆盖参数。"
    )
    
    parser.add_argument(
        "--kb",
        type=int,
        default=None, # 默认不覆盖
        help="通过命令行覆盖配置文件中的 kb 值。\n"
             "这是一个整数值。"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=f"results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="指定用于保存所有实验结果摘要的CSV文件路径。"
    )

    args = parser.parse_args()
    
    if args.all and args.kb is not None:
        parser.error("参数冲突: 不能在运行所有实验 (--all) 的同时指定覆盖参数 (--kb)。")
        return

    if not args.all and not args.experiment:
        parser.print_help()
        print("\n请提供一个操作: 指定一个实验 (-e) 或运行所有实验 (-a)。")
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
        print(f"所有实验运行完毕。共收集到 {len(all_results)} 份结果。")
        
        df = pd.DataFrame(all_results)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            df.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"结果摘要已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果到CSV文件时出错: {e}")


if __name__ == "__main__":
    main()
