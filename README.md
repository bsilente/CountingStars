🚀 LEO卫星网络中流量测量算法仿真平台
本项目是一个基于Python的离散事件仿真平台，旨在模拟和评估在动态变化的低地球轨道（LEO）卫星网络拓扑中，多种流量测量算法的性能。该平台是为复现相关学术论文中的实验结果而设计的。

📝 项目简介
随着LEO卫星星座（如Starlink和Iridium）的部署，网络拓扑结构变得高度动态。在这种环境下，准确、高效地测量网络流量（即大象流检测）对于网络管理、拥塞控制和安全至关重要。本项目模拟了一个动态的卫星网络环境，其中卫星节点间的链路会随时间变化。平台实现了多种经典的流量测量算法，并对它们在不同网络负载和拓扑下的关键性能指标（如准确性、错误率、吞吐量、延迟等）进行评估。

本项目实现并比较了以下算法：

CountingStars: 一种基于内存池的精确计数方法。

Elastic Sketch: 一种结合重部（Heavy Part）和轻部（Light Part）的哈希表结构，用于流量估计。

Count-Min Sketch: 一种经典的基于多哈希函数的流量估计算法。

BF+CM Sketch (FlowLiDAR): 一种结合布隆过滤器（Bloom Filter）和CM-Sketch的混合方法，用于高效识别和计数新出现的流。

🏗️ 文件结构
.
├── configs/                  # 存放不同实验场景的配置文件
│   ├── base_config.py
│   ├── Iridium_0_1.py
│   └── ...
├── data/                     # 存放拓扑和流量数据
│   ├── adj/                  # 存放邻接矩阵文件 (topology_t_X.csv)
│   └── traffic/              # 存放流量矩阵文件 (traffic_t_X.csv)
├── results/                  # 存放仿真结果的输出目录
├── main_simulation.py        # 仿真主逻辑，包括事件处理和统计数据收集
├── run_experiments.py        # 实验运行脚本，用于启动和管理不同的实验配置
├── simulation_core.py        # 仿真核心组件和测量算法的实现
├── tcp_flow.py               # TCP流量生成器的实现
├── utils.py                  # 辅助函数 (读取数据, 寻路算法)
└── README.md                 # 项目说明文件

🛠️ 环境准备
克隆代码库:

git clone <your-repository-url>
cd <your-repository-name>

安装依赖:
本项目依赖于 pandas 和 numpy。建议创建一个虚拟环境并安装依赖。

pip install pandas numpy

或者，你可以创建一个 requirements.txt 文件并写入以下内容:

pandas
numpy

然后运行:

pip install -r requirements.txt

准备数据:

将描述网络拓扑随时间变化的邻接矩阵文件（例如 topology_t_0.csv, topology_t_1.csv, ...）放入 data/adj/ 目录中。

将描述节点间流量需求的流量矩阵文件（例如 traffic_t_0.csv, traffic_t_1.csv, ...）放入 data/traffic/ 目录中。

确保文件名和路径与 configs 目录中配置文件里的设置相匹配。

🔬 如何复现论文结果
复现实验的核心是运行 run_experiments.py 脚本。该脚本会加载指定的配置文件，启动仿真，并输出结果。

1. 理解配置文件
所有的实验参数都在 configs/ 目录下的Python文件中定义。例如，configs/Iridium_0_1.py 可能定义了使用铱星（Iridium）拓扑和0.1负载的实验场景。关键配置项包括：

NUM_SATELLITES: 卫星节点的数量。

SIMULATION_END_TIME: 仿真总时长（秒）。

TRAFFIC_STOP_TIME: 流量生成停止时间（秒）。

ADJACENCY_MATRIX_DIR, TRAFFIC_MATRIX_DIR: 数据文件所在的目录。

MEMORY_POOL_SIZE, ELASTIC_SKETCH_TOTAL_MEMORY_BYTES, etc.: 各个测量算法的内存大小配置。

ENABLE_ELASTIC_SKETCH, ENABLE_CM_SKETCH, ENABLE_BFCM_SKETCH: 布尔开关，用于决定在仿真中启用哪些算法。

2. 运行单个实验
你可以通过 -e 或 --experiment 参数指定要运行的实验名称。实验名称在 run_experiments.py 文件的 EXPERIMENTS 字典中定义。

例如，要运行 "Iridium星座在0.5负载下" 的实验：

python run_experiments.py -e Iridium_load_0_5

仿真将在终端中显示进度和实时统计数据。结束后，会打印出所有测量算法的详细性能指标，包括：

ARE (Average Relative Error): 平均相对误差

WMRE (Weighted Mean Relative Error): 加权平均相对误差

Throughput (Mbps): 吞吐量

Avg Latency (ms): 平均延迟

3. 运行所有预定义的实验
要一次性运行 EXPERIMENTS 字典中定义的所有实验，并把结果汇总到一个CSV文件中，使用 -a 或 --all 参数。

python run_experiments.py -a -o results/summary_all_experiments.csv

-a 标志会依次执行所有实验。

-o 参数指定了输出CSV文件的路径。如果目录不存在，脚本会自动创建。

4. 调整内存大小（关键参数）
论文中的一个关键比较维度是不同内存大小下各算法的性能。你可以通过 --kb 参数在命令行覆盖配置文件中的内存设置。

例如，要在 Starlink_load_0_9 实验中将每个节点的总内存设置为 256 KB来运行仿真：

python run_experiments.py -e Starlink_load_0_9 --kb 256

该命令会临时修改与 kb相关的内存参数，运行仿真，并输出在该内存限制下的性能结果。这对于生成论文中关于内存效率的图表至关重要。

📊 输出说明
终端输出:

仿真开始和结束时，会显示总体统计数据，包括总发送/接收数据包、丢包率、吞吐量和延迟。

接着，会分算法展示性能评估结果，包括不同误差指标（ARE, WMRE, RE）的计算值。

CSV文件输出:

当使用 -a 或 -o 参数时，会生成一个CSV文件。

该文件包含了每次实验的配置参数（如实验名称、最终的kb值）以及所有记录的性能指标。这非常便于使用其他工具（如Excel, Jupyter Notebook）进行后续的数据分析和绘图。

🧠 代码逻辑简述
run_experiments.py: 实验的入口，负责解析命令行参数，加载对应的config模块，并调用主仿真函数。

main_simulation.py: 包含了仿真的核心驱动逻辑 main_runner。它初始化网络，调度周期性的拓扑和流量更新 (update_topology_and_flows)，并在仿真结束后调用 calculate_error_metrics 等函数来计算和打印所有统计数据。

simulation_core.py: 定义了网络的基本元素（Node, Link, Packet）和事件调度器（Simulator）。最重要的是，它完整实现了 ElasticSketch, CountMinSketch, 和 BFCMSketch 等核心算法。

tcp_flow.py: 定义了TcpFlow类，它根据流量矩阵的需求生成数据包，并模拟TCP的行为。

utils.py: 提供了读取拓扑/流量矩阵和使用Dijkstra算法计算最短路径等实用功能。

希望这份文档能帮助你顺利地复现和展示你的研究成果！
