# config.py
# 存储全局配置常量
load = 1.0
kb = 10
memory_all = 128 * kb # 单位：8B
# memory_all = 50 # 单位：8B

# --- 网络参数 ---
LINK_DATA_RATE_MBPS = 1000000000.0
LINK_DELAY_S = 0.001
PACKET_SIZE_BYTES = 1024
QUEUE_SIZE_PACKETS = 1
NODE_PROCESSING_DELAY_S = 0.000001 #解析包延迟
NUM_PARSING_MODULES_PER_PORT = 16
INTERVAL = 1/load/128/2.5


# --- 仿真时间控制 ---
GLOBAL_START_TIME = 0.0
TRAFFIC_STOP_TIME = 1.0
DRAIN_DURATION = 1.0
SIMULATION_END_TIME = TRAFFIC_STOP_TIME + DRAIN_DURATION

# --- 拓扑和流量设置 ---
NUM_SATELLITES = 66
PORT = 5000

# --- 方法 1: 原有测量方法参数 ---
MEMORY_POOL_SIZE = memory_all # 每个卫星内存池中可以存储的键值对上限



# --- 方法 2: Elastic Sketch 测量方法参数 ---
ENABLE_ELASTIC_SKETCH = True 
ELASTIC_SKETCH_TOTAL_MEMORY_BYTES = memory_all * 8
ELASTIC_SKETCH_HEAVY_PART_RATIO = 0.4
ELASTIC_SKETCH_LAMBDA = 2
ELASTIC_SKETCH_HEAVY_BUCKET_SIZE_BYTES = 24 
ELASTIC_SKETCH_LIGHT_PART_COUNTER_SIZE_BYTES = 1 


# --- 方法 3: Count-Min Sketch 测量方法参数 (修改后) ---
ENABLE_CM_SKETCH = True
# 预设的固定内存预算 (单位: 字节)
CM_SKETCH_MEMORY_BYTES = memory_all * 8
# 目标置信度 (1 - delta)，用于计算深度 d
CM_SKETCH_TARGET_DELTA = 0.003
# 每个计数器的大小 (单位: 字节)，用于计算
CM_SKETCH_COUNTER_SIZE_BYTES = 4


# --- 方法 4: BF+CM Sketch (FlowLiDAR) 测量方法参数 ---
ENABLE_BFCM_SKETCH = True
# 总内存预算 (单位: 字节)
BFCM_SKETCH_TOTAL_MEMORY_BYTES = memory_all * 8
# 布隆过滤器占总内存的比例
BFCM_SKETCH_BF_RATIO = 0.1
# 布隆过滤器使用的哈希函数数量 (k)
BFCM_SKETCH_BF_HASH_FUNCTIONS = 4
# Count-Min Sketch 使用的哈希函数数量 (d)
BFCM_SKETCH_CM_HASH_FUNCTIONS = 4
# Count-Min Sketch 中每个计数器的大小 (单位: 字节)
BFCM_SKETCH_COUNTER_SIZE_BYTES = 4 # 32-bit counters


# # --- **新增**: 动态拓扑相关 ---
# ADJACENCY_MATRIX_DIR = "Adjacency_matrices_time_starlink" # 存放邻接矩阵的目录
# ADJACENCY_MATRIX_TEMPLATE = "{}.csv" # 邻接矩阵文件名模板
# # --- 结束新增 ---

# # --- 流量矩阵文件模板 ---
# TRAFFIC_MATRIX_DIR = f"demand_starlink_0.5"
# TRAFFIC_MATRIX_TEMPLATE = "{}.csv"

# # --- 输入/输出文件名 ---
# IP_OUTPUT_CSV_FILENAME = "ip_assignments\\satellite_ip_assignments_{}s.csv" #文件名加入时间戳区分
# LOG_FILENAME = "simulation_log_enhanced.txt"
# # **移除**: 不再需要静态邻接矩阵文件名
# # DATA_CSV_FILENAME = "data.csv"
# PORT_STATS_CSV_FILENAME = "satellite_port_rx_stats.csv"
# DETAILED_STATS_CSV_TEMPLATE = "result\\Satellite_{}_detailed_rx_stats.csv"


# --- **新增**: 动态拓扑相关 ---
ADJACENCY_MATRIX_DIR = "Adjacency_matrices_time_yixing" # 存放邻接矩阵的目录
ADJACENCY_MATRIX_TEMPLATE = "{}.csv" # 邻接矩阵文件名模板
# --- 结束新增 ---

# --- 流量矩阵文件模板 ---
TRAFFIC_MATRIX_DIR = f"demand_yixing_0.1"
TRAFFIC_MATRIX_TEMPLATE = "{}.csv"

# --- 输入/输出文件名 ---
IP_OUTPUT_CSV_FILENAME = "ip_assignments\\satellite_ip_assignments_{}s.csv" #文件名加入时间戳区分
LOG_FILENAME = "simulation_log_enhanced.txt"
# **移除**: 不再需要静态邻接矩阵文件名
# DATA_CSV_FILENAME = "data.csv"
PORT_STATS_CSV_FILENAME = "satellite_port_rx_stats.csv"
DETAILED_STATS_CSV_TEMPLATE = "result\\Satellite_{}_detailed_rx_stats.csv"
