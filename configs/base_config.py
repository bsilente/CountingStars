# --- Memory Settings ---
inter = 1.0
kb = 10
memory_all = 128 * kb # unit：8B

# --- Network Parameters ---
LINK_DATA_RATE_MBPS = 1000000000.0
LINK_DELAY_S = 0.001
PACKET_SIZE_BYTES = 1024
QUEUE_SIZE_PACKETS = 1
NODE_PROCESSING_DELAY_S = 0.000001
NUM_PARSING_MODULES_PER_PORT = 16
INTERVAL = 1/inter/128/2.5

# --- Simulation time control ---
GLOBAL_START_TIME = 0.0
TRAFFIC_STOP_TIME = 1.0
DRAIN_DURATION = 1.0
SIMULATION_END_TIME = TRAFFIC_STOP_TIME + DRAIN_DURATION

# --- Topology and traffic settings ---
NUM_SATELLITES = 66
PORT = 5000

# --- Method 1: CountingStars Parameters ---
MEMORY_POOL_SIZE = memory_all

# --- Method 2: Elastic Sketch Measurement of Parameter Values ---
ENABLE_ELASTIC_SKETCH = True 
ELASTIC_SKETCH_TOTAL_MEMORY_BYTES = memory_all * 8
ELASTIC_SKETCH_HEAVY_PART_RATIO = 0.4
ELASTIC_SKETCH_LAMBDA = 2
ELASTIC_SKETCH_HEAVY_BUCKET_SIZE_BYTES = 24 
ELASTIC_SKETCH_LIGHT_PART_COUNTER_SIZE_BYTES = 1 

# --- Method 3: Count-Min Sketch Measurement of Method Parameters ---
ENABLE_CM_SKETCH = True
CM_SKETCH_MEMORY_BYTES = memory_all * 8
CM_SKETCH_TARGET_DELTA = 0.003
CM_SKETCH_COUNTER_SIZE_BYTES = 4

# --- Method 4: BF+CM Sketch (FlowLiDAR) measure method parameters ---
ENABLE_BFCM_SKETCH = True
BFCM_SKETCH_TOTAL_MEMORY_BYTES = memory_all * 8
BFCM_SKETCH_BF_RATIO = 0.1
BFCM_SKETCH_BF_HASH_FUNCTIONS = 4
BFCM_SKETCH_CM_HASH_FUNCTIONS = 4
BFCM_SKETCH_COUNTER_SIZE_BYTES = 4

# --- Adjacency matrix ---
ADJACENCY_MATRIX_DIR = "Adjacency_matrices_time_yixing"
ADJACENCY_MATRIX_TEMPLATE = "{}.csv"


# --- Traffic matrix ---
TRAFFIC_MATRIX_DIR = f"demand_yixing_0.1"
TRAFFIC_MATRIX_TEMPLATE = "{}.csv"

# --- Output file ---
IP_OUTPUT_CSV_FILENAME = "ip_assignments\\satellite_ip_assignments_{}s.csv"
LOG_FILENAME = "simulation_log_enhanced.txt"
PORT_STATS_CSV_FILENAME = "satellite_port_rx_stats.csv"
DETAILED_STATS_CSV_TEMPLATE = "result\\Satellite_{}_detailed_rx_stats.csv"
