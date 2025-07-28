from .base_config import *
inter = 1.0
kb = 500
memory_all = 128 * kb

# --- Simulation time control ---
GLOBAL_START_TIME = 0.0
TRAFFIC_STOP_TIME = 1.0
DRAIN_DURATION = 1.0
SIMULATION_END_TIME = TRAFFIC_STOP_TIME + DRAIN_DURATION

# --- Topology and traffic settings ---
NUM_SATELLITES = 1584
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
ADJACENCY_MATRIX_DIR = "Adjacency_matrices_time_starlink"
ADJACENCY_MATRIX_TEMPLATE = "{}.csv"

# --- Traffic matrix ---
TRAFFIC_MATRIX_DIR = f"demand_starlink_0.5"
TRAFFIC_MATRIX_TEMPLATE = "{}.csv"
