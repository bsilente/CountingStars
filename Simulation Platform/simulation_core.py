import uuid
import heapq
import traceback
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Deque, Callable, TYPE_CHECKING
from collections import deque
import config

REROUTE_PORT = -2

@dataclass
class HeavyBucket:
    flow_key: Optional[Tuple[int, int]] = None
    positive_votes: int = 0
    negative_votes: int = 0
    flag: bool = False 

class ElasticSketch:
    def __init__(self):
        heavy_memory = config.ELASTIC_SKETCH_TOTAL_MEMORY_BYTES * config.ELASTIC_SKETCH_HEAVY_PART_RATIO
        light_memory = config.ELASTIC_SKETCH_TOTAL_MEMORY_BYTES - heavy_memory
        num_heavy_buckets = int(heavy_memory // config.ELASTIC_SKETCH_HEAVY_BUCKET_SIZE_BYTES)
        num_light_counters = int(light_memory // config.ELASTIC_SKETCH_LIGHT_PART_COUNTER_SIZE_BYTES)
        self.heavy_part: List[HeavyBucket] = [HeavyBucket() for _ in range(num_heavy_buckets)]
        self.light_part: List[int] = [0] * num_light_counters
        self.light_counter_max = 255 

    def _hash_to_heavy(self, flow_key: Tuple[int, int]) -> int:
        if not self.heavy_part: return -1
        return hash(flow_key) % len(self.heavy_part)

    def _hash_to_light(self, flow_key: Tuple[int, int]) -> int:
        if not self.light_part: return -1
        return hash(str(flow_key) + "light") % len(self.light_part)

    def _add_to_light_part(self, flow_key: Tuple[int, int], value: int):
        light_index = self._hash_to_light(flow_key)
        if light_index != -1:
            self.light_part[light_index] = min(self.light_counter_max, self.light_part[light_index] + value)

    def insert(self, flow_key: Tuple[int, int]) -> bool:
        heavy_index = self._hash_to_heavy(flow_key)
        if heavy_index == -1:
            self._add_to_light_part(flow_key, 1)
            return False
        
        bucket = self.heavy_part[heavy_index]
        if bucket.flow_key is None:
            bucket.flow_key = flow_key
            bucket.positive_votes = 1
            return True
        elif bucket.flow_key == flow_key:
            bucket.positive_votes += 1
            return True
        else:
            bucket.negative_votes += 1
            if bucket.negative_votes >= config.ELASTIC_SKETCH_LAMBDA * bucket.positive_votes:
                evicted_flow_key = bucket.flow_key
                evicted_votes = bucket.positive_votes
                if evicted_flow_key is not None:
                    self._add_to_light_part(evicted_flow_key, evicted_votes)
                bucket.flow_key = flow_key
                bucket.positive_votes = 1
                bucket.negative_votes = 0
                return True
            else:
                self._add_to_light_part(flow_key, 1)
                return False

    def query(self, flow_key: Tuple[int, int]) -> int:
        estimated_count = 0
        
        light_index = self._hash_to_light(flow_key)
        if light_index != -1:
            estimated_count += self.light_part[light_index]
            
        heavy_index = self._hash_to_heavy(flow_key)
        if heavy_index != -1:
            bucket = self.heavy_part[heavy_index]
            if bucket.flow_key == flow_key:
                estimated_count += bucket.positive_votes
                
        return estimated_count

class CountMinSketch:
    def __init__(self, memory_bytes=None, delta=None, counter_size_bytes=None, d=None):
        mem_bytes = memory_bytes if memory_bytes is not None else config.CM_SKETCH_MEMORY_BYTES
        cnt_size = counter_size_bytes if counter_size_bytes is not None else config.CM_SKETCH_COUNTER_SIZE_BYTES

        if cnt_size <= 0:
            raise ValueError("Counter size must be positive")
        if d is not None:
            self.d = int(d)
        else:
            dlt = delta if delta is not None else config.CM_SKETCH_TARGET_DELTA
            if dlt <= 0 or dlt >= 1:
                raise ValueError("Delta must be in the range (0, 1)")
            self.d = int(math.ceil(math.log(1 / dlt)))

        if self.d <= 0:
             self.d = 1
        
        total_counters = mem_bytes // cnt_size
        self.w = int(total_counters // self.d)
        if self.w <= 0:
            raise ValueError(f"Memory budget {mem_bytes}B is too small for the target d={self.d} to allocate at least one column.")
        
        self.counts = [[0] * self.w for _ in range(self.d)] # self.w is now guaranteed to be an integer
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(self.d)]

    def _hash(self, flow_key: Tuple[int, int], seed_index: int) -> int:
        return hash((flow_key, self.hash_seeds[seed_index])) % self.w

    def insert(self, flow_key: Tuple[int, int]):
        for i in range(self.d):
            col_index = self._hash(flow_key, i)
            self.counts[i][col_index] += 1

    def query(self, flow_key: Tuple[int, int]) -> int:
        min_count = float('inf')
        for i in range(self.d):
            col_index = self._hash(flow_key, i)
            min_count = min(min_count, self.counts[i][col_index])
        return int(min_count)

class LazyBloomFilter:
    def __init__(self):
        bf_memory_bytes = config.BFCM_SKETCH_TOTAL_MEMORY_BYTES * config.BFCM_SKETCH_BF_RATIO
        self.k = config.BFCM_SKETCH_BF_HASH_FUNCTIONS
        if self.k <= 0:
            raise ValueError("BFCM_SKETCH_BF_HASH_FUNCTIONS must be positive")
        
        total_bits = int(bf_memory_bytes * 8)
        bits_per_array = total_bits // self.k
        if bits_per_array <= 0:
            raise ValueError(f"Memory allocated for BF is not sufficient to support {self.k} arrays.")

        self.m = bits_per_array
        self.arrays = [[0] * self.m for _ in range(self.k)]
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(self.k)]

    def _hash(self, flow_key: Tuple[int, int], seed_index: int) -> int:
        return hash((flow_key, self.hash_seeds[seed_index])) % self.m

    def check_and_set(self, flow_key: Tuple[int, int]) -> bool:
        for i in range(self.k):
            index = self._hash(flow_key, i)
            if self.arrays[i][index] == 0:
                self.arrays[i][index] = 1
                return True
        return False

class BFCMSketch:
    def __init__(self):
        print(f"BF+CM Sketch (FlowLiDAR) initializing:")
        print(f"  Total memory: {config.BFCM_SKETCH_TOTAL_MEMORY_BYTES} bytes")
        bf_mem = config.BFCM_SKETCH_TOTAL_MEMORY_BYTES * config.BFCM_SKETCH_BF_RATIO
        cm_mem = config.BFCM_SKETCH_TOTAL_MEMORY_BYTES - bf_mem
        print(f"  - Bloom Filter (Lazy) memory: {bf_mem} bytes, k={config.BFCM_SKETCH_BF_HASH_FUNCTIONS}")
        print(f"  - CM Sketch memory: {cm_mem} bytes, d={config.BFCM_SKETCH_CM_HASH_FUNCTIONS}")

        self.lazy_bf = LazyBloomFilter()
        self.cm_sketch = CountMinSketch(
            memory_bytes=cm_mem,
            counter_size_bytes=config.BFCM_SKETCH_COUNTER_SIZE_BYTES,
            d=config.BFCM_SKETCH_CM_HASH_FUNCTIONS
        )


@dataclass
class Ipv4Address:
    address: str
    def __str__(self): return self.address

@dataclass
class Packet:
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: int = -1; dest_id: int = -1
    flow_id: str = ""; seq_num: int = 0
    creation_time: float = 0.0
    size_bytes: int = config.PACKET_SIZE_BYTES
    path: List['Node'] = field(default_factory=list)
    path_index: int = 0; sender_id: int = -1

@dataclass
class FlowStats:
    tx_bytes: int = 0; rx_bytes: int = 0
    tx_packets: int = 0; rx_packets: int = 0
    lost_packets_queue: int = 0
    lost_packets_module_busy: int = 0
    lost_packets_other: int = 0
    packets_rerouted: int = 0
    latencies: List[float] = field(default_factory=list)
    simulator_ref: Optional['Simulator'] = field(default=None, repr=False)

    def record_latency(self, packet_creation_time: float):
        if self.simulator_ref:
            latency = self.simulator_ref.now() - packet_creation_time
            self.latencies.append(latency)
        else: print("Warning: FlowStats is not associated with a Simulator, cannot record latency.")

    def get_avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def get_total_lost_packets(self) -> int:
        return self.lost_packets_queue + self.lost_packets_module_busy + self.lost_packets_other


class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.interfaces: List[Dict] = []
        self.links: Dict[int, 'Link'] = {}
        self.port_rx_bytes: Dict[int, int] = {}
        self.detailed_port_rx_bytes: Dict[int, Dict[int, int]] = {}
        self.port_module_busy: Dict[int, List[bool]] = {}
        self.port_next_module_index: Dict[int, int] = {}
        self.flow_memory: Dict[Tuple[int, int], int] = {}
        
        if config.ENABLE_ELASTIC_SKETCH:
            self.elastic_sketch = ElasticSketch()
        else:
            self.elastic_sketch = None
            
        if config.ENABLE_CM_SKETCH:
            self.cm_sketch = CountMinSketch()
        else:
            self.cm_sketch = None
            
        if config.ENABLE_BFCM_SKETCH:
            self.bfcm_sketch = BFCMSketch()
            self.bfcm_control_plane_counts: Dict[Tuple[int, int], int] = {}
        else:
            self.bfcm_sketch = None
            self.bfcm_control_plane_counts = {}

        self.true_flow_counts: Dict[Tuple[int, int], int] = {}


    def add_interface(self, port: int, ip: str, link: 'Link'):
        if port in [iface['port'] for iface in self.interfaces]:
             print(f"Warning: Attempting to add an existing port {port} to node {self.node_id}.")
             new_port = port
             while new_port in [iface['port'] for iface in self.interfaces]:
                 new_port += 1
             print(f"  -> Using new port number {new_port} instead.")
             port = new_port
        self.interfaces.append({"port": port, "ip": Ipv4Address(ip), "link": link})
        neighbor = link.node1 if link.node2 == self else link.node2
        self.links[neighbor.node_id] = link
        self.port_rx_bytes.setdefault(port, 0)
        self.port_module_busy.setdefault(port, [False] * config.NUM_PARSING_MODULES_PER_PORT)
        self.port_next_module_index.setdefault(port, 0)

    def remove_interface_for_link(self, link_to_remove: 'Link'):
        removed_port = -1
        neighbor_id = -1
        new_interfaces = []
        for iface in self.interfaces:
            if iface.get('link') is link_to_remove:
                removed_port = iface.get('port')
                neighbor_node = link_to_remove.get_other_node(self)
                neighbor_id = neighbor_node.node_id
            else:
                new_interfaces.append(iface)
        self.interfaces = new_interfaces
        if neighbor_id != -1 and neighbor_id in self.links and self.links[neighbor_id] is link_to_remove:
            del self.links[neighbor_id]

    def has_ip(self, ip: str) -> bool:
        return any(iface["ip"].address == ip for iface in self.interfaces)

    def get_link_to(self, neighbor_node_id: int) -> Optional['Link']:
        return self.links.get(neighbor_node_id)

    def increment_port_rx_bytes(self, port_number: int, bytes_received: int):
        self.port_rx_bytes[port_number] = self.port_rx_bytes.get(port_number, 0) + bytes_received

    def increment_actual_port_sender_rx_bytes(self, actual_arrival_port: int, sender_id: int, bytes_received: int):
        if actual_arrival_port not in self.detailed_port_rx_bytes:
            self.detailed_port_rx_bytes[actual_arrival_port] = {}
        port_dict = self.detailed_port_rx_bytes[actual_arrival_port]
        port_dict[sender_id] = port_dict.get(sender_id, 0) + bytes_received


class Link:
    def __init__(self, node1: Node, node2: Node, data_rate_mbps: float, delay_s: float, queue_size_packets: int):
        self.node1 = node1; self.node2 = node2
        self.data_rate_bps = data_rate_mbps * 1e6
        self.transmission_time_s = (config.PACKET_SIZE_BYTES * 8) / self.data_rate_bps if self.data_rate_bps > 0 else float('inf')
        self.delay_s = delay_s; self.queue_limit_packets = queue_size_packets
        self.queue1_2: Deque[Packet] = deque(); self.is_busy1_2: bool = False
        self.queue2_1: Deque[Packet] = deque(); self.is_busy2_1: bool = False

    def get_queue_and_busy_flag(self, source_node: Node) -> Tuple[Deque[Packet], bool]:
        if source_node == self.node1: return self.queue1_2, self.is_busy1_2
        elif source_node == self.node2: return self.queue2_1, self.is_busy2_1
        else: raise ValueError(f"Source node {source_node.node_id} does not belong to this link...")

    def set_busy_flag(self, source_node: Node, busy: bool):
        if source_node == self.node1: self.is_busy1_2 = busy
        elif source_node == self.node2: self.is_busy2_1 = busy
        else: raise ValueError(f"Source node {source_node.node_id} does not belong to this link...")

    def get_other_node(self, node: Node) -> Node:
        return self.node2 if node == self.node1 else self.node1

    def get_packets_in_queues(self) -> List[Tuple[Packet, Node]]:
        packets = []
        for pkt in self.queue1_2: packets.append((pkt, self.node1))
        for pkt in self.queue2_1: packets.append((pkt, self.node2))
        return packets

class Ipv4AddressHelper:
    def __init__(self, base: str, mask: str):
        self.base_ip = [int(x) for x in base.split(".")]
        self.mask = [int(x) for x in mask.split(".")]
        self.current_subnet = 0; self.subnet_increment = 4
        self.initial_base = list(self.base_ip)
        self.initial_subnet = self.current_subnet

    def reset(self):
        pass

    def new_network(self):
        self.current_subnet += self.subnet_increment
        if self.current_subnet >= 256:
            self.base_ip[2] = 0; self.current_subnet = 0
            self.base_ip[1] += 1
            if self.base_ip[1] >= 256: raise OverflowError("IP Address range exceeded")
        self.base_ip[2] = self.current_subnet

    def assign(self, link: Link) -> List[Tuple[Node, Ipv4Address]]:
         retry_count = 0; max_retries = 256 * 256
         temp_base = list(self.base_ip); temp_subnet = self.current_subnet
         while retry_count < max_retries:
            ip1_str = f"{self.base_ip[0]}.{self.base_ip[1]}.{self.base_ip[2]}.1"
            ip2_str = f"{self.base_ip[0]}.{self.base_ip[1]}.{self.base_ip[2]}.2"
            conflict = False
            if link.node1.has_ip(ip1_str) or link.node1.has_ip(ip2_str): conflict = True
            if link.node2.has_ip(ip1_str) or link.node2.has_ip(ip2_str): conflict = True
            if not conflict:
                port1 = 1
                while port1 in [iface['port'] for iface in link.node1.interfaces]: port1 += 1
                port2 = 1
                while port2 in [iface['port'] for iface in link.node2.interfaces]: port2 += 1
                link.node1.add_interface(port1, ip1_str, link)
                link.node2.add_interface(port2, ip2_str, link)
                return [(link.node1, Ipv4Address(ip1_str)), (link.node2, Ipv4Address(ip2_str))]
            else:
                self.new_network(); retry_count += 1
                if self.base_ip == temp_base and self.current_subnet == temp_subnet:
                     raise RuntimeError(f"Failed to find non-conflicting IP subnet for link {link.node1.node_id}-{link.node2.node_id}.")
         raise RuntimeError(f"Failed to assign IP address after maximum retries for link {link.node1.node_id}-{link.node2.node_id}.")

class Simulator:
    current_time: float = config.GLOBAL_START_TIME
    events: List[Tuple[float, int, int, Callable, tuple]] = []
    stop_time: float = config.SIMULATION_END_TIME
    event_count: int = 0

    @classmethod
    def reset(cls):
        cls.current_time = config.GLOBAL_START_TIME
        cls.events = []
        cls.event_count = 0

    @classmethod
    def schedule(cls, delay: float, callback: Callable, args: tuple = (), priority: int = 10):
        schedule_time = cls.current_time + delay
        if schedule_time < cls.stop_time - 1e-9:
            cls.event_count += 1
            heapq.heappush(cls.events, (schedule_time, priority, cls.event_count, callback, args))

    @classmethod
    def run(cls):
        print(f"--- Simulation started (Duration: {cls.stop_time - cls.current_time}s, Traffic/Topology updates stop at: {config.TRAFFIC_STOP_TIME}s) ---")
        while cls.events:
            time, prio, count, callback, args = heapq.heappop(cls.events)
            if time >= cls.stop_time - 1e-9: cls.events = []; break
            if time > cls.current_time: cls.current_time = time
            try: callback(*args)
            except Exception as e:
                print(f"Error: Event callback {callback.__name__} raised an exception at time {cls.current_time:.9f}! Error: {e}")
                traceback.print_exc()
        cls.current_time = min(cls.current_time, cls.stop_time)
        print(f"--- Simulation ended at {cls.current_time:.6f}s ---")

    @classmethod
    def now(cls) -> float: return cls.current_time

    @classmethod
    def stop_simulation(cls): print("--- Simulation was forcibly stopped ---"); cls.events = []
