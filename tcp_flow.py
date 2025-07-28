from typing import List, TYPE_CHECKING, Callable
import math
import config
from utils import find_shortest_path
from simulation_core import Packet, FlowStats, Simulator, Node
import random

if TYPE_CHECKING:
    pass

class TcpFlow:
    def __init__(self, flow_id: str, source: 'Node', dest: 'Node', nodes: List['Node'],
                 initial_target_rate_mbps: float,
                 packet_entry_handler: Callable):
        self.flow_id = flow_id
        self.source = source
        self.dest = dest
        self.packet_size_bytes = config.PACKET_SIZE_BYTES
        self.stats = FlowStats(simulator_ref=Simulator)
        self.nodes = nodes
        self.path: List[Node] = find_shortest_path(nodes, source.node_id, dest.node_id)
        self.next_seq_num = 0
        self.is_started = False
        self.is_generating = False
        self.packet_entry_handler = packet_entry_handler

        self.current_target_rate_bps = initial_target_rate_mbps * 1e6
        self.interval_s = self._calculate_interval(self.current_target_rate_bps)

        if not self.path:
            print(f"Error: Initially unable to find a path for flow {self.flow_id} (from {source.node_id+1} to {dest.node_id+1}).")
        else:
             self.print_path_info("Initial")

    def _calculate_interval(self, rate_bps):
        if rate_bps <= 1e-9:
            return float('inf')
        bits_per_packet = self.packet_size_bytes * 8
        packets_per_second = rate_bps / bits_per_packet
        if packets_per_second < 1:
            return float('inf')
        integer_packets = math.floor(packets_per_second)
        return 1.0 / (integer_packets + 1)

    def update_path(self, new_path: List['Node']):
        if self.path != new_path:
            self.path = new_path
            if not self.path:
                print(f"Warning: T={Simulator.now():.6f} Path for flow {self.flow_id} is invalid after update, stopping generation.")
                self.update_target_rate(0.0)

    def print_path_info(self, prefix=""):
        if not self.path:
            return

        path_repr_parts = []
        source_node = self.path[0]
        if len(self.path) > 1:
            next_node = self.path[1]; link_to_next = source_node.get_link_to(next_node.node_id)
            exit_port = -1
            if link_to_next:
                for iface in source_node.interfaces:
                    if iface.get('link') is link_to_next: exit_port = iface.get('port', -1); break
            path_repr_parts.append(f"N{source_node.node_id + 1}(out:{exit_port if exit_port != -1 else '?'})")
        else: path_repr_parts.append(f"N{source_node.node_id + 1}(source is dest)")

        for i in range(len(self.path) - 1):
            current_node = self.path[i]; next_node = self.path[i+1]
            link = current_node.get_link_to(next_node.node_id)
            entry_port = -1; exit_port_next = -1
            if link:
                for iface in next_node.interfaces:
                    if iface.get('link') is link: entry_port = iface.get('port', -1); break
                if i < len(self.path) - 2:
                    next_next_node = self.path[i+2]; link_to_next_next = next_node.get_link_to(next_next_node.node_id)
                    if link_to_next_next:
                        for iface in next_node.interfaces:
                            if iface.get('link') is link_to_next_next: exit_port_next = iface.get('port', -1); break
                    path_repr_parts.append(f"N{next_node.node_id + 1}(in:{entry_port if entry_port != -1 else '?'}/out:{exit_port_next if exit_port_next != -1 else '?'})")
                else: path_repr_parts.append(f"N{next_node.node_id + 1}(in:{entry_port if entry_port != -1 else '?'})")
            else: path_repr_parts.append(f"N{next_node.node_id + 1}(Link Error!)")
        path_description = " -> ".join(path_repr_parts)
        print(f"Flow {self.flow_id}: {prefix} Port Path {path_description}")
        print(f"  Current Rate: {self.current_target_rate_bps/1e6:.2f} Mbps, Current Interval: {self.interval_s:.6f} s")

    def update_target_rate(self, new_rate_mbps: float):
        new_rate_bps = new_rate_mbps * 1e6
        was_generating = self.is_generating
        rate_changed = abs(self.current_target_rate_bps - new_rate_bps) > 1e-9
        self.current_target_rate_bps = new_rate_bps
        self.interval_s = self._calculate_interval(self.current_target_rate_bps)

        if new_rate_bps > 1e-9:
            self.is_generating = True
            if (not was_generating or rate_changed) and self.is_started and self.path:
                 if Simulator.now() < config.TRAFFIC_STOP_TIME - 1e-9:
                    Simulator.schedule(self.interval_s, self.send_packet, priority=20)
        else:
            self.is_generating = False

    def start(self):
        if not self.path or self.is_started: return
        self.is_started = True
        if self.current_target_rate_bps > 1e-9:
            self.is_generating = True
            print(f"Starting flow {self.flow_id} at T={Simulator.now():.6f}")
            Simulator.schedule(self.interval_s, self.send_packet, priority=20)
        else:
            self.is_generating = False

    def send_packet(self):
        current_time = Simulator.now()
        if not self.is_generating or not self.path or current_time >= config.TRAFFIC_STOP_TIME - 1e-9 or self.interval_s == float('inf'):
            if self.is_generating: self.is_generating = False
            return

        pkt = Packet(
            source_id=self.source.node_id, dest_id=self.dest.node_id,
            flow_id=self.flow_id, seq_num=self.next_seq_num,
            creation_time=current_time, size_bytes=self.packet_size_bytes,
            path=self.path,
            path_index=0, sender_id=-1
        )
        self.stats.tx_packets += 1
        self.stats.tx_bytes += pkt.size_bytes
        self.next_seq_num += 1

        if self.packet_entry_handler:
            Simulator.schedule(0.0, self.packet_entry_handler, args=(pkt, self.source, -1), priority=15)
        else:
            print(f"Error: Flow {self.flow_id} has no packet_entry_handler set!")
            return

        next_send_time = current_time + self.interval_s
        if self.interval_s != float('inf') and next_send_time < config.TRAFFIC_STOP_TIME - 1e-9:
            Simulator.schedule(self.interval_s + 0.5 * config.INTERVAL - random.random() * config.INTERVAL, self.send_packet, priority=20)
            # Simulator.schedule(self.interval_s, self.send_packet, priority=20)
        else:
            self.is_generating = False

    def receive_packet(self, packet: Packet):
        self.stats.rx_packets += 1
        self.stats.rx_bytes += packet.size_bytes
        self.stats.record_latency(packet.creation_time)

    def handle_congestion_event(self):
        pass
