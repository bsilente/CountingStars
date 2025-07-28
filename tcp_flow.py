# tcp_flow.py
# 定义 TcpFlow 类 - 支持动态更新速率和路径

from typing import List, TYPE_CHECKING, Callable
import math

# 导入配置和工具函数
import config
from utils import find_shortest_path # 保持导入
# 导入核心组件
from simulation_core import Packet, FlowStats, Simulator, Node
import random

if TYPE_CHECKING:
    pass

class TcpFlow:
    """
    模拟数据流 (固定间隔发送，但间隔和路径可动态更新)
    - 发送间隔根据当前的目标速率计算。
    - 目标速率由外部根据流量矩阵更新。
    - 路径由外部根据拓扑变化更新。
    - 不包含拥塞控制逻辑。
    """
    def __init__(self, flow_id: str, source: 'Node', dest: 'Node', nodes: List['Node'],
                 initial_target_rate_mbps: float,
                 packet_entry_handler: Callable):
        self.flow_id = flow_id
        self.source = source
        self.dest = dest
        self.packet_size_bytes = config.PACKET_SIZE_BYTES
        self.stats = FlowStats(simulator_ref=Simulator)
        self.nodes = nodes # 保存对节点列表的引用（可能需要更新？）
        # 初始路径计算后保存
        self.path: List[Node] = find_shortest_path(nodes, source.node_id, dest.node_id)
        self.next_seq_num = 0
        self.is_started = False
        self.is_generating = False
        self.packet_entry_handler = packet_entry_handler

        self.current_target_rate_bps = initial_target_rate_mbps * 1e6
        self.interval_s = self._calculate_interval(self.current_target_rate_bps)

        if not self.path:
            print(f"错误: 初始无法为流 {self.flow_id} (从 {source.node_id+1} 到 {dest.node_id+1}) 找到路径。")
        else:
             self.print_path_info("初始") # 调用打印函数

    def _calculate_interval(self, rate_bps):
        """计算包生成间隔，确保每秒发包数>=1且为整数，间隔为1/(n+1)"""
        if rate_bps <= 1e-9:
            return float('inf')
        # 计算每秒发包数
        bits_per_packet = self.packet_size_bytes * 8
        packets_per_second = rate_bps / bits_per_packet
        # 若发包数小于1，返回无穷大，阻止包生成
        if packets_per_second < 1:
            return float('inf')
        # 取整数包数
        integer_packets = math.floor(packets_per_second)
        # 间隔为 1 秒除以 (包数 + 1)
        return 1.0 / (integer_packets + 1)

    def update_path(self, new_path: List['Node']):
        """更新流的传输路径"""
        if self.path != new_path: # 仅在路径实际改变时更新
            self.path = new_path
            if not self.path:
                print(f"警告: T={Simulator.now():.6f} 流 {self.flow_id} 的路径更新后无效，停止生成。")
                self.update_target_rate(0.0) # 路径无效则停止流量
            # else: # 路径有效，不需要打印，避免过多日志
            #     self.print_path_info("更新后")

    def print_path_info(self, prefix=""):
        """打印当前路径的端口信息"""
        if not self.path:
            # print(f"流 {self.flow_id}: {prefix}路径无效或不存在。") # 减少打印
            return

        path_repr_parts = []
        source_node = self.path[0]
        if len(self.path) > 1:
            next_node = self.path[1]; link_to_next = source_node.get_link_to(next_node.node_id)
            exit_port = -1
            if link_to_next:
                for iface in source_node.interfaces:
                    if iface.get('link') is link_to_next: exit_port = iface.get('port', -1); break
            path_repr_parts.append(f"N{source_node.node_id + 1}(出:{exit_port if exit_port != -1 else '?'})")
        else: path_repr_parts.append(f"N{source_node.node_id + 1}(源即目的)")

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
                    path_repr_parts.append(f"N{next_node.node_id + 1}(入:{entry_port if entry_port != -1 else '?'}/出:{exit_port_next if exit_port_next != -1 else '?'})")
                else: path_repr_parts.append(f"N{next_node.node_id + 1}(入:{entry_port if entry_port != -1 else '?'})")
            else: path_repr_parts.append(f"N{next_node.node_id + 1}(链路错误!)") # 路径无效
        path_description = " -> ".join(path_repr_parts)
        print(f"流 {self.flow_id}: {prefix}端口路径 {path_description}")
        # 打印速率和间隔信息
        print(f"  当前速率: {self.current_target_rate_bps/1e6:.2f} Mbps, 当前间隔: {self.interval_s:.6f} s")

    def update_target_rate(self, new_rate_mbps: float):
        """更新流的目标发送速率和对应的发送间隔"""
        new_rate_bps = new_rate_mbps * 1e6
        was_generating = self.is_generating
        rate_changed = abs(self.current_target_rate_bps - new_rate_bps) > 1e-9
        self.current_target_rate_bps = new_rate_bps
        self.interval_s = self._calculate_interval(self.current_target_rate_bps)

        if new_rate_bps > 1e-9:
            self.is_generating = True
            if (not was_generating or rate_changed) and self.is_started and self.path: # 确保有路径
                 if Simulator.now() < config.TRAFFIC_STOP_TIME - 1e-9:
                    # 延迟 interval_s 秒发送第一个包
                    Simulator.schedule(self.interval_s, self.send_packet, priority=20)
        else:
            self.is_generating = False

    def start(self):
        """开始发送数据包（如果初始速率和路径允许）"""
        if not self.path or self.is_started: return
        self.is_started = True
        if self.current_target_rate_bps > 1e-9:
            self.is_generating = True
            print(f"启动流 {self.flow_id} at T={Simulator.now():.6f}")
            # 延迟 interval_s 秒发送第一个包
            Simulator.schedule(self.interval_s, self.send_packet, priority=20)
        else:
            self.is_generating = False
            # print(f"流 {self.flow_id} 已创建但初始速率为0或路径无效，暂不启动发送。")

    def send_packet(self):
        """创建并发送（调度到达第一个队列）一个数据包，并根据当前间隔调度下一次发送"""
        current_time = Simulator.now()
        if not self.is_generating or not self.path or current_time >= config.TRAFFIC_STOP_TIME - 1e-9 or self.interval_s == float('inf'):
            if self.is_generating: self.is_generating = False
            return

        pkt = Packet(
            source_id=self.source.node_id, dest_id=self.dest.node_id,
            flow_id=self.flow_id, seq_num=self.next_seq_num,
            creation_time=current_time, size_bytes=self.packet_size_bytes,
            path=self.path, # 使用当前路径
            path_index=0, sender_id=-1
        )
        self.stats.tx_packets += 1
        self.stats.tx_bytes += pkt.size_bytes
        self.next_seq_num += 1

        if self.packet_entry_handler:
            Simulator.schedule(0.0, self.packet_entry_handler, args=(pkt, self.source, -1), priority=15)
        else:
            print(f"错误: 流 {self.flow_id} 没有设置 packet_entry_handler！")
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
        pass # 固定间隔发送，忽略拥塞信号
