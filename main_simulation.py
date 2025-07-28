# main_simulation.py
# 主仿真脚本，整合所有模块并运行

import sys
import traceback
import csv
import os
from typing import Dict, List, Tuple, Callable
from collections import deque
from collections import Counter

# 注意: 这里的 'import config' 会被 run_experiments.py 动态替换
# 所以我们不需要在这里修改它
import config

# 导入核心组件和引擎
from simulation_core import Packet, FlowStats, Ipv4Address, Node, Link, Simulator, Ipv4AddressHelper, REROUTE_PORT

# 导入工具函数
from utils import read_adjacency_matrix, find_shortest_path, read_traffic_matrix
# 导入流量类
from tcp_flow import TcpFlow

# --- 全局变量 ---
# 这些变量将在 main_runner() 函数中被重置
active_flows: Dict[str, 'TcpFlow'] = {}
nodes_list: List[Node] = []
links_list: List[Link] = []
ip_helper = Ipv4AddressHelper("10.1.1.0", "255.255.255.252")
lost_packets_module_real = 0

# 方法 1: 原有测量方法的统计变量
count1 = 0
total_processed_packets = 0

# 方法 2: Elastic Sketch 测量方法的统计变量
elastic_sketch_hits = 0
total_processed_for_elastic = 0

# 方法 3: Count-Min Sketch 测量方法的统计变量
cm_sketch_estimated_hits = 0
total_processed_for_cm = 0

# [新增] 方法 4: BF+CM Sketch (FlowLiDAR) 测量方法的统计变量
total_processed_for_bfcm = 0


# --- 事件处理函数 ---
def _process_packet_at_node(packet: Packet, current_node: Node, arrival_port: int, module_was_busy: bool):
    """
    在节点处理延迟结束后，实际处理数据包的函数。
    包含端口统计、目的地检查、转发或丢弃逻辑。
    新增一个布尔参数 module_was_busy 来控制测量逻辑。
    """
    global nodes_list, count1, total_processed_packets
    global elastic_sketch_hits, total_processed_for_elastic
    global cm_sketch_estimated_hits, total_processed_for_cm
    global total_processed_for_bfcm

    # --- [BUG修复]：仅在模块不繁忙 且 数据包从物理端口到达时，才执行所有测量和计数逻辑 ---
    # 这个修复确保了所有测量的分子和分母都来自同一数据群体
    if not module_was_busy and arrival_port != -1 and arrival_port != REROUTE_PORT:
        if packet.source_id != -1 and packet.dest_id != -1:
            flow_key = (packet.source_id, packet.dest_id)

            # --- 上帝视角：记录真实值 ---
            current_node.true_flow_counts[flow_key] = current_node.true_flow_counts.get(flow_key, 0) + 1

            # --- 原有测量方法 (内存池) ---
            total_processed_packets += 1
            if flow_key in current_node.flow_memory:
                current_node.flow_memory[flow_key] += 1
                count1 += 1
            else:
                if len(current_node.flow_memory) < config.MEMORY_POOL_SIZE:
                    current_node.flow_memory[flow_key] = 1
                    count1 += 1
            
            # --- Elastic Sketch 测量方法 ---
            if config.ENABLE_ELASTIC_SKETCH and current_node.elastic_sketch:
                total_processed_for_elastic += 1
                if current_node.elastic_sketch.insert(flow_key):
                    elastic_sketch_hits += 1
            
            # --- Count-Min Sketch 测量方法 ---
            if config.ENABLE_CM_SKETCH and current_node.cm_sketch:
                total_processed_for_cm += 1
                if current_node.cm_sketch.query(flow_key) > 0:
                    cm_sketch_estimated_hits += 1
                current_node.cm_sketch.insert(flow_key)

            # --- [新增] BF+CM Sketch (FlowLiDAR) 测量方法 ---
            if config.ENABLE_BFCM_SKETCH and current_node.bfcm_sketch:
                total_processed_for_bfcm += 1
                # 使用 Lazy Update BF 检查
                is_newly_detected = current_node.bfcm_sketch.lazy_bf.check_and_set(flow_key)
                if is_newly_detected:
                    # 如果是新检测到的（或流的前k个包），则在“控制平面”计数器中+1
                    current_node.bfcm_control_plane_counts[flow_key] = current_node.bfcm_control_plane_counts.get(flow_key, 0) + 1
                else:
                    # 如果流的所有k个BF位都已设置，则将其余的包计入CM Sketch
                    current_node.bfcm_sketch.cm_sketch.insert(flow_key)


    # --- [注意]：后续的端口统计和数据包转发逻辑不受上述修改影响，保持原样 ---

    # --- 端口总字节数统计 ---
    if arrival_port != -1 and arrival_port != REROUTE_PORT:
         try: current_node.increment_port_rx_bytes(arrival_port, packet.size_bytes)
         except Exception as e: print(f"错误: T={Simulator.now():.9f} Node {current_node.node_id} Port {arrival_port} 增加总字节数出错: {e}")

    # --- 详细端口统计 ---
    sender_id = packet.sender_id
    if arrival_port != -1 and arrival_port != REROUTE_PORT and sender_id != -1:
        try: current_node.increment_actual_port_sender_rx_bytes(arrival_port, sender_id, packet.size_bytes)
        except Exception as e: print(f"错误: T={Simulator.now():.9f} Node {current_node.node_id} 实际端口 {arrival_port} 为发送者 {sender_id} 增加详细字节数出错: {e}")

    # --- 处理重注入的数据包 ---
    if arrival_port == REROUTE_PORT:
        remaining_path = find_shortest_path(nodes_list, current_node.node_id, packet.dest_id)
        if not remaining_path or len(remaining_path) < 2:
            print(f"  -> 重路由失败: T={Simulator.now():.6f} 找不到新路径从 {current_node.node_id+1} 到 {packet.dest_id+1}。丢弃包 {packet.seq_num} (流 {packet.flow_id})。")
            flow = active_flows.get(packet.flow_id)
            if flow: flow.stats.lost_packets_other += 1
            return
        else:
            packet.path = remaining_path
            packet.path_index = 0

    # 检查路径有效性
    if packet.path_index >= len(packet.path) or packet.path[packet.path_index].node_id != current_node.node_id:
         print(f"警告: T={Simulator.now():.9f} 包 {packet.packet_id} seq {packet.seq_num} 的路径在节点 {current_node.node_id+1} 失效 (期望: {packet.path[packet.path_index].node_id+1 if packet.path_index < len(packet.path) else '越界'})。丢弃该包。")
         flow = active_flows.get(packet.flow_id);
         if flow: flow.stats.lost_packets_other += 1
         return

    # 检查是否到达最终目的地
    if current_node.node_id == packet.dest_id:
        flow = active_flows.get(packet.flow_id)
        if flow: flow.receive_packet(packet)
        else: print(f"错误: T={Simulator.now():.9f} 流 {packet.flow_id} 未找到...")
    else:
        # 需要转发数据包
        packet.sender_id = current_node.node_id
        packet.path_index += 1
        if packet.path_index >= len(packet.path):
            print(f"错误: T={Simulator.now():.9f} 包 {packet.packet_id} seq {packet.seq_num} 路径索引错误 (转发时)。")
            flow = active_flows.get(packet.flow_id);
            if flow: flow.stats.lost_packets_other += 1
            return

        next_node_obj = packet.path[packet.path_index]
        link_to_next = current_node.get_link_to(next_node_obj.node_id)
        if not link_to_next:
            print(f"警告: T={Simulator.now():.9f} 从 {current_node.node_id+1} 到 {next_node_obj.node_id+1} 链路未找到 (可能拓扑已变)。尝试重路由包 {packet.seq_num} (流 {packet.flow_id})。")
            flow = active_flows.get(packet.flow_id)
            if flow: flow.stats.packets_rerouted += 1

            new_remaining_path = find_shortest_path(nodes_list, current_node.node_id, packet.dest_id)
            if not new_remaining_path or len(new_remaining_path) < 2:
                 print(f"  -> 重路由失败: T={Simulator.now():.6f} 找不到新路径从 {current_node.node_id+1} 到 {packet.dest_id+1}。丢弃包。")
                 if flow: flow.stats.lost_packets_other += 1
                 return
            else:
                 packet.path = new_remaining_path
                 packet.path_index = 0
                 if packet.path_index + 1 >= len(packet.path):
                      print(f"错误: T={Simulator.now():.9f} 重路由后路径无效 (仅当前节点)。丢弃包 {packet.seq_num}。")
                      if flow: flow.stats.lost_packets_other += 1
                      return
                 packet.path_index = 1
                 next_node_obj = packet.path[packet.path_index]
                 link_to_next = current_node.get_link_to(next_node_obj.node_id)
                 if not link_to_next:
                      print(f"错误: T={Simulator.now():.9f} 重路由后仍然找不到到下一跳 {next_node_obj.node_id+1} 的链路！丢弃包 {packet.seq_num}。")
                      if flow: flow.stats.lost_packets_other += 1
                      return

        queue, is_busy = link_to_next.get_queue_and_busy_flag(current_node)
        if len(queue) < link_to_next.queue_limit_packets:
            queue.append(packet)
            if not is_busy: Simulator.schedule(0.0, handle_transmission_start, args=(link_to_next, current_node), priority=5)
        else:
            flow = active_flows.get(packet.flow_id)
            if flow:
                flow.stats.lost_packets_queue += 1
                flow.handle_congestion_event()

def _clear_module_processing_lock(current_node: Node, port_num: int, module_idx: int):
    if port_num not in current_node.port_module_busy or module_idx >= config.NUM_PARSING_MODULES_PER_PORT:
        return
    current_node.port_module_busy[port_num][module_idx] = False

def handle_packet_arrival(packet: Packet, current_node: Node, arrival_port: int):
    global lost_packets_module_real
    if arrival_port == -1 or arrival_port == REROUTE_PORT:
        # 对于重路由或源节点发出的包，不检查模块繁忙，直接处理
        Simulator.schedule(config.NODE_PROCESSING_DELAY_S,
                           _process_packet_at_node,
                           args=(packet, current_node, arrival_port, False), # 传入 module_was_busy=False
                           priority=11)
        return
    
    current_node.port_module_busy.setdefault(arrival_port, [False] * config.NUM_PARSING_MODULES_PER_PORT)
    current_node.port_next_module_index.setdefault(arrival_port, 0)
    module_idx = current_node.port_next_module_index[arrival_port]
    current_node.port_next_module_index[arrival_port] = (module_idx + 1) % config.NUM_PARSING_MODULES_PER_PORT
    is_module_busy = current_node.port_module_busy[arrival_port][module_idx]

    # [BUG修复] lost_packets_module_real 应该在检查繁忙之前增加，因为它代表所有试图进入的包
    lost_packets_module_real += 1
    
    if not is_module_busy:
        # 模块不繁忙，正常处理
        current_node.port_module_busy[arrival_port][module_idx] = True
        Simulator.schedule(config.NODE_PROCESSING_DELAY_S, _process_packet_at_node, args=(packet, current_node, arrival_port, False), priority=11)
        Simulator.schedule(config.NODE_PROCESSING_DELAY_S, _clear_module_processing_lock, args=(current_node, arrival_port, module_idx), priority=10)
    else:
        # 模块繁忙，记录丢包
        flow = active_flows.get(packet.flow_id)
        if flow:
            flow.stats.lost_packets_module_busy += 1
        Simulator.schedule(config.NODE_PROCESSING_DELAY_S, _process_packet_at_node, args=(packet, current_node, arrival_port, True), priority=11)


def handle_transmission_start(link: Link, source_node: Node):
    queue, is_busy = link.get_queue_and_busy_flag(source_node)
    if is_busy: return
    if not queue: link.set_busy_flag(source_node, False); return
    packet_to_send = queue.popleft()
    link.set_busy_flag(source_node, True)
    Simulator.schedule(link.transmission_time_s, handle_transmission_end, args=(link, source_node, packet_to_send), priority=5)

def handle_transmission_end(link: Link, source_node: Node, transmitted_packet: Packet):
    dest_node = link.get_other_node(source_node)
    link.set_busy_flag(source_node, False)
    arrival_port = -1; found_port = False
    for iface in dest_node.interfaces:
        if iface.get('link') is link: arrival_port = iface.get('port', -1); found_port = True; break
    if not found_port:
        print(f"警告: T={Simulator.now():.9f} 在目的节点 {dest_node.node_id+1} 找不到链路端口！")
        arrival_port = -1
    Simulator.schedule(link.delay_s, handle_packet_arrival, args=(transmitted_packet, dest_node, arrival_port), priority=10)
    queue, _ = link.get_queue_and_busy_flag(source_node)
    if queue: Simulator.schedule(0.0, handle_transmission_start, args=(link, source_node), priority=6)

def print_simulation_time(log_file_handle):
    if log_file_handle and not log_file_handle.closed:
        log_file_handle.write(f"当前仿真时间: {Simulator.now():.6f} s\n"); log_file_handle.flush()
        if Simulator.now() < Simulator.stop_time - 1.0: Simulator.schedule(1.0, print_simulation_time, args=(log_file_handle,), priority=100)

def display_progress():
    current_time = Simulator.now()
    total_time = Simulator.stop_time
    if total_time <= 0: return
    progress = max(0.0, min(1.0, (current_time - config.GLOBAL_START_TIME) / (total_time - config.GLOBAL_START_TIME)))
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'仿真进度: |{bar}| {progress*100:.1f}% ({current_time:.2f}s / {total_time:.2f}s)   ', end='\r')
    if current_time < total_time - 1e-9:
        Simulator.schedule(0.2, display_progress, priority=110)

def update_topology_and_flows(current_second: int, log_file_handle):
    """在每个整数秒开始时，更新拓扑、IP、流路径和流速率"""
    global active_flows, nodes_list, links_list, ip_helper
    
    current_sim_time = Simulator.now()
    log_prefix = f"T={current_sim_time:.6f}s [Update {current_second}s]:"

    print(f"\n{log_prefix} 开始更新拓扑和流量...")
    if log_file_handle and not log_file_handle.closed:
        log_file_handle.write(f"{log_prefix} 开始更新拓扑和流量...\n")

    new_adj_matrix = read_adjacency_matrix(current_second)
    if not new_adj_matrix:
        print(f"{log_prefix} 错误: 无法加载邻接矩阵，跳过本秒更新。")
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.write(f"{log_prefix} 错误: 无法加载邻接矩阵，跳过本秒更新。\n")
        next_second = current_second + 1
        if next_second <= int(config.TRAFFIC_STOP_TIME):
             next_update_time = float(next_second)
             delay = max(0.0, next_update_time - Simulator.now())
             Simulator.schedule(delay, update_topology_and_flows, args=(next_second, log_file_handle), priority=15)
        return

    current_links_set = set()
    for link in links_list:
        n1, n2 = link.node1.node_id, link.node2.node_id
        current_links_set.add(tuple(sorted((n1, n2))))
    new_links_set = set()
    for i in range(config.NUM_SATELLITES):
        for j in range(i + 1, config.NUM_SATELLITES):
            if new_adj_matrix[i][j]:
                new_links_set.add((i, j))
    links_to_add = new_links_set - current_links_set
    links_to_remove_tuples = current_links_set - new_links_set

    links_removed_count = 0
    packets_to_reroute_count = 0
    links_list_new_iteration = []

    for link in list(links_list):
        link_tuple = tuple(sorted((link.node1.node_id, link.node2.node_id)))
        if link_tuple in links_to_remove_tuples:
            links_removed_count += 1
            packets_in_link_queues = link.get_packets_in_queues()
            if packets_in_link_queues:
                 log_msg_reroute = f"{log_prefix} 链路 {link.node1.node_id+1}-{link.node2.node_id+1} 断开，尝试重注入 {len(packets_in_link_queues)} 个包..."
                 if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg_reroute + "\n")

            for packet, sending_node in packets_in_link_queues:
                packets_to_reroute_count += 1
                flow = active_flows.get(packet.flow_id)
                if flow: flow.stats.packets_rerouted += 1
                Simulator.schedule(0.0, handle_packet_arrival, args=(packet, sending_node, REROUTE_PORT), priority=9)

            link.node1.remove_interface_for_link(link)
            link.node2.remove_interface_for_link(link)
        else:
            links_list_new_iteration.append(link)
    links_list = links_list_new_iteration

    if links_removed_count > 0: print(f"{log_prefix} 移除了 {links_removed_count} 条链路。")
    if packets_to_reroute_count > 0:
         log_msg_drop = f"{log_prefix} 共尝试重注入 {packets_to_reroute_count} 个来自断开链路队列的数据包。"
         print(log_msg_drop)
         if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg_drop + "\n")

    links_added_count = 0
    for link_tuple in links_to_add:
        i, j = link_tuple
        link = Link(nodes_list[i], nodes_list[j], config.LINK_DATA_RATE_MBPS, config.LINK_DELAY_S, config.QUEUE_SIZE_PACKETS)
        links_list.append(link)
        try:
            ip_helper.assign(link); ip_helper.new_network(); links_added_count += 1
        except Exception as e:
            print(f"ERROR: IP Assignment failed for new link {i}-{j}: {e}"); Simulator.stop_simulation(); return
    if links_added_count > 0: print(f"{log_prefix} 新增了 {links_added_count} 条链路。")

    ip_filename = config.IP_OUTPUT_CSV_FILENAME.format(current_second)
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(ip_filename), exist_ok=True)
        with open(ip_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['SatelliteID', 'PortNumber', 'IPAddress']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for node in nodes_list:
                if node.interfaces:
                    sorted_interfaces = sorted(node.interfaces, key=lambda iface: iface['port'])
                    for iface in sorted_interfaces:
                        writer.writerow({'SatelliteID': node.node_id + 1, 'PortNumber': iface['port'], 'IPAddress': iface['ip'].address})
    except Exception as e: print(f"{log_prefix} 导出 IP 分配时出错: {e}")


    print(f"{log_prefix} 更新现有流的路径...")
    flows_to_stop = []
    for flow_id, flow in active_flows.items():
        new_path = find_shortest_path(nodes_list, flow.source.node_id, flow.dest.node_id)
        flow.update_path(new_path)
        if not new_path and flow.is_generating: flows_to_stop.append(flow_id)
    for flow_id in flows_to_stop:
        if flow_id in active_flows:
             log_msg = f"{log_prefix} 停止流 {flow_id}，因为在新拓扑下找不到路径。"
             if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg + "\n")
             active_flows[flow_id].update_target_rate(0.0)

    traffic_matrix = read_traffic_matrix(current_second)
    if not traffic_matrix:
        print(f"{log_prefix} 警告: 无法加载第 {current_second} 秒的流量矩阵，本时段无流量更新。")
    else:
        print(f"{log_prefix} 根据流量矩阵调整流...")
        active_in_this_second = set()
        for i in range(config.NUM_SATELLITES):
            for j in range(config.NUM_SATELLITES):
                if i == j: continue
                rate_mbps = traffic_matrix[i][j]
                flow_id_display = f"F_{i+1}-{j+1}"
                existing_flow = active_flows.get(flow_id_display)

                if rate_mbps > 1e-9:
                    active_in_this_second.add(flow_id_display)
                    if existing_flow:
                        if not existing_flow.path:
                             existing_flow.update_target_rate(0.0); continue
                        if abs(existing_flow.current_target_rate_bps/1e6 - rate_mbps) > 1e-9:
                            log_msg = f"{log_prefix} 更新流 {flow_id_display} 速率为 {rate_mbps:.2f} Mbps"
                            if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg + "\n")
                            existing_flow.update_target_rate(rate_mbps)
                    else:
                        new_path = find_shortest_path(nodes_list, i, j)
                        if new_path:
                            log_msg = f"{log_prefix} 创建新流 {flow_id_display} 速率 {rate_mbps:.2f} Mbps"
                            if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg + "\n")
                            new_flow = TcpFlow(flow_id_display, nodes_list[i], nodes_list[j], nodes_list,
                                             initial_target_rate_mbps=rate_mbps,
                                             packet_entry_handler=handle_packet_arrival)
                            active_flows[flow_id_display] = new_flow
                            new_flow.start()
                else:
                    if existing_flow and existing_flow.is_generating:
                        log_msg = f"{log_prefix} 停止流 {flow_id_display} (需求速率为 0)"
                        if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg + "\n")
                        existing_flow.update_target_rate(0.0)

        for flow_id in list(active_flows.keys()):
             if flow_id not in active_in_this_second:
                 if active_flows[flow_id].is_generating:
                      log_msg = f"{log_prefix} 停止上时段遗留流 {flow_id} (本时段无需求)"
                      if log_file_handle and not log_file_handle.closed: log_file_handle.write(log_msg + "\n")
                      active_flows[flow_id].update_target_rate(0.0)


    next_second = current_second + 1
    if next_second <= int(config.TRAFFIC_STOP_TIME):
        next_update_time = float(next_second)
        delay = max(0.0, next_update_time - Simulator.now())
        Simulator.schedule(delay, update_topology_and_flows, args=(next_second, log_file_handle), priority=15)

def calculate_error_metrics(true_counts: Counter, estimated_counts: Counter) -> Tuple[float, float, float]:
    """计算 ARE, WMRE, RE 三个误差指标"""
    # ARE: 平均相对误差
    total_relative_error = 0
    flow_count = len(true_counts)
    if flow_count > 0:
        for flow_key, true_count in true_counts.items():
            if true_count > 0:
                estimated_count = estimated_counts.get(flow_key, 0)
                total_relative_error += abs(estimated_count - true_count) / true_count
        are = (total_relative_error / flow_count) * 100
    else:
        are = 0

    # RE: 总体相对误差
    true_sum = sum(true_counts.values())
    estimated_sum = sum(estimated_counts.values())
    re = abs(estimated_sum - true_sum) / true_sum * 100 if true_sum > 0 else 0

    # WMRE: 加权平均相对误差
    true_dist = Counter(true_counts.values())
    est_dist = Counter(estimated_counts.values())
    all_sizes = set(true_dist.keys()) | set(est_dist.keys())
    
    numerator = 0
    denominator = 0
    for size in all_sizes:
        n_i = true_dist.get(size, 0)
        n_hat_i = est_dist.get(size, 0)
        numerator += abs(n_i - n_hat_i)
        if (n_i + n_hat_i) > 0:
            denominator += (n_i + n_hat_i)
    
    wmre = (numerator / (denominator/2)) if denominator > 0 else 0
    
    return are, wmre, re

# ==============================================================================
# [MODIFIED] The main logic is now wrapped in main_runner
# It now returns a dictionary of results.
# ==============================================================================
def main_runner(config_module, show_progress=True) -> Dict:
    """
    这是新的仿真主函数，被 run_experiments.py 调用。
    它执行一次完整的仿真，打印所有结果，并返回一个包含关键指标的字典。
    """
    global active_flows, nodes_list, links_list, ip_helper
    global count1, total_processed_packets, lost_packets_module_real
    global elastic_sketch_hits, total_processed_for_elastic
    global cm_sketch_estimated_hits, total_processed_for_cm
    global total_processed_for_bfcm
    
    # --- 重置所有全局状态 ---
    active_flows.clear()
    links_list.clear()
    nodes_list.clear()
    ip_helper.reset()
    
    count1 = 0
    total_processed_packets = 0
    lost_packets_module_real = 0
    elastic_sketch_hits = 0
    total_processed_for_elastic = 0
    cm_sketch_estimated_hits = 0
    total_processed_for_cm = 0
    total_processed_for_bfcm = 0

    Simulator.reset()
    log_file_handle = None # 简化，暂不使用文件日志
    
    # --- 运行仿真 ---
    try:
        print("创建初始节点...")
        nodes_list = [Node(i) for i in range(config_module.NUM_SATELLITES)]
        first_update_second = int(config_module.GLOBAL_START_TIME)
        Simulator.schedule(0.0, update_topology_and_flows, args=(first_update_second, log_file_handle), priority=10)
        if show_progress:
            Simulator.schedule(1.0, print_simulation_time, args=(log_file_handle,), priority=100)
            Simulator.schedule(0.01, display_progress, priority=105)
        Simulator.run()
        if show_progress:
            print() # 清除进度条的行
    except Exception as e:
        print(f"!! 仿真期间发生严重错误: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()

    # ==============================================================================
    # --- 结果统计与打印 (从原始的 if __name__ == '__main__' 移入) ---
    # ==============================================================================
    
    print(f"\n--- 单次仿真完成 ---")
    
    # --- 详细流量统计 ---
    print("\n--- 仿真结果统计 ---")
    total_tx_bytes = 0; total_rx_bytes = 0
    total_lost_queue = 0; total_lost_module_busy = 0; total_lost_other = 0
    total_packets_rerouted = 0
    total_in_flight = 0; all_latencies = []

    if not active_flows: print("没有活动的流量进行统计。")
    else:
        header = f"{'Flow ID':<15} | {'Tx Pkts':>7} | {'Rx Pkts':>7} | {'Lost LinkQ':>8} | {'Lost ModBusy':>10} | {'Lost O':>6} | {'Rerouted':>8} | {'InFlight':>8} | {'Loss %':>7} | {'Avg Lat(ms)':>11} | {'Throughput(Mbps)':>16}"
        print(header)
        print("-" * (len(header) + 2))
        for flow_id, flow in sorted(active_flows.items()):
            stats = flow.stats
            total_tx_packets_flow = stats.tx_packets; total_received = stats.rx_packets
            lost_q = stats.lost_packets_queue; lost_mb = stats.lost_packets_module_busy; lost_o = stats.lost_packets_other
            rerouted = stats.packets_rerouted
            total_lost = stats.get_total_lost_packets()
            in_flight_packets = max(0, total_tx_packets_flow - total_received - total_lost )

            packet_loss_ratio = (total_lost / total_tx_packets_flow) * 100 if total_tx_packets_flow > 0 else 0.0
            throughput_duration = min(Simulator.current_time, config_module.TRAFFIC_STOP_TIME) - config_module.GLOBAL_START_TIME
            throughput_mbps = (stats.rx_bytes * 8.0 / throughput_duration) / 1e6 if throughput_duration > 1e-9 else 0.0
            avg_latency_ms = stats.get_avg_latency() * 1000

            print(f"{flow_id:<15} | {total_tx_packets_flow:>7} | {total_received:>7} | {lost_q:>8} | {lost_mb:>10} | {lost_o:>6} | {rerouted:>8} | {in_flight_packets:>8} | {packet_loss_ratio:>6.2f}% | {avg_latency_ms:.3f} | {throughput_mbps:>16.6f}")

            total_tx_bytes += stats.tx_bytes; total_rx_bytes += stats.rx_bytes
            total_lost_queue += lost_q; total_lost_module_busy += lost_mb; total_lost_other += lost_o
            total_packets_rerouted += rerouted
            total_in_flight += in_flight_packets; all_latencies.extend(stats.latencies)
        print("-" * (len(header) + 2))

    # --- 总体性能统计 ---
    overall_tx_packets = sum(f.stats.tx_packets for f in active_flows.values())
    overall_rx_packets = sum(f.stats.rx_packets for f in active_flows.values())
    overall_total_lost_linkq_other = total_lost_queue + total_lost_other
    overall_loss_ratio = (overall_total_lost_linkq_other / overall_tx_packets) * 100 if overall_tx_packets > 0 else 0.0
    throughput_duration = min(Simulator.current_time, config_module.TRAFFIC_STOP_TIME) - config_module.GLOBAL_START_TIME
    overall_throughput_mbps = (total_rx_bytes * 8.0 / throughput_duration) / 1e6 if throughput_duration > 1e-9 else 0.0
    overall_avg_latency_ms = (sum(all_latencies) / len(all_latencies)) * 1000 if all_latencies else 0.0
    count_loss_ratio = total_lost_module_busy / lost_packets_module_real * 100 if lost_packets_module_real > 0 else 0.0

    print("\n--- 总体统计 ---")
    print(f"总发送数据包: {overall_tx_packets}")
    print(f"总接收数据包: {overall_rx_packets}")
    print(f"总丢失数据包 (链路出口队列): {total_lost_queue}")
    print(f"总丢失数据包 (模块繁忙): {total_lost_module_busy}")
    print(f"总丢失数据包 (其他): {total_lost_other}")
    print(f"总丢失数据包 (检测到的): {overall_total_lost_linkq_other}")
    print(f"总尝试重路由数据包: {total_packets_rerouted}")
    print(f"总在途数据包 (仿真结束时): {total_in_flight}")
    print(f"整体(检测到的)丢包率: {overall_loss_ratio:.2f}%")
    print(f"count(检测到的)丢包率: {count_loss_ratio:.2f}%")
    print(f"总发送字节: {total_tx_bytes}")
    print(f"总接收字节: {total_rx_bytes}")
    print(f"聚合吞吐量 (基于 {throughput_duration:.2f}s 流量生成时间): {overall_throughput_mbps:.6f} Mbps")
    print(f"平均端到端延迟: {overall_avg_latency_ms:.3f} ms")
    print(f"总节点应count数：{lost_packets_module_real}")
    
    # --- 测量方法性能评估 ---
    print("\n--- 测量方法性能评估 ---")

    global_true_counts = Counter()
    for node in nodes_list:
        global_true_counts.update(node.true_flow_counts)
    
    results = {} # 创建字典用于返回结果

    # --- 1. 原有方法 (固定内存池) ---
    print("\n--- 1. 原有方法 (固定内存池) ---")
    print(f"内存池大小 (MEMORY_POOL_SIZE): {config_module.MEMORY_POOL_SIZE*8}")
    
    est_counts_mem = Counter()
    for node in nodes_list:
        est_counts_mem.update(node.flow_memory)
    
    numerator_mem = sum(est_counts_mem.values())
    denominator = lost_packets_module_real
    new_miss_rate_mem = (1 - numerator_mem / denominator) * 100 if denominator > 0 else 0.0
    print(f"新定义 -> 失误率: {new_miss_rate_mem:.2f}% (基于总到达包)")
    
    are_mem, wmre_mem, re_mem = calculate_error_metrics(global_true_counts, est_counts_mem)
    print(f"真实误差 -> ARE: {are_mem:.4f}%, WMRE: {wmre_mem:.4f}, RE: {re_mem:.4f}%")
    results['mem_pool_miss_rate_%'] = new_miss_rate_mem
    results['mem_pool_are_%'] = are_mem
    results['mem_pool_wmre'] = wmre_mem
    results['mem_pool_re_%'] = re_mem

    # --- 2. Elastic Sketch ---
    if config_module.ENABLE_ELASTIC_SKETCH:
        print("\n--- 2. 新方法 (Elastic Sketch) ---")
        print(f"总内存 (BYTES): {config_module.ELASTIC_SKETCH_TOTAL_MEMORY_BYTES}")

        est_counts_es = Counter()
        for node in nodes_list:
            if node.elastic_sketch:
                for flow_key in node.true_flow_counts:
                    est_counts_es[flow_key] += node.elastic_sketch.query(flow_key)
        
        numerator_es = sum(est_counts_es.values())
        new_miss_rate_es = (1 - numerator_es / denominator) * 100 if denominator > 0 else 0.0
        print(f"新定义 -> 失误率: {new_miss_rate_es:.2f}% (基于总到达包)")

        are_es, wmre_es, re_es = calculate_error_metrics(global_true_counts, est_counts_es)
        print(f"真实误差 -> ARE: {are_es:.4f}%, WMRE: {wmre_es:.4f}, RE: {re_es:.4f}%")
        results['es_miss_rate_%'] = new_miss_rate_es
        results['es_are_%'] = are_es
        results['es_wmre'] = wmre_es
        results['es_re_%'] = re_es

    # --- 3. Count-Min Sketch ---
    if config_module.ENABLE_CM_SKETCH:
        print("\n--- 3. 新方法 (Count-Min Sketch) ---")
        print(f"内存预算 (BYTES): {config_module.CM_SKETCH_MEMORY_BYTES}, 目标 Delta: {config_module.CM_SKETCH_TARGET_DELTA}")

        est_counts_cm = Counter()
        for node in nodes_list:
            if node.cm_sketch:
                for flow_key in node.true_flow_counts:
                    est_counts_cm[flow_key] += node.cm_sketch.query(flow_key)
        
        numerator_cm = sum(est_counts_cm.values())
        new_miss_rate_cm = (1 - numerator_cm / denominator) * 100 if denominator > 0 else 0.0
        print(f"新定义 -> 失误率: {new_miss_rate_cm:.2f}% (基于总到达包)")

        are_cm, wmre_cm, re_cm = calculate_error_metrics(global_true_counts, est_counts_cm)
        print(f"真实误差 -> ARE: {are_cm:.4f}%, WMRE: {wmre_cm:.4f}, RE: {re_cm:.4f}%")
        results['cm_miss_rate_%'] = new_miss_rate_cm
        results['cm_are_%'] = are_cm
        results['cm_wmre'] = wmre_cm
        results['cm_re_%'] = re_cm

    # --- 4. BF+CM Sketch (FlowLiDAR) ---
    if config_module.ENABLE_BFCM_SKETCH:
        print("\n--- 4. 新方法 (BF+CM Sketch / FlowLiDAR) ---")
        print(f"总内存 (BYTES): {config_module.BFCM_SKETCH_TOTAL_MEMORY_BYTES}")
        print(f"  (BF 内存占比: {config_module.BFCM_SKETCH_BF_RATIO*100}%, CM 内存占比: {(1-config_module.BFCM_SKETCH_BF_RATIO)*100}%)")

        est_counts_bfcm = Counter()
        for node in nodes_list:
            if node.bfcm_sketch:
                for flow_key in node.true_flow_counts:
                    cp_count = node.bfcm_control_plane_counts.get(flow_key, 0)
                    cm_count = node.bfcm_sketch.cm_sketch.query(flow_key)
                    est_counts_bfcm[flow_key] += (cp_count + cm_count)

        numerator_bfcm = sum(est_counts_bfcm.values())
        new_miss_rate_bfcm = (1 - numerator_bfcm / denominator) * 100 if denominator > 0 else 0.0
        print(f"新定义 -> 失误率: {new_miss_rate_bfcm:.2f}% (基于总到达包)")

        are_bfcm, wmre_bfcm, re_bfcm = calculate_error_metrics(global_true_counts, est_counts_bfcm)
        print(f"真实误差 -> ARE: {are_bfcm:.4f}%, WMRE: {wmre_bfcm:.4f}, RE: {re_bfcm:.4f}%")
        results['bfcm_miss_rate_%'] = new_miss_rate_bfcm
        results['bfcm_are_%'] = are_bfcm
        results['bfcm_wmre'] = wmre_bfcm
        results['bfcm_re_%'] = re_bfcm

    # --- ATM 指标计算 ---
    print("\n--- ATM (Average Traffic Memory) 指标 ---")
    total_recorded_flows = sum(len(node.flow_memory) for node in nodes_list)
    denominator_atm = config_module.NUM_SATELLITES * config_module.memory_all
    atm_metric = total_recorded_flows / denominator_atm if denominator_atm > 0 else 0.0

    print(f"基准方案记录的总流数: {total_recorded_flows}")
    print(f"ATM 计算分母 (NUM_SATELLITES * memory_all): {denominator_atm}")
    print(f"计算出的 ATM: {atm_metric:.6f}")
    results['atm_metric'] = atm_metric

    # --- 补充关键网络性能指标到返回字典 ---
    results['throughput_mbps'] = overall_throughput_mbps
    results['avg_latency_ms'] = overall_avg_latency_ms
    results['loss_rate_link_queue_%'] = overall_loss_ratio
    results['loss_rate_module_busy_%'] = count_loss_ratio

    return results


# ==============================================================================
# [MODIFIED] The old __main__ block is replaced with a new one
# that guides the user to the new entry point.
# ==============================================================================
if __name__ == "__main__":
    print("--- 直接运行 main_simulation.py ---")
    print("--- 这将使用默认的 config.py (如果存在) 或最后加载的配置 ---")
    print("--- 推荐使用 'python run_experiments.py' 来运行实验 ---")
    
    try:
        print("测试模式。")
    except ImportError:
        print("\n错误: 无法直接运行。")
        print("请确保您已创建 'configs/base_config.py' 文件。")
        print("建议通过 'python run_experiments.py' 启动。")
    except Exception as e:
        print(f"\n直接运行时发生错误: {e}")

