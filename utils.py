# utils.py
# 包含辅助函数

import pandas as pd
import numpy as np
import heapq
import sys
import csv
import os
from typing import List, Tuple, TYPE_CHECKING

# 导入配置常量和所需类
import config
# 导入 simulation_core 中的类
from simulation_core import Ipv4Address, Node, Link, Ipv4AddressHelper


def read_adjacency_matrix(second: int) -> List[List[bool]]:
    """读取指定秒数的邻接矩阵CSV文件"""
    filename = config.ADJACENCY_MATRIX_TEMPLATE.format(second)
    filepath = os.path.join(config.ADJACENCY_MATRIX_DIR, filename)
    matrix = []
    try:
        # print(f"读取邻接矩阵: {filepath} ...") # 可选调试
        df = pd.read_csv(filepath, header=None)
        matrix_np = df.to_numpy()
        def safe_to_bool(x):
            if pd.isna(x): return False
            s = str(x).strip();
            if s.isdigit(): return int(s) != 0
            try: return float(s) != 0.0
            except ValueError: return False
        bool_matrix = np.vectorize(safe_to_bool)(matrix_np)
        if bool_matrix.shape[0] != bool_matrix.shape[1]: raise ValueError("邻接矩阵必须为方阵")
        if bool_matrix.shape[0] != config.NUM_SATELLITES: raise ValueError(f"矩阵大小与卫星数量不匹配")
        # print(f"成功从 '{filepath}' 解析 {bool_matrix.shape[0]}x{bool_matrix.shape[1]} 邻接矩阵.") # 可选调试
        matrix = bool_matrix.tolist()
    except FileNotFoundError:
        print(f"警告: 未找到时间 {second}s 的邻接矩阵文件 '{filepath}'。")
        return []
    except Exception as e:
        print(f"解析邻接矩阵文件 '{filepath}' 时出错: {e}")
        return []
    return matrix


def find_shortest_path(nodes: List['Node'], source_node_id: int, dest_node_id: int) -> List['Node']:
    """使用 Dijkstra 算法查找最短路径 (基于跳数)"""
    graph = {node.node_id: [] for node in nodes}
    for node in nodes:
        # 使用 node.links 这个实时更新的字典
        for neighbor_id, link in node.links.items():
            graph[node.node_id].append((neighbor_id, 1))

    distances = {node.node_id: float('inf') for node in nodes}
    predecessors = {node.node_id: None for node in nodes}
    pq = [(0, source_node_id)]; distances[source_node_id] = 0
    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_id == dest_node_id: break
        if current_distance > distances[current_id]: continue
        if current_id not in graph: continue # 可能节点暂时没有连接
        for neighbor_id, weight in graph[current_id]:
            distance = current_distance + weight
            if distance < distances[neighbor_id]:
                distances[neighbor_id] = distance; predecessors[neighbor_id] = current_id
                heapq.heappush(pq, (distance, neighbor_id))
    path_nodes = []
    current_id = dest_node_id
    if distances[dest_node_id] == float('inf'): return []
    while current_id is not None:
        node_obj = next((n for n in nodes if n.node_id == current_id), None)
        if node_obj: path_nodes.append(node_obj)
        else: print(f"错误: 路径重构中未找到节点 ID {current_id}。"); return []
        current_id = predecessors[current_id]
    return path_nodes[::-1]

# build_satellite_topo 函数已移除

def read_traffic_matrix(second: int) -> List[List[float]]:
    """读取指定秒数的流量矩阵CSV文件"""
    filename = config.TRAFFIC_MATRIX_TEMPLATE.format(second)
    filepath = os.path.join(config.TRAFFIC_MATRIX_DIR, filename)
    matrix = []
    try:
        # print(f"读取流量矩阵: {filepath} ...")
        df = pd.read_csv(filepath, header=None)
        if df.shape[0] != config.NUM_SATELLITES or df.shape[1] != config.NUM_SATELLITES:
            print(f"错误: 流量矩阵文件 '{filepath}' 的维度 ({df.shape}) 与卫星数量 ({config.NUM_SATELLITES}) 不匹配。")
            return []
        matrix = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.tolist()
        # print(f"成功读取流量矩阵 {filepath}")
    except FileNotFoundError:
        print(f"警告: 未找到时间 {second}s 的流量矩阵文件 '{filepath}'。将使用零流量。")
        matrix = [[0.0] * config.NUM_SATELLITES for _ in range(config.NUM_SATELLITES)]
    except Exception as e:
        print(f"读取流量矩阵文件 '{filepath}' 时出错: {e}")
        matrix = [[0.0] * config.NUM_SATELLITES for _ in range(config.NUM_SATELLITES)]
    return matrix

