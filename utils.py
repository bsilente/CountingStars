import pandas as pd
import numpy as np
import heapq
import sys
import csv
import os
from typing import List, Tuple, TYPE_CHECKING
import config
from simulation_core import Ipv4Address, Node, Link, Ipv4AddressHelper


def read_adjacency_matrix(second: int) -> List[List[bool]]:
    filename = config.ADJACENCY_MATRIX_TEMPLATE.format(second)
    filepath = os.path.join(config.ADJACENCY_MATRIX_DIR, filename)
    matrix = []
    try:
        df = pd.read_csv(filepath, header=None)
        matrix_np = df.to_numpy()
        def safe_to_bool(x):
            if pd.isna(x): return False
            s = str(x).strip();
            if s.isdigit(): return int(s) != 0
            try: return float(s) != 0.0
            except ValueError: return False
        bool_matrix = np.vectorize(safe_to_bool)(matrix_np)
        if bool_matrix.shape[0] != bool_matrix.shape[1]: raise ValueError("Adjacency matrix must be square")
        if bool_matrix.shape[0] != config.NUM_SATELLITES: raise ValueError(f"Matrix size does not match the number of satellites")
        matrix = bool_matrix.tolist()
    except FileNotFoundError:
        print(f"Warning: Adjacency matrix file '{filepath}' not found for time {second}s.")
        return []
    except Exception as e:
        print(f"Error parsing adjacency matrix file '{filepath}': {e}")
        return []
    return matrix


def find_shortest_path(nodes: List['Node'], source_node_id: int, dest_node_id: int) -> List['Node']:
    graph = {node.node_id: [] for node in nodes}
    for node in nodes:
        for neighbor_id, link in node.links.items():
            graph[node.node_id].append((neighbor_id, 1))

    distances = {node.node_id: float('inf') for node in nodes}
    predecessors = {node.node_id: None for node in nodes}
    pq = [(0, source_node_id)]; distances[source_node_id] = 0
    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_id == dest_node_id: break
        if current_distance > distances[current_id]: continue
        if current_id not in graph: continue
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
        else: print(f"Error: Node ID {current_id} not found during path reconstruction."); return []
        current_id = predecessors[current_id]
    return path_nodes[::-1]

def read_traffic_matrix(second: int) -> List[List[float]]:
    filename = config.TRAFFIC_MATRIX_TEMPLATE.format(second)
    filepath = os.path.join(config.TRAFFIC_MATRIX_DIR, filename)
    matrix = []
    try:
        df = pd.read_csv(filepath, header=None)
        if df.shape[0] != config.NUM_SATELLITES or df.shape[1] != config.NUM_SATELLITES:
            print(f"Error: The dimensions ({df.shape}) of the traffic matrix file '{filepath}' do not match the number of satellites ({config.NUM_SATELLITES}).")
            return []
        matrix = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.tolist()
    except FileNotFoundError:
        print(f"Warning: Traffic matrix file '{filepath}' not found for time {second}s. Will use zero traffic.")
        matrix = [[0.0] * config.NUM_SATELLITES for _ in range(config.NUM_SATELLITES)]
    except Exception as e:
        print(f"Error reading traffic matrix file '{filepath}': {e}")
        matrix = [[0.0] * config.NUM_SATELLITES for _ in range(config.NUM_SATELLITES)]
    return matrix
