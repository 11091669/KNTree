
import networkx as nx
import numpy as np
import queue
import functools
import os

# 获取算法模块所在目录的父目录
ALGORITHM_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVE_DIR = os.path.dirname(ALGORITHM_DIR)

TOPO_FILES = {
    'Equinix': os.path.join(SOLVE_DIR, 'topo/topo-with-comments/Equinix/'),
    'G_Scale': os.path.join(SOLVE_DIR, 'topo/topo-with-comments/G_Scale/'),
    'IDN': os.path.join(SOLVE_DIR, 'topo/topo-with-comments/IDN/'),
    'TestTopo': os.path.join(SOLVE_DIR, 'topo/topo-with-comments/TestTopo/'),
    'Expander': os.path.join(SOLVE_DIR, 'topo/topo-with-comments/Expander/'),
}

def createGraph(Edge, Delay = 0):  # 绘制网络图
    G = nx.Graph()
    G.add_edges_from(Edge, bandwidth=200)  # 设置节点权重

    return G

def getTopoGraph(name):
    """
    根据名称获取拓扑图
    """
    if name in TOPO_FILES:
        topo_file = TOPO_FILES[name]
        linksFilesPath = TOPO_FILES[name] + 'links.txt'
        links = []
        with open(linksFilesPath, 'r') as f:
            for line in f:
            # 分割每行的数据并提取前两个字段
                parts = line.strip().split()
        
                # 提取前两个字段作为边 (node1, node2)
                if len(parts) >= 2:
                    node1, node2 = int(parts[0]), int(parts[1])
                    links.append((node1, node2))
        return createGraph(links)
    else:
        raise ValueError(f"Topology '{name}' not found in predefined topologies.")
    