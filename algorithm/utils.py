
import networkx as nx
import itertools
import json
import os
from typing import List, Set, Tuple, Dict, Any
from gurobipy import GRB


def is_leaf(T: nx.Graph, node: int) -> bool:
    """
    检查节点是否是树T中的叶节点。
    叶节点的度为1（对于孤立的单节点树，度为0）。
    """
    degree = T.degree(node)
    return degree <= 1


def save_model_to_json(model: Any, filename: str, k: int, n: int, G: nx.Graph, name: str) -> Dict:
    """
    将模型结果保存到 JSON 文件

    Args:
        model: Gurobi优化模型
        filename: 输出文件名
        k: 生成树数量
        n: 故障节点数量
        G: 拓扑图
        name: 拓扑名称

    Returns:
        Dict: 包含模型结果的字典
    """
    status_dict = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.NUMERIC: "numeric",
        GRB.SUBOPTIMAL: "suboptimal",
    }

    result = {
        "basic_info": {
            "topology_name": name,
            "tree_count": k,
            "failure_nodes": n,
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges())
        },
        "solve_status": {
            "status": status_dict.get(model.status, f"unknown_{model.status}"),
            "status_code": model.status
        },
        "trees": [],
        "link_utilization": {},
        "summary": {}
    }

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
        result["solve_status"]["objective_value"] = float(model.objVal)
        result["solve_status"]["solve_time"] = float(model.Runtime)
        if model.MIPGap is not None:
            result["solve_status"]["mip_gap"] = float(model.MIPGap)

        r_min_var = model.getVarByName("r_min")
        if r_min_var is not None:
            result["solve_status"]["r_min"] = float(r_min_var.x)

        # 获取每棵树的信息
        E = [(i,j) for (i,j) in list(G.edges())] + [(j,i) for (i,j) in list(G.edges())]

        for t in range(k):
            tree_info = {
                "tree_id": t,
                "bandwidth": None,
                "non_leaf_nodes": [],
                "edges": []
            }

            w_var = model.getVarByName(f"w_{t}")
            if w_var is not None:
                tree_info["bandwidth"] = float(w_var.x)

            # 非叶节点
            for i in G.nodes():
                p_var = model.getVarByName(f"p_{t}_{i}")
                if p_var is not None and p_var.x > 0.5:
                    tree_info["non_leaf_nodes"].append(i)

            # 树中的边
            for (i, j) in E:
                x_var = model.getVarByName(f"x_{t}_{i}_{j}")
                if x_var is not None and x_var.x > 0.5:
                    tree_info["edges"].append([i, j])

            result["trees"].append(tree_info)

        # 链路带宽利用率
        total_used_all = 0.0
        total_max_all = 0.0

        for (i, j) in G.edges():
            max_bandwidth = G.edges[i, j]['bandwidth']
            total_used = 0.0

            for t in range(k):
                w_var = model.getVarByName(f"w_{t}")
                x_var1 = model.getVarByName(f"x_{t}_{i}_{j}")
                x_var2 = model.getVarByName(f"x_{t}_{j}_{i}")

                if w_var is not None:
                    used_t = w_var.x * ((x_var1.x if x_var1 else 0) + (x_var2.x if x_var2 else 0))
                    total_used += used_t

            total_used_all += total_used
            total_max_all += max_bandwidth

            utilization = (total_used / max_bandwidth) * 100 if max_bandwidth > 0 else 0
            result["link_utilization"][f"{i}-{j}"] = {
                "max_bandwidth": float(max_bandwidth),
                "used_bandwidth": float(total_used),
                "utilization_percent": float(utilization)
            }

        # 总体统计
        total_utilization = (total_used_all / total_max_all) * 100 if total_max_all > 0 else 0
        result["summary"] = {
            "total_max_bandwidth": float(total_max_all),
            "total_used_bandwidth": float(total_used_all),
            "total_utilization_percent": float(total_utilization)
        }

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 保存到 JSON 文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def load_model_from_json(filename: str) -> Dict:
    """
    从 JSON 文件加载模型结果

    Args:
        filename: JSON 文件名

    Returns:
        Dict: 包含模型结果的字典
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_tree_from_json(result: Dict) -> List[nx.Graph]:
    """
    从 JSON 结果字典中解析出树结构

    Args:
        result: JSON 加载的结果字典

    Returns:
        List[nx.Graph]: 树的列表
    """
    trees = []
    for tree_info in result.get("trees", []):
        tree = nx.Graph()
        for edge in tree_info.get("edges", []):
            tree.add_edge(edge[0], edge[1])
        trees.append(tree)
    return trees


def get_tree_edges(result: Dict, tree_id: int) -> List[Tuple[int, int]]:
    """
    获取指定树的边列表

    Args:
        result: JSON 加载的结果字典
        tree_id: 树的ID

    Returns:
        List[Tuple[int, int]]: 边列表
    """
    trees = result.get("trees", [])
    if 0 <= tree_id < len(trees):
        return [tuple(edge) for edge in trees[tree_id].get("edges", [])]
    return []


def get_tree_bandwidth(result: Dict, tree_id: int) -> float:
    """
    获取指定树的带宽

    Args:
        result: JSON 加载的结果字典
        tree_id: 树的ID

    Returns:
        float: 带宽值
    """
    trees = result.get("trees", [])
    if 0 <= tree_id < len(trees):
        return trees[tree_id].get("bandwidth", 0.0)
    return 0.0
