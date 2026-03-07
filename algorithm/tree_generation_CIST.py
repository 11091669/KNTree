"""
Tree generation algorithms for fault-tolerant network topologies.

Current model focuses on feasibility:
- Find k spanning trees in graph G
- Internal-node disjointness across trees
- Edge disjointness across trees
- No residual-bandwidth constraints
"""

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from typing import Dict


def genTree(G: nx.Graph, k: int, n: int = None) -> gp.Model:
    """
    Build a feasibility model that searches k trees satisfying:
    1) each tree is a spanning tree
    2) internal nodes are mutually exclusive across trees
    3) each undirected edge can appear in at most one tree

    Args:
        G: topology graph
        k: number of trees
        n: reserved for API compatibility (unused)

    Returns:
        Gurobi model
    """
    model = gp.Model("CIST_feasibility")

    V = list(G.nodes())
    E_undirected = list(G.edges())
    E = E_undirected + [(j, i) for i, j in E_undirected]
    N = len(V)

    if N == 0:
        raise ValueError("Input graph is empty.")

    source_node = V[0]

    x = {}  # x[t,i,j] = 1 if directed edge (i,j) is chosen in tree t
    p = {}  # p[t,i] = 1 if node i is internal (non-leaf) in tree t
    f = {}  # flow for connectivity

    for t in range(k):
        for i in V:
            p[t, i] = model.addVar(vtype=GRB.BINARY, name=f"p_{t}_{i}")

        for i, j in E:
            x[t, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{i}_{j}")
            f[t, i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{t}_{i}_{j}")

    # Feasibility objective only
    model.setObjective(0, GRB.MINIMIZE)

    # 1) Tree structure constraints
    for t in range(k):
        # Exactly N-1 selected directed edges in tree t
        model.addConstr(gp.quicksum(x[t, i, j] for i, j in E) == N - 1, name=f"tree_size_{t}")

        # An undirected edge can be used in at most one direction in each tree
        for i, j in E_undirected:
            model.addConstr(x[t, i, j] + x[t, j, i] <= 1, name=f"edge_dir_once_{t}_{i}_{j}")

        # Single-commodity flow connectivity
        model.addConstr(
            gp.quicksum(f[t, source_node, j] for j in V if (source_node, j) in E) == N - 1,
            name=f"source_flow_{t}",
        )

        for i in V:
            if i == source_node:
                continue
            inflow = gp.quicksum(f[t, j, i] for j in V if (j, i) in E)
            outflow = gp.quicksum(f[t, i, j] for j in V if (i, j) in E)
            model.addConstr(inflow == outflow + 1, name=f"flow_balance_{t}_{i}")

        for i, j in E:
            model.addConstr(f[t, i, j] <= (N - 1) * x[t, i, j], name=f"flow_edge_{t}_{i}_{j}")

    # 2) Internal-node definition by degree in each tree
    for t in range(k):
        for i in V:
            deg_i_t = gp.quicksum(x[t, i, j] + x[t, j, i] for j in V if (i, j) in E)

            # p=1 => degree >= 2
            model.addConstr(deg_i_t >= 2 * p[t, i], name=f"non_leaf_lb_{t}_{i}")

            # p=0 => degree <=1 ; p=1 => degree can be up to N-1
            model.addConstr(deg_i_t <= 1 + (N - 2) * p[t, i], name=f"non_leaf_ub_{t}_{i}")

    # 3) Internal-node disjointness across trees
    for i in V:
        model.addConstr(gp.quicksum(p[t, i] for t in range(k)) <= 1, name=f"internal_disjoint_{i}")

    # 4) Edge disjointness across trees
    for i, j in E_undirected:
        model.addConstr(
            gp.quicksum(x[t, i, j] + x[t, j, i] for t in range(k)) <= 1,
            name=f"edge_disjoint_{i}_{j}",
        )

    return model


def add_more_trees(G: nx.Graph, model1: gp.Model, k: int, extra_k: int) -> gp.Model:
    """Deprecated: second-phase residual-bandwidth model has been removed."""
    raise NotImplementedError("add_more_trees is deprecated after removing residual-bandwidth constraints.")


def model_info_to_file(model: gp.Model, filename: str, k: int, G: nx.Graph):
    """Write selected tree edges and edge-weight summaries to file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("CIST feasibility solution\n")
        f.write(f"status: {model.status}\n")
        f.write(f"objective: {model.objVal if model.SolCount > 0 else 'N/A'}\n")
        f.write(f"k: {k}\n\n")

        total_weight = 0.0
        for t in range(k):
            f.write(f"Tree {t}\n")
            f.write("Edges:\n")
            tree_weight = 0.0

            for i, j in G.edges():
                x_ij = model.getVarByName(f"x_{t}_{i}_{j}")
                x_ji = model.getVarByName(f"x_{t}_{j}_{i}")
                w = G.edges[i, j].get("weight", 1.0)

                if x_ij is not None and x_ij.x > 0.5:
                    f.write(f"  ({i}, {j}), weight={w}\n")
                    tree_weight += w
                elif x_ji is not None and x_ji.x > 0.5:
                    f.write(f"  ({j}, {i}), weight={w}\n")
                    tree_weight += w

            total_weight += tree_weight
            f.write(f"Tree {t} total_weight: {tree_weight}\n\n")

        f.write(f"All trees total_weight: {total_weight}\n")


def model_info_to_file_two_phase(
    model2: gp.Model,
    filename: str,
    k: int,
    extra_k: int,
    G: nx.Graph,
    model1: gp.Model,
    n: int = None,
):
    """Deprecated: second-phase residual-bandwidth model has been removed."""
    raise NotImplementedError("model_info_to_file_two_phase is deprecated after removing residual-bandwidth constraints.")


def run(
    name: str,
    k: int,
    n: int,
    extra_k: int = None,
    time_limit: int = 5400,
    mip_gap: float = 0.01,
    threads: int = 10,
) -> Dict:
    """
    Solve feasibility model for k edge-disjoint / internal-node-disjoint trees.

    Args:
        name: topology name
        k: number of trees
        n: reserved for API compatibility (unused)
        extra_k: reserved for API compatibility (unused)
        time_limit: solver time limit (seconds)
        mip_gap: reserved for API compatibility (unused)
        threads: solver threads
    """
    from GraphFunc import getTopoGraph

    G = getTopoGraph(name)
    model = genTree(G, k, n)

    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", threads)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Found feasible solution (OPTIMAL), obj={model.objVal}")
        out_file = f"model_info_feasible_topo_{name}_n={n}_k={k}.txt"
        model_info_to_file(model, out_file, k, G)
    elif model.status in (GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
        print(f"Found feasible solution before stop, status={model.status}, obj={model.objVal}")
        out_file = f"model_info_feasible_partial_topo_{name}_n={n}_k={k}.txt"
        model_info_to_file(model, out_file, k, G)
    else:
        print(f"No feasible solution found, status={model.status}")
        return {"model": model, "feasible": False}

    return {"model": model, "feasible": True, "output": out_file}
