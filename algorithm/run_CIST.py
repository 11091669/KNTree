import os
import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from algorithm.tree_generation_CIST import genTree
from algorithm.GraphFunc import getTopoGraph

# Result directory
RESULT_DIR = "result_CIST"


def _status_to_text(status: int) -> str:
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
    return status_dict.get(status, f"unknown_{status}")


def _edge_bandwidth(G, i, j) -> float:
    # Graph may not carry bandwidth attr on every edge; default to 0.0
    return float(G.edges[i, j].get("bandwidth", 0.0))


def _extract_tree_info(model: gp.Model, G, k: int):
    """
    Extract tree edges/non-leaf nodes from model.
    Tree bandwidth is defined as the minimum bandwidth over all edges in that tree.
    """
    trees = []

    undirected_edges = list(G.edges())

    for t in range(k):
        selected_edges = []
        non_leaf_nodes = []

        for i in G.nodes():
            p_var = model.getVarByName(f"p_{t}_{i}")
            if p_var is not None and p_var.x > 0.5:
                non_leaf_nodes.append(i)

        for i, j in undirected_edges:
            x_ij = model.getVarByName(f"x_{t}_{i}_{j}")
            x_ji = model.getVarByName(f"x_{t}_{j}_{i}")
            if x_ij is not None and x_ij.x > 0.5:
                selected_edges.append((i, j))
            elif x_ji is not None and x_ji.x > 0.5:
                selected_edges.append((j, i))

        edge_bws = [_edge_bandwidth(G, u, v) for (u, v) in selected_edges]
        tree_bandwidth = min(edge_bws) if edge_bws else 0.0

        trees.append(
            {
                "tree_id": t,
                "bandwidth": float(tree_bandwidth),
                "non_leaf_nodes": non_leaf_nodes,
                "edges": [[u, v] for u, v in selected_edges],
            }
        )

    return trees


def _build_result(model: gp.Model, G, name: str, k: int):
    trees = _extract_tree_info(model, G, k) if model.SolCount > 0 else []

    result = {
        "basic_info": {
            "topology_name": name,
            "tree_count": k,
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
        },
        "solve_status": {
            "status": _status_to_text(model.status),
            "status_code": model.status,
            "solve_time": float(model.Runtime),
        },
        "trees": trees,
        "link_utilization": {},
        "summary": {},
    }

    if model.SolCount > 0:
        result["solve_status"]["objective_value"] = float(model.objVal)
        if model.MIPGap is not None:
            result["solve_status"]["mip_gap"] = float(model.MIPGap)

    # Edge usage under CIST edge exclusivity:
    # if an edge belongs to a tree, occupied bandwidth on that edge equals that tree's bandwidth.
    total_used_all = 0.0
    total_max_all = 0.0

    # Build quick lookup: undirected edge -> occupied bandwidth
    occupied_by_edge = {}
    for tree in trees:
        bw = float(tree["bandwidth"])
        for u, v in tree["edges"]:
            a, b = (u, v) if (u, v) in G.edges else (v, u)
            key = tuple(sorted((a, b)))
            occupied_by_edge[key] = bw

    for i, j in G.edges():
        key = tuple(sorted((i, j)))
        max_bw = _edge_bandwidth(G, i, j)
        used_bw = float(occupied_by_edge.get(key, 0.0))

        total_used_all += used_bw
        total_max_all += max_bw

        util = (used_bw / max_bw * 100.0) if max_bw > 0 else 0.0
        result["link_utilization"][f"{i}-{j}"] = {
            "max_bandwidth": max_bw,
            "used_bandwidth": used_bw,
            "utilization_percent": util,
        }

    total_util = (total_used_all / total_max_all * 100.0) if total_max_all > 0 else 0.0
    result["summary"] = {
        "total_max_bandwidth": float(total_max_all),
        "total_used_bandwidth": float(total_used_all),
        "total_utilization_percent": float(total_util),
    }

    return result


def run(name: str, k: int):
    """Run CIST feasibility model and save result json."""
    save_dir = os.path.join(RESULT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    G = getTopoGraph(name)
    model = genTree(G, k)

    # Feasibility model: no objective optimization target beyond finding a valid solution
    model.setParam("Threads", 10)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Found feasible CIST solution, status=OPTIMAL, obj={model.objVal}")
    elif model.status in (GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL) and model.SolCount > 0:
        print(f"Found feasible CIST solution before stop, status={model.status}, obj={model.objVal}")
    else:
        print(f"No feasible CIST solution found, status={model.status}")

    filename = os.path.join(save_dir, f"k={k}.json")
    result = _build_result(model, G, name, k)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Result saved to: {filename}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run CIST tree generation model")
    parser.add_argument(
        "--topo",
        type=str,
        default="Equinix",
        help="Topology name (default: Equinix)",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Tree counts to run (default: 2 3)",
    )

    args = parser.parse_args()

    for k in args.k_list:
        print(f"\n{'=' * 50}")
        print(f"Topology: {args.topo}, tree_count: {k}")
        print(f"{'=' * 50}")
        run(args.topo, k)


if __name__ == "__main__":
    main()
