"""
Tree generation algorithms for fault-tolerant network topologies.
"""

import time
import gurobipy as gp
from gurobipy import GRB
import itertools
import networkx as nx
from typing import List, Set, Tuple, Dict

M = 1000  # 大M值

def genTree(G: nx.Graph, k: int, n: int) -> gp.Model:
    """
    输入一个图生成一组最大化剩余带宽的树
    
    Args:
        G: 拓扑图
        k: 生成树的数量
        n: 故障节点数量
    
    Returns:
        gp.Model: Gurobi优化模型
    """
    # 创建模型
    model = gp.Model()

    # 创建节点集合V
    V = G.nodes()
    # 创建边集合E
    E = G.edges()
    # 把E转换为双向边集合
    E = [(i, j) for i, j in E] + [(j, i) for i, j in E]

    W = G.edges(data=True)  # 带权边集合，包含边的权重信息
    bandwidths = [data['bandwidth'] for u, v, data in W]
    Wmax = max(bandwidths)  # 计算最大值

    source_node = next(iter(V))  # 假设源节点为第一个节点


    # 创建变量
    x = {}  # x[t,e_ij] 边e_ij是否在树t中
    p = {}   # p[t,i] 点i在树t中是否为非叶节点
    w = {}   # w[t] 树t分配的带宽
    r_min = model.addVar(name="r_min")  # 最小剩余带宽

    # 初始化变量
    for t in range(k):
        w[t] = model.addVar(name=f"w_{t}", lb=0.0)
        for i in V:
            p[t, i] = model.addVar(name=f"p_{t}_{i}", vtype=GRB.BINARY)
        for e in E:
            i, j = e
            x[t, i, j] = model.addVar(name=f"x_{t}_{i}_{j}", vtype=GRB.BINARY)

    # 目标函数：最大化r_min
    model.setObjective(r_min, GRB.MAXIMIZE)

    # 生成所有可能的节点故障集合
    all_S = list(itertools.combinations(V, n))


    # 剩余带宽约束
    for S in all_S:
        # r_S 表示所有满足条件的树的带宽之和
        r_S = model.addVar(name=f"r_S_{S}", lb=0.0)

        # 存储每个树t对r_S的贡献
        y_list = []
        for t in range(k):
            # z_{t,S} = 1 当且仅当S中所有节点在树t中都是叶节点（prod(1-p_{t,i})=1）
            z = model.addVar(name=f"z_{t}_{S}", vtype=GRB.BINARY)

            for i in S:
                model.addConstr(z <= 1 - p[t, i], name=f"z_ub_{t}_{S}_{i}")  # 任一i非叶则z=0
            model.addConstr(z >= gp.quicksum(1 - p[t, i] for i in S) - (len(S) - 1),
                        name=f"z_lb_{t}_{S}")  # 所有i为叶则z=1


            # y_{t,S} = w_t * z_{t,S}（树t对r_S的贡献）
            y = model.addVar(name=f"y_{t}_{S}", lb=0.0)
            model.addConstr(y <= Wmax * z, name=f"y_ub1_{t}_{S}")
            model.addConstr(y <= w[t], name=f"y_ub2_{t}_{S}")
            model.addConstr(y >= w[t] + Wmax * (z - 1), name=f"y_lb_{t}_{S}")

            y_list.append(y)

        # r_S 等于所有y的总和（即满足条件的树的带宽之和）
        model.addConstr(r_S == gp.quicksum(y_list), name=f"r_S_sum_{S}")

        # r_min 是所有r_S的最小值
        model.addConstr(r_min <= r_S, name=f"r_min_{S}")


    # 生成树约束
    for t in range(k):
        # 每个树有n-1条边
        model.addConstr(gp.quicksum(x[t, i, j] for i, j in E) == len(V) - 1,
                        name=f"tree_size_{t}")

        # 每条边只选一次
        for i, j in E:
            model.addConstr(x[t, i, j] + x[t, j, i] <= 1, name=f"edge_once_{t}_{i}_{j}")

        
        # 定义流变量：f[t,i,j]表示边(i,j)上的流量
        f = {}
        for (i, j) in E:
            f[t, i, j] = model.addVar(lb=0.0, name=f"flow_{t}_{i}_{j}")

        # 源点流出总流量 = len(V)-1（覆盖所有其他节点）
        model.addConstr(gp.quicksum(f[t, source_node, j] for j in V if (source_node, j) in E) == len(V) - 1,
                    name=f"source_flow_{t}")

        # 非源点节点：流入量 = 流出量 + 1（确保每个节点被到达）
        for i in V:
            if i != source_node:
                inflow = gp.quicksum(f[t, j, i] for j in V if (j, i) in E)
                outflow = gp.quicksum(f[t, i, j] for j in V if (i, j) in E)
                model.addConstr(inflow == outflow + 1, name=f"flow_balance_{t}_{i}")

        # 流量只能沿选中的边传递（大M约束）
        for (i, j) in E:
            model.addConstr(f[t, i, j] <= M * x[t, i, j], name=f"flow_edge_{t}_{i}_{j}")


    # 非叶节点约束
    for t in range(k):
        for i in V:
            # 非叶节点约束
            model.addConstr(gp.quicksum(x[t, i, j] + x[t, j, i] for j in V if (i,j) in E) >= 2 * p[t, i], name=f"non_leaf_lb_{t}_{i}")
            model.addConstr(gp.quicksum(x[t, i, j] + x[t, j, i] for j in V if (i,j) in E) <= M * p[t, i] + 1, name=f"non_leaf_ub_{t}_{i}")


    # 权重约束
    for e in W:
        i, j, w_ij = e
        model.addConstr(gp.quicksum(w[t] * (x[t, i, j] + x[t, j, i]) for t in range(k)) <= w_ij['bandwidth'],
                        name=f"bandwidth_constraint_{i}_{j}")

    return model


#在剩余图中找生成树
def add_more_trees(G: nx.Graph, model1: gp.Model, k: int, extra_k: int) -> gp.Model:
    """
    Args:
        G: 拓扑图
        model1: 第一阶段的优化模型
        k: 第一阶段生成的树数量
        extra_k: 第二阶段额外生成的树数量
    
    Returns:
        gp.Model: 第二阶段的优化模型
    """
    model2 = gp.Model()

    # 创建节点集合V
    V = G.nodes()
    # 创建边集合E
    E = G.edges()
    # 把E转换为双向边集合
    E = [(i, j) for i, j in E] + [(j, i) for i, j in E]

    W = G.edges(data=True)  # 带权边集合，包含边的权重信息
    
    source_node = next(iter(V))  # 假设源节点为第一个节点

    # 复制每条边的带宽（避免修改G）
    bandwidth_used = {(i, j): data['bandwidth'] for i, j, data in G.edges(data=True)}

    # 计算第一阶段树的占用
    for (i, j) in G.edges():
        total_used = 0.0
        for t in range(k):
            w_t = model1.getVarByName(f"w_{t}").x
            total_used += w_t * (
                model1.getVarByName(f"x_{t}_{i}_{j}").x +
                model1.getVarByName(f"x_{t}_{j}_{i}").x
            )
        bandwidth_used[(i, j)] -= total_used


    # 新变量 - 只为阶段二创建变量
    x2 = {}
    w2 = {}

    # 为阶段二树创建变量
    for t_idx in range(extra_k):
        t = k + t_idx  # 实际树索引
        w2[t] = model2.addVar(lb=0.0, name=f"w2_{t}")
        for e in E:
            i, j = e
            x2[t, i, j] = model2.addVar(name=f"x2_{t}_{i}_{j}", vtype=GRB.BINARY)

    # 为阶段二树添加树结构约束
    for t_idx in range(extra_k):
        t = k + t_idx  # 实际树索引
        # 树边数量约束：双向边之和为 len(V)-1
        model2.addConstr(gp.quicksum(x2[t,i,j] for (i,j) in E) == len(V)-1, 
                        name=f"tree_size2_{t}")

        # 每条边只能选一个方向（或者都不选）
        for (i,j) in E:
            model2.addConstr(x2[t,i,j] + x2[t,j,i] <= 1, name=f"edge_once2_{t}_{i}_{j}")

        # 定义流变量：f[t,i,j]表示边(i,j)上的流量
        f = {}
        for (i, j) in E:
            f[t, i, j] = model2.addVar(lb=0.0, name=f"flow2_{t}_{i}_{j}")


        # 源点流出总流量 = len(V)-1（覆盖所有其他节点）
        model2.addConstr(gp.quicksum(f[t, source_node, j] for j in V if (source_node, j) in E) == len(V) - 1,
                    name=f"source_flow2_{t}")

        # 非源点节点：流入量 = 流出量 + 1（确保每个节点被到达）
        for i in V:
            if i != source_node:
                inflow = gp.quicksum(f[t, j, i] for j in V if (j, i) in E)
                outflow = gp.quicksum(f[t, i, j] for j in V if (i, j) in E)
                model2.addConstr(inflow == outflow + 1, name=f"flow_balance2_{t}_{i}")

        # 流量只能沿选中的边传递（大M约束）
        for (i, j) in E:
            model2.addConstr(f[t, i, j] <= M * x2[t, i, j], name=f"flow_edge2_{t}_{i}_{j}")

    # 权重约束
    for e in W:
        i, j, w_ij = e
        model2.addConstr(gp.quicksum(w2[t] * (x2[t, i, j] + x2[t, j, i]) for t in range(k, k+extra_k)) <= bandwidth_used[(i, j)],
                        name=f"bandwidth_constraint_{i}_{j}")


    # 目标：最大化带宽和
    model2.setObjective(gp.quicksum(w2[t] for t in range(k, k+extra_k)), GRB.MAXIMIZE)

    return model2


def model_info_to_file(model: gp.Model, filename: str, k: int, G: nx.Graph):
    """
    输出模型信息到文件
    
    Args:
        model: Gurobi优化模型
        filename: 输出文件名
        k: 树的数量
        G: 拓扑图
    """
    with open(filename, 'w') as f:
        f.write(f"最优目标值: {model.objVal}\n")
        for t in range(k):
            f.write(f"\n树 {t} 的带宽分配: {model.getVarByName(f'w_{t}').x}")
            f.write("树中的边:\n")
            for e in G.edges():
                i, j = e
                if model.getVarByName(f"x_{t}_{i}_{j}").x > 0.5:
                    f.write(f"  边 ({i},{j})\n")
                if model.getVarByName(f"x_{t}_{j}_{i}").x > 0.5:
                    f.write(f"  边 ({j},{i})\n")
        f.write("\n===== 链路带宽利用率统计 =====\n")
        total_used_all = 0.0  # 所有链路的总占用带宽
        total_max_all = 0.0    # 所有链路的总最大可用带宽
        for (i, j) in G.edges():
            # 1. 获取链路的最大可用带宽
            max_bandwidth = G.edges[i, j]['bandwidth']
            # 2. 计算该链路被所有树占用的总带宽
            total_used = 0.0
            for t in range(k):
                w_t = model.getVarByName(f"w_{t}").x  # 树t的带宽
                total_used += w_t * (model.getVarByName(f"x_{t}_{i}_{j}").x + model.getVarByName(f"x_{t}_{j}_{i}").x)  # 累加树t对该链路的占用
            # 3. 累加至总变量
            total_used_all += total_used
            total_max_all += max_bandwidth

        # 5. 计算并输出总利用率
        total_utilization = (total_used_all / total_max_all) * 100  # 百分比
        f.write("===== 总带宽利用率统计 =====\n")
        f.write(f"所有链路的总最大带宽: {total_max_all:.2f}\n")
        f.write(f"所有链路的总占用带宽: {total_used_all:.2f}\n")
        f.write(f"总带宽利用率: {total_utilization:.2f}%\n")


def model_info_to_file_two_phase(model2: gp.Model, filename: str, k: int, extra_k: int, G: nx.Graph, model1: gp.Model, n: int = None):
    """
    输出两阶段完整的树的信息和带宽利用率
    
    Args:
        model2: 第二阶段的优化模型
        filename: 输出文件名
        k: 第一阶段生成的树数量
        extra_k: 第二阶段额外生成的树数量
        G: 拓扑图
        model1: 第一阶段的优化模型
        n: 节点故障数量
    """
    total_k = k + extra_k
    
    with open(filename, 'w') as f:
        
        # 基本参数信息
        topo_name = filename.split("_topo_")[-1].replace(".txt", "")
        f.write("【基本参数】\n")
        f.write("  拓扑名称: {}\n".format(topo_name))
        f.write("  第一阶段树数量 (k): {}\n".format(k))
        f.write("  第二阶段树数量 (extra_k): {}\n".format(extra_k))
        f.write("  节点故障数量 (n): {}\n".format(n if n is not None else 'N/A'))
        f.write("  总树数量: {}\n\n".format(total_k))
        
        # 第二阶段目标值
        f.write("【优化目标】\n")
        f.write("  阶段一目标: 最小剩余带宽最大化\n")
        if model1.status == GRB.OPTIMAL:
            f.write("  阶段一最优最小剩余带宽: {:.2f}\n".format(model1.getVarByName("r_min").x))
        f.write("  阶段二目标: 最大化第二阶段新增带宽利用率\n")
        f.write("  阶段二最优目标值: {:.2f}\n\n".format(model2.objVal))
        
        # ==================== 阶段一结果 ====================
        f.write("=" * 60 + "\n")
        f.write("           阶段一：满足故障约束的生成树\n")
        f.write("=" * 60 + "\n\n")
        
        E = [(i,j) for (i,j) in list(G.edges())] + [(j,i) for (i,j) in list(G.edges())]
        
        for t in range(k):
            f.write(f"--- 树 {t} ---\n")
            f.write(f"带宽分配: {model1.getVarByName(f'w_{t}').x:.2f}\n")
            f.write("树中的边:\n")
            for (i,j) in E:
                if model1.getVarByName(f"x_{t}_{i}_{j}").x > 0.5:
                    f.write(f"  ({i}, {j})\n")
            f.write("\n")
        
        # 阶段一带宽统计
        phase1_used = 0.0
        phase1_max = 0.0
        for (i, j) in G.edges():
            max_bw = G.edges[i, j]['bandwidth']
            used = 0.0
            for t in range(k):
                w_t = model1.getVarByName(f"w_{t}").x
                used += w_t * (model1.getVarByName(f"x_{t}_{i}_{j}").x + 
                               model1.getVarByName(f"x_{t}_{j}_{i}").x)
            phase1_used += used
            phase1_max += max_bw
        phase1_util = (phase1_used / phase1_max) * 100 if phase1_max > 0 else 0
        f.write("===== 总带宽利用率统计 =====\n")
        f.write(f"所有链路的总最大带宽: {phase1_max:.2f}\n")
        f.write(f"所有链路的总占用带宽: {phase1_used:.2f}\n")
        f.write(f"总带宽利用率: {phase1_util:.2f}%\n\n")
        
        # ==================== 阶段二结果 ====================
        f.write("=" * 60 + "\n")
        f.write("           阶段二：利用剩余带宽的生成树\n")
        f.write("=" * 60 + "\n\n")
        
        for t in range(k, total_k):
            f.write(f"--- 树 {t} (阶段二) ---\n")
            f.write("带宽分配: {:.2f}\n".format(model2.getVarByName("w2_{}".format(t)).x))
            f.write("树中的边:\n")
            for e in G.edges():
                i, j = e
                if model2.getVarByName("x2_{}_{}_{}".format(t, i, j)).x > 0.5:
                    f.write("  ({}, {})\n".format(i, j))
                if model2.getVarByName("x2_{}_{}_{}".format(t, j, i)).x > 0.5:
                    f.write("  ({}, {})\n".format(j, i))
            f.write("\n")
        
        # 阶段二后的总带宽统计
        f.write("===== 总带宽利用率统计（阶段一+阶段二） =====\n")
        total_used_all = phase1_used
        total_max_all = phase1_max
        
        # 阶段二占用
        for (i, j) in G.edges():
            max_bandwidth = G.edges[i, j]['bandwidth']
            total_used = 0.0
            for t in range(k, total_k):
                w_t = model2.getVarByName(f'w2_{t}').x
                total_used += w_t * (model2.getVarByName(f'x2_{t}_{i}_{j}').x + model2.getVarByName(f'x2_{t}_{j}_{i}').x)
            total_used_all += total_used

        total_utilization = (total_used_all / total_max_all) * 100 if total_max_all > 0 else 0
        f.write(f"所有链路的总最大带宽: {total_max_all:.2f}\n")
        f.write(f"所有链路的总占用带宽: {total_used_all:.2f}\n")
        f.write(f"总带宽利用率: {total_utilization:.2f}%\n\n")
        
        # 对比总结
        f.write("===== 优化效果总结 =====\n")
        f.write(f"仅阶段一的带宽利用率: {phase1_util:.2f}%\n")
        f.write(f"加入阶段二后的总带宽利用率: {total_utilization:.2f}%\n")
        f.write(f"提升: {total_utilization - phase1_util:.2f}%")


def run(name: str, k: int, n: int, extra_k: int = None, 
                              time_limit: int = 5400, mip_gap: float = 0.01, 
                              threads: int = 10) -> Dict:
    """
    Args:
        name: 拓扑名称
        k: 生成树数量
        n: 故障节点数量
        extra_k: 第二阶段额外生成的树数量（默认等于k）
        time_limit: 求解时间限制（秒）
        mip_gap: MIP最优性差距
        threads: 线程数
    """
    from GraphFunc import getTopoGraph
    
    if extra_k is None:
        extra_k = k  # 默认第二阶段生成与第一阶段相同数量的树
    
    G = getTopoGraph(name)  # 获取拓扑图
    E = G.edges()

    # ========== 阶段一：生成满足故障约束的树 ==========
    model = genTree(G, k, n)

    # 设置基础参数
    model.setParam('MIPGap', mip_gap)  # 1% 的最优性差距
    model.setParam('TimeLimit', time_limit)  # 90 分钟超时
    model.setParam('Threads', threads)  # 最大线程数

    # 启用回调函数检查停滞情况
    global best_obj, start_time, last_improvement_time

    best_obj = float('-inf')  # 初始化为负无穷大
    start_time = time.time()
    last_improvement_time = start_time

    def termination_callback(model, where):
        global best_obj, start_time, last_improvement_time
        if where == GRB.Callback.MIPSOL:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            # 如果当前目标值比最佳值大，则更新
            if current_obj > best_obj + 1e-6:  # 考虑浮点数精度，使用+1e-6而非直接相等
                best_obj = current_obj
                last_improvement_time = time.time()
                print(f"新最优解: {best_obj}, 耗时: {time.time()-start_time:.2f}s")
            
            # 若 30 分钟无改进，终止
            if time.time() - last_improvement_time > 1800:
                print(f"停滞时间过长（30 分钟），终止求解")
                model.terminate()

    model.optimize(termination_callback)

    # 输出阶段一结果
    if model.status == GRB.OPTIMAL:
        print(f"阶段一最优解: {model.objVal}")
        print("最小剩余带宽:", model.getVarByName("r_min").x)
        model_info_to_file(model, f"model_info_phase1_optimal_topo_{name}_n={n}_k={k}.txt", k, G)
    elif model.status in (GRB.INTERRUPTED, GRB.TIME_LIMIT):
        print(f"\n阶段一求解被中断，当前最优目标值: {model.objVal}")
        model_info_to_file(model, f"model_info_phase1_no_opt_topo_{name}_n={n}_k={k}.txt", k, G)
    else:
        print(f"阶段一模型未找到最优解。状态: {model.status}")
        return

    return model

    # # ========== 阶段二：利用剩余带宽生成更多树 ==========
    # model2 = add_more_trees(G, model, k, extra_k)

    # # 设置阶段二参数
    # model2.setParam('MIPGap', mip_gap)
    # model2.setParam('TimeLimit', time_limit)
    # model2.setParam('Threads', threads)
    
    # # 输出两阶段总体结果
    # if model2.status == GRB.OPTIMAL:
    #     print(f"阶段二最优解: {model2.objVal}")
    #     model_info_to_file_two_phase(model2, f"model_info_two_phase_topo_{name}_n={n}_k={k}_extra={extra_k}.txt", k, extra_k, G, model, n)
    # elif model2.status in (GRB.INTERRUPTED, GRB.TIME_LIMIT):
    #     print(f"阶段二求解被中断，当前最优目标值: {model2.objVal}")
    #     model_info_to_file_two_phase(model2, f"model_info_two_phase_topo_{name}_n={n}_k={k}_extra={extra_k}.txt", k, extra_k, G, model, n)
    # else:
    #     print(f"阶段二模型未找到最优解。状态: {model2.status}")
    
    # return {
    #     'phase1_model': model,
    #     'phase2_model': model2,
    #     'phase1_file': f"model_info_phase1_optimal_topo_{name}_n={n}_k={k}.txt",
    #     'phase2_file': f"model_info_two_phase_topo_{name}_n={n}_k={k}_extra={extra_k}.txt"
    # }