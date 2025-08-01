import time
import gurobipy as gp
from gurobipy import GRB
import itertools
import networkx as nx
from GraphFunc import getTopoGraph

M = 1000  # 大M值


# 输入一个图生成一组最大化剩余带宽的树
def genTree(G, k, n):
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

def model_info_to_file(model, filename, k, G):
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

            # # 4. 输出单条链路的利用率
            # utilization = (total_used / max_bandwidth) * 100  # 百分比
            # print(f"链路 ({i},{j}):")
            # print(f"  最大带宽: {max_bandwidth}")
            # print(f"  占用总带宽: {total_used:.2f}")
            # print(f"  利用率: {utilization:.2f}%\n")

        # 5. 计算并输出总利用率
        total_utilization = (total_used_all / total_max_all) * 100  # 百分比
        f.write("===== 总带宽利用率统计 =====\n")
        f.write(f"所有链路的总最大带宽: {total_max_all:.2f}\n")
        f.write(f"所有链路的总占用带宽: {total_used_all:.2f}\n")
        f.write(f"总带宽利用率: {total_utilization:.2f}%\n")



# 输入：
# name: 拓扑名称
# k: 生成树数量
# n: 故障节点数量
def run(name, k, n):
    G = getTopoGraph(name)  # 获取拓扑图

    # 创建并求解模型
    model = genTree(G, k, n)

    # 设置基础参数
    model.setParam('MIPGap', 0.01)  # 1% 的最优性差距
    model.setParam('TimeLimit', 5400)  # 90 分钟超时
    model.setParam('Threads', 10)  # 最大线程数


    # 启用回调函数检查停滞情况
    global best_obj, start_time, last_improvement_time

    best_obj = float('-inf')  # 初始化为负无穷大
    start_time = time.time()
    last_improvement_time = start_time

    def termination_callback(model, where):
        global best_obj, start_time, last_improvement_time
        if where == GRB.Callback.MIPSOL:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            # 修改比较逻辑：如果当前目标值比最佳值大，则更新
            if current_obj > best_obj + 1e-6:  # 考虑浮点数精度，使用+1e-6而非直接相等
                best_obj = current_obj
                last_improvement_time = time.time()
                print(f"新最优解: {best_obj}, 耗时: {time.time()-start_time:.2f}s")
            
            # 若 30 分钟无改进，终止
            if time.time() - last_improvement_time > 1800:
                print(f"停滞时间过长（30 分钟），终止求解")
                model.terminate()

    model.optimize(termination_callback)

    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"最优解: {model.objVal}")
        print("最小剩余带宽:", model.getVarByName("r_min").x)
        model_info_to_file(model, f"model_info_optimal_k={k}_n={n}_topo_{name}.txt", k, G)
    elif model.status in (GRB.INTERRUPTED, GRB.TIME_LIMIT):  # 求解被中断（超时）
        print("\n===== 求解被中断，但已找到可行解 =====")
        print(f"当前最优目标值: {model.objVal}")  # 当前最优解的目标函数值
        model_info_to_file(model, f"model_info_no_opt_k={k}_n={n}_topo_{name}.txt", k, G)
    else:
        print("模型未找到最优解。状态:", model.status)

# if __name__ == "__main__":
#     # 参数设置
#     G = getTopoGraph('Equinix')  # 获取拓扑图

#     k = 3  # 生成树数量
#     n = 1  # 故障节点数量

#     # 创建并求解模型
#     model = genTree(G,k,n)
#     model.optimize()

#     # 输出结果
#     if model.status == GRB.OPTIMAL:
#         print(f"最优解: {model.objVal}")
#         print("最小剩余带宽:", model.getVarByName("r_min").x)
#     elif model.status == GRB.INTERRUPTED:  # 求解被中断（如手动终止、超时）
#         print("\n===== 求解被中断，但已找到可行解 =====")
#         print(f"当前最优目标值: {model.objVal}")  # 当前最优解的目标函数值
#     else:
#         print("模型未找到最优解。状态:", model.status)
#         model.computeIIS()  # 计算不可行的最小约束集
#         if model.IISMinimal:
#             print("IIS is minimal.")
#         model.write("model.ilp")  # 输出模型的不可行约束子集




