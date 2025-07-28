import gurobipy as gp
from gurobipy import GRB
import itertools
import networkx as nx

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


    # 创建变量
    x = {}  # x[t,e_ij] 边e_ij是否在树t中
    dp = {}  # dp[t,i] 点i在树t中的深度
    p = {}   # p[t,i] 点i在树t中是否为非叶节点
    w = {}   # w[t] 树t分配的带宽
    r_min = model.addVar(name="r_min")  # 最小剩余带宽

    # 初始化变量
    for t in range(k):
        w[t] = model.addVar(name=f"w_{t}", lb=0.0)
        for i in V:
            dp[t, i] = model.addVar(name=f"dp_{t}_{i}", vtype=GRB.INTEGER)
            p[t, i] = model.addVar(name=f"p_{t}_{i}", vtype=GRB.BINARY)
        for e in E:
            i, j = e
            x[t, i, j] = model.addVar(name=f"x_{t}_{i}_{j}", vtype=GRB.BINARY)

    # 目标函数：最大化r_min
    model.setObjective(r_min, GRB.MAXIMIZE)

    # 生成所有可能的节点故障集合
    all_S = list(itertools.combinations(V, n))
    print("所有可能的节点故障集合:", all_S)


    # 剩余带宽约束
    for S in all_S:
        # r_S 表示所有满足条件的树的带宽之和
        print(f"处理节点故障集合 S: {S}")
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
            Wmax = 100  # w[t]的上界，需根据实际问题调整
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

        # source = 0
        # # 定义流变量：f[t,i,j]表示边(i,j)上的流量
        # f = {}
        # for (i, j) in E:
        #     f[t, i, j] = model.addVar(lb=0.0, name=f"flow_{t}_{i}_{j}")

        # # 源点流出总流量 = len(V)-1（覆盖所有其他节点）
        # model.addConstr(gp.quicksum(f[t, source, j] for j in V if (source, j) in E) == len(V) - 1,
        #             name=f"source_flow_{t}")

        # # 非源点节点：流入量 = 流出量 + 1（确保每个节点被到达）
        # for i in V:
        #     if i != source:
        #         inflow = gp.quicksum(f[t, j, i] for j in V if (j, i) in E)
        #         outflow = gp.quicksum(f[t, i, j] for j in V if (i, j) in E)
        #         model.addConstr(inflow == outflow + 1, name=f"flow_balance_{t}_{i}")

        # # 流量只能沿选中的边传递（大M约束）
        # for (i, j) in E:
        #     model.addConstr(f[t, i, j] <= M * x[t, i, j], name=f"flow_edge_{t}_{i}_{j}")
            # 定义节点是否被选中的变量
        node = model.addVars(range(len(V)), vtype=GRB.BINARY, name=f"node_{t}")

        for i in range(len(V)):
            # 如果节点i被选中，则至少有一条入边或出边
            model.addConstr(gp.quicksum(x[t, i, j] for j in V if (i, j) in E) +
                            gp.quicksum(x[t, j, i] for j in V if (j, i) in E) >= node[i],
                            name=f"node_edge_{t}_{i}")

            # 如果节点i有边，则必须被选中
            model.addConstr(gp.quicksum(x[t, i, j] for j in V if (i, j) in E) +
                            gp.quicksum(x[t, j, i] for j in V if (j, i) in E) <= M * node[i],
                            name=f"edge_node_{t}_{i}")

        # 强制所有节点都被选中
        model.addConstr(gp.quicksum(node[i] for i in range(len(V))) == len(V), name=f"all_nodes_{t}")


    # 非叶节点约束
    for t in range(k):
        for i in V:
            # 非叶节点约束
            model.addConstr(gp.quicksum(x[t, i, j] + x[t, j, i] for j in V if (i,j) in E) >= 2 * p[t, i], name=f"non_leaf_lb_{t}_{i}")
            model.addConstr(gp.quicksum(x[t, i, j] + x[t, j, i] for j in V if (i,j) in E) <= M * p[t, i] + 1, name=f"non_leaf_ub_{t}_{i}")


    # 权重约束
    for e in W:
        i, j, w_ij = e
        model.addConstr(gp.quicksum(w[t] * (x[t, i, j] + x[t, j, i]) for t in range(k)) <= w_ij['weight'],
                        name=f"weight_constraint_{i}_{j}")

    return model

# 示例使用
if __name__ == "__main__":
    # 参数设置

    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 6), (3, 4),
        (3, 6), (4, 5), (4, 6), (4, 7), (5, 7), (6, 7)]  # 边集合

    # 创建一个无向图
    G = nx.Graph()

    # 添加边
    G.add_edges_from(E, weight=100)


    k = 2  # 生成树数量
    n = 1  # 故障节点数量

    # print("生成的图:", G.edges(data=True))

    # 创建并求解模型
    model = genTree(G,k,n)
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"最优解: {model.objVal}")
        # 输出其他感兴趣的变量值
        for t in range(k):
            print(f"\n树 {t} 的带宽分配: {model.getVarByName(f'w_{t}').x}")
            print("树中的边:")
            for e in E:
                i, j = e
                if model.getVarByName(f"x_{t}_{i}_{j}").x > 0.5:
                    print(f"  边 ({i},{j})")
                if model.getVarByName(f"x_{t}_{j}_{i}").x > 0.5:
                    print(f"  边 ({j},{i})")

        print("最小剩余带宽:", model.getVarByName("r_min").x)
    else:
        print("模型未找到最优解。状态:", model.status)
