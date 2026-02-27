"""
Tree selection algorithms for choosing optimal tree combinations.
"""

import networkx as nx
from typing import List, Set, Tuple, Dict
import itertools
import copy
import gurobipy as gp
from gurobipy import GRB
from .utils import is_leaf


def check_single_node_feasibility(tree_set: List[nx.Graph]) -> bool:
    """
    检查单节点可行性：树集合中的每个节点至少在一棵树上为叶子节点
    
    假设：树集合中的每棵树都包含相同的存活节点集合
    
    Args:
        tree_set: 树集合（每棵树都包含相同的存活节点）
    
    Returns:
        bool: 如果满足单节点可行性返回True，否则返回False
    """
    if not tree_set:
        return True
    
    # 获取第一棵树的所有节点（所有树应该包含相同的节点集合）
    nodes_in_trees = set(tree_set[0].nodes())
    
    for node in nodes_in_trees:
        # 检查该节点是否在至少一棵树中是叶节点
        is_leaf_in_any_tree = False
        for T in tree_set:
            if node in T.nodes() and is_leaf(T, node):
                is_leaf_in_any_tree = True
                break
        
        if not is_leaf_in_any_tree:
            return False
    
    return True


def greedy_fault_check(tree_set: List[nx.Graph], V: Set[int], n_fault: int) -> bool:
    """
    贪心算法检查：尝试选择n个节点破坏可行性
    
    Args:
        tree_set: 树集合
        V: 所有节点集合
        n_fault: 故障节点数量n
    
    Returns:
        bool: 如果能破坏可行性返回True（即找到n个非叶节点），否则返回False
    """
    remaining_nodes = set(V)
    selected_nodes = set()
    
    for _ in range(n_fault):
        if not remaining_nodes:
            break
        
        # 找到破坏最多的节点（即在最多棵树中为非叶节点的点）
        max_violations = -1
        best_node = None
        
        for node in remaining_nodes:
            # 计算该节点在多少棵树中为非叶节点
            non_leaf_count = 0
            for T in tree_set:
                if node in T.nodes() and not is_leaf(T, node):
                    non_leaf_count += 1
            
            if non_leaf_count > max_violations:
                max_violations = non_leaf_count
                best_node = node
        
        # 如果没有找到非叶节点，停止
        if max_violations == 0:
            break
        
        selected_nodes.add(best_node)
        remaining_nodes.remove(best_node)
    
    # 检查选中的节点是否都是非叶节点
    for node in selected_nodes:
        all_non_leaf = True
        for T in tree_set:
            if node in T.nodes() and is_leaf(T, node):
                all_non_leaf = False
                break
        if all_non_leaf:
            return True  # 能破坏可行性
    
    return False


def _compute_structural_bandwidth(tree_set: List[nx.Graph], node: int) -> int:
    """
    计算节点的结构性带宽：节点在多少棵树中为叶子节点（内部函数）。
    
    Args:
        tree_set: 树集合
        node: 节点
    
    Returns:
        int: 节点为叶节点的树的数量
    """
    leaf_count = 0
    for T in tree_set:
        if node in T.nodes() and is_leaf(T, node):
            leaf_count += 1
    
    return leaf_count


def _compute_link_overlap(tree_set: List[nx.Graph]) -> float:
    """
    计算树集合的链路重叠度（内部函数）。
    
    Args:
        tree_set: 树集合
    
    Returns:
        float: 链路重叠度（重叠边占总边数的比例）
    """
    if len(tree_set) < 2:
        return 0.0
    
    # 收集所有树的边
    edge_sets = []
    total_edges = 0
    for T in tree_set:
        edges = set(T.edges())
        edge_sets.append(edges)
        total_edges += len(edges)
    
    if total_edges == 0:
        return 0.0
    
    # 计算所有边对的重叠次数
    overlap_count = 0
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            overlap = len(edge_sets[i] & edge_sets[j])
            overlap_count += overlap
    
    # 归一化：重叠次数 / 总边数
    overlap_degree = (2 * overlap_count) / total_edges if total_edges > 0 else 0.0
    
    return overlap_degree


def _compute_non_leaf_frequency(tree_set: List[nx.Graph], node: int) -> int:
    """
    计算节点作为非叶节点的频率（内部函数）。
    
    Args:
        tree_set: 树集合
        node: 节点
    
    Returns:
        int: 节点为非叶节点的树的数量
    """
    non_leaf_count = 0
    for T in tree_set:
        if node in T.nodes() and not is_leaf(T, node):
            non_leaf_count += 1
    
    return non_leaf_count


def phase1_tree_selection(
    CandidateSets: List[List[nx.Graph]], 
    V: Set[int], 
    n_fault: int,
    top_k: int = 10,
    structural_bandwidth_threshold: int = None,
    link_overlap_threshold: float = None
) -> List[List[nx.Graph]]:
    """
    阶段一：结构可行性选树
    
    依次从候选集中每棵树的候选树中选一棵，进行以下判断：
    1. 检查单节点可行性：每个节点至少在一棵树上为叶子节点
    2. 贪心算法，选n个节点看能不能破坏可行性
    3. 选最有可能的topk进入第二阶段
    
    Args:
        CandidateSets: 候选树集合的列表（每个元素是一个树对应的所有候选树）
        V: 所有节点集合
        n_fault: 故障节点数量n
        top_k: 选择前k个最优的树组合
        structural_bandwidth_threshold: 结构性带宽阈值（可选）
        link_overlap_threshold: 链路重叠度阈值（可选）
    
    Returns:
        List[List[nx.Graph]]: 选择的top-k个树组合的列表
    """
    # 生成所有可能的树组合
    print(f"生成所有可能的树组合...")
    all_combinations = list(itertools.product(*CandidateSets))
    print(f"总共有 {len(all_combinations)} 种可能的组合")
    
    # 存储所有满足阶段1条件的树组合及其评分
    feasible_combinations = []
    
    # 遍历所有组合
    for idx, tree_set in enumerate(all_combinations):
        if idx % 1000 == 0:
            print(f"已检查 {idx}/{len(all_combinations)} 个组合...")
        
        # 步骤1：检查单节点可行性
        if not check_single_node_feasibility(tree_set):
            continue
        
        # 步骤2：贪心算法检查n节点故障
        can_break = greedy_fault_check(tree_set, V, n_fault)
        if can_break:
            continue  # 能破坏可行性，不满足要求
        
        # 如果通过前面的检查，计算评分指标
        # 计算结构性带宽（每个节点为叶节点的树的数量）
        structural_bandwidths = []
        for node in V:
            sb = _compute_structural_bandwidth(tree_set, node)
            structural_bandwidths.append(sb)
        
        avg_structural_bandwidth = sum(structural_bandwidths) / len(structural_bandwidths) if structural_bandwidths else 0
        
        # 计算链路重叠度
        link_overlap = _compute_link_overlap(tree_set)
        
        # 计算非叶节点频率（找出最常为非叶的节点）
        non_leaf_freqs = {}
        for node in V:
            non_leaf_freqs[node] = _compute_non_leaf_frequency(tree_set, node)
        
        # 排序找出最常为非叶的节点
        top_non_leaf_nodes = sorted(non_leaf_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # 计算评分（可以根据实际需求调整）
        # 这里我们倾向于：结构性带宽低、链路重叠度低、非叶节点频率低
        score = 0
        
        # 结构性带宽评分（越低越好）
        if structural_bandwidth_threshold is not None:
            # 如果有阈值，检查是否满足
            max_sb = max(structural_bandwidths) if structural_bandwidths else 0
            if max_sb > structural_bandwidth_threshold:
                continue  # 不满足阈值，跳过
        score += (100 - avg_structural_bandwidth) * 0.4
        
        # 链路重叠度评分（越低越好）
        if link_overlap_threshold is not None:
            if link_overlap > link_overlap_threshold:
                continue  # 不满足阈值，跳过
        score += (100 - link_overlap * 100) * 0.3
        
        # 非叶节点频率评分（越低越好）
        max_non_leaf_freq = max(non_leaf_freqs.values()) if non_leaf_freqs else 0
        score += (10 - max_non_leaf_freq) * 0.3
        
        feasible_combinations.append({
            'tree_set': tree_set,
            'score': score,
            'structural_bandwidths': structural_bandwidths,
            'link_overlap': link_overlap,
            'non_leaf_freqs': non_leaf_freqs
        })
    
    print(f"找到 {len(feasible_combinations)} 个满足阶段1条件的组合")
    
    # 按评分排序，选择top-k
    feasible_combinations.sort(key=lambda x: x['score'], reverse=True)
    selected = feasible_combinations[:top_k]
    
    print(f"\n选择了前 {len(selected)} 个最优组合：")
    for i, combo in enumerate(selected):
        print(f"组合 {i+1}:")
        print(f"  评分: {combo['score']:.2f}")
        print(f"  平均结构性带宽: {sum(combo['structural_bandwidths'])/len(combo['structural_bandwidths']):.2f}")
        print(f"  链路重叠度: {combo['link_overlap']:.4f}")
        print(f"  最常为非叶的节点: {sorted(combo['non_leaf_freqs'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # 返回选中的树组合
    return [combo['tree_set'] for combo in selected]


def phase1_tree_selection_with_pruning(
    CandidateSets: List[List[nx.Graph]], 
    V: Set[int], 
    n_fault: int,
    top_k: int = 10,
    structural_bandwidth_threshold: int = None,
    link_overlap_threshold: float = None,
    max_combinations_to_check: int = 100000
) -> List[List[nx.Graph]]:
    """
    阶段一：结构可行性选树（带剪枝优化版本）
    
    为了避免组合爆炸，使用启发式剪枝策略：
    1. 限制检查的组合数量
    2. 使用贪心策略优先检查更有希望的组合
    
    Args:
        CandidateSets: 候选树集合的列表
        V: 所有节点集合
        n_fault: 故障节点数量n
        top_k: 选择前k个最优的树组合
        structural_bandwidth_threshold: 结构性带宽阈值（可选）
        link_overlap_threshold: 链路重叠度阈值（可选）
        max_combinations_to_check: 最大检查的组合数量
    
    Returns:
        List[List[nx.Graph]]: 选择的top-k个树组合的列表
    """
    print(f"带剪枝的树选择，最大检查组合数: {max_combinations_to_check}")
    
    # 存储所有满足阶段1条件的树组合及其评分
    feasible_combinations = []
    
    # 使用随机采样或启发式策略来选择组合
    total_possible = 1
    for candidate_list in CandidateSets:
        total_possible *= len(candidate_list)
    
    print(f"总共有 {total_possible} 种可能的组合")
    
    # 如果组合数量在限制范围内，全部检查
    if total_possible <= max_combinations_to_check:
        all_combinations = list(itertools.product(*CandidateSets))
        combinations_to_check = all_combinations
    else:
        # 否则，使用随机采样
        print(f"组合数量过多，使用随机采样策略")
        import random
        combinations_to_check = []
        for _ in range(max_combinations_to_check):
            combo = []
            for candidate_list in CandidateSets:
                combo.append(random.choice(candidate_list))
            combinations_to_check.append(combo)
    
    # 遍历所有要检查的组合
    for idx, tree_set in enumerate(combinations_to_check):
        if idx % 1000 == 0:
            print(f"已检查 {idx}/{len(combinations_to_check)} 个组合...")
        
        # 步骤1：检查单节点可行性
        if not check_single_node_feasibility(tree_set):
            continue
        
        # 步骤2：贪心算法检查n节点故障
        can_break = greedy_fault_check(tree_set, V, n_fault)
        if can_break:
            continue  # 能破坏可行性，不满足要求
        
        # 如果通过前面的检查，计算评分指标
        # 计算结构性带宽
        structural_bandwidths = []
        for node in V:
            sb = _compute_structural_bandwidth(tree_set, node)
            structural_bandwidths.append(sb)
        
        avg_structural_bandwidth = sum(structural_bandwidths) / len(structural_bandwidths) if structural_bandwidths else 0
        
        # 计算链路重叠度
        link_overlap = _compute_link_overlap(tree_set)
        
        # 计算非叶节点频率
        non_leaf_freqs = {}
        for node in V:
            non_leaf_freqs[node] = _compute_non_leaf_frequency(tree_set, node)
        
        # 检查阈值约束
        if structural_bandwidth_threshold is not None:
            max_sb = max(structural_bandwidths) if structural_bandwidths else 0
            if max_sb > structural_bandwidth_threshold:
                continue
        
        if link_overlap_threshold is not None:
            if link_overlap > link_overlap_threshold:
                continue
        
        # 计算评分
        score = 0
        score += (100 - avg_structural_bandwidth) * 0.4
        score += (100 - link_overlap * 100) * 0.3
        max_non_leaf_freq = max(non_leaf_freqs.values()) if non_leaf_freqs else 0
        score += (10 - max_non_leaf_freq) * 0.3
        
        feasible_combinations.append({
            'tree_set': tree_set,
            'score': score,
            'structural_bandwidths': structural_bandwidths,
            'link_overlap': link_overlap,
            'non_leaf_freqs': non_leaf_freqs
        })
    
    print(f"找到 {len(feasible_combinations)} 个满足阶段1条件的组合")
    
    # 按评分排序，选择top-k
    feasible_combinations.sort(key=lambda x: x['score'], reverse=True)
    selected = feasible_combinations[:top_k]
    
    print(f"\n选择了前 {len(selected)} 个最优组合：")
    for i, combo in enumerate(selected):
        print(f"组合 {i+1}:")
        print(f"  评分: {combo['score']:.2f}")
        print(f"  平均结构性带宽: {sum(combo['structural_bandwidths'])/len(combo['structural_bandwidths']):.2f}")
        print(f"  链路重叠度: {combo['link_overlap']:.4f}")
        print(f"  最常为非叶的节点: {sorted(combo['non_leaf_freqs'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # 返回选中的树组合
    return [combo['tree_set'] for combo in selected]


def _is_tree_alive_after_fault(T: nx.Graph, S: Set[int]) -> bool:
    """
    判断树T在故障集合S下是否存活
    
    存活条件：故障集合S中的所有节点都为树T的叶子节点
    
    Args:
        T: 树
        S: 故障节点集合
    
    Returns:
        bool: 如果树存活返回True，否则返回False
    """
    V_T = set(T.nodes())
    
    # 如果故障集合为空，树自动存活
    if len(S) == 0:
        return True
    
    # 计算树中包含的故障节点
    fault_nodes_in_T = V_T & S
    
    # 检查所有在树中的故障节点是否都是叶节点
    for node in fault_nodes_in_T:
        if not is_leaf(T, node):
            return False
    
    return True


def phase2_bandwidth_optimization(
    tree_set: List[nx.Graph],
    G: nx.Graph,
    n_fault: int,
    time_limit: int = 5400,
    mip_gap: float = 0.01,
    threads: int = 10
) -> Tuple[gp.Model, float]:
    """
    阶段二：在固定树集下做带宽MILP优化
    
    Args:
        tree_set: 固定的树集合（从阶段一选出）
        G: 拓扑图，包含链路带宽信息
        n_fault: 故障节点数量n
        time_limit: 求解时间限制（秒），默认5400（90分钟）
        mip_gap: MIP最优性差距，默认0.01（1%）
        threads: 线程数，默认10
    
    Returns:
        Tuple[gp.Model, float]: 优化后的模型和最小剩余带宽r_min
    """
    print("=" * 60)
    print("阶段二：带宽MILP优化")
    print("=" * 60)
    
    V = G.nodes()
    E = G.edges()
    k = len(tree_set)  # 树的数量
    
    print(f"树数量: {k}")
    print(f"节点数量: {len(V)}")
    print(f"故障节点数: {n_fault}")
    
    # 创建模型
    model = gp.Model("Phase2_Bandwidth_Optimization")
    
    # 变量定义
    w = {}  # w[t] 树t分配的带宽
    for t in range(k):
        w[t] = model.addVar(lb=0.0, name=f"w_{t}")
    
    r_min = model.addVar(lb=0.0, name="r_min")
    
    # 目标函数：最大化r_min
    model.setObjective(r_min, GRB.MAXIMIZE)
    
    # 生成所有可能的n个节点的故障集合
    all_S = list(itertools.combinations(V, n_fault))
    print(f"需要考虑的故障集合数量: {len(all_S)}")
    
    # 预计算每个树在每种故障场景下的存活状态
    # survives[t, idx] = 1 当且仅当树t在故障集合all_S[idx]下存活
    survives = {}
    
    # 检查每个故障场景下是否有至少一棵树存活
    # 如果某个故障场景下所有树都不存活，则该树组合不可行，直接返回
    for idx, S in enumerate(all_S):
        S_set = set(S)
        any_survive = False
        
        for t in range(k):
            T = tree_set[t]
            alive = _is_tree_alive_after_fault(T, S_set)
            survives[t, idx] = 1 if alive else 0
            
            if alive:
                any_survive = True
        
        # 如果该故障场景下所有树都不存活，返回0（优化结果必然为0）
        if not any_survive:
            print(f"\n故障集合 {S} 下所有树都不存活，该树组合不可行")
            print(f"返回最小剩余带宽: 0.0")
            # 创建一个空模型返回
            empty_model = gp.Model("Infeasible_Tree_Set")
            return empty_model, 0.0
    
    # 剩余带宽约束
    for idx, S in enumerate(all_S):
        if idx % 1000 == 0 and len(all_S) > 1000:
            print(f"处理故障集合 {idx}/{len(all_S)}...")
        
        # r_S 表示所有在故障集合S下存活的树的带宽之和
        r_S = model.addVar(lb=0.0, name=f"r_S_{S}")
        
        # 存储每个树t对r_S的贡献
        y_list = []
        for t in range(k):
            # 如果树t在这个故障场景下不存活，则贡献为0
            if survives[t, idx] == 0:
                y = model.addVar(lb=0.0, ub=0.0, name=f"y_{t}_{S}")
            else:
                # 树t存活，贡献为w[t]
                y = model.addVar(lb=0.0, name=f"y_{t}_{S}")
                model.addConstr(y == w[t], name=f"y_eq_{t}_{S}")
            
            y_list.append(y)
        
        # r_S 等于所有y的总和（即存活的树的带宽之和）
        model.addConstr(r_S == gp.quicksum(y_list), name=f"r_S_sum_{S}")
        
        # r_min 是所有r_S的最小值
        model.addConstr(r_min <= r_S, name=f"r_min_{S}")
    
    # 链路带宽约束
    print("添加链路带宽约束...")
    for (i, j) in E:
        max_bandwidth = G.edges[i, j]['bandwidth']
        total_used = gp.quicksum(
            w[t] * (1 if (i, j) in tree_set[t].edges() or (j, i) in tree_set[t].edges() else 0)
            for t in range(k)
        )
        model.addConstr(total_used <= max_bandwidth, name=f"bandwidth_{i}_{j}")
    
    # 设置求解参数
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('Threads', threads)
    model.setParam('OutputFlag', 1)
    
    print("\n开始求解...")
    model.optimize()
    
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"\n找到最优解！")
        print(f"最小剩余带宽 r_min: {r_min.X:.4f}")
        print(f"\n各树带宽分配:")
        for t in range(k):
            print(f"  树 {t}: {w[t].X:.4f}")
    elif model.status in (GRB.INTERRUPTED, GRB.TIME_LIMIT):
        print(f"\n求解被中断或超时")
        print(f"当前最优目标值: {model.objVal:.4f}")
        print(f"当前最小剩余带宽 r_min: {r_min.X:.4f}")
        print(f"\n各树带宽分配:")
        for t in range(k):
            print(f"  树 {t}: {w[t].X:.4f}")
    else:
        print(f"\n模型求解失败，状态码: {model.status}")
    
    return model, r_min.X if model.status in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT) else 0.0


def full_two_phase_optimization(
    CandidateSets: List[List[nx.Graph]],
    G: nx.Graph,
    V: Set[int],
    n_fault: int,
    top_k_phase1: int = 10,
    time_limit_phase2: int = 5400,
    mip_gap: float = 0.01,
    threads: int = 10
) -> Dict:
    """
    完整的两阶段优化流程
    
    Args:
        CandidateSets: 候选树集合
        G: 拓扑图
        V: 节点集合
        n_fault: 故障节点数量
        top_k_phase1: 阶段一选择top-k个组合
        time_limit_phase2: 阶段二求解时间限制
        mip_gap: MIP最优性差距
        threads: 线程数
    
    Returns:
        Dict: 包含优化结果的字典
    """
    print("\n" + "=" * 60)
    print("开始两阶段优化流程")
    print("=" * 60)
    
    # 阶段一：结构可行性选树
    print("\n========== 阶段一：结构可行性选树 ==========")
    phase1_results = phase1_tree_selection(
        CandidateSets=CandidateSets,
        V=V,
        n_fault=n_fault,
        top_k=top_k_phase1
    )
    
    print(f"\n阶段一完成，选择了 {len(phase1_results)} 个树组合进入阶段二")
    
    # 阶段二：对每个组合进行带宽优化
    print("\n========== 阶段二：带宽MILP优化 ==========")
    
    best_model = None
    best_r_min = float('-inf')
    best_tree_set = None
    best_index = -1
    
    for idx, tree_set in enumerate(phase1_results):
        print(f"\n优化树组合 {idx + 1}/{len(phase1_results)}...")
        
        model, r_min = phase2_bandwidth_optimization(
            tree_set=tree_set,
            G=G,
            n_fault=n_fault,
            time_limit=time_limit_phase2,
            mip_gap=mip_gap,
            threads=threads
        )
        
        if r_min > best_r_min:
            best_r_min = r_min
            best_model = model
            best_tree_set = tree_set
            best_index = idx
    
    print("\n" + "=" * 60)
    print("两阶段优化完成")
    print("=" * 60)
    print(f"\n最优结果:")
    print(f"  最优树组合索引: {best_index + 1}")
    print(f"  最小剩余带宽: {best_r_min:.4f}")
    
    return {
        'best_model': best_model,
        'best_r_min': best_r_min,
        'best_tree_set': best_tree_set,
        'best_index': best_index,
        'all_phase1_results': phase1_results
    }


def evaluate_tree_combinations(tree_combinations: List[List[nx.Graph]], V: Set[int], n_fault: int) -> Dict:
    """
    评估多个树组合的质量
    
    Args:
        tree_combinations: 树组合列表
        V: 节点集合
        n_fault: 故障节点数量
    
    Returns:
        Dict: 评估结果
    """
    evaluation_results = {
        'total_combinations': len(tree_combinations),
        'feasible_combinations': 0,
        'average_score': 0,
        'best_score': float('-inf'),
        'worst_score': float('inf'),
        'feasibility_breakdown': []
    }
    
    scores = []
    
    for idx, tree_set in enumerate(tree_combinations):
        # 检查单节点可行性
        single_node_feasible = check_single_node_feasibility(tree_set)
        
        # 检查n节点故障可行性
        n_node_feasible = not greedy_fault_check(tree_set, V, n_fault)
        
        # 计算评分
        structural_bandwidths = [_compute_structural_bandwidth(tree_set, node) for node in V]
        avg_structural_bandwidth = sum(structural_bandwidths) / len(structural_bandwidths) if structural_bandwidths else 0
        link_overlap = _compute_link_overlap(tree_set)
        
        score = (100 - avg_structural_bandwidth) * 0.4 + (100 - link_overlap * 100) * 0.6
        
        scores.append(score)
        
        # 记录可行性
        if single_node_feasible and n_node_feasible:
            evaluation_results['feasible_combinations'] += 1
        
        evaluation_results['feasibility_breakdown'].append({
            'index': idx,
            'single_node_feasible': single_node_feasible,
            'n_node_feasible': n_node_feasible,
            'score': score,
            'structural_bandwidth': avg_structural_bandwidth,
            'link_overlap': link_overlap
        })
    
    if scores:
        evaluation_results['average_score'] = sum(scores) / len(scores)
        evaluation_results['best_score'] = max(scores)
        evaluation_results['worst_score'] = min(scores)
    
    return evaluation_results


def print_evaluation_results(evaluation_results: Dict):
    """
    打印评估结果
    
    Args:
        evaluation_results: 评估结果字典
    """
    print(f"\n树组合评估结果:")
    print(f"总组合数: {evaluation_results['total_combinations']}")
    print(f"可行组合数: {evaluation_results['feasible_combinations']}")
    print(f"平均评分: {evaluation_results['average_score']:.2f}")
    print(f"最高评分: {evaluation_results['best_score']:.2f}")
    print(f"最低评分: {evaluation_results['worst_score']:.2f}")
    
    print(f"\n可行性详情:")
    for result in evaluation_results['feasibility_breakdown']:
        status = "可行" if result['single_node_feasible'] and result['n_node_feasible'] else "不可行"
        print(f"  组合 {result['index']}: {status}, 评分: {result['score']:.2f}")