"""
Tree recovery algorithms for handling node failures in network topologies.
"""

import networkx as nx
from typing import List, Set, Tuple, Dict
from .utils import is_leaf


def _get_connected_components(T: nx.Graph) -> List[Set[int]]:
    """
    获取树T的所有连通分量。
    """
    components = []
    for nodes in nx.connected_components(T):
        components.append(set(nodes))
    return components


def _find_component_by_node(components: List[Set[int]], node: int) -> Set[int]:
    """
    找到包含特定节点的组件。
    """
    for component in components:
        if node in component:
            return component
    return None


def _get_outgoing_edges(G: nx.Graph, T: nx.Graph, component: Set[int]) -> List[Tuple[int, int]]:
    """
    获取原图中从组件出发的所有出边（内部函数）。
    边(u, v)是出边，如果u在组件中，且v在T的其他分量中。
    """
    outgoing_edges = []
    remaining_nodes = set(T.nodes())
    
    for u in component:
        for v in G.neighbors(u):
            # v必须在T中
            # 且v不能在与u相同的分量中
            if v in remaining_nodes and v not in component:
                outgoing_edges.append((u, v))
    
    return outgoing_edges


def _has_outgoing_edge(G: nx.Graph, T: nx.Graph, component: Set[int]) -> bool:
    """
    检查组件在原图中是否至少有一条出边。
    """
    return len(_get_outgoing_edges(G, T, component)) > 0


def _rebuild_tree(T_original: nx.Graph, selected_edges: List[Tuple[int, int]], failed_nodes: int = None) -> nx.Graph:
    """
    使用原始树结构加上连接组件的选定边来重建树。
    
    Args:
        T_original: 原始树
        selected_edges: 连接组件的选定边
        failed_nodes: 故障节点列表或单个故障节点
    
    Returns:
        nx.Graph: 重建后的树
    """
    T_new = nx.Graph()
    
    # 处理failed_nodes参数
    if failed_nodes is None:
        failed_nodes = []
    elif isinstance(failed_nodes, int):
        failed_nodes = [failed_nodes]
    
    failed_set = set(failed_nodes)
    
    # 复制除故障节点外的所有节点
    for node in T_original.nodes():
        if node not in failed_set:
            T_new.add_node(node)
    
    # 复制所有不涉及故障节点的原始边
    for u, v in T_original.edges():
        if u not in failed_set and v not in failed_set:
            T_new.add_edge(u, v)
    
    # 添加连接组件的选定边
    T_new.add_edges_from(selected_edges)
    
    return T_new


def LocalRecoveryWithPruning(G: nx.Graph, Tau_input: List[nx.Graph], failed_nodes: List[int]) -> List[List[nx.Graph]]:
    """
    输入：
        G = (V, E) - 原始图
        Tau_input = {T_1, ..., T_m} - 输入树集合
        failed_nodes [f_1, ..., f_k] - 故障节点列表
    
    输出：
        CandidateSets - Tau_output候选树集合集合（候选树集合的列表）
    
    该算法在节点故障后尝试恢复树。
    对于故障节点是叶节点的树，可以直接使用（移除故障节点）。
    对于损坏的树，搜索所有可能的组件重连方式。
    """
    CandidateSets = []
    BrokenTrees = []
    
    # 将树分为故障节点是叶节点的树和损坏的树
    for T in Tau_input:
        # 获取树中实际存在的故障节点
        failed_in_tree = [f for f in failed_nodes if f in T.nodes()]
        
        if not failed_in_tree:
            # 没有故障节点在树中，可以直接使用
            CandidateSets.append([T])
        elif all(is_leaf(T, f) for f in failed_in_tree):
            # 所有故障节点都是叶节点，直接移除即可
            T_copy = T.copy()
            for f in failed_in_tree:
                if f in T_copy.nodes():
                    T_copy.remove_node(f)
            CandidateSets.append([T_copy])
        else:
            # 至少有一个故障节点不是叶节点，需要重连组件
            # 先移除所有叶节点故障，然后处理剩余的非叶节点故障
            T_copy = T.copy()
            leaf_failures = [f for f in failed_in_tree if is_leaf(T, f)]
            non_leaf_failures = [f for f in failed_in_tree if not is_leaf(T, f)]
            
            # 移除叶节点故障
            for f in leaf_failures:
                if f in T_copy.nodes():
                    T_copy.remove_node(f)
            
            # 保存所有故障节点信息，用于重建树
            T_copy.failed_nodes_to_remove = non_leaf_failures
            T_copy.leaf_failures_removed = leaf_failures
            BrokenTrees.append(T_copy)
    
    # 处理每个损坏的树
    for T in BrokenTrees:
        # 获取需要移除的故障节点
        if hasattr(T, 'failed_nodes_to_remove'):
            # 如果树已经被预处理过（移除了叶节点故障）
            failed_to_remove = T.failed_nodes_to_remove
            # 获取原始树中的所有故障节点（包括叶节点和非叶节点）
            all_failed_in_tree = T.failed_nodes_to_remove + getattr(T, 'leaf_failures_removed', [])
            current_tree = T
        else:
            # 普通情况，移除所有故障节点
            failed_to_remove = [f for f in failed_nodes if f in T.nodes()]
            all_failed_in_tree = failed_to_remove
            current_tree = T
        
        # 从树中移除非叶节点故障（叶节点已经被预先移除）
        T_without_failed = current_tree.copy()
        for f in failed_to_remove:
            if f in T_without_failed.nodes():
                T_without_failed.remove_node(f)
        
        # 获取移除后的连通分量
        Components = _get_connected_components(T_without_failed)
        
        # 剪枝：如果任何组件没有出边，此树不可行
        is_feasible = True
        for Ci in Components:
            if not _has_outgoing_edge(G, T_without_failed, Ci):
                is_feasible = False
                break
        
        if not is_feasible:
            # 跳过此树（被剪枝）
            continue
        
        # 存储此损坏树的候选树
        CandidateTrees_T = []
        
        # 定义递归搜索函数
        def search(Components_set: List[Set[int]], E_sel: List[Tuple[int, int]]):
            """
            递归搜索，找到所有连接组件的可能方式。
            
            输入：
                Components_set - 连通分量列表
                E_sel - 目前已选定的边列表
            """
            # 只剩下一个分量，树已完成
            if len(Components_set) == 1:
                # 重建树，移除所有故障节点（包括叶节点和非叶节点）
                T_new = _rebuild_tree(current_tree, E_sel, all_failed_in_tree)
                CandidateTrees_T.append(T_new)
                return
            
            # 选择最小的分量（启发式策略）
            Ci = min(Components_set, key=len)
            
            # 尝试从此分量出发的所有出边
            outgoing = _get_outgoing_edges(G, T_without_failed, Ci)
            
            for e in outgoing:
                u, v = e
                
                # 找出v所属的分量（如果v在剩余树中）
                Cv = _find_component_by_node(Components_set, v)
                
                if Cv is None:
                    # v不在任何分量中（可能在树外或是故障节点）
                    # 跳过这条边
                    continue
                
                # 合并分量Ci和Cv
                C_new = Ci.union(Cv)
                
                # 创建新的分量集合：移除Ci和Cv，添加C_new
                new_components = [comp for comp in Components_set if comp != Ci and comp != Cv]
                new_components.append(C_new)
                
                # 递归搜索，使用合并后的分量和更新的边集合
                new_E_sel = E_sel + [(u, v)]
                search(new_components, new_E_sel)
        
        # 开始递归搜索
        search(Components, [])
        
        # 将此损坏树的所有候选树添加到结果中
        if CandidateTrees_T:
            CandidateSets.append(CandidateTrees_T)
    
    return CandidateSets


def extract_trees_from_model(model, k: int, G: nx.Graph) -> List[nx.Graph]:
    """
    从优化模型中提取树结构
    
    Args:
        model: Gurobi优化模型
        k: 树的数量
        G: 拓扑图
    
    Returns:
        List[nx.Graph]: 提取的树列表
    """
    T_output = []
    
    for t in range(k):
        # 创建新树
        T = nx.Graph()
        
        # 获取图中所有节点
        V = G.nodes()
        T.add_nodes_from(V)
        
        # 遍历图中所有边，检查哪些边在树t中
        for i, j in G.edges():
            # 检查边 (i, j)
            var_ij = model.getVarByName(f"x_{t}_{i}_{j}")
            # 检查边 (j, i)
            var_ji = model.getVarByName(f"x_{t}_{j}_{i}")
            
            # 如果任一方向的边在树中，添加该边
            if (var_ij and var_ij.X > 0.5) or (var_ji and var_ji.X > 0.5):
                T.add_edge(i, j)
        
        T_output.append(T)
    
    return T_output


def evaluate_recovery_quality(candidate_sets: List[List[nx.Graph]], failed_node: int) -> Dict:
    """
    评估恢复候选树的质量
    
    Args:
        candidate_sets: 候选树集合列表
        failed_node: 故障节点
    
    Returns:
        Dict: 评估指标
    """
    quality_metrics = {
        'failed_node': failed_node,
        'total_candidates': 0,
        'feasible_candidates': 0,
        'avg_tree_size': 0,
        'connectivity_scores': []
    }
    
    total_trees = 0
    total_size = 0
    
    for candidate_set in candidate_sets:
        quality_metrics['total_candidates'] += len(candidate_set)
        
        for tree in candidate_set:
            # 检查连通性
            if nx.is_connected(tree):
                quality_metrics['feasible_candidates'] += 1
                quality_metrics['connectivity_scores'].append(1.0)
            else:
                quality_metrics['connectivity_scores'].append(0.0)
            
            total_trees += 1
            total_size += tree.number_of_nodes()
    
    if total_trees > 0:
        quality_metrics['avg_tree_size'] = total_size / total_trees
    
    return quality_metrics


def print_recovery_results(candidate_sets: List[List[nx.Graph]], failed_node: int):
    """
    打印恢复结果
    
    Args:
        candidate_sets: 候选树集合列表
        failed_node: 故障节点
    """
    print(f"\n故障节点 {failed_node} 的恢复结果:")
    print(f"候选集合总数: {len(candidate_sets)}")
    
    for i, trees in enumerate(candidate_sets):
        print(f"\n候选集合 {i+1}:")
        print(f"  树的数量: {len(trees)}")
        for j, T in enumerate(trees):
            print(f"  树 {j+1}: {T.number_of_nodes()} 个节点, {T.number_of_edges()} 条边")
            print(f"    节点: {sorted(T.nodes())}")
            print(f"    边: {sorted(T.edges())}")
    
    # 评估质量
    quality = evaluate_recovery_quality(candidate_sets, failed_node)
    print(f"\n质量评估:")
    print(f"  总候选数: {quality['total_candidates']}")
    print(f"  可行候选数: {quality['feasible_candidates']}")
    print(f"  平均树大小: {quality['avg_tree_size']:.2f}")
    print(f"  连通性得分: {sum(quality['connectivity_scores'])/len(quality['connectivity_scores']):.2f}")


def print_candidate_sets(CandidateSets: List[List[nx.Graph]]):
    """
    打印候选集合信息，用于调试。
    """
    print(f"找到的候选集合总数: {len(CandidateSets)}")
    for i, trees in enumerate(CandidateSets):
        print(f"\n候选集合 {i+1}:")
        print(f"  树的数量: {len(trees)}")
        for j, T in enumerate(trees):
            print(f"  树 {j+1}: {T.number_of_nodes()} 个节点, {T.number_of_edges()} 条边")
            print(f"    节点: {sorted(T.nodes())}")
            print(f"    边: {sorted(T.edges())}")