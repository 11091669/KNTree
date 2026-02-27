"""
Tree recovery function for handling node failures in network topologies.

该模块提供了树恢复的主函数，结合候选树生成和优化选择，
在节点故障后找到能够容忍错误并具有最优带宽的新树组合。
"""

import networkx as nx
from typing import List, Set, Union, Dict
from .tree_candidates import LocalRecoveryWithPruning
from .tree_selection import full_two_phase_optimization


def tree_recovery(
    G: nx.Graph,
    Tau_input: List[nx.Graph],
    failed_nodes: Union[int, List[int]],
    V: Set[int] = None,
    top_k_phase1: int = 10,
    time_limit_phase2: int = 5400,
    mip_gap: float = 0.01,
    threads: int = 10
) -> Dict:
    """
    树恢复函数：在给定图、树集合和故障节点的情况下，找到能够容忍错误并具有最优带宽的新树组合。
    
    函数流程：
    1. 利用 tree_candidates (LocalRecoveryWithPruning) 生成一组候选树
    2. 将候选树输入到 tree_selection (full_two_phase_optimization) 中选出最优解
    
    Args:
        G: 网络拓扑图 (V, E)，边需包含 'bandwidth' 属性
        Tau_input: 输入树集合 {T_1, ..., T_m}
        failed_nodes: 故障节点，可以是单个节点（int）或节点列表（List[int]）
        V: 所有节点集合，如果为None则使用G.nodes()
        top_k_phase1: 阶段一选择top-k个树组合进行带宽优化，默认10
        time_limit_phase2: 阶段二求解时间限制（秒），默认5400（90分钟）
        mip_gap: MIP最优性差距，默认0.01（1%）
        threads: 求解器线程数，默认10
    
    Returns:
        Dict: 包含恢复结果的字典，包含以下键：
            - 'best_tree_set': 最优树组合 (List[nx.Graph])
            - 'best_r_min': 最小剩余带宽 (float)
            - 'best_model': Gurobi优化模型
            - 'best_index': 最优树组合在阶段一候选中的索引
            - 'all_phase1_results': 阶段一所有候选结果
            - 'failed_nodes': 故障节点列表
            - 'n_fault': 故障节点数量
            - 'candidate_sets': 候选树集合
    """
    print("=" * 70)
    print("树恢复算法 - 开始")
    print("=" * 70)
    
    # 参数标准化：统一将failed_nodes转为列表
    if isinstance(failed_nodes, int):
        failed_nodes_list = [failed_nodes]
    else:
        failed_nodes_list = failed_nodes
    
    n_fault = len(failed_nodes_list)
    
    print(f"\n输入参数:")
    print(f"  图节点数: {G.number_of_nodes()}")
    print(f"  图边数: {G.number_of_edges()}")
    print(f"  输入树数量: {len(Tau_input)}")
    print(f"  故障节点: {failed_nodes_list}")
    print(f"  故障节点数量: {n_fault}")
    
    # 确定节点集合V
    if V is None:
        V = set(G.nodes())
    else:
        V = set(V)
    
    print(f"  节点集合大小: {len(V)}")
    
    # ========== 阶段A：候选树生成 ==========
    print("\n" + "=" * 70)
    print("阶段A：候选树生成 (LocalRecoveryWithPruning)")
    print("=" * 70)
    
    CandidateSets = LocalRecoveryWithPruning(
        G=G,
        Tau_input=Tau_input,
        failed_nodes=failed_nodes_list
    )
    
    print(f"\n候选树生成完成:")
    print(f"  原始树数量: {len(Tau_input)}")
    print(f"  候选集合数量: {len(CandidateSets)}")
    
    for i, trees in enumerate(CandidateSets):
        print(f"  原始树 {i+1} 的候选树数量: {len(trees)}")
    
    # 检查是否有候选树
    if not CandidateSets or all(len(trees) == 0 for trees in CandidateSets):
        print("\n错误：没有生成任何有效的候选树！")
        return {
            'best_tree_set': None,
            'best_r_min': 0.0,
            'best_model': None,
            'best_index': -1,
            'all_phase1_results': [],
            'failed_nodes': failed_nodes_list,
            'n_fault': n_fault,
            'candidate_sets': CandidateSets,
            'error': 'No valid candidate trees generated'
        }
    
    # ========== 阶段B：两阶段优化 ==========
    print("\n" + "=" * 70)
    print("阶段B：两阶段优化 (结构可行性选树 + 带宽MILP优化)")
    print("=" * 70)
    
    try:
        optimization_result = full_two_phase_optimization(
            CandidateSets=CandidateSets,
            G=G,
            V=V,
            n_fault=n_fault,
            top_k_phase1=top_k_phase1,
            time_limit_phase2=time_limit_phase2,
            mip_gap=mip_gap,
            threads=threads
        )
        
        # ========== 汇总结果 ==========
        print("\n" + "=" * 70)
        print("树恢复算法 - 完成")
        print("=" * 70)
        
        result = {
            'best_tree_set': optimization_result['best_tree_set'],
            'best_r_min': optimization_result['best_r_min'],
            'best_model': optimization_result['best_model'],
            'best_index': optimization_result['best_index'],
            'all_phase1_results': optimization_result['all_phase1_results'],
            'failed_nodes': failed_nodes_list,
            'n_fault': n_fault,
            'candidate_sets': CandidateSets
        }
        
        print(f"\n最终结果摘要:")
        print(f"  故障节点: {failed_nodes_list}")
        print(f"  最优剩余带宽: {result['best_r_min']:.4f}")
        print(f"  最优树组合索引: {result['best_index'] + 1}/{len(result['all_phase1_results'])}")
        
        if result['best_tree_set'] is not None:
            print(f"  最优树集合包含 {len(result['best_tree_set'])} 棵树")
            for i, T in enumerate(result['best_tree_set']):
                edges = list(T.edges())
                print(f"    树 {i+1}: {T.number_of_nodes()} 个节点, {len(edges)} 条边")
                print(f"    边: {edges}")
        
        return result
        
    except Exception as e:
        print(f"\n错误：优化过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'best_tree_set': None,
            'best_r_min': 0.0,
            'best_model': None,
            'best_index': -1,
            'all_phase1_results': [],
            'failed_nodes': failed_nodes_list,
            'n_fault': n_fault,
            'candidate_sets': CandidateSets,
            'error': str(e)
        }


def tree_recovery_simple(
    G: nx.Graph,
    Tau_input: List[nx.Graph],
    failed_node: int,
    V: Set[int] = None
) -> List[nx.Graph]:
    """
    简化版树恢复函数：只执行候选树生成，返回第一个可行的树组合。
    
    Args:
        G: 网络拓扑图
        Tau_input: 输入树集合
        failed_node: 单个故障节点
        V: 所有节点集合，如果为None则使用G.nodes()
    
    Returns:
        List[nx.Graph]: 恢复后的树组合（如果成功），否则返回空列表
    """
    print("=" * 70)
    print("简化版树恢复算法 (不进行带宽优化)")
    print("=" * 70)
    
    # 确定节点集合V
    if V is None:
        V = set(G.nodes())
    else:
        V = set(V)
    
    print(f"故障节点: {failed_node}")
    
    # 生成候选树
    CandidateSets = LocalRecoveryWithPruning(
        G=G,
        Tau_input=Tau_input,
        failed_nodes=[failed_node]
    )
    
    print(f"生成的候选集合数量: {len(CandidateSets)}")
    
    # 检查是否所有树都有候选
    if len(CandidateSets) < len(Tau_input):
        print("警告：部分树无法恢复")
        return []
    
    # 返回每个候选集合的第一个树（最简单的策略）
    recovered_trees = [trees[0] for trees in CandidateSets if trees]
    
    print(f"恢复后的树数量: {len(recovered_trees)}")
    
    return recovered_trees


def evaluate_recovery_quality(
    original_trees: List[nx.Graph],
    recovered_trees: List[nx.Graph],
    G: nx.Graph,
    failed_nodes: Union[int, List[int]]
) -> Dict:
    """
    评估树恢复的质量
    
    Args:
        original_trees: 原始树集合
        recovered_trees: 恢复后的树集合
        G: 网络拓扑图
        failed_nodes: 故障节点
    
    Returns:
        Dict: 评估指标
    """
    if isinstance(failed_nodes, int):
        failed_nodes = [failed_nodes]
    
    metrics = {
        'failed_nodes': failed_nodes,
        'num_original_trees': len(original_trees),
        'num_recovered_trees': len(recovered_trees),
        'all_recovered': len(recovered_trees) == len(original_trees),
        'avg_tree_size_original': 0,
        'avg_tree_size_recovered': 0,
        'node_coverage_original': set(),
        'node_coverage_recovered': set(),
        'trees_without_failed_nodes': 0
    }
    
    # 计算原始树的统计信息
    total_size_original = 0
    for T in original_trees:
        total_size_original += T.number_of_nodes()
        metrics['node_coverage_original'].update(T.nodes())
    
    metrics['avg_tree_size_original'] = total_size_original / len(original_trees) if original_trees else 0
    
    # 计算恢复后树的统计信息
    total_size_recovered = 0
    failed_set = set(failed_nodes)
    
    for T in recovered_trees:
        total_size_recovered += T.number_of_nodes()
        metrics['node_coverage_recovered'].update(T.nodes())
        
        # 检查树是否不包含故障节点
        if not set(T.nodes()) & failed_set:
            metrics['trees_without_failed_nodes'] += 1
    
    metrics['avg_tree_size_recovered'] = total_size_recovered / len(recovered_trees) if recovered_trees else 0
    
    # 计算覆盖率
    metrics['coverage_ratio'] = (
        len(metrics['node_coverage_recovered']) / len(metrics['node_coverage_original'])
        if metrics['node_coverage_original'] else 0
    )
    
    return metrics


def print_recovery_summary(result: Dict):
    """
    打印树恢复结果的摘要
    
    Args:
        result: tree_recovery函数返回的结果字典
    """
    print("\n" + "=" * 70)
    print("树恢复结果摘要")
    print("=" * 70)
    
    if 'error' in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"\n故障信息:")
    print(f"  故障节点: {result['failed_nodes']}")
    print(f"  故障节点数量: {result['n_fault']}")
    
    print(f"\n候选树生成:")
    print(f"  候选集合数量: {len(result['candidate_sets'])}")
    for i, trees in enumerate(result['candidate_sets']):
        print(f"  树 {i+1}: {len(trees)} 个候选")
    
    print(f"\n优化结果:")
    print(f"  最优剩余带宽: {result['best_r_min']:.4f}")
    print(f"  最优组合索引: {result['best_index'] + 1}")
    
    if result['best_tree_set']:
        print(f"\n最优树集合:")
        for i, T in enumerate(result['best_tree_set']):
            print(f"  树 {i+1}:")
            print(f"    节点数: {T.number_of_nodes()}")
            print(f"    边数: {T.number_of_edges()}")
            print(f"    节点列表: {sorted(T.nodes())}")