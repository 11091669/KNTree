"""
Algorithm package for tree generation, recovery, and selection.

This package provides modules for:
- Tree generation with fault tolerance
- Tree recovery after node failures
- Tree selection based on structural and bandwidth criteria
"""

from .tree_generation import genTree, add_more_trees, model_info_to_file, model_info_to_file_two_phase
from .tree_recovery import (
    LocalRecoveryWithPruning,
    extract_trees_from_model,
    batch_recovery,
    evaluate_recovery_quality,
    print_recovery_results,
    print_candidate_sets
)
from .tree_selection import (
    phase1_tree_selection, 
    phase1_tree_selection_with_pruning, 
    phase2_bandwidth_optimization, 
    full_two_phase_optimization, 
    evaluate_tree_combinations, 
    print_evaluation_results,
    check_single_node_feasibility,
    greedy_fault_check
)
from .utils import is_leaf

__version__ = "1.0.0"
__all__ = [
    # Tree generation
    'genTree',
    'add_more_trees',
    'model_info_to_file',
    'model_info_to_file_two_phase',
    
    # Tree recovery
    'LocalRecoveryWithPruning',
    'extract_trees_from_model',
    'batch_recovery',
    'evaluate_recovery_quality',
    'print_recovery_results',
    'print_candidate_sets',
    
    # Tree selection
    'phase1_tree_selection',
    'phase1_tree_selection_with_pruning',
    'phase2_bandwidth_optimization',
    'full_two_phase_optimization',
    'evaluate_tree_combinations',
    'print_evaluation_results',
    
    # Utility functions
    'is_leaf',
    'check_single_node_feasibility',
    'greedy_fault_check',
    
    # Parser function
    'parse_model_info_file',
]

def parse_model_info_file(filename: str) -> dict:
    """
    解析model_info文件，提取树结构和带宽信息
    
    Args:
        filename: model_info文件路径，例如：
                 - model_info_optimal_k=2_n=1_topo_TestTopo.txt
                 - model_info_two_phase_topo_G_Scale_n=1_k=3_extra=2.txt
    
    Returns:
        dict: 包含解析结果的字典
              {
                  'optimal_value': float,  # 最优目标值
                  'trees': [  # 树列表
                      {
                          'id': int,  # 树编号
                          'bandwidth': float,  # 带宽分配
                          'edges': list of tuples  # 边列表 [(u, v), ...]
                      },
                      ...
                  ],
                  'total_max_bandwidth': float,  # 总最大带宽
                  'total_used_bandwidth': float,  # 总占用带宽
                  'bandwidth_utilization': float,  # 带宽利用率（百分比）
                  'phase2_info': dict or None  # 阶段二信息（如果是两阶段文件）
              }
    
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件格式不正确
    """
    import re
    import os
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    result = {
        'optimal_value': 0.0,
        'trees': [],
        'total_max_bandwidth': 0.0,
        'total_used_bandwidth': 0.0,
        'bandwidth_utilization': 0.0,
        'phase2_info': None
    }
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析最优目标值
    opt_match = re.search(r'最优目标值:\s*([\d.]+)', content)
    if opt_match:
        result['optimal_value'] = float(opt_match.group(1))
    
    # 检查是否是两阶段文件
    if '阶段一' in content and '阶段二' in content:
        # 两阶段文件格式
        result['phase2_info'] = {
            'phase1_trees': [],
            'phase2_trees': [],
            'phase1_utilization': 0.0,
            'total_utilization': 0.0,
            'improvement': 0.0
        }
        
        # 解析阶段一目标值
        phase1_match = re.search(r'阶段一最优最小剩余带宽:\s*([\d.]+)', content)
        if phase1_match:
            result['phase2_info']['phase1_min_residual'] = float(phase1_match.group(1))
        
        # 解析阶段二目标值
        phase2_match = re.search(r'阶段二最优目标值:\s*([\d.]+)', content)
        if phase2_match:
            result['phase2_info']['phase2_objective'] = float(phase2_match.group(1))
        
        # 解析阶段一的树
        phase1_section = re.search(r'阶段一.*?(?=阶段二|$)', content, re.DOTALL)
        if phase1_section:
            tree_matches = re.finditer(r'--- 树 (\d+) ---\n带宽分配:\s*([\d.]+)\n树中的边:\s*((?:\s*边 \([^)]+\)\n)+)', 
                                     phase1_section.group(0))
            for match in tree_matches:
                tree_id = int(match.group(1))
                bandwidth = float(match.group(2))
                edges_text = match.group(3)
                edges = re.findall(r'边 \((\d+),\s*(\d+)\)', edges_text)
                result['phase2_info']['phase1_trees'].append({
                    'id': tree_id,
                    'bandwidth': bandwidth,
                    'edges': [(int(u), int(v)) for u, v in edges]
                })
        
        # 解析阶段二的树
        phase2_section = re.search(r'阶段二.*?(?=总带宽利用率统计|优化效果总结|$)', content, re.DOTALL)
        if phase2_section:
            tree_matches = re.finditer(r'--- 树 (\d+) \(阶段二\) ---\n带宽分配:\s*([\d.]+)\n树中的边:\s*((?:\s*\([^)]+\)\n)+)', 
                                     phase2_section.group(0))
            for match in tree_matches:
                tree_id = int(match.group(1))
                bandwidth = float(match.group(2))
                edges_text = match.group(3)
                edges = re.findall(r'\((\d+),\s*(\d+)\)', edges_text)
                result['phase2_info']['phase2_trees'].append({
                    'id': tree_id,
                    'bandwidth': bandwidth,
                    'edges': [(int(u), int(v)) for u, v in edges]
                })
        
        # 解析阶段一利用率
        phase1_util_match = re.search(r'仅阶段一的带宽利用率:\s*([\d.]+)%', content)
        if phase1_util_match:
            result['phase2_info']['phase1_utilization'] = float(phase1_util_match.group(1))
        
        # 解析总利用率
        total_util_match = re.search(r'加入阶段二后的总带宽利用率:\s*([\d.]+)%', content)
        if total_util_match:
            result['phase2_info']['total_utilization'] = float(total_util_match.group(1))
        
        # 解析提升值
        improvement_match = re.search(r'提升:\s*([\d.]+)%', content)
        if improvement_match:
            result['phase2_info']['improvement'] = float(improvement_match.group(1))
        
        # 合并所有树
        result['trees'] = result['phase2_info']['phase1_trees'] + result['phase2_info']['phase2_trees']
        
    else:
        # 单阶段文件格式
        # 解析树信息
        tree_pattern = r'树 (\d+) 的带宽分配:\s*([\d.]+)\s*树中的边:\s*((?:\s*边 \([^)]+\)\n)*)'
        tree_matches = re.finditer(tree_pattern, content)
        
        for match in tree_matches:
            tree_id = int(match.group(1))
            bandwidth = float(match.group(2))
            edges_text = match.group(3)
            
            # 解析边
            edges = re.findall(r'边 \((\d+),\s*(\d+)\)', edges_text)
            result['trees'].append({
                'id': tree_id,
                'bandwidth': bandwidth,
                'edges': [(int(u), int(v)) for u, v in edges]
            })
    
    # 解析带宽利用率统计
    max_bandwidth_match = re.search(r'所有链路的总最大带宽:\s*([\d.]+)', content)
    if max_bandwidth_match:
        result['total_max_bandwidth'] = float(max_bandwidth_match.group(1))
    
    used_bandwidth_match = re.search(r'所有链路的总占用带宽:\s*([\d.]+)', content)
    if used_bandwidth_match:
        result['total_used_bandwidth'] = float(used_bandwidth_match.group(1))
    
    utilization_match = re.search(r'总带宽利用率:\s*([\d.]+)%', content)
    if utilization_match:
        result['bandwidth_utilization'] = float(utilization_match.group(1))
    
    return result
