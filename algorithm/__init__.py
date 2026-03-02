"""
Algorithm package for tree generation, recovery, and selection.

This package provides modules for:
- Tree generation with fault tolerance
- Tree recovery after node failures
- Tree selection based on structural and bandwidth criteria
"""

from .tree_generation import genTree, add_more_trees, model_info_to_file, model_info_to_file_two_phase
from .tree_candidates import (
    LocalRecoveryWithPruning,
    evaluate_recovery_quality as evaluate_recovery_quality_candidates,
    print_recovery_results,
    print_candidate_sets
)
from .tree_recovery import (
    tree_recovery,
    tree_recovery_simple,
    evaluate_recovery_quality,
    print_recovery_summary
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
from .utils import (
    is_leaf,
    save_model_to_json,
    load_model_from_json,
    parse_tree_from_json,
    get_tree_edges,
    get_tree_bandwidth
)

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
    'evaluate_recovery_quality_candidates',
    'print_recovery_results',
    'print_candidate_sets',
    'tree_recovery',
    'tree_recovery_simple',
    'evaluate_recovery_quality',
    'print_recovery_summary',

    
    # Tree selection
    'phase1_tree_selection',
    'phase1_tree_selection_with_pruning',
    'phase2_bandwidth_optimization',
    'full_two_phase_optimization',
    'evaluate_tree_combinations',
    'print_evaluation_results',
    
    # Utility functions
    'is_leaf',
    'save_model_to_json',
    'load_model_from_json',
    'parse_tree_from_json',
    'get_tree_edges',
    'get_tree_bandwidth',
    'check_single_node_feasibility',
    'greedy_fault_check',

    # Parser function
    'parse_model_info_file',
]