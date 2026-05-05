"""
Code deduplication utilities for Pass@k evaluation
Groups identical code (ignoring whitespace) to avoid redundant testing
"""
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import hashlib


def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments, whitespace, and indentation
    This allows comparison of functionally identical code with different formatting
    
    Supports both C and Verilog comment styles:
    - Single-line comments: // ...
    - Multi-line comments: /* ... */
    
    Args:
        code: Source code string
        
    Returns:
        Normalized code string with comments and whitespace removed
    """
    # Step 1: Remove multi-line comments /* ... */
    # Use non-greedy matching to handle multiple comments correctly
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Step 2: Remove single-line comments // ...
    # Match // followed by anything until end of line
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    
    # Step 3: Remove all whitespace characters (spaces, tabs, newlines, etc.)
    normalized = re.sub(r'\s+', '', code)
    
    return normalized


def compute_code_hash(code: str) -> str:
    """
    Compute a hash of normalized code for efficient comparison
    
    Args:
        code: Source code string
        
    Returns:
        SHA256 hash of normalized code
    """
    normalized = normalize_code(code)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def group_runs_by_code(runs):
    """
    Group runs by their code content (ignoring whitespace)
    
    Args:
        runs: List of (test_case, run_index, code) tuples
        
    Returns:
        Dict mapping code_hash to list of runs with that code
    """
    groups = defaultdict(list)
    
    for test_case, run_index, code in runs:
        code_hash = compute_code_hash(code)
        groups[code_hash].append((test_case, run_index, code))
    
    return dict(groups)


def select_representative_runs(groups):
    """
    Select one representative run from each group for evaluation
    
    Args:
        groups: Dict mapping code_hash to list of runs
        
    Returns:
        Tuple of:
        - List of representative runs to evaluate
        - Dict mapping (uuid, run_index) to (representative_uuid, representative_run_index)
    """
    representatives = []
    run_mapping = {}
    
    for code_hash, runs in groups.items():
        # CRITICAL FIX: Prioritize run_0 as representative for Pass@1 consistency
        # Sort runs by run_index to ensure run_0 is selected first
        sorted_runs = sorted(runs, key=lambda r: r[1])  # Sort by run_index (r[1])
        representative = sorted_runs[0]
        
        representatives.append(representative)
        
        # Map all runs in this group to the representative
        rep_test_case, rep_run_index, _ = representative
        for test_case, run_index, code in runs:
            run_mapping[(test_case.uuid, run_index)] = (rep_test_case.uuid, rep_run_index)
    
    return representatives, run_mapping


def get_deduplication_stats(groups):
    """
    Get statistics about code deduplication
    
    Args:
        groups: Dict mapping code_hash to list of runs
        
    Returns:
        Dict with deduplication statistics
    """
    total_runs = sum(len(runs) for runs in groups.values())
    unique_codes = len(groups)
    duplicates = total_runs - unique_codes
    
    # Group size distribution
    group_sizes = [len(runs) for runs in groups.values()]
    max_group_size = max(group_sizes) if group_sizes else 0
    avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
    
    return {
        'total_runs': total_runs,
        'unique_codes': unique_codes,
        'duplicate_runs': duplicates,
        'runs_to_evaluate': unique_codes,
        'max_group_size': max_group_size,
        'avg_group_size': avg_group_size,
        'deduplication_rate': duplicates / total_runs if total_runs > 0 else 0
    }