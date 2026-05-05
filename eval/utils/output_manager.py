"""
Output manager for eval system
Handles real-time JSON output updates
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
import glob
import shutil

from models.eval_models import EvalResult


class OutputManager:
    """
    Manages eval outputs with real-time updates
    
    Directory structure:
        eval/outputs/
        └── eval_{security-hint}_{model}_{timestamp}/
            ├── eval_info.json
            ├── eval_result.json
            ├── CWE-1234/
            │   ├── {uuid1}/
            │   └── {uuid2}/
            └── CWE-5678/
                └── ...
    """
    
    def __init__(self, output_dir: Path, eval_name: str = None, security_hint: int = 0):
        """
        Initialize output manager
        
        Args:
            output_dir: Base output directory
            eval_name: Model name for this eval run
            security_hint: Security hint level (0, 1, 2)
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create eval directory: eval_{security-hint}_{model}_{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hint_str = f"hint{security_hint}"
        if eval_name:
            # Extract model name from path and sanitize for directory name
            # Use basename to get just the model name, then replace unsafe characters
            # Also need to handle paths like /root/.cache/huggingface/hub/models--Namespace--ModelName
            # where we want to extract just "Namespace-ModelName" (replace -- with -)
            from pathlib import Path as PathLib
            # Split by '/' and take the last part that looks like a model name
            # This handles both simple names and complex paths
            path_parts = eval_name.split('/')
            model_name = path_parts[-1] if path_parts else eval_name
            
            # Handle HuggingFace cache paths: models--Namespace--ModelName -> Namespace-ModelName
            if model_name.startswith('models--'):
                model_name = model_name.split('models--', 1)[-1]
                # Replace HuggingFace's -- separator with single -
                model_name = model_name.replace('--', '-')
            
            # Replace characters that are unsafe for directory names
            # Replace /, :, \ with - (but keep . for version numbers)
            model_name = model_name.replace("/", "-").replace(":", "-").replace("\\", "-")
            eval_dir_name = f"eval_{hint_str}_{model_name}_{timestamp}"
        else:
            eval_dir_name = f"eval_{hint_str}_{timestamp}"
        
        self.eval_dir = self.output_dir / eval_dir_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Result files
        self.result_file = self.eval_dir / "eval_result.json"
        self.info_file = self.eval_dir / "eval_info.json"
        
        # Thread-safe lock
        self._lock = Lock()
        
        # Logger
        self.logger = logging.getLogger("eval.output_manager")
        
        # Initialize result structure
        self._initialize_result_file()
        
        # Pass@k configuration
        self.pass_at_k = 1
        
        # Code deduplication: run_mapping for copying results to duplicates
        self.run_mapping = None
        
        # Track completed seeds for real-time Pass@k statistics
        # Key: uuid, Value: set of completed run_indices
        self._completed_runs_by_seed: Dict[str, set] = {}
        self._completed_seeds: set = set()  # Seeds with all k runs completed
    
    def _initialize_result_file(self):
        """Initialize result file with empty structure"""
        initial_data = {
            "summary": {
                # Requirement-level: average pass rates across seeds
                "avg_first_iter_functional_pass_rate": 0.0,
                "avg_first_iter_security_pass_rate": 0.0,
                "avg_final_functional_pass_rate": 0.0,
                "avg_final_security_pass_rate": 0.0,
                # Seed-level: ratio of seeds with 100% pass rate
                "seed_first_iter_functional_pass_rate": 0.0,
                "seed_first_iter_security_pass_rate": 0.0,
                "seed_final_functional_pass_rate": 0.0,
                "seed_final_security_pass_rate": 0.0,
                # Average iterations
                "avg_iterations": 0.0
            },
            "results": []
        }
        
        with self._lock:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def set_eval_info(self, **kwargs):
        """
        Set eval run information with all parameters
        
        Args:
            **kwargs: All eval parameters to save
        """
        # Extract Pass@k config
        self.pass_at_k = kwargs.get('pass_at_k', 1)
        
        eval_info = {
            "eval_dir": str(self.eval_dir),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "command_line": " ".join(sys.argv),
            "completed_cases": 0,
            "successful_cases": 0,
            **kwargs
        }
        
        with self._lock:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(eval_info, f, ensure_ascii=False, indent=2)
    
    def add_result(self, result: EvalResult):
        """
        Add or update an eval result
        
        Args:
            result: EvalResult to add/update
        """
        with self._lock:
            data = self._read_result_file()
            
            # Check if result already exists
            # For Pass@k mode, use (uuid, run_index) as unique key
            # For Pass@1 mode, use uuid only
            result_dict = result.to_dict()
            run_index = result_dict.get("run_index", 0)
            existing_idx = None
            for i, r in enumerate(data["results"]):
                # If run_index > 0, use (uuid, run_index) as key
                if run_index > 0:
                    if r["uuid"] == result.uuid and r.get("run_index", 0) == run_index:
                        existing_idx = i
                        break
                else:
                    # Pass@1 mode: use uuid only
                    if r["uuid"] == result.uuid:
                        existing_idx = i
                        break
            
            if existing_idx is not None:
                data["results"][existing_idx] = result_dict
            else:
                data["results"].append(result_dict)
            
            # Track completed runs for Pass@k real-time statistics
            if self.pass_at_k > 1 and result_dict.get("end_time"):
                uuid = result.uuid
                if uuid not in self._completed_runs_by_seed:
                    self._completed_runs_by_seed[uuid] = set()
                self._completed_runs_by_seed[uuid].add(run_index)
                
                # Check if this seed has completed all k runs
                if len(self._completed_runs_by_seed[uuid]) >= self.pass_at_k:
                    # Verify run_0 exists (critical for Pass@1 metrics)
                    if 0 in self._completed_runs_by_seed[uuid]:
                        self._completed_seeds.add(uuid)
            
            # Update summary averages
            self._update_summary(data)
            
            self._write_result_file(data)
            
            # Update eval_info stats
            self._update_eval_info_stats(data)
            
            # Return whether this seed just completed all runs
            return (self.pass_at_k > 1 and
                    result.uuid in self._completed_seeds and
                    len(self._completed_runs_by_seed.get(result.uuid, set())) == self.pass_at_k)
    
    def add_result_from_dict(self, result_dict: Dict[str, Any]):
        """
        Add or update an eval result from dictionary
        
        Args:
            result_dict: Result dictionary to add/update
        """
        with self._lock:
            data = self._read_result_file()
            
            # Check if result already exists
            # For Pass@k mode, use (uuid, run_index) as unique key
            # For Pass@1 mode, use uuid only
            run_index = result_dict.get("run_index", 0)
            existing_idx = None
            for i, r in enumerate(data["results"]):
                # If run_index > 0, use (uuid, run_index) as key
                if run_index > 0:
                    if r["uuid"] == result_dict.get("uuid") and r.get("run_index", 0) == run_index:
                        existing_idx = i
                        break
                else:
                    # Pass@1 mode: use uuid only
                    if r["uuid"] == result_dict.get("uuid"):
                        existing_idx = i
                        break
            
            if existing_idx is not None:
                data["results"][existing_idx] = result_dict
            else:
                data["results"].append(result_dict)
            
            # Track completed runs for Pass@k real-time statistics
            if self.pass_at_k > 1 and result_dict.get("end_time"):
                uuid = result_dict.get("uuid")
                if uuid not in self._completed_runs_by_seed:
                    self._completed_runs_by_seed[uuid] = set()
                self._completed_runs_by_seed[uuid].add(run_index)
                
                # Check if this seed has completed all k runs
                if len(self._completed_runs_by_seed[uuid]) >= self.pass_at_k:
                    # Verify run_0 exists (critical for Pass@1 metrics)
                    if 0 in self._completed_runs_by_seed[uuid]:
                        self._completed_seeds.add(uuid)
            
            # Update summary averages
            self._update_summary(data)
            
            self._write_result_file(data)
            
            # Update eval_info stats
            self._update_eval_info_stats(data)
            
            # Return whether this seed just completed all runs
            uuid = result_dict.get("uuid")
            return (self.pass_at_k > 1 and
                    uuid in self._completed_seeds and
                    len(self._completed_runs_by_seed.get(uuid, set())) == self.pass_at_k)
    
    def set_run_mapping(self, run_mapping: Dict[Tuple[str, int], Tuple[str, int]]):
        """
        Set run_mapping for code deduplication
        
        Args:
            run_mapping: Dict mapping (uuid, run_index) to (representative_uuid, representative_run_index)
        """
        self.run_mapping = run_mapping
    
    def copy_result_to_duplicates_immediately(self, representative_result: Dict[str, Any]):
        """
        Immediately copy a representative result to its duplicate runs
        This is called right after a representative run completes to ensure
        real-time statistics include all runs (not just representatives)
        
        Args:
            representative_result: The result dict of the representative run
        """
        if not self.run_mapping:
            return
        
        rep_uuid = representative_result.get('uuid')
        rep_run_index = representative_result.get('run_index', 0)
        rep_cwe_id = representative_result.get('cwe_id')
        
        # Find all runs that map to this representative
        duplicates_to_copy = []
        for (target_uuid, target_run_idx), (mapped_uuid, mapped_run_idx) in self.run_mapping.items():
            if mapped_uuid == rep_uuid and mapped_run_idx == rep_run_index:
                # Skip the representative itself
                if target_uuid == rep_uuid and target_run_idx == rep_run_index:
                    continue
                duplicates_to_copy.append((target_uuid, target_run_idx))
        
        # Copy result to each duplicate
        for target_uuid, target_run_idx in duplicates_to_copy:
            copied_result = representative_result.copy()
            copied_result['uuid'] = target_uuid
            copied_result['run_index'] = target_run_idx
            copied_result['copied_from'] = f"{rep_uuid}_run_{rep_run_index}"
            copied_result['copied_immediately'] = True  # Mark for debugging
            
            # Add the copied result
            self.add_result_from_dict(copied_result)
            
    
    def finalize(self):
        """Mark eval as complete"""
        with self._lock:
            # CRITICAL FIX: Copy results to duplicates BEFORE computing final statistics
            # This ensures all duplicate runs have results before statistics are computed
            if self.run_mapping:
                self.logger.info("Copying results to duplicate runs in finalize()...")
                self._copy_results_to_duplicates()
            
            data = self._read_result_file()
            # Final summary update
            self._update_summary(data)
            self._write_result_file(data)
            
            # Update eval_info end time
            if self.info_file.exists():
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                info["end_time"] = datetime.now().isoformat()
                self._update_eval_info_stats(data, info)
                with open(self.info_file, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
    
    def _copy_results_to_duplicates(self):
        """
        Copy evaluation results from representative runs to duplicate runs
        This is called in finalize() to avoid race conditions in concurrent execution
        
        Note: This method assumes it's called within a lock context
        """
        if not self.run_mapping:
            return
        
        # Read current results
        data = self._read_result_file()
        results = data.get('results', [])
        
        # Build a lookup for existing results
        result_lookup = {}
        for result in results:
            uuid = result.get('uuid')
            run_index = result.get('run_index', 0)
            result_lookup[(uuid, run_index)] = result
        
        # Copy results
        copied_count = 0
        for (target_uuid, target_run_idx), (rep_uuid, rep_run_idx) in self.run_mapping.items():
            # Skip if target already has a result (may have been copied immediately)
            if (target_uuid, target_run_idx) in result_lookup:
                continue
            
            # Skip if this is the representative itself
            if target_uuid == rep_uuid and target_run_idx == rep_run_idx:
                continue
            
            # Find representative result
            rep_result = result_lookup.get((rep_uuid, rep_run_idx))
            if not rep_result:
                self.logger.warning(f"Representative result not found for {rep_uuid} run {rep_run_idx}")
                continue
            
            # Copy result with updated uuid and run_index
            copied_result = rep_result.copy()
            copied_result['uuid'] = target_uuid
            copied_result['run_index'] = target_run_idx
            copied_result['copied_from'] = f"{rep_uuid}_run_{rep_run_idx}"
            
            # Add to results list (not calling add_result_from_dict to avoid nested locks)
            results.append(copied_result)
            result_lookup[(target_uuid, target_run_idx)] = copied_result
            copied_count += 1
            
        
        if copied_count > 0:
            self.logger.info(f"Copied {copied_count} results to duplicate runs")
            # Write updated results back
            data['results'] = results
            self._write_result_file(data)
    
    def _update_eval_info_stats(self, data: Dict[str, Any], info: Dict[str, Any] = None):
        """Update eval_info with current stats"""
        if info is None and self.info_file.exists():
            with open(self.info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
        
        if info:
            info["completed_cases"] = len([
                r for r in data["results"] if r.get("end_time") is not None
            ])
            info["successful_cases"] = len([
                r for r in data["results"] if r.get("success", False)
            ])
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
    
    def _update_summary(self, data: Dict[str, Any]):
        """
        Update summary averages based on all completed results
        
        Args:
            data: Full result data dict
        """
        results = data.get("results", [])
        # Only include results that actually ran tests (total_iterations > 0)
        # Exclude API errors or other failures that prevented test execution
        valid_results = [
            r for r in results 
            if r.get("end_time") is not None and r.get("total_iterations", 0) > 0
        ]
        
        if not valid_results:
            return
        
        n = len(valid_results)
        
        # Calculate averages
        summary = data.get("summary", {})
        
        # First iteration averages
        summary["avg_first_iter_functional_pass_rate"] = sum(
            r.get("first_iter_functional_pass_rate", 0) for r in valid_results
        ) / n
        summary["avg_first_iter_security_pass_rate"] = sum(
            r.get("first_iter_security_pass_rate", 0) for r in valid_results
        ) / n
        # Final iteration averages
        summary["avg_final_functional_pass_rate"] = sum(
            r.get("final_functional_pass_rate", 0) for r in valid_results
        ) / n
        summary["avg_final_security_pass_rate"] = sum(
            r.get("final_security_pass_rate", 0) for r in valid_results
        ) / n
        
        # Seed-level pass rates (ratio of seeds with 100% pass)
        summary["seed_first_iter_functional_pass_rate"] = sum(
            1 for r in valid_results if r.get("first_iter_functional_pass_rate", 0) == 1.0
        ) / n
        summary["seed_first_iter_security_pass_rate"] = sum(
            1 for r in valid_results if r.get("first_iter_security_pass_rate", 0) == 1.0
        ) / n
        summary["seed_final_functional_pass_rate"] = sum(
            1 for r in valid_results if r.get("final_functional_pass_rate", 0) == 1.0
        ) / n
        summary["seed_final_security_pass_rate"] = sum(
            1 for r in valid_results if r.get("final_security_pass_rate", 0) == 1.0
        ) / n
        
        # Average iterations
        summary["avg_iterations"] = sum(
            r.get("total_iterations", 0) for r in valid_results
        ) / n
        
        data["summary"] = summary
    
    def _read_result_file(self) -> Dict[str, Any]:
        """Read current result file"""
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_result_file(self, data: Dict[str, Any]):
        """Write to result file"""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_eval_dir(self) -> Path:
        """Get eval directory path"""
        return self.eval_dir
    
    def get_result_file(self) -> Path:
        """Get result file path"""
        return self.result_file
    
    def save_code(self, uuid: str, cwe_id: str, iteration: int, code: str, language: str, run_index: int = 0):
        """
        Save generated code to file
        
        Directory structure: eval_dir/CWE-xxxx/uuid/run_N/code_iter{n}.{ext}
        Always use run_N format for consistency
        
        Args:
            uuid: Test case UUID
            cwe_id: CWE ID (e.g., "CWE-1234")
            iteration: Iteration number
            code: Code content
            language: Programming language
            run_index: Run index for Pass@k mode (0-based)
        """
        # Create case directory: eval_dir/CWE-xxxx/uuid/run_N/
        # Always use run_N format for consistency
        case_dir = self.eval_dir / cwe_id / uuid / f"run_{run_index}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine extension
        ext = "v" if language == "verilog" else "c"
        
        # Save code
        code_file = case_dir / f"code_iter{iteration}.{ext}"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def save_initial_context(self, uuid: str, cwe_id: str, context: List[Any], run_index: int = 0):
        """
        Save initial LLM context (chat history) for code deduplication
        
        Directory structure: eval_dir/CWE-xxxx/uuid/run_N/initial_context.json
        Always use run_N format for consistency
        
        Args:
            uuid: Test case UUID
            cwe_id: CWE ID
            context: List of chat messages (system, user, assistant)
            run_index: Run index for Pass@k mode (0-based)
        """
        # Always use run_N format for consistency
        case_dir = self.eval_dir / cwe_id / uuid / f"run_{run_index}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert messages to serializable format
        context_data = []
        for msg in context:
            msg_dict = {
                'type': msg.__class__.__name__,
                'content': msg.content
            }
            context_data.append(msg_dict)
        
        context_file = case_dir / f"initial_context.json"
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, ensure_ascii=False, indent=2)
    
    def load_initial_context(self, uuid: str, cwe_id: str, run_index: int = 0) -> Optional[List[Dict[str, str]]]:
        """
        Load saved initial LLM context
        
        Args:
            uuid: Test case UUID
            cwe_id: CWE ID
            run_index: Run index for Pass@k mode (0-based)
            
        Returns:
            List of message dicts or None if not found
        """
        # Always use run_N format for consistency
        case_dir = self.eval_dir / cwe_id / uuid / f"run_{run_index}"
        
        context_file = case_dir / f"initial_context.json"
        if not context_file.exists():
            return None
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def compute_pass_at_k_statistics(self, k: int, log_callback=None, realtime: bool = False):
        """
        Compute Pass@k statistics from multiple runs
        
        Args:
            k: Requested k value
            log_callback: Optional callback function for logging progress
            realtime: If True, only compute statistics for completed seeds (all k runs done)
        """
        with self._lock:
            data = self._read_result_file()
            results = data.get('results', [])
            
            # Group results by UUID
            runs_by_uuid: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                uuid = result.get('uuid')
                if uuid:
                    if uuid not in runs_by_uuid:
                        runs_by_uuid[uuid] = []
                    runs_by_uuid[uuid].append(result)
            
            # For realtime avg_actual_k calculation, count ALL completed runs
            # not just seeds that finished all k runs
            total_completed_runs = 0
            total_seeds_with_runs = 0
            for uuid, runs in runs_by_uuid.items():
                valid_runs = [
                    r for r in runs
                    if r.get('end_time') and r.get('total_iterations', 0) > 0
                ]
                if valid_runs:
                    total_completed_runs += len(valid_runs)
                    total_seeds_with_runs += 1
            
            # Compute Pass@k for each seed
            pass_at_k_results = []
            
            for uuid, runs in runs_by_uuid.items():
                # In realtime mode, only process seeds that have completed all k runs
                if realtime and uuid not in self._completed_seeds:
                    continue
                
                # Filter valid runs
                valid_runs = [
                    r for r in runs
                    if r.get('end_time') and r.get('total_iterations', 0) > 0
                ]
                
                if not valid_runs:
                    continue
                
                # Sort valid runs by run_index to ensure consistent ordering
                # This is CRITICAL: we need run_0 to be first for Pass@1 metrics
                # CRITICAL FIX (Bug 25): Treat None as 0 for backward compatibility with old data
                valid_runs = sorted(valid_runs, key=lambda r: r.get('run_index') if r.get('run_index') is not None else 0)
                
                # CRITICAL FIX: Ensure run_0 exists for Pass@1 metrics
                # If run_0 is missing, skip this seed's Pass@1 statistics
                # CRITICAL FIX (Bug 25): Accept both run_index=0 and run_index=None as run_0
                # This handles old data where run_index might be None (from Pass@1 mode)
                run_0 = None
                for r in valid_runs:
                    run_idx = r.get('run_index')
                    if run_idx == 0 or run_idx is None:
                        run_0 = r
                        break
                
                if not run_0:
                    # Skip this seed if run_0 is not available
                    # This prevents using run_1 or run_2 as Pass@1 baseline
                    self.logger.warning(f"Skipping seed {uuid}: run_0 not found in valid runs")
                    continue
                
                actual_k = len(valid_runs)
                
                # Functional Pass@k: at least one run with all functional passed
                functional_pass_at_k = any(
                    r.get('success', False) for r in valid_runs
                )
                
                # Count functional-passed runs (for statistics only, not for security calculation)
                functional_passed_runs = [
                    r for r in valid_runs if r.get('success', False)
                ]
                
                # MODIFIED: Security Pass@k now considers ALL runs, not just functional-passed
                # Old logic: only consider runs where functional passed
                # New logic: check if any run has all security passed (regardless of functional)
                security_pass_at_k = any(
                    r.get('final_security_pass_rate', 0) == 1.0
                    for r in valid_runs
                )
                
                # Compute Pass@1 metrics from run_0 (guaranteed to exist after check above)
                first_run = run_0
                
                # Aggregate all runs for requirement-level stats
                all_functional_rates = [r.get('final_functional_pass_rate', 0) for r in valid_runs]
                
                # Pass@k should take the BEST rate (maximum) among all runs
                # This is because Pass@k means "at least one success in k attempts"
                best_functional_rate = max(all_functional_rates) if all_functional_rates else 0
                
                # MODIFIED: Security rates now use BEST rate among ALL runs (not just functional-passed)
                # This aligns with the new denominator logic (all security requirements)
                best_security_rate = max(
                    (r.get('final_security_pass_rate', 0) for r in valid_runs),
                    default=0
                )
                
                pass_at_k_result = {
                    'uuid': uuid,
                    'cwe_id': run_0.get('cwe_id'),
                    'language': run_0.get('language'),
                    'requested_k': k,
                    'actual_k': actual_k,
                    
                    # Pass@1 metrics (from run_0 only - guaranteed to be run_index=0)
                    'pass_at_1': {
                        'run_index': 0,  # Explicitly mark this is from run_0
                        'functional_pass': run_0.get('success', False),
                        # MODIFIED: Security pass now independent of functional pass
                        'security_pass': run_0.get('final_security_pass_rate', 0) == 1.0,
                        'first_iter_functional_pass_rate': run_0.get('first_iter_functional_pass_rate', 0),
                        'first_iter_security_pass_rate': run_0.get('first_iter_security_pass_rate', 0),
                        'final_functional_pass_rate': run_0.get('final_functional_pass_rate', 0),
                        'final_security_pass_rate': run_0.get('final_security_pass_rate', 0),
                        'total_iterations': run_0.get('total_iterations', 0)
                    },
                    
                    # Pass@k metrics
                    'pass_at_k': {
                        'functional_pass': functional_pass_at_k,
                        'security_pass': security_pass_at_k,
                        'best_functional_pass_rate': best_functional_rate,  # Changed from avg to best (max)
                        # MODIFIED: Always include best_security_pass_rate (not conditional on functional pass)
                        'best_security_pass_rate': best_security_rate,
                        'valid_runs': actual_k,
                        'functional_passed_runs': len(functional_passed_runs)
                    },
                    
                    # All runs details
                    'all_runs': valid_runs
                }
                
                pass_at_k_results.append(pass_at_k_result)
            
            # Compute summary statistics
            summary = self._compute_pass_at_k_summary(pass_at_k_results, k)
            
            # CRITICAL FIX: In realtime mode, use actual completed runs for avg_actual_k
            # not just the average of completed seeds
            if realtime and total_seeds_with_runs > 0:
                realtime_avg_actual_k = total_completed_runs / total_seeds_with_runs
            else:
                realtime_avg_actual_k = summary['avg_actual_k']
            
            # Save Pass@k results
            pass_at_k_data = {
                'pass_at_k_config': {
                    'requested_k': k,
                    'avg_actual_k': realtime_avg_actual_k,
                    'seeds_with_full_k': summary['seeds_with_full_k'],
                    'seeds_degraded': summary['seeds_degraded'],
                    'total_seeds': len(pass_at_k_results)
                },
                'summary': summary,
                'results': pass_at_k_results
            }
            
            # Write to separate Pass@k result file
            pass_at_k_file = self.eval_dir / f"eval_result_pass_at_{k}.json"
            with open(pass_at_k_file, 'w', encoding='utf-8') as f:
                json.dump(pass_at_k_data, f, ensure_ascii=False, indent=2)
            
            # Call log callback if provided
            if log_callback:
                log_callback(summary, k, len(pass_at_k_results), realtime)
    
    def _compute_pass_at_k_summary(
        self,
        pass_at_k_results: List[Dict[str, Any]],
        k: int
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for Pass@k
        """
        if not pass_at_k_results:
            return {}
        
        n = len(pass_at_k_results)
        
        # Count seeds with full k runs
        seeds_with_full_k = sum(
            1 for r in pass_at_k_results if r['actual_k'] == k
        )
        seeds_degraded = sum(
            1 for r in pass_at_k_results if 0 < r['actual_k'] < k
        )
        
        # Average actual k
        avg_actual_k = sum(r['actual_k'] for r in pass_at_k_results) / n
        
        # Pass@1 summary (from first runs - run_0 only)
        # These metrics should match exactly with Pass@1 mode (eval_all.sh)
        pass_at_1_functional = sum(
            1 for r in pass_at_k_results if r['pass_at_1']['functional_pass']
        ) / n
        
        # MODIFIED: Pass@1 security now uses ALL security requirements as denominator
        # Old logic: only count seeds where functional passed
        # New logic: count all security requirements regardless of functional pass
        total_security_reqs_pass1 = 0
        passed_security_reqs_pass1 = 0
        for r in pass_at_k_results:
            # Get security pass rate from run_0
            security_rate = r['pass_at_1'].get('final_security_pass_rate', 0)
            # Count security requirements from all_runs[0] (run_0)
            all_runs = r.get('all_runs', [])
            if all_runs:
                run_0 = all_runs[0]  # run_0 is guaranteed to be first after sorting
                iterations = run_0.get('iterations', [])
                if iterations:
                    final_iter = iterations[-1]
                    security_results = final_iter.get('security_results', [])
                    num_security_reqs = len(security_results)
                    if num_security_reqs > 0:
                        total_security_reqs_pass1 += num_security_reqs
                        passed_security_reqs_pass1 += security_rate * num_security_reqs
        
        pass_at_1_security = passed_security_reqs_pass1 / total_security_reqs_pass1 if total_security_reqs_pass1 > 0 else 0
        
        # Pass@k summary
        pass_at_k_functional = sum(
            1 for r in pass_at_k_results if r['pass_at_k']['functional_pass']
        ) / n
        
        # MODIFIED: Pass@k security now uses ALL security requirements as denominator
        # Old logic: only count seeds where functional passed
        # New logic: count all security requirements regardless of functional pass
        total_security_reqs_passk = 0
        passed_security_reqs_passk = 0
        for r in pass_at_k_results:
            # Get best security pass rate across all runs
            best_security_rate = r['pass_at_k'].get('best_security_pass_rate', 0)
            # If best_security_pass_rate is None (no functional-passed runs), treat as 0
            if best_security_rate is None:
                best_security_rate = 0
            
            # Count security requirements from any run (they should all have same number)
            all_runs = r.get('all_runs', [])
            if all_runs:
                # Use first run to count requirements
                first_run = all_runs[0]
                iterations = first_run.get('iterations', [])
                if iterations:
                    final_iter = iterations[-1]
                    security_results = final_iter.get('security_results', [])
                    num_security_reqs = len(security_results)
                    if num_security_reqs > 0:
                        total_security_reqs_passk += num_security_reqs
                        passed_security_reqs_passk += best_security_rate * num_security_reqs
        
        pass_at_k_security = passed_security_reqs_passk / total_security_reqs_passk if total_security_reqs_passk > 0 else 0
        
        # Requirement-level averages for Pass@k
        # Use best (max) functional rate, not average
        best_functional_rate = sum(
            r['pass_at_k']['best_functional_pass_rate'] for r in pass_at_k_results
        ) / n
        
        # MODIFIED: Security rate now uses ALL security requirements
        # Average of best security rates across all seeds (not just functional-passed)
        best_security_rate = passed_security_reqs_passk / total_security_reqs_passk if total_security_reqs_passk > 0 else 0
        
        return {
            'avg_actual_k': avg_actual_k,
            'seeds_with_full_k': seeds_with_full_k,
            'seeds_degraded': seeds_degraded,
            
            # Pass@1 metrics
            'pass_at_1': {
                'seed_functional_pass_rate': pass_at_1_functional,
                'seed_security_pass_rate': pass_at_1_security,
                'avg_first_iter_functional_pass_rate': sum(
                    r['pass_at_1']['first_iter_functional_pass_rate'] for r in pass_at_k_results
                ) / n,
                'avg_first_iter_security_pass_rate': sum(
                    r['pass_at_1']['first_iter_security_pass_rate'] for r in pass_at_k_results
                ) / n,
                'avg_final_functional_pass_rate': sum(
                    r['pass_at_1']['final_functional_pass_rate'] for r in pass_at_k_results
                ) / n,
                # MODIFIED: Security average now uses ALL seeds, not just functional-passed
                # This matches the new denominator logic (all security requirements)
                'avg_final_security_pass_rate': passed_security_reqs_pass1 / total_security_reqs_pass1 if total_security_reqs_pass1 > 0 else 0,
                'avg_iterations': sum(
                    r['pass_at_1']['total_iterations'] for r in pass_at_k_results
                ) / n
            },
            
            # Pass@k metrics
            'pass_at_k': {
                'seed_functional_pass_rate': pass_at_k_functional,
                'seed_security_pass_rate': pass_at_k_security,
                'avg_functional_pass_rate': best_functional_rate,  # Changed: use best instead of avg
                'avg_security_pass_rate': best_security_rate
            }
        }
