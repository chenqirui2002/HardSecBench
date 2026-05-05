"""
Data loader for eval system
Loads result.json files from test data directory
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from models.eval_models import TestCase


class DataLoader:
    """Load test cases from data generation outputs"""
    
    def __init__(self):
        self.logger = logging.getLogger("eval.data_loader")
    
    def load_batchall_test_cases(self, data_path: str, filter_report_path: str = None) -> Tuple[List[TestCase], Dict[str, Dict[str, str]]]:
        """
        Load all test cases from a batchall data directory
        
        Args:
            data_path: Path to batchall output directory containing result.json
            filter_report_path: Optional path to filter_report.json for testbench quality filtering
            
        Returns:
            Tuple of (List of TestCase objects, Dict of CWE info {cwe_id: {name, description}})
        """
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Check for filter report first
        if filter_report_path:
            filter_report_file = Path(filter_report_path)
            if filter_report_file.exists():
                self.logger.info(f"Using filter report: {filter_report_file}")
                return self._load_from_filter_report(filter_report_file, data_dir)
        
        # Prefer result_filtered.json if exists, otherwise use result.json
        result_filtered = data_dir / "result_filtered.json"
        result_file = data_dir / "result.json"
        
        if result_filtered.exists():
            self.logger.info(f"Using filtered result: {result_filtered}")
            result_file = result_filtered
        elif not result_file.exists():
            raise FileNotFoundError(f"result.json not found in: {data_path}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is batchall format (has 'cwes' key)
        if "cwes" not in data:
            raise ValueError(f"Not a batchall format result.json (missing 'cwes' key)")
        
        test_cases = []
        cwe_info = {}
        cwes_data = data.get("cwes", {})
        
        for cwe_id, cwe_data in cwes_data.items():
            # Extract CWE info
            cwe_info[cwe_id] = {
                "name": cwe_data.get("cwe_name", ""),
                "description": cwe_data.get("cwe_description", "")
            }
            
            seeds = cwe_data.get("seeds", [])
            
            for seed in seeds:
                test_case = self._parse_seed_to_test_case(
                    seed, data_dir, cwe_id, is_batchall=True
                )
                if test_case:
                    test_cases.append(test_case)
        
        self.logger.info(f"Loaded {len(test_cases)} test cases from batchall {data_path}")
        return test_cases, cwe_info
    
    def _load_from_filter_report(self, filter_report_path: Path, data_dir: Path) -> Tuple[List[TestCase], Dict[str, Dict[str, str]]]:
        """
        Load test cases from filter_report.json
        
        Args:
            filter_report_path: Path to filter_report.json
            data_dir: Base data directory containing seed folders
            
        Returns:
            Tuple of (List of TestCase objects, Dict of CWE info)
        """
        with open(filter_report_path, 'r', encoding='utf-8') as f:
            filter_data = json.load(f)
        
        # Get effective seeds from filter report
        effective_seeds = {}
        for seed_detail in filter_data.get("seed_details", []):
            if seed_detail.get("is_effective", False):
                seed_uuid = seed_detail["seed_uuid"]
                effective_seeds[seed_uuid] = seed_detail
        
        self.logger.info(f"Filter report contains {len(effective_seeds)} effective seeds out of {filter_data.get('summary', {}).get('total_seeds', 0)} total")
        
        # Load the original result.json to get full seed information
        result_file = data_dir / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"result.json not found in: {data_dir}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        if "cwes" not in result_data:
            raise ValueError(f"Not a batchall format result.json (missing 'cwes' key)")
        
        test_cases = []
        cwe_info = {}
        cwes_data = result_data.get("cwes", {})
        
        for cwe_id, cwe_data in cwes_data.items():
            # Extract CWE info
            cwe_info[cwe_id] = {
                "name": cwe_data.get("cwe_name", ""),
                "description": cwe_data.get("cwe_description", "")
            }
            
            seeds = cwe_data.get("seeds", [])
            
            for seed in seeds:
                seed_uuid = seed.get("seed_uuid")
                
                # Only load seeds that are in the effective seeds list
                if seed_uuid not in effective_seeds:
                    continue
                
                test_case = self._parse_seed_to_test_case(
                    seed, data_dir, cwe_id, is_batchall=True
                )
                if test_case:
                    test_cases.append(test_case)
        
        self.logger.info(f"Loaded {len(test_cases)} effective test cases from filter report")
        return test_cases, cwe_info
    
    def _parse_seed_to_test_case(
        self, 
        seed: Dict[str, Any], 
        data_dir: Path,
        cwe_id: str,
        is_batchall: bool = False
    ) -> Optional[TestCase]:
        """
        Parse a seed dict to TestCase object
        
        Args:
            seed: Seed data dict
            data_dir: Base data directory
            cwe_id: CWE ID for this seed
            is_batchall: Whether this is batchall format (seed in cwe subdir)
            
        Returns:
            TestCase object or None if invalid
        """
        # Skip failed or incomplete seeds
        if seed.get("status") != "completed":
            self.logger.warning(f"Skipping incomplete seed: {seed.get('seed_uuid')}")
            return None
        
        # Only load seeds that passed all tests in data generation
        if not seed.get("passed", False):
            self.logger.info(f"Skipping seed (not passed): {seed.get('seed_uuid')}")
            return None
        
        # If tb_validated field exists, also check it
        # Seeds must pass both: passed=True AND tb_validated=True (if present)
        if "tb_validated" in seed and not seed.get("tb_validated", True):
            self.logger.info(f"Skipping seed (tb_validated=False): {seed.get('seed_uuid')}")
            return None
        
        # Build seed directory path
        if is_batchall:
            # Batchall: seed dir is under cwe subdir (e.g., cwe-1209/uuid)
            cwe_dir_name = f"cwe-{cwe_id.replace('CWE-', '').lower()}"
            seed_dir = data_dir / cwe_dir_name / seed["seed_uuid"]
        else:
            # Single batch: seed dir is directly under data_dir
            seed_dir = data_dir / seed["seed_uuid"]
        
        if not seed_dir.exists():
            self.logger.warning(f"Seed directory not found: {seed_dir}")
            return None
        
        return TestCase(
            uuid=seed["seed_uuid"],
            cwe_id=cwe_id,
            language=seed["language"],
            problem_description=seed.get("problem_description", ""),
            module_or_function_name=seed.get("module_or_function_name", ""),
            function_requirements=seed.get("function_requirements", []),
            security_requirements=seed.get("security_requirements", []),
            input_specification=seed.get("input_specification", ""),
            output_specification=seed.get("output_specification", ""),
            golden_file=seed.get("golden_file", ""),
            functional_test_files=seed.get("functional_test_files", {}),
            security_test_files=seed.get("security_test_files", {}),
            seed_dir=str(seed_dir)
        )
    
    def load_test_cases(self, data_path: str) -> Tuple[List[TestCase], Dict[str, Dict[str, str]]]:
        """
        Load all test cases from a data directory
        
        Args:
            data_path: Path to batch output directory containing result.json
            
        Returns:
            Tuple of (List of TestCase objects, Dict of CWE info {cwe_id: {name, description}})
        """
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Find result.json
        result_file = data_dir / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"result.json not found in: {data_path}")
        
        # Load result.json
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        cwe_info = {}
        seeds = data.get("seeds", [])
        
        # Extract CWE info from batch_info if available
        batch_info = data.get("batch_info", {})
        if batch_info:
            cwe_id = batch_info.get("cwe_id", "")
            if cwe_id:
                cwe_info[cwe_id] = {
                    "name": batch_info.get("cwe_name", ""),
                    "description": batch_info.get("cwe_description", "")
                }
        
        for seed in seeds:
            # Use shared parsing logic
            cwe_id = seed.get("cwe_id", "")
            test_case = self._parse_seed_to_test_case(seed, data_dir, cwe_id)
            if test_case:
                test_cases.append(test_case)
        
        self.logger.info(f"Loaded {len(test_cases)} test cases from {data_path}")
        return test_cases, cwe_info
    
    def load_test_file(self, seed_dir: str, filename: str) -> Optional[str]:
        """
        Load a test file content
        
        Args:
            seed_dir: Seed directory path
            filename: Test file name
            
        Returns:
            File content or None if not found
        """
        file_path = Path(seed_dir) / filename
        if not file_path.exists():
            self.logger.warning(f"Test file not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_golden_code(self, seed_dir: str, golden_file: str) -> Optional[str]:
        """
        Load golden code for reference
        
        Args:
            seed_dir: Seed directory path
            golden_file: Golden code filename
            
        Returns:
            Golden code content or None
        """
        return self.load_test_file(seed_dir, golden_file)
