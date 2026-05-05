"""
Batch output manager for handling batch generation outputs with real-time updates
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock

from config.settings import config


class BatchOutputManager:
    """
    Manages batch generation outputs with:
    - Timestamped batch directories
    - UUID-based seed directories
    - Real-time result.json updates
    """
    
    def __init__(self, base_output_dir: Path, batch_name: str = None):
        """
        Initialize batch output manager
        
        Args:
            base_output_dir: Base directory for all outputs (e.g., data_generation/outputs)
            batch_name: Optional batch name prefix (e.g., CWE-1234)
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch directory with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.LLM_MODEL.replace("/", "-")
        if batch_name:
            batch_dir_name = f"batch_{batch_name}_{timestamp}_{model_name}"
        else:
            batch_dir_name = f"batch_{timestamp}_{model_name}"
        
        self.batch_dir = self.base_output_dir / batch_dir_name
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Result file path
        self.result_file = self.batch_dir / "result.json"
        
        # Thread-safe lock for updating result.json
        self._lock = Lock()
        
        # Initialize empty result structure
        self._initialize_result_file()
    
    def _initialize_result_file(self):
        """Initialize result.json with empty structure"""
        initial_data = {
            "batch_info": {
                "batch_dir": str(self.batch_dir),
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_seeds": 0,
                "completed_seeds": 0,
                "successful_seeds": 0
            },
            "seeds": []
        }
        
        with self._lock:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def create_seed_directory(self, seed_index: int = None) -> tuple[str, Path]:
        """
        Create a new seed directory with UUID
        
        Args:
            seed_index: Optional seed index for tracking
            
        Returns:
            Tuple of (seed_uuid, seed_directory_path)
        """
        seed_uuid = str(uuid.uuid4())
        seed_dir = self.batch_dir / seed_uuid
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        return seed_uuid, seed_dir
    
    def add_seed_entry(
        self,
        seed_uuid: str,
        cwe_id: str,
        task_id: str,
        problem_description: str = None,
        language: str = None,
        function_requirements: List[str] = None,
        security_requirements: List[str] = None,
        input_specification: str = None,
        output_specification: str = None,
        module_or_function_name: str = None
    ):
        """
        Add a new seed entry to result.json
        
        Args:
            seed_uuid: Unique seed identifier
            cwe_id: CWE identifier
            task_id: Task identifier
            problem_description: Problem description text
            language: Programming language (verilog/c)
            function_requirements: List of functional requirements
            security_requirements: List of security requirements
            input_specification: Input specification
            output_specification: Output specification
            module_or_function_name: Module or function name
        """
        seed_entry = {
            "seed_uuid": seed_uuid,
            "cwe_id": cwe_id,
            "task_id": task_id,
            "problem_description": problem_description or "",
            "language": language or "unknown",
            "module_or_function_name": module_or_function_name or "",
            "function_requirements": function_requirements or [],
            "security_requirements": security_requirements or [],
            "input_specification": input_specification or "",
            "output_specification": output_specification or "",
            "status": "in_progress",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "iterations": {},  # Dict: {iteration_num: [issue_types]} e.g. {"1": ["tb_issue"], "2": ["PASS"]}
            "passed": False,
            "golden_file": None,
            "functional_test_files": {},
            "security_test_files": {},
            "functional_pass_rate": 0.0,
            "security_pass_rate": 0.0,
            "error": None
        }
        
        with self._lock:
            data = self._read_result_file()
            data["seeds"].append(seed_entry)
            data["batch_info"]["total_seeds"] = len(data["seeds"])
            self._write_result_file(data)
    
    def update_seed_result(
        self,
        seed_uuid: str,
        iterations: dict = None,
        passed: bool = None,
        golden_file: str = None,
        functional_test_files: Dict[str, str] = None,
        security_test_files: Dict[str, str] = None,
        functional_pass_rate: float = None,
        security_pass_rate: float = None,
        error: str = None,
        status: str = None,
        problem_description: str = None,
        function_requirements: list = None,
        security_requirements: list = None,
        input_specification: str = None,
        output_specification: str = None
    ):
        """
        Update seed result in result.json
        
        Args:
            seed_uuid: Unique seed identifier
            iterations: Number of iterations
            passed: Whether all tests passed
            golden_file: Path to golden code file
            functional_test_files: Dict of {filename: explanation} for functional tests
            security_test_files: Dict of {filename: explanation} for security tests
            functional_pass_rate: Functional test pass rate (0.0-1.0)
            security_pass_rate: Security test pass rate (0.0-1.0)
            error: Error message if failed
            status: Status update ("in_progress", "completed", "failed")
            problem_description: Updated problem description (if revised by Architect)
            function_requirements: Updated functional requirements (if revised)
            security_requirements: Updated security requirements (if revised)
            input_specification: Updated input specification (if revised)
            output_specification: Updated output specification (if revised)
        """
        with self._lock:
            data = self._read_result_file()
            
            # Find seed entry
            seed_entry = None
            for seed in data["seeds"]:
                if seed["seed_uuid"] == seed_uuid:
                    seed_entry = seed
                    break
            
            if not seed_entry:
                raise ValueError(f"Seed {seed_uuid} not found in result.json")
            
            # Update fields
            if iterations is not None:
                seed_entry["iterations"] = iterations
            if passed is not None:
                seed_entry["passed"] = passed
            if golden_file is not None:
                seed_entry["golden_file"] = golden_file
            if functional_test_files is not None:
                seed_entry["functional_test_files"] = functional_test_files
            if security_test_files is not None:
                seed_entry["security_test_files"] = security_test_files
            if functional_pass_rate is not None:
                seed_entry["functional_pass_rate"] = functional_pass_rate
            if security_pass_rate is not None:
                seed_entry["security_pass_rate"] = security_pass_rate
            if error is not None:
                seed_entry["error"] = error
            if status is not None:
                seed_entry["status"] = status
                if status in ["completed", "failed"]:
                    seed_entry["end_time"] = datetime.now().isoformat()
            
            # Update problem fields if revised by Architect
            if problem_description is not None:
                seed_entry["problem_description"] = problem_description
            if function_requirements is not None:
                seed_entry["function_requirements"] = function_requirements
            if security_requirements is not None:
                seed_entry["security_requirements"] = security_requirements
            if input_specification is not None:
                seed_entry["input_specification"] = input_specification
            if output_specification is not None:
                seed_entry["output_specification"] = output_specification
            
            # Update batch statistics
            data["batch_info"]["completed_seeds"] = sum(
                1 for s in data["seeds"] if s["status"] in ["completed", "failed"]
            )
            data["batch_info"]["successful_seeds"] = sum(
                1 for s in data["seeds"] if s["passed"]
            )
            
            self._write_result_file(data)
    
    def finalize_batch(self):
        """Mark batch as complete"""
        with self._lock:
            data = self._read_result_file()
            data["batch_info"]["end_time"] = datetime.now().isoformat()
            self._write_result_file(data)
    
    def _read_result_file(self) -> Dict[str, Any]:
        """Read current result.json"""
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_result_file(self, data: Dict[str, Any]):
        """Write to result.json"""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_seed_dir(self, seed_uuid: str) -> Path:
        """Get seed directory path"""
        return self.batch_dir / seed_uuid
    
    def get_batch_dir(self) -> Path:
        """Get batch directory path"""
        return self.batch_dir
    
    def get_result_file_path(self) -> Path:
        """Get result.json path"""
        return self.result_file


class BatchAllOutputManager:
    """
    Manages batchall generation outputs with:
    - Timestamped batchall directory
    - Per-CWE subdirectories
    - Aggregated result.json
    """
    
    def __init__(self, base_output_dir: Path):
        """
        Initialize batchall output manager
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batchall directory with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.LLM_MODEL.replace("/", "-")
        batchall_dir_name = f"batchall_{timestamp}_{model_name}"
        
        self.batchall_dir = self.base_output_dir / batchall_dir_name
        self.batchall_dir.mkdir(parents=True, exist_ok=True)
        
        # Result file path
        self.result_file = self.batchall_dir / "result.json"
        
        # Thread-safe lock for updating result.json
        self._lock = Lock()
        
        # Initialize empty result structure
        self._initialize_result_file()
    
    def _initialize_result_file(self):
        """Initialize result.json with empty structure"""
        initial_data = {
            "batchall_info": {
                "batchall_dir": str(self.batchall_dir),
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_cwes": 0,
                "completed_cwes": 0,
                "successful_cwes": 0,
                "total_seeds": 0,
                "completed_seeds": 0,
                "successful_seeds": 0
            },
            "cwes": {}
        }
        
        with self._lock:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def create_cwe_directory(self, cwe_id: str) -> Path:
        """
        Create a directory for a specific CWE
        
        Args:
            cwe_id: CWE identifier (e.g., CWE-1243)
            
        Returns:
            Path to CWE directory
        """
        cwe_dir = self.batchall_dir / cwe_id.lower()
        cwe_dir.mkdir(parents=True, exist_ok=True)
        return cwe_dir
    
    def add_cwe_entry(self, cwe_id: str, cwe_name: str, total_seeds: int):
        """
        Add a new CWE entry to result.json
        
        Args:
            cwe_id: CWE identifier
            cwe_name: CWE name
            total_seeds: Total number of seeds for this CWE
        """
        cwe_entry = {
            "cwe_id": cwe_id,
            "cwe_name": cwe_name,
            "status": "in_progress",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_seeds": total_seeds,
            "completed_seeds": 0,
            "successful_seeds": 0,
            "seeds": []
        }
        
        with self._lock:
            data = self._read_result_file()
            data["cwes"][cwe_id] = cwe_entry
            data["batchall_info"]["total_cwes"] = len(data["cwes"])
            data["batchall_info"]["total_seeds"] += total_seeds
            self._write_result_file(data)
    
    def add_seed_to_cwe(
        self,
        cwe_id: str,
        seed_uuid: str,
        task_id: str,
        problem_description: str = None,
        language: str = None,
        function_requirements: List[str] = None,
        security_requirements: List[str] = None,
        input_specification: str = None,
        output_specification: str = None,
        module_or_function_name: str = None
    ):
        """Add a seed entry to a CWE"""
        seed_entry = {
            "seed_uuid": seed_uuid,
            "task_id": task_id,
            "problem_description": problem_description or "",
            "language": language or "unknown",
            "module_or_function_name": module_or_function_name or "",
            "function_requirements": function_requirements or [],
            "security_requirements": security_requirements or [],
            "input_specification": input_specification or "",
            "output_specification": output_specification or "",
            "status": "in_progress",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "iterations": {},
            "passed": False,
            "golden_file": None,
            "functional_test_files": {},
            "security_test_files": {},
            "functional_pass_rate": 0.0,
            "security_pass_rate": 0.0,
            "error": None
        }
        
        with self._lock:
            data = self._read_result_file()
            if cwe_id in data["cwes"]:
                data["cwes"][cwe_id]["seeds"].append(seed_entry)
            self._write_result_file(data)
    
    def update_seed_result(
        self,
        cwe_id: str,
        seed_uuid: str,
        iterations: dict = None,
        passed: bool = None,
        golden_file: str = None,
        functional_test_files: Dict[str, str] = None,
        security_test_files: Dict[str, str] = None,
        functional_pass_rate: float = None,
        security_pass_rate: float = None,
        error: str = None,
        status: str = None,
        problem_description: str = None,
        function_requirements: list = None,
        security_requirements: list = None,
        input_specification: str = None,
        output_specification: str = None
    ):
        """Update seed result in result.json"""
        with self._lock:
            data = self._read_result_file()
            
            if cwe_id not in data["cwes"]:
                return
            
            # Find seed entry
            seed_entry = None
            for seed in data["cwes"][cwe_id]["seeds"]:
                if seed["seed_uuid"] == seed_uuid:
                    seed_entry = seed
                    break
            
            if not seed_entry:
                return
            
            # Update fields
            if iterations is not None:
                seed_entry["iterations"] = iterations
            if passed is not None:
                seed_entry["passed"] = passed
            if golden_file is not None:
                seed_entry["golden_file"] = golden_file
            if functional_test_files is not None:
                seed_entry["functional_test_files"] = functional_test_files
            if security_test_files is not None:
                seed_entry["security_test_files"] = security_test_files
            if functional_pass_rate is not None:
                seed_entry["functional_pass_rate"] = functional_pass_rate
            if security_pass_rate is not None:
                seed_entry["security_pass_rate"] = security_pass_rate
            if error is not None:
                seed_entry["error"] = error
            if status is not None:
                seed_entry["status"] = status
                if status in ["completed", "failed"]:
                    seed_entry["end_time"] = datetime.now().isoformat()
            
            # Update problem fields if revised
            if problem_description is not None:
                seed_entry["problem_description"] = problem_description
            if function_requirements is not None:
                seed_entry["function_requirements"] = function_requirements
            if security_requirements is not None:
                seed_entry["security_requirements"] = security_requirements
            if input_specification is not None:
                seed_entry["input_specification"] = input_specification
            if output_specification is not None:
                seed_entry["output_specification"] = output_specification
            
            # Update CWE statistics
            cwe_data = data["cwes"][cwe_id]
            cwe_data["completed_seeds"] = sum(
                1 for s in cwe_data["seeds"] if s["status"] in ["completed", "failed"]
            )
            cwe_data["successful_seeds"] = sum(
                1 for s in cwe_data["seeds"] if s["passed"]
            )
            
            # Update batchall statistics
            data["batchall_info"]["completed_seeds"] = sum(
                cwe["completed_seeds"] for cwe in data["cwes"].values()
            )
            data["batchall_info"]["successful_seeds"] = sum(
                cwe["successful_seeds"] for cwe in data["cwes"].values()
            )
            
            self._write_result_file(data)
    
    def finalize_cwe(self, cwe_id: str, status: str = "completed"):
        """Mark a CWE as complete"""
        with self._lock:
            data = self._read_result_file()
            if cwe_id in data["cwes"]:
                data["cwes"][cwe_id]["status"] = status
                data["cwes"][cwe_id]["end_time"] = datetime.now().isoformat()
                
                # Update completed CWEs count
                data["batchall_info"]["completed_cwes"] = sum(
                    1 for cwe in data["cwes"].values() 
                    if cwe["status"] in ["completed", "failed"]
                )
                data["batchall_info"]["successful_cwes"] = sum(
                    1 for cwe in data["cwes"].values()
                    if cwe["status"] == "completed" and cwe["successful_seeds"] > 0
                )
            self._write_result_file(data)
    
    def finalize_batchall(self):
        """Mark batchall as complete"""
        with self._lock:
            data = self._read_result_file()
            data["batchall_info"]["end_time"] = datetime.now().isoformat()
            self._write_result_file(data)
    
    def _read_result_file(self) -> Dict[str, Any]:
        """Read current result.json"""
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_result_file(self, data: Dict[str, Any]):
        """Write to result.json"""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_batchall_dir(self) -> Path:
        """Get batchall directory path"""
        return self.batchall_dir
    
    def get_result_file_path(self) -> Path:
        """Get result.json path"""
        return self.result_file
