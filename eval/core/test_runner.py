"""
Test runner for eval system
Runs existing testbenches against newly generated code
"""
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging
import tempfile
import shutil

from models.eval_models import TestCase, RequirementResult


@dataclass
class CompileResult:
    """Result of compilation"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class SimulationResult:
    """Result of simulation/execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int


class TestRunner:
    """
    Run testbenches/tests against generated code
    """
    
    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize test runner
        
        Args:
            work_dir: Working directory for test execution
        """
        self.work_dir = Path(work_dir).resolve() if work_dir else Path(tempfile.mkdtemp(prefix="eval_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("eval.test_runner")
    
    def run_all_tests(
        self,
        test_case: TestCase,
        code: str
    ) -> Tuple[List[RequirementResult], List[RequirementResult]]:
        """
        Run all tests (functional and security) for a test case
        
        Args:
            test_case: Test case with test files
            code: Generated code to test
            
        Returns:
            (functional_results, security_results)
        """
        # CRITICAL FIX: Use work_dir directly, don't create UUID subdirectory
        # The work_dir already includes the UUID path from eval_orchestrator
        case_dir = self.work_dir
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Write generated code to file
        ext = "v" if test_case.language == "verilog" else "c"
        code_file = case_dir / f"{test_case.module_or_function_name}.{ext}"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Run functional tests
        functional_results = self._run_tests(
            test_case=test_case,
            code=code,
            code_file=code_file,
            case_dir=case_dir,
            test_files=test_case.functional_test_files,
            requirements=test_case.function_requirements,
            test_type="functional"
        )
        
        # Run security tests
        security_results = self._run_tests(
            test_case=test_case,
            code=code,
            code_file=code_file,
            case_dir=case_dir,
            test_files=test_case.security_test_files,
            requirements=test_case.security_requirements,
            test_type="security"
        )
        
        return functional_results, security_results
    
    def run_single_test(
        self,
        test_case: TestCase,
        code: str,
        tb_file: str,
        tb_code: str,
        requirement_index: int,
        requirement_text: str,
        requirement_type: str
    ) -> RequirementResult:
        """
        Run a single test
        
        Args:
            test_case: Test case
            code: Generated code
            tb_file: Testbench filename
            tb_code: Testbench code
            requirement_index: Requirement index
            requirement_text: Requirement text
            requirement_type: 'functional' or 'security'
            
        Returns:
            RequirementResult
        """
        # CRITICAL FIX: Use work_dir directly, don't create UUID subdirectory
        # The work_dir already includes the UUID path from eval_orchestrator
        # Structure: work_dir = eval_dir/CWE-xxxx/UUID/run_N/
        case_dir = self.work_dir
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Write code to expected filename
        ext = "v" if test_case.language == "verilog" else "c"
        code_file = case_dir / f"{test_case.module_or_function_name}.{ext}"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Write testbench
        tb_path = case_dir / tb_file
        with open(tb_path, 'w', encoding='utf-8') as f:
            f.write(tb_code)
        
        # Run test
        if test_case.language == "verilog":
            result = self._run_verilog_test(code_file, tb_path, case_dir)
        else:
            result = self._run_c_test(code_file, tb_path, case_dir)
        
        # Check pass/fail
        passed = self._check_pass_fail(result.stdout, test_case.language)
        
        return RequirementResult(
            requirement_index=requirement_index,
            requirement_text=requirement_text,
            requirement_type=requirement_type,
            tb_file=tb_file,
            tb_code=tb_code,
            passed=passed,
            output=result.stdout,
            error_message=result.stderr if not result.success else None
        )
    
    def _run_tests(
        self,
        test_case: TestCase,
        code: str,
        code_file: Path,
        case_dir: Path,
        test_files: Dict[str, str],
        requirements: List[str],
        test_type: str
    ) -> List[RequirementResult]:
        """Run a set of tests"""
        results = []
        seed_dir = Path(test_case.seed_dir)
        
        for tb_filename, explanation in test_files.items():
            # Extract requirement index from filename (e.g., tb_functional_req_0_xxx.v)
            req_index = self._extract_req_index(tb_filename)
            if req_index is None or req_index >= len(requirements):
                self.logger.warning(f"Invalid requirement index in {tb_filename}")
                continue
            
            requirement_text = requirements[req_index]
            
            # Load testbench code
            tb_source = seed_dir / tb_filename
            if not tb_source.exists():
                self.logger.warning(f"Testbench not found: {tb_source}")
                results.append(RequirementResult(
                    requirement_index=req_index,
                    requirement_text=requirement_text,
                    requirement_type=test_type,
                    tb_file=tb_filename,
                    tb_code="",
                    passed=False,
                    output="",
                    error_message=f"Testbench file not found: {tb_filename}"
                ))
                continue
            
            with open(tb_source, 'r', encoding='utf-8') as f:
                tb_code = f.read()
            
            # Copy testbench to case dir
            tb_path = case_dir / tb_filename
            with open(tb_path, 'w', encoding='utf-8') as f:
                f.write(tb_code)
            
            # Run test
            if test_case.language == "verilog":
                result = self._run_verilog_test(code_file, tb_path, case_dir)
            else:
                result = self._run_c_test(code_file, tb_path, case_dir)
            
            # Check pass/fail
            passed = self._check_pass_fail(result.stdout, test_case.language)
            
            results.append(RequirementResult(
                requirement_index=req_index,
                requirement_text=requirement_text,
                requirement_type=test_type,
                tb_file=tb_filename,
                tb_code=tb_code,
                passed=passed,
                output=result.stdout,
                error_message=result.stderr if not result.success else None
            ))
        
        return results
    
    def _run_verilog_test(
        self,
        design_file: Path,
        tb_file: Path,
        work_dir: Path
    ) -> SimulationResult:
        """Run Verilog simulation"""
        sim_out = work_dir / "sim.out"
        
        # Compile with iverilog
        cmd = f"iverilog -g2012 -o '{sim_out}' '{design_file}' '{tb_file}'"
        exit_code, stdout, stderr = self._run_command(cmd, work_dir)
        
        if exit_code != 0:
            return SimulationResult(
                success=False,
                stdout=stdout,
                stderr=f"Compilation failed:\n{stderr}",
                exit_code=exit_code
            )
        
        # Run simulation (with strict timeout to prevent runaway processes)
        cmd = f"vvp '{sim_out}'"
        exit_code, stdout, stderr = self._run_command(cmd, work_dir, timeout=15)
        
        # Cleanup
        if sim_out.exists():
            sim_out.unlink()
        
        # Clean up VCD files generated by testbench
        for vcd_file in work_dir.glob("*.vcd"):
            try:
                vcd_file.unlink()
            except Exception:
                pass
        
        return SimulationResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def _run_c_test(
        self,
        code_file: Path,
        test_file: Path,
        work_dir: Path
    ) -> SimulationResult:
        """Run C test"""
        exe_out = work_dir / "test.out"
        
        # Compile code and test together
        cmd = f"gcc -Wall -o '{exe_out}' '{code_file}' '{test_file}'"
        exit_code, stdout, stderr = self._run_command(cmd, work_dir)
        
        if exit_code != 0:
            return SimulationResult(
                success=False,
                stdout=stdout,
                stderr=f"Compilation failed:\n{stderr}",
                exit_code=exit_code
            )
        
        # Run test
        exit_code, stdout, stderr = self._run_command(str(exe_out), work_dir, timeout=30)
        
        # Cleanup
        if exe_out.exists():
            exe_out.unlink()
        
        return SimulationResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def _run_command(
        self,
        cmd: str,
        cwd: Path,
        timeout: int = 30
    ) -> Tuple[int, str, str]:
        """Run shell command with proper timeout and process cleanup"""
        import os
        import signal
        
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True  # Create new process group for clean kill
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                # Kill the entire process group (including vvp child processes)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already dead
                process.wait()
                return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            # Cleanup on any error
            if process is not None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                process.wait()
            return -1, "", f"Command failed: {str(e)}"
    
    def _check_pass_fail(self, output: str, language: str) -> bool:
        """
        Check if test passed based on output
        
        Verilog: looks for [TEST] PASS/FAIL
        C: looks for [CASE-N] PASS/FAIL
        
        Note: We check for explicit FAIL markers in test result format,
        not arbitrary "FAIL" substring (which could appear in descriptions
        like "validation failed as expected").
        """
        if not output:
            return False
        
        output_upper = output.upper()
        
        if language == "verilog":
            # Check for [TEST] FAIL marker
            if "[TEST] FAIL" in output_upper:
                return False
            return "[TEST] PASS" in output_upper
        else:
            # C language: check for [CASE-N] FAIL pattern
            # Match patterns like [CASE-1] FAIL, [CASE-2] FAIL, etc.
            fail_pattern = r'\[CASE-\d+\]\s*FAIL'
            if re.search(fail_pattern, output_upper):
                return False
            # Check for at least one PASS
            pass_pattern = r'\[CASE-\d+\]\s*PASS'
            return bool(re.search(pass_pattern, output_upper))
    
    def _extract_req_index(self, filename: str) -> Optional[int]:
        """Extract requirement index from filename"""
        # Pattern: tb_functional_req_0_xxx.v or tb_security_req_1_xxx.c
        match = re.search(r'req_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None
    
    def update_testbench(
        self,
        test_case: TestCase,
        tb_file: str,
        new_tb_code: str
    ):
        """
        Update a testbench file with new code
        
        Args:
            test_case: Test case
            tb_file: Testbench filename
            new_tb_code: New testbench code
        """
        # CRITICAL FIX: Use work_dir directly, don't create UUID subdirectory
        # The work_dir already includes the UUID path from eval_orchestrator
        case_dir = self.work_dir
        case_dir.mkdir(parents=True, exist_ok=True)
        
        tb_path = case_dir / tb_file
        with open(tb_path, 'w', encoding='utf-8') as f:
            f.write(new_tb_code)
    
    def cleanup(self):
        """Clean up work directory"""
        if self.work_dir.exists() and str(self.work_dir).startswith(tempfile.gettempdir()):
            shutil.rmtree(self.work_dir, ignore_errors=True)
