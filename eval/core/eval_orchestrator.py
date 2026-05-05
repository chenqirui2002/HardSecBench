"""
Eval orchestrator - main evaluation loop
"""
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime
import logging

from models.eval_models import (
    TestCase, EvalResult, IterationResult,
    RequirementResult
)
from agents import TargetLLM, Collaborator
from agents.target_llm import HINT_NONE, HINT_GENERAL, HINT_CWE
from core.data_loader import DataLoader
from core.test_runner import TestRunner
from utils.output_manager import OutputManager
from config.settings import config

# Maximum attempts for compilation fix
MAX_COMPILE_ATTEMPTS = 3


class EvalOrchestrator:
    """
    Main orchestrator for the evaluation process
    
    Flow:
    1. Target LLM generates code
    2. Run tests (functional + security)
    3. Collaborator provides feedback for functional failures
    4. Iterate until functional tests pass or max iterations reached
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_iterations: int,
        max_attempts: int,
        output_manager: OutputManager,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        quiet: bool = False,
        security_hint_level: int = HINT_NONE,
        cwe_info: Optional[Dict[str, Dict[str, str]]] = None,
        engine: Optional[str] = None,
        max_timeout: Optional[int] = None,
        generate_only: bool = False,
        **engine_kwargs
    ):
        """
        Initialize eval orchestrator
        
        Args:
            model_name: Name of the model to evaluate
            temperature: Temperature for target LLM
            max_iterations: Maximum iterations per test case
            max_attempts: Maximum attempts for TB fixing per iteration
            output_manager: Output manager for saving results
            api_key: API key for target LLM
            base_url: Base URL for target LLM
            quiet: If True, suppress detailed console output (for batchall mode)
            security_hint_level: Security hint level (0=none, 1=general, 2=cwe)
            cwe_info: Dict mapping cwe_id to {name, description}
            engine: LLM engine type ("openai" or "transformers"), uses config if None
            max_timeout: Max timeout for API requests in seconds (default: 120)
            generate_only: If True, only generate code once without iteration (for small models)
            **engine_kwargs: Additional engine-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.max_attempts = max_attempts
        self.output_manager = output_manager
        self.quiet = quiet
        self.security_hint_level = security_hint_level
        self.cwe_info = cwe_info or {}
        self.max_timeout = max_timeout or config.TARGET_TIMEOUT
        self.generate_only = generate_only
        
        self.logger = logging.getLogger("eval.orchestrator")
        
        # Determine engine and prepare kwargs
        target_engine = engine or config.TARGET_ENGINE
        target_engine_kwargs = engine_kwargs.copy()
        
        # Add config-based engine kwargs if using transformers
        if target_engine == "transformers" and not engine_kwargs:
            target_engine_kwargs = {
                "device": config.TARGET_DEVICE,
                "load_in_8bit": config.TARGET_LOAD_IN_8BIT,
                "load_in_4bit": config.TARGET_LOAD_IN_4BIT,
                "torch_dtype": config.TARGET_TORCH_DTYPE,
            }
        
        # Initialize agents
        self.target_llm = TargetLLM(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            security_hint_level=security_hint_level,
            engine=target_engine,
            max_timeout=self.max_timeout,
            generate_only=generate_only,
            **target_engine_kwargs
        )
        self.collaborator = Collaborator()
        
        # Test runner (per case)
        self.test_runner = None
    
    def _log_detail(self, msg: str, level: str = "info"):
        """Log detailed message (suppressed in quiet mode for console, always to file)"""
        if self.quiet:
            # In quiet mode, only log to file (DEBUG level won't show on console)
            self.logger.debug(msg)
        else:
            getattr(self.logger, level)(msg)
    
    def evaluate_test_case(self, test_case: TestCase, run_index: int = 0, preloaded_context: Optional[List[Dict[str, str]]] = None) -> EvalResult:
        """
        Evaluate a single test case
        
        Args:
            test_case: Test case to evaluate
            run_index: Index of this run for Pass@k mode (0-based)
            preloaded_context: Optional preloaded initial context (for code deduplication)
            
        Returns:
            EvalResult with all iteration results
        """
        self._log_detail(f"Evaluating test case: {test_case.uuid} (run {run_index})")
        self._log_detail(f"  CWE: {test_case.cwe_id}, Language: {test_case.language}")
        
        # Store run_index for Pass@k tracking
        self.current_run_index = run_index
        
        # Reset target_llm history to avoid state pollution between test cases
        self.target_llm.reset_history()
        
        # If preloaded context is provided, restore it
        if preloaded_context:
            self._log_detail("  Using preloaded initial context")
            self.target_llm.restore_context_from_dict(preloaded_context)
        
        # Initialize result
        result = EvalResult(
            uuid=test_case.uuid,
            cwe_id=test_case.cwe_id,
            language=test_case.language,
            model_name=self.model_name,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            max_attempts=self.max_attempts
        )
        
        # Add run_index to result for Pass@k tracking
        result.run_index = run_index
        
        # Create test runner for this case
        # CRITICAL FIX: Use consistent directory structure for both Pass@1 and Pass@k
        # Structure: eval_dir/CWE-xxxx/UUID/run_N/
        # This matches output_manager's save_code() and save_testbench() paths
        # TestRunner will use this as work_dir and create files directly in it
        work_dir = self.output_manager.get_eval_dir() / test_case.cwe_id / test_case.uuid / f"run_{run_index}"
        self.test_runner = TestRunner(work_dir=work_dir)
        
        # Store current TB codes
        current_tb_codes: Dict[str, str] = {}
        
        try:
            # Step 1: Generate initial code (skip if preloaded context provided)
            if not preloaded_context:
                self._log_detail("  Step 1: Generating initial code...")
                
                # CRITICAL FIX: Always use compilation check, even in generate_only mode
                # This ensures the initial code is compilable before running tests
                # generate_only mode only skips iteration, not compilation verification
                code = self._generate_code_with_compile_check(
                    test_case=test_case,
                    is_initial=True
                )
                
                # Save code
                self.output_manager.save_code(
                    test_case.uuid, test_case.cwe_id, 0, code, test_case.language, run_index=run_index
                )
                
                # CRITICAL FIX: Save initial context for later reuse (e.g., in code deduplication)
                # This ensures the file exists for all runs, not just in deduplication mode
                initial_context = self.target_llm.get_messages()
                self.output_manager.save_initial_context(
                    test_case.uuid, test_case.cwe_id, initial_context, run_index=run_index
                )
            else:
                # Extract code from preloaded context (last AI message)
                self._log_detail("  Step 1: Using code from preloaded context...")
                code = self.target_llm._extract_code(
                    preloaded_context[-1]['content'],
                    test_case.language
                )
            
            # Generate only mode: skip iteration, just run tests once
            if self.generate_only:
                self._log_detail("  Generate only mode: Running tests without iteration...")
                
                # Run tests once
                functional_results, security_results = self._run_tests_with_current_tb(
                    test_case, code, current_tb_codes
                )
                
                # Calculate pass rates
                functional_pass_rate = self._calculate_pass_rate(functional_results)
                security_pass_rate = self._calculate_pass_rate(security_results)
                
                self._log_detail(f"    Functional: {functional_pass_rate:.1%}, Security: {security_pass_rate:.1%}")
                
                # Create single iteration result
                iter_result = IterationResult(
                    iteration_number=1,
                    code=code,
                    functional_results=functional_results,
                    security_results=security_results,
                    functional_pass_rate=functional_pass_rate,
                    security_pass_rate=security_pass_rate,
                    all_functional_passed=all(r.passed for r in functional_results)
                )
                
                result.iterations.append(iter_result)
                
                # Finalize result
                result.first_iter_functional_pass_rate = functional_pass_rate
                result.first_iter_security_pass_rate = security_pass_rate
                result.final_functional_pass_rate = functional_pass_rate
                result.final_security_pass_rate = security_pass_rate
                result.success = iter_result.all_functional_passed
                result.total_iterations = 1
                result.end_time = datetime.now().isoformat()
                
                # Save result
                self.output_manager.add_result(result)
                
                self._log_detail(f" Completed (generate only): success={result.success}")
                self._log_detail(f" Rates: functional={result.final_functional_pass_rate:.1%}, security={result.final_security_pass_rate:.1%}")
                
                return result
            
            # Main iteration loop (normal mode)
            for iteration in range(self.max_iterations):
                self._log_detail(f"  Iteration {iteration + 1}/{self.max_iterations}")
                
                # Save code at start of iteration (for potential rollback)
                code_at_iteration_start = code
                
                # Step 2: Run tests
                self._log_detail("    Running tests...")
                functional_results, security_results = self._run_tests_with_current_tb(
                    test_case, code, current_tb_codes
                )
                
                # Calculate pass rates
                functional_pass_rate = self._calculate_pass_rate(functional_results)
                security_pass_rate = self._calculate_pass_rate(security_results)
                
                self._log_detail(f"    Functional: {functional_pass_rate:.1%}, Security: {security_pass_rate:.1%}")
                
                # Create iteration result
                iter_result = IterationResult(
                    iteration_number=iteration + 1,
                    code=code,
                    functional_results=functional_results,
                    security_results=security_results,
                    functional_pass_rate=functional_pass_rate,
                    security_pass_rate=security_pass_rate
                )
                
                # Check if all functional tests passed
                all_functional_passed = all(r.passed for r in functional_results)
                iter_result.all_functional_passed = all_functional_passed
                
                # Step 3: Collaborator feedback (only for functional failures)
                if not all_functional_passed:
                    failed_functional = [r for r in functional_results if not r.passed]
                    
                    if failed_functional:
                        self._log_detail("    Collaborator generating feedback...")
                        feedback = self.collaborator.generate_feedback(
                            code=code,
                            failed_functional_results=failed_functional,
                            language=test_case.language
                        )
                        
                        if feedback:
                            iter_result.collaborator_message = feedback
                            formatted_feedback = self.collaborator.format_feedback_for_llm(feedback)
                            
                            # Generate fixed code with compilation verification
                            self._log_detail("    Target LLM fixing code...")
                            new_code = self._fix_code_with_compile_check(
                                test_case=test_case,
                                feedback=formatted_feedback,
                                fallback_code=code_at_iteration_start
                            )
                            
                            # Check if code was rolled back
                            if new_code == code_at_iteration_start:
                                self._log_detail("    Code rolled back to iteration start due to compilation failure", "warning")
                                iter_result.code_rolled_back = True
                            else:
                                # Save new code only if it's different and compiles
                                code = new_code
                                self.output_manager.save_code(
                                    test_case.uuid, test_case.cwe_id,
                                    iteration + 1, code, test_case.language, run_index=run_index
                                )
                
                result.iterations.append(iter_result)
                
                # Check stop condition
                if all_functional_passed:
                    self._log_detail("    All functional tests passed!")
                    break
            
            # Finalize result
            first_iter = result.iterations[0] if result.iterations else None
            final_iter = result.iterations[-1] if result.iterations else None
            
            # First iteration pass rates
            if first_iter:
                result.first_iter_functional_pass_rate = first_iter.functional_pass_rate
                result.first_iter_security_pass_rate = first_iter.security_pass_rate
            
            # Final iteration pass rates
            if final_iter:
                result.final_functional_pass_rate = final_iter.functional_pass_rate
                result.final_security_pass_rate = final_iter.security_pass_rate
                result.success = final_iter.all_functional_passed
            
            result.total_iterations = len(result.iterations)
            result.end_time = datetime.now().isoformat()
            
        except Exception as e:
            known_model_output_errors = {
                "Extracted code is too short or empty",
                "Generated feedback is too short or empty",
            }

            if isinstance(e, ValueError) and str(e) in known_model_output_errors:
                self.logger.error(f"Error evaluating test case: {e}")
            else:
                self.logger.exception(f"Error evaluating test case: {e}")

            result.error = str(e)
            result.end_time = datetime.now().isoformat()
        
        # Save result
        self.output_manager.add_result(result)
        
        self._log_detail(f" Completed: success={result.success}, iterations={result.total_iterations}")
        self._log_detail(f" Final rates: functional={result.final_functional_pass_rate:.1%}, security={result.final_security_pass_rate:.1%}")
        
        return result
    
    def _run_tests_with_current_tb(
        self,
        test_case: TestCase,
        code: str,
        current_tb_codes: Dict[str, str]
    ) -> Tuple[List[RequirementResult], List[RequirementResult]]:
        """
        Run tests using current (potentially modified) TB codes
        """
        functional_results = []
        security_results = []
        seed_dir = Path(test_case.seed_dir)
        
        # Run functional tests
        for tb_file, explanation in test_case.functional_test_files.items():
            req_index = self.test_runner._extract_req_index(tb_file)
            if req_index is None or req_index >= len(test_case.function_requirements):
                continue
            
            # Use current TB code or load from file
            if tb_file in current_tb_codes:
                tb_code = current_tb_codes[tb_file]
            else:
                tb_path = seed_dir / tb_file
                if tb_path.exists():
                    with open(tb_path, 'r', encoding='utf-8') as f:
                        tb_code = f.read()
                    current_tb_codes[tb_file] = tb_code
                else:
                    continue
            
            result = self.test_runner.run_single_test(
                test_case=test_case,
                code=code,
                tb_file=tb_file,
                tb_code=tb_code,
                requirement_index=req_index,
                requirement_text=test_case.function_requirements[req_index],
                requirement_type="functional"
            )
            functional_results.append(result)
        
        # Run security tests
        for tb_file, explanation in test_case.security_test_files.items():
            req_index = self.test_runner._extract_req_index(tb_file)
            if req_index is None or req_index >= len(test_case.security_requirements):
                continue
            
            # Use current TB code or load from file
            if tb_file in current_tb_codes:
                tb_code = current_tb_codes[tb_file]
            else:
                tb_path = seed_dir / tb_file
                if tb_path.exists():
                    with open(tb_path, 'r', encoding='utf-8') as f:
                        tb_code = f.read()
                    current_tb_codes[tb_file] = tb_code
                else:
                    continue
            
            result = self.test_runner.run_single_test(
                test_case=test_case,
                code=code,
                tb_file=tb_file,
                tb_code=tb_code,
                requirement_index=req_index,
                requirement_text=test_case.security_requirements[req_index],
                requirement_type="security"
            )
            security_results.append(result)
        
        return functional_results, security_results
    
    def _compile_code(self, test_case: TestCase, code: str) -> Tuple[bool, str]:
        """
        Try to compile the code and return (success, error_message)
        Also validates that module/function name matches expected name
        """
        import subprocess
        import tempfile
        import os
        import signal
        import re
        from pathlib import Path
        
        # CRITICAL FIX: First check if module/function name is correct
        # This prevents issues where LLM uses wrong name (e.g., mem_ecc_driver instead of topmodule)
        expected_name = test_case.module_or_function_name
        
        if test_case.language == "verilog":
            # Check for "module <name>" declaration
            module_pattern = r'\bmodule\s+(\w+)\s*[#(;]'
            matches = re.findall(module_pattern, code)
            if not matches:
                return False, f"No module declaration found in code"
            
            # Check if expected module name exists
            if expected_name not in matches:
                found_names = ", ".join(matches)
                return False, f"Module name mismatch: expected '{expected_name}', but found '{found_names}'"
        else:
            # C: Check for function definition
            # Pattern: return_type function_name(...)
            func_pattern = rf'\b\w+\s+{re.escape(expected_name)}\s*\('
            if not re.search(func_pattern, code):
                return False, f"Function '{expected_name}' not found in code"
        
        # Create temp directory for compilation test
        with tempfile.TemporaryDirectory(prefix="compile_check_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write code to file
            ext = "v" if test_case.language == "verilog" else "c"
            code_file = temp_path / f"{test_case.module_or_function_name}.{ext}"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            if test_case.language == "verilog":
                # Verilog: just syntax check with iverilog
                cmd = f"iverilog -g2012 -t null '{code_file}'"
            else:
                # C: compile to object file (no linking)
                cmd = f"gcc -c -Wall -fsyntax-only '{code_file}'"
            
            process = None
            try:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=temp_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True
                )
                try:
                    stdout, stderr = process.communicate(timeout=30)
                    if process.returncode == 0:
                        return True, ""
                    else:
                        return False, stderr
                except subprocess.TimeoutExpired:
                    # Kill the entire process group
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    return False, "Compilation timeout"
            except Exception as e:
                if process is not None:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                    process.wait()
                return False, str(e)
    
    def _generate_code_with_compile_check(
        self,
        test_case: TestCase,
        is_initial: bool = True
    ) -> str:
        """
        Generate code and iterate until it compiles successfully
        
        Args:
            test_case: Test case
            is_initial: Whether this is initial code generation
            
        Returns:
            Code that compiles successfully
        """
        # Get CWE info if available
        cwe_name = None
        cwe_description = None
        if self.security_hint_level == HINT_CWE and test_case.cwe_id in self.cwe_info:
            cwe_name = self.cwe_info[test_case.cwe_id].get("name")
            cwe_description = self.cwe_info[test_case.cwe_id].get("description")
        
        # Generate initial code
        code = self.target_llm.generate_initial_code(
            test_case,
            cwe_name=cwe_name,
            cwe_description=cwe_description
        )
        
        # For transformers engine, skip compilation check if code extraction failed
        # This avoids unnecessary retries when the model output is not parseable
        if self.target_llm.engine == "transformers":
            # Check if code looks valid (has module/function keyword)
            if test_case.language == "verilog":
                has_valid_code = "module" in code.lower()
            else:
                has_valid_code = any(keyword in code for keyword in ["void", "int", "uint", "char", "float", "double"])
            
            if not has_valid_code:
                self._log_detail("    Transformers engine: Code extraction failed, skipping compilation check", "warning")
                return code
        
        # Check compilation
        success, error_msg = self._compile_code(test_case, code)
        
        if success:
            self._log_detail("    Initial code compiles successfully")
            return code
        
        # For transformers engine, don't retry compilation failures
        # Small models often can't fix compilation errors effectively
        if self.target_llm.engine == "transformers":
            self._log_detail("    Transformers engine: Compilation failed, skipping retry", "warning")
            return code
        
        # Iterate to fix compilation errors (only for API-based engines)
        for attempt in range(MAX_COMPILE_ATTEMPTS):
            self._log_detail(f"    Compilation failed (attempt {attempt + 1}/{MAX_COMPILE_ATTEMPTS}), fixing...")
            self.logger.debug(f"    Compile error: {error_msg[:200]}...")
            
            # Generate feedback for compilation error
            compile_feedback = self._build_compile_error_feedback(error_msg, test_case.language)
            
            # Ask LLM to fix
            code = self.target_llm.fix_code(compile_feedback)
            
            # Check again
            success, error_msg = self._compile_code(test_case, code)
            
            if success:
                self._log_detail(f"    Code compiles successfully after {attempt + 1} fix(es)")
                # Compress history to remove intermediate compilation error messages
                self.target_llm.compress_history_after_compile_fix()
                return code
        
        # Return last attempt even if it doesn't compile
        self._log_detail(f"    Code still doesn't compile after {MAX_COMPILE_ATTEMPTS} attempts", "warning")
        return code
    
    def _fix_code_with_compile_check(
        self,
        test_case: TestCase,
        feedback: str,
        fallback_code: str
    ) -> str:
        """
        Fix code based on feedback and iterate until it compiles successfully
        
        Args:
            test_case: Test case
            feedback: Feedback from collaborator
            fallback_code: Code to rollback to if compilation fails
            
        Returns:
            Fixed code that compiles successfully, or fallback_code if all attempts fail
        """
        # Record position before applying feedback
        messages_before_feedback = len(self.target_llm.get_messages())
        
        # Apply feedback fix
        code = self.target_llm.fix_code(feedback)
        
        # Check compilation
        success, error_msg = self._compile_code(test_case, code)
        
        if success:
            return code
        
        # Iterate to fix compilation errors
        for attempt in range(MAX_COMPILE_ATTEMPTS):
            self._log_detail(f"    Fixed code doesn't compile (attempt {attempt + 1}/{MAX_COMPILE_ATTEMPTS}), fixing...")
            
            # Generate feedback for compilation error
            compile_feedback = self._build_compile_error_feedback(error_msg, test_case.language)
            
            # Ask LLM to fix
            code = self.target_llm.fix_code(compile_feedback)
            
            # Check again
            success, error_msg = self._compile_code(test_case, code)
            
            if success:
                self._log_detail(f"    Code compiles successfully after {attempt + 1} compilation fix(es)")
                # Compress history: remove intermediate compilation error messages
                # Keep messages up to feedback, then only the final successful response
                self._compress_compile_fix_history(messages_before_feedback)
                return code
        
        # Rollback to fallback code if compilation still fails
        self._log_detail(f"    Code still doesn't compile after {MAX_COMPILE_ATTEMPTS} attempts, rolling back to previous version", "warning")
        
        # Rollback chat history to before the failed fix attempt
        self.target_llm.chat_history.messages = self.target_llm.chat_history.messages[:messages_before_feedback]
        
        return fallback_code
    
    def _compress_compile_fix_history(self, messages_before_feedback: int):
        """
        Compress history after compilation fix.
        Keep all messages before feedback + feedback + final successful response.
        Remove intermediate compilation error messages.
        """
        messages = self.target_llm.get_messages()
        if len(messages) <= messages_before_feedback + 2:
            return
        
        # Keep: messages before feedback + feedback (human) + final response (AI)
        messages_to_keep = messages[:messages_before_feedback] + [messages[-2], messages[-1]]
        self.target_llm.chat_history.messages = messages_to_keep
        
        removed_count = len(messages) - len(messages_to_keep)
        if removed_count > 0:
            self._log_detail(f"    Context compressed: Removed {removed_count} compilation fix messages")
    
    def _build_compile_error_feedback(self, error_msg: str, language: str) -> str:
        """
        Build feedback message for compilation errors
        """
        lang_name = "Verilog" if language == "verilog" else "C"
        return f"""Your code has compilation errors. Please fix them.

## Compilation Error
```
{error_msg}
```

Please provide a corrected version of your {lang_name} code that compiles without errors.
Make sure to:
1. Include all necessary headers/imports
2. Fix any syntax errors
3. Ensure all types are properly defined

Output your corrected code in a markdown code block."""
    
    def _is_compilation_failure(self, result: RequirementResult) -> bool:
        """
        Check if the failure is due to compilation error
        Compilation failures are always CODE_ISSUE, not TB_ISSUE
        
        This checks for the specific error message format from TestRunner
        """
        if not result.error_message:
            return False
        
        # Check for compilation failure markers from test_runner
        compilation_markers = [
            "Compilation failed:",
            "Compilation timeout",
            "Command failed:"
        ]
        
        return any(marker in result.error_message for marker in compilation_markers)
    
    
    def _calculate_pass_rate(self, results: List[RequirementResult]) -> float:
        """Calculate pass rate from results"""
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return passed / len(results)
