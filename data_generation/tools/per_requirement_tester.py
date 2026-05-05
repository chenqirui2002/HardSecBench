"""
Per-Requirement Testbench Architecture

One requirement → One testbench file
Simple, focused, easy to fix
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from models.single_generation import CodeLanguage, GenerationTask, GeneratedCode, ProblemDescription
from tools import AgentToolbox
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from utils.logger_manager import get_logger
import json
import re


@dataclass
class SingleRequirementTest:
    """Test result for a single requirement"""
    requirement_index: int
    requirement_text: str
    suite_type: str  # 'functional' or 'security'
    tb_code: str
    tb_file: str
    passed: bool
    output: str
    error_message: Optional[str] = None
    locked: bool = False  # NEW: Locked after passing, won't re-test
    expected_output: Optional[str] = None  # NEW: Expected output for deterministic check
    explanation: Optional[str] = None  # NEW: Test explanation from Red Team


class PerRequirementTester:
    """
    Generate and run one testbench per requirement
    Simple prompts, no parsing complexity
    """
    
    def __init__(self, red_teamer, max_tb_compile_attempts: int = 5):
        self.red_teamer = red_teamer
        self.logger = get_logger()
        self.toolbox = red_teamer.toolbox
        self.max_tb_compile_attempts = max_tb_compile_attempts
        # Per-requirement chat histories: {"functional-req-0": history, ...}
        self.req_chat_histories = {}
        # Track output directory for saving tests
        self.output_dir = None
    
    def _get_or_create_req_history(self, suite_type: str, req_index: int, system_context: str = None):
        """
        Get or create chat history for a specific requirement.
        Each requirement has its own independent conversation context.
        
        Args:
            suite_type: 'functional' or 'security'
            req_index: Requirement index
            system_context: Optional system context for new history
        """
        req_id = f"{suite_type}-req-{req_index}"
        if req_id not in self.req_chat_histories:
            from langchain_core.messages import SystemMessage
            history = InMemoryChatMessageHistory()
            if system_context:
                history.add_message(SystemMessage(content=system_context))
            self.req_chat_histories[req_id] = history
        return self.req_chat_histories[req_id]
    
    def clear_req_histories(self):
        """Clear all per-requirement histories (call at start of new task)"""
        self.req_chat_histories = {}
    
    def clear_suite_histories(self, suite_type: str):
        """
        Clear chat histories for a specific suite type.
        Called when requirements are revised and TBs need to be regenerated.
        
        Args:
            suite_type: 'functional' or 'security'
        """
        keys_to_remove = [k for k in self.req_chat_histories if k.startswith(f"{suite_type}-req-")]
        for key in keys_to_remove:
            del self.req_chat_histories[key]
        self.logger.info(f"Cleared {len(keys_to_remove)} chat histories for {suite_type} suite")
    
    def _get_test_term(self, task: GenerationTask) -> str:
        """Get language-appropriate term for test code"""
        return "TB" if task.language == CodeLanguage.VERILOG else "test"
    
    def generate_per_requirement_tests(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        code: GeneratedCode,
        suite_type: str,
        requirements: List[str],
        system_context: str,
        output_dir: Path = None
    ) -> List[SingleRequirementTest]:
        """
        Generate one TB per requirement
        
        Args:
            suite_type: 'functional' or 'security'
            requirements: List of requirement strings
            system_context: System context for Red Team (used to create per-req histories)
            output_dir: Directory to save test files (optional)
        """
        results = []
        
        # Store output_dir for saving tests
        if output_dir:
            self.output_dir = output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, requirement in enumerate(requirements):
            test_term = "TB" if task.language == CodeLanguage.VERILOG else "test"
            self.logger.info(f"  Generating {test_term} for {suite_type} requirement {i+1}/{len(requirements)}")
            
            # Get or create independent chat history for this requirement
            req_history = self._get_or_create_req_history(suite_type, i, system_context)
            
            # Generate single-requirement TB with explanation
            tb_code, explanation = self._generate_single_requirement_tb(
                task, problem, code, requirement, i, suite_type, req_history,
                max_attempts=self.max_tb_compile_attempts
            )
            
            if not tb_code:
                self.logger.error(f"  Failed to generate {test_term} for requirement {i}")
                results.append(SingleRequirementTest(
                    requirement_index=i,
                    requirement_text=requirement,
                    suite_type=suite_type,
                    tb_code="",
                    tb_file="",
                    passed=False,
                    output="",
                    error_message="TB generation failed"
                ))
                continue
            
            # Log the explanation
            self.logger.info(f"  Explanation: {explanation}")
            
            # Run single-requirement TB
            ext = "v" if task.language == CodeLanguage.VERILOG else "c"
            base_name = "topmodule" if task.language == CodeLanguage.VERILOG else "module"
            tb_file = f"tb_{suite_type}_req_{i}_{base_name}.{ext}"
            
            # Save test file if output_dir is set
            if self.output_dir:
                self.red_teamer._save_test_to_file(task, tb_code, tb_file, self.output_dir)
            
            test_result = self._run_single_requirement_tb(
                task, code, tb_code, tb_file
            )
            
            # C language now uses deterministic comparison (no LLM judging needed)
            # Verilog uses [TEST] PASS/FAIL format
            # Both return passed=True/False directly
            
            # Final safety check: ensure passed is never None
            if test_result['passed'] is None:
                test_result['passed'] = False
            
            test_obj = SingleRequirementTest(
                requirement_index=i,
                requirement_text=requirement,
                suite_type=suite_type,
                tb_code=tb_code,
                tb_file=tb_file,
                passed=test_result['passed'],
                output=test_result['output'],
                error_message=test_result.get('error'),
                locked=False,  # Initially not locked
                explanation=explanation  # Store explanation
            )
            
            # Auto-lock if passed
            if test_obj.passed:
                test_obj.locked = True
                # Save expected_output for deterministic check on future reruns
                # Verilog: actual output (contains [TEST] PASS)
                # C: actual output (contains [CASE-N] RETURN=xxx)
                test_obj.expected_output = test_result['output']
                status = "✓ PASS (locked)"
            else:
                status = "✗ FAIL"
            
            self.logger.info(f"  {suite_type} req {i}: {status}")
            results.append(test_obj)
        
        return results
    
    def rerun_unlocked_tests(
        self,
        task: GenerationTask,
        code: GeneratedCode,
        previous_results: List[SingleRequirementTest],
        verify_locked: bool = True,
        chat_history=None
    ) -> List[SingleRequirementTest]:
        """
        Rerun only unlocked (failed) tests with new code
        Optionally verify locked tests with deterministic check (no LLM)
        """
        new_results = []
        
        for prev_test in previous_results:
            if prev_test.locked:
                if verify_locked:
                    # Verify locked test with deterministic check (no LLM needed)
                    self.logger.info(f"  Verifying {prev_test.suite_type} req {prev_test.requirement_index} (deterministic)...")
                    test_result = self._run_single_requirement_tb(
                        task, code, prev_test.tb_code, prev_test.tb_file
                    )
                    
                    # Deterministic check: compare output
                    still_passed = self._deterministic_check(
                        test_result['output'], 
                        prev_test.expected_output
                    )
                    
                    if still_passed:
                        self.logger.info(f"    ✓ Still PASS (no LLM needed)")
                        new_results.append(prev_test)  # Keep locked
                    else:
                        self.logger.warning(f"    ✗ Test result changed! Unlocking...")
                        # Unlock and treat as failed - test result changed from PASS to FAIL
                        prev_test.locked = False
                        prev_test.passed = False
                        prev_test.output = test_result['output']
                        prev_test.error_message = test_result.get('error', 'Test result changed from PASS to FAIL after code modification')
                        new_results.append(prev_test)
                else:
                    # Trust locked status, don't verify
                    self.logger.info(f"  {prev_test.suite_type} req {prev_test.requirement_index}: ✓ PASS (locked, skipped)")
                    new_results.append(prev_test)
            else:
                # Rerun unlocked test
                self.logger.info(f"  Rerunning {prev_test.suite_type} req {prev_test.requirement_index}...")
                
                test_result = self._run_single_requirement_tb(
                    task, code, prev_test.tb_code, prev_test.tb_file
                )
                
                # C language now uses deterministic comparison (no LLM judging needed)
                # Verilog uses [TEST] PASS/FAIL format
                # Both return passed=True/False directly
                
                # Safety check: ensure passed is never None
                if test_result['passed'] is None:
                    test_result['passed'] = False  # Default to False if comparison failed
                
                # Update test result
                new_test = SingleRequirementTest(
                    requirement_index=prev_test.requirement_index,
                    requirement_text=prev_test.requirement_text,
                    suite_type=prev_test.suite_type,
                    tb_code=prev_test.tb_code,
                    tb_file=prev_test.tb_file,
                    passed=test_result['passed'],
                    output=test_result['output'],
                    error_message=test_result.get('error'),
                    locked=False,
                    explanation=prev_test.explanation
                )
                
                # Auto-lock if now passed
                if new_test.passed:
                    new_test.locked = True
                    # Save expected_output for deterministic check on future reruns
                    # Verilog: actual output (contains [TEST] PASS)
                    # C: actual output (contains [CASE-N] RETURN=xxx)
                    new_test.expected_output = test_result['output']
                    status = "✓ PASS (now locked)"
                else:
                    status = "✗ FAIL"
                
                self.logger.info(f"    → {status}")
                new_results.append(new_test)
        
        return new_results
    
    def _extract_explanation_from_response(self, response: str) -> str:
        """Extract test explanation from LLM response"""
        import re
        
        # Look for "Test Explanation:" section
        match = re.search(r'Test Explanation:(.*?)(?:```|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            explanation = match.group(1).strip()
            # Clean up - take first few sentences
            sentences = explanation.split('.')
            if len(sentences) > 3:
                explanation = '. '.join(sentences[:3]) + '.'
            return explanation
        
        # Fallback: take first paragraph before code
        lines = response.split('\n')
        explanation_lines = []
        for line in lines:
            if '```' in line or 'module ' in line:
                break
            if line.strip() and not line.strip().startswith('#'):
                explanation_lines.append(line.strip())
                if len(explanation_lines) >= 3:
                    break
        
        if explanation_lines:
            return ' '.join(explanation_lines)
        
        return "No explanation provided"
    
    def _deterministic_check(self, actual_output: str, expected_output: str) -> bool:
        """
        Deterministic check: compare actual vs expected output
        No LLM needed
        
        Both Verilog and C now use PASS/FAIL format:
        - Verilog: [TEST] PASS/FAIL
        - C: [CASE-N] PASS/FAIL
        """
        if not expected_output:
            return False
        
        # Normalize outputs (remove whitespace variations)
        actual_normalized = ' '.join(actual_output.split())
        expected_normalized = ' '.join(expected_output.split())
        
        # Check if it's Verilog format (has [TEST] marker)
        if "[TEST]" in expected_normalized:
            # Verilog: Check PASS/FAIL status
            expected_pass = "[TEST] PASS" in expected_normalized
            actual_pass = "[TEST] PASS" in actual_normalized
            return expected_pass == actual_pass
        else:
            # C language: Check [CASE-N] PASS/FAIL status (same logic as Verilog)
            # Expected should have all PASS, no FAIL
            expected_has_fail = re.search(r'\[CASE-\d+\]\s*FAIL', expected_normalized, re.IGNORECASE)
            actual_has_fail = re.search(r'\[CASE-\d+\]\s*FAIL', actual_normalized, re.IGNORECASE)
            
            expected_has_pass = re.search(r'\[CASE-\d+\]\s*PASS', expected_normalized, re.IGNORECASE)
            actual_has_pass = re.search(r'\[CASE-\d+\]\s*PASS', actual_normalized, re.IGNORECASE)
            
            # Both should have same PASS/FAIL status
            # If expected passed (has PASS, no FAIL), actual should also pass
            expected_passed = expected_has_pass and not expected_has_fail
            actual_passed = actual_has_pass and not actual_has_fail
            
            return expected_passed == actual_passed
    
    def _generate_single_requirement_tb(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        code: GeneratedCode,
        requirement: str,
        req_index: int,
        suite_type: str,
        chat_history,
        max_attempts: int = 5
    ) -> str:
        """
        Generate TB for a single requirement with validation loop
        
        Ensures TB is compilable before returning
        """
        
        if task.language == CodeLanguage.VERILOG:
            prompt = self._create_verilog_single_req_prompt(
                task, problem, requirement, req_index, suite_type
            )
        else:
            prompt = self._create_c_single_req_prompt(
                task, problem, requirement, req_index, suite_type
            )
        
        # Use chat history
        chat_history.add_message(HumanMessage(content=prompt))
        
        for attempt in range(max_attempts):
            response = self.red_teamer.llm.invoke(chat_history.messages)
            
            # C language: LLM generates complete test code with PASS/FAIL judgments
            if task.language == CodeLanguage.C:
                from tools.c_function_tester import CFunctionTester
                
                try:
                    # Extract C code from response
                    tb_code = self.red_teamer._extract_code_from_response(response.content)
                    if not tb_code:
                        raise ValueError("No C code found in response")
                    
                    # Count test cases from code
                    import re
                    case_count = len(re.findall(r'\[CASE-\d+\]', tb_code))
                    self.logger.info(f"    Executing {case_count} test case(s)...")
                    
                    # Execute test code
                    tester = CFunctionTester()
                    function_file = self.toolbox.files.base_dir / "module.c"
                    
                    exec_result = tester.run_test(
                        function_file=function_file,
                        test_code=tb_code
                    )
                    
                    # Log command output (DEBUG level)
                    self.logger.debug(f"[C Test Execution] {suite_type} req {req_index}")
                    if exec_result["compile_success"]:
                        self.logger.debug(f"Compile: SUCCESS")
                        self.logger.debug(f"Output: {exec_result['raw_stdout']}")
                        self.logger.debug(f"Passed: {exec_result['passed']}")
                    else:
                        self.logger.debug(f"Compile: FAILED\nError:\n{exec_result['compile_error']}")
                    if not exec_result["run_success"] and exec_result.get('run_error'):
                        self.logger.debug(f"Run Error:\n{exec_result['run_error']}")
                    
                    if not exec_result["compile_success"]:
                        raise Exception(f"Compile error: {exec_result['compile_error']}")
                    
                    if not exec_result["run_success"]:
                        raise Exception(f"Run error: {exec_result['run_error']}")
                    
                    explanation = self._extract_explanation_from_response(response.content)
                    
                    # Mark as compiled successfully
                    compile_ok = True
                    
                except Exception as e:
                    self.logger.info(f"    Attempt {attempt+1}: Test generation/execution failed - {str(e)}")
                    if attempt < max_attempts - 1:
                        chat_history.add_message(AIMessage(content=response.content))
                        chat_history.add_message(HumanMessage(content=f"Error: {str(e)}\nProvide complete C test code with self-contained PASS/FAIL judgments. Use printf(\"[CASE-N] PASS\\n\") or printf(\"[CASE-N] FAIL: reason\\n\")."))
                        continue
                    else:
                        return "", f"Failed: {str(e)}"
            
            # Verilog: extract code as before
            else:
                tb_code = self.red_teamer._extract_code_from_response(response.content)
                explanation = self._extract_explanation_from_response(response.content)
                
                if not tb_code:
                    self.logger.warning(f"    Attempt {attempt+1}: No TB code extracted")
                    if attempt < max_attempts - 1:
                        chat_history.add_message(AIMessage(content=response.content))
                        chat_history.add_message(HumanMessage(content="No code found in your response. Please provide the complete testbench code."))
                        continue
                    else:
                        return "", explanation
                
                compile_ok = False  # Will be set by compilation below
            
            # Validate compilation
            ext = "v" if task.language == CodeLanguage.VERILOG else "c"
            base_name = "topmodule" if task.language == CodeLanguage.VERILOG else "module"
            tb_file = f"tb_{suite_type}_req_{req_index}_{base_name}.{ext}"
            # Use relative path - toolbox will prepend work_dir
            self.toolbox.write_code(tb_file, tb_code, task.language.value)
            tb_path = self.toolbox.files.base_dir / tb_file
            
            # Try to compile (C already compiled above, Verilog compiles here)
            if task.language == CodeLanguage.VERILOG:
                compile_ok, run_result = self.red_teamer._run_verilog_test(task, code, str(tb_path))
                compile_result = None  # Verilog doesn't return compile_result separately
                
                # Log command output (DEBUG level)
                self.logger.debug(f"[Verilog TB Generation] Attempt {attempt+1}: {tb_file}")
                if compile_ok and run_result and run_result.success:
                    self.logger.debug(f"Compile & Run: SUCCESS\nOutput:\n{run_result.stdout}")
                else:
                    self.logger.debug(f"Compile & Run: FAILED")
                    if run_result:
                        if run_result.stderr:
                            self.logger.debug(f"Error:\n{run_result.stderr}")
                        if run_result.stdout:
                            self.logger.debug(f"Stdout:\n{run_result.stdout}")
            # C language: already executed above, compile_ok already set
            
            if compile_ok:
                if task.language == CodeLanguage.VERILOG:
                    self.logger.info(f"    ✓ TB compiled successfully (attempt {attempt+1})")
                else:
                    self.logger.info(f"    ✓ Test executed successfully (attempt {attempt+1})")
                
                # Context compression: Remove compilation failure messages
                # Only keep the final successful response
                if attempt > 0:
                    # Find the initial prompt position
                    initial_prompt_idx = None
                    for i, msg in enumerate(chat_history.messages):
                        if "Single-Requirement Testbench Generation" in msg.content or "Single-Requirement Test Case Generation" in msg.content:
                            initial_prompt_idx = i
                            break
                    
                    if initial_prompt_idx is not None:
                        # Keep messages before initial prompt + successful response
                        messages_to_keep = chat_history.messages[:initial_prompt_idx + 1]
                        chat_history.messages = messages_to_keep
                        self.logger.info(f"    🗜️ Context compressed: Removed {attempt} failed attempts")
                
                chat_history.add_message(AIMessage(content=response.content))
                return tb_code, explanation
            else:
                # Compilation/execution failed
                # Note: C language failures are handled above in except block
                if task.language == CodeLanguage.VERILOG:
                    error_msg = run_result.stderr if run_result else "Unknown error"
                else:
                    # C language - should not reach here (handled above)
                    error_msg = "Test failed (see error above)"
                self.logger.info(f"    Attempt {attempt+1}: Test failed")
                
                if attempt < max_attempts - 1:
                    # Ask Red Team to fix
                    chat_history.add_message(AIMessage(content=response.content))
                    fix_prompt = f"""Your testbench has compilation errors:

{error_msg[:500]}

Please fix the compilation errors and provide the corrected testbench code."""
                    chat_history.add_message(HumanMessage(content=fix_prompt))
                else:
                    # Last attempt failed
                    test_term = "TB" if task.language == CodeLanguage.VERILOG else "test"
                    self.logger.error(f"    ✗ {test_term} compilation failed after {max_attempts} attempts")
                    chat_history.add_message(AIMessage(content=response.content))
                    return tb_code, explanation  # Return anyway, will be marked as failed
        
        return "", explanation
    
    def _build_all_requirements_context(
        self,
        problem: ProblemDescription,
        current_suite_type: str,
        current_req_index: int
    ) -> str:
        """
        Build context string showing all requirements.
        Marks the current requirement being tested.
        
        Security requirements are shown as context to avoid ambiguity,
        but with a clear warning not to test security behavior in functional TBs.
        """
        lines = ["## Requirements Context"]
        
        # Add functional requirements (always shown)
        if problem.function_requirements:
            lines.append("\n**Functional Requirements:**")
            for i, req in enumerate(problem.function_requirements):
                marker = " <- (current target)" if (current_suite_type == "functional" and i == current_req_index) else ""
                lines.append(f"- [FUNCTIONAL-REQ-{i}]: {req}{marker}")
        
        # Add security requirements as context (to avoid ambiguity)
        if problem.security_requirements:
            if current_suite_type == "security":
                lines.append("\n**Security Requirements (These describe SECURE behavior that SHOULD be implemented):**")
            else:
                lines.append("\n**Security Requirements (Context Only - DO NOT test in functional TB):**")
            for i, req in enumerate(problem.security_requirements):
                marker = " <- (current target)" if (current_suite_type == "security" and i == current_req_index) else ""
                lines.append(f"- [SECURITY-REQ-{i}]: {req}{marker}")
        
        # Add warning for functional testbenches
        if current_suite_type == "functional":
            lines.append("\n**WARNING**: You are generating a FUNCTIONAL testbench. Do NOT test or assume any security behavior. Your TB should ONLY verify the target functional requirement.")
        elif current_suite_type == "security":
            lines.append("\n**NOTE**: Security requirements describe SECURE behavior. Your test should PASS if the security measure IS implemented, and FAIL if it is MISSING.")
        
        return "\n".join(lines)
    
    def _create_verilog_single_req_prompt(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        requirement: str,
        req_index: int,
        suite_type: str
    ) -> str:
        """Simple Verilog TB prompt for ONE requirement only"""
        
        all_reqs_context = self._build_all_requirements_context(problem, suite_type, req_index)
        
        return f"""# Single-Requirement Testbench Generation

## Problem
{problem.question}

{all_reqs_context}

## Target Requirement
**[{suite_type.upper()}-REQ-{req_index}]**: {requirement}

## Module
**Name**: `topmodule`
**Inputs**: {problem.input_specification}
**Outputs**: {problem.output_specification}

## Syntax
- Target: Icarus Verilog 12.0, Verilog-2001/2005
- Output: `$display("[TEST] PASS/FAIL: reason");`

## Rules
1. Instantiate: `topmodule uut(...);`
2. Test ONLY using module outputs
3. Must not conflict with other requirements
4. Your test file should ONLY contain logic for the target requirement
5. Do NOT assume or test any behavior not explicitly stated in the target requirement

## Response
1. **Test Explanation**: 2-3 sentences
2. **Testbench Code**: Complete testbench"""
    
    def _create_c_single_req_prompt(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        requirement: str,
        req_index: int,
        suite_type: str
    ) -> str:
        """Generate self-contained test with PASS/FAIL judgments"""
        
        all_reqs_context = self._build_all_requirements_context(problem, suite_type, req_index)
        
        return f"""# Single-Requirement Test Case Generation

## Problem
{problem.question}

{all_reqs_context}

## Target Requirement
**[{suite_type.upper()}-REQ-{req_index}]**: {requirement}

## Function
**Name**: `module`
**Inputs**: {problem.input_specification}
**Outputs**: {problem.output_specification}

## Rules
1. Declare: `extern ... module(...);`
2. Test return value AND pointer outputs
3. Output: `printf("[CASE-N] PASS/FAIL: reason\\n");`
4. Use block scope to isolate test cases
5. Never use extern for global variables
6. Your test file should ONLY contain logic for the target requirement
7. Do NOT assume or test any behavior not explicitly stated in the target requirement

## Response
1. **Test Explanation**: 2-3 sentences
2. **Test Code**: Complete C code with main(), 1-3 test cases"""
    
    def _run_single_requirement_tb(
        self,
        task: GenerationTask,
        code: GeneratedCode,
        tb_code: str,
        tb_file: str
    ) -> dict:
        """
        Run a single-requirement test
        - Verilog: Execute testbench file
        - C: Parse test inputs from tb_code, execute with CFunctionTester
        """
        
        if task.language == CodeLanguage.VERILOG:
            # Verilog: Write TB file and execute
            tb_path = Path(self.toolbox.work_dir) / tb_file
            self.toolbox.write_code(tb_path, tb_code, task.language.value)
            
            compile_ok, run_result = self.red_teamer._run_verilog_test(
                task, code, str(tb_path)
            )
            
            # Log command output (DEBUG level)
            self.logger.debug(f"[Verilog Test] {tb_file}")
            if compile_ok and run_result and run_result.success:
                self.logger.debug(f"Compile & Run: SUCCESS\nOutput:\n{run_result.stdout}")
            else:
                self.logger.debug(f"Compile & Run: FAILED")
                if run_result:
                    if run_result.stderr:
                        self.logger.debug(f"Error:\n{run_result.stderr}")
                    if run_result.stdout:
                        self.logger.debug(f"Stdout:\n{run_result.stdout}")
            
            if not compile_ok or not run_result or not run_result.success:
                error = run_result.stderr if run_result else "Unknown error"
                return {
                    'passed': False,
                    'output': run_result.stdout if run_result else "",
                    'error': f"Compilation/execution failed: {error[:200]}"
                }
            
            output = run_result.stdout
            
            if not output or output.strip() == '':
                diagnostic = f"""No output detected. Possible causes:
1. Test crashed during execution
2. $display statements not reached
3. Test finished too early (missing waits)
4. Simulation timeout

Stderr: {run_result.stderr[:150] if run_result.stderr else 'None'}"""
                return {'passed': False, 'output': '', 'error': diagnostic}
            
            # Verilog uses [TEST] PASS/FAIL format (same logic as C)
            # First check for FAIL, then check for PASS
            if "[TEST] FAIL" in output:
                error_detail = "Test executed but found '[TEST] FAIL' in output."
                return {'passed': False, 'output': output, 'error': error_detail}
            elif "[TEST] PASS" in output:
                return {'passed': True, 'output': output}
            else:
                error_detail = "Test executed but did not output '[TEST] PASS'. Check test logic or code behavior."
                return {'passed': False, 'output': output, 'error': error_detail}
        
        else:
            # C: Execute test code, check for PASS/FAIL in output (same as Verilog)
            from tools.c_function_tester import CFunctionTester
            
            try:
                # Execute with CFunctionTester
                tester = CFunctionTester()
                function_file = self.toolbox.files.base_dir / "module.c"
                
                exec_result = tester.run_test(
                    function_file=function_file,
                    test_code=tb_code
                )
                
                # Log command output (DEBUG level)
                self.logger.debug(f"[C Test Execution] {tb_file}")
                if exec_result["compile_success"]:
                    self.logger.debug(f"Compile: SUCCESS")
                    self.logger.debug(f"Output: {exec_result['raw_stdout']}")
                    self.logger.debug(f"Passed: {exec_result['passed']}")
                else:
                    self.logger.debug(f"Compile: FAILED\nError:\n{exec_result['compile_error']}")
                if not exec_result["run_success"] and exec_result.get('run_error'):
                    self.logger.debug(f"Run Error:\n{exec_result['run_error']}")
                
                if not exec_result["compile_success"]:
                    return {
                        'passed': False,
                        'output': '',
                        'error': f"Compile error: {exec_result['compile_error'][:200]}"
                    }
                
                if not exec_result["run_success"]:
                    return {
                        'passed': False,
                        'output': exec_result['raw_stdout'],
                        'error': f"Run error: {exec_result['run_error'][:200]}"
                    }
                
                # Use PASS/FAIL detection from CFunctionTester (same logic as Verilog)
                if exec_result['passed']:
                    return {
                        'passed': True,
                        'output': exec_result['raw_stdout'],
                        'error': None
                    }
                else:
                    error_msg = "Test executed but contains FAIL or no PASS detected."
                    return {
                        'passed': False,
                        'output': exec_result['raw_stdout'],
                        'error': error_msg
                    }
                
            except Exception as e:
                return {'passed': False, 'output': '', 'error': f'Execution error: {str(e)}'}
    
    def fix_single_requirement_tb(
        self,
        task: GenerationTask,
        code: GeneratedCode,
        suite_type: str,
        req_index: int,
        requirement_text: str,
        arbiter_guidance: Optional[str] = None
    ) -> Tuple[Optional[SingleRequirementTest], None]:
        """
        Fix a single failing TB based on Arbiter's guidance.
        Uses the per-requirement chat history to preserve context.
        
        Args:
            arbiter_guidance: Guidance from Arbiter about what to fix
        
        Returns:
            Tuple of (fixed_test, None)
            - fixed_test: The fixed test result, or None if fix failed
            - Second element is always None (kept for API compatibility)
        """
        
        requirement_id = f"{suite_type.upper()}-REQ-{req_index}"
        
        # Different fix prompts for Verilog and C
        if task.language == CodeLanguage.VERILOG:
            fix_prompt = self._build_verilog_fix_prompt(
                requirement_id, requirement_text, arbiter_guidance, suite_type
            )
        else:
            fix_prompt = self._build_c_fix_prompt(
                requirement_id, requirement_text, arbiter_guidance, suite_type
            )
        
        # Use per-requirement independent history (preserves context from TB generation)
        req_history = self._get_or_create_req_history(suite_type, req_index)
        
        req_history.add_message(HumanMessage(content=fix_prompt))
        response = self.red_teamer.llm.invoke(req_history.messages)
        req_history.add_message(AIMessage(content=response.content))
        
        # Extract explanation from response
        fixed_explanation = self._extract_explanation_from_response(response.content)
        
        # Process response based on language
        if task.language == CodeLanguage.VERILOG:
            # Extract fixed TB code
            fixed_tb_code = self.red_teamer._extract_code_from_response(response.content)
            
            if not fixed_tb_code:
                self.logger.warning(f"  ✗ No fixed TB code extracted")
                return None, None
        else:
            # C language: Extract complete test code with PASS/FAIL judgments
            from tools.c_function_tester import CFunctionTester
            
            try:
                # Extract C code from response
                fixed_tb_code = self.red_teamer._extract_code_from_response(response.content)
                if not fixed_tb_code:
                    self.logger.warning(f"  No fixed C test code extracted")
                    return None, None
                
                # Execute test code
                tester = CFunctionTester()
                function_file = self.toolbox.files.base_dir / "module.c"
                
                exec_result = tester.run_test(
                    function_file=function_file,
                    test_code=fixed_tb_code
                )
                
                if not exec_result["compile_success"]:
                    self.logger.warning(f"  Fixed test compile failed: {exec_result['compile_error'][:100]}")
                    return None, None
                
                if not exec_result["run_success"]:
                    self.logger.warning(f"  Fixed test run failed: {exec_result['run_error'][:100]}")
                    return None, None
                
                # Use PASS/FAIL detection (same as Verilog)
                passed = exec_result['passed']
                error_msg = None if passed else "Test contains FAIL or no PASS detected"
                
                tb_file = f"tb_{suite_type}_req_{req_index}_module.c"
                
                # Save fixed test to file if output_dir is set
                if self.output_dir:
                    self.red_teamer._save_test_to_file(task, fixed_tb_code, tb_file, self.output_dir)
                
                test_term = "test"
                if passed:
                    self.logger.info(f"  Fixed {test_term} for requirement {req_index}")
                else:
                    self.logger.warning(f"  Fixed {test_term} still fails for requirement {req_index}")
                
                return SingleRequirementTest(
                    requirement_index=req_index,
                    requirement_text=requirement_text,
                    suite_type=suite_type,
                    tb_code=fixed_tb_code,
                    tb_file=tb_file,
                    explanation=fixed_explanation,
                    passed=passed,
                    output=exec_result['raw_stdout'],
                    error_message=error_msg,
                    locked=passed,
                    expected_output=exec_result['raw_stdout'] if passed else None
                ), None
                
            except Exception as e:
                self.logger.warning(f"  Failed to parse/execute fixed C test: {str(e)}")
                return None, None
        
        # Verilog path continues here
        # Generate TB file path
        ext = "v"
        base_name = "topmodule"
        tb_file = f"tb_{suite_type}_req_{req_index}_{base_name}.{ext}"
        
        # Save fixed TB to file if output_dir is set
        if self.output_dir:
            self.red_teamer._save_test_to_file(task, fixed_tb_code, tb_file, self.output_dir)
        
        # Run fixed TB
        test_result = self._run_single_requirement_tb(
            task, code, fixed_tb_code, tb_file
        )
        
        test_term = "TB"
        
        if test_result['passed']:
            self.logger.info(f"  ✓ Fixed {test_term} for requirement {req_index}")
        else:
            self.logger.warning(f"  ✗ Fixed {test_term} still fails for requirement {req_index}")
        
        # Always return the result (whether passed or failed)
        return SingleRequirementTest(
            requirement_index=req_index,
            requirement_text=requirement_text,
            suite_type=suite_type,
            tb_code=fixed_tb_code,
            tb_file=tb_file,
            explanation=fixed_explanation,
            passed=test_result['passed'],
            output=test_result['output'],
            error_message=test_result.get('error'),
            locked=test_result['passed'],
            expected_output=test_result.get('expected_output', test_result['output']) if test_result['passed'] else None
        ), None
    
    def _build_verilog_fix_prompt(
        self,
        requirement_id: str,
        requirement_text: str,
        arbiter_guidance: Optional[str],
        suite_type: str = "functional"
    ) -> str:
        """Build SIMPLIFIED fix prompt for Verilog testbench"""
        warning = ""
        if suite_type == "functional":
            warning = "\n\n**WARNING**: This is a FUNCTIONAL testbench. Do NOT test or assume any security behavior. Only verify the functional requirement."
        
        return f"""# Fix Testbench for [{requirement_id}]

**Requirement**: {requirement_text}

**Fix**: {arbiter_guidance if arbiter_guidance else "Fix timing or signal logic issues."}{warning}

## Output Format
1. **Explanation**: 1-2 sentences explaining what you changed and why
2. **Code**: The complete corrected testbench code"""
    
    def _build_c_fix_prompt(
        self,
        requirement_id: str,
        requirement_text: str,
        arbiter_guidance: Optional[str],
        suite_type: str = "functional"
    ) -> str:
        """Build SIMPLIFIED fix prompt for C test code"""
        warning = ""
        if suite_type == "functional":
            warning = "\n\n**WARNING**: This is a FUNCTIONAL test. Do NOT test or assume any security behavior. Only verify the functional requirement."
        
        return f"""# Fix Test for [{requirement_id}]

**Requirement**: {requirement_text}

**Fix**: {arbiter_guidance if arbiter_guidance else "Fix expected values, pointer verification, or function call."}{warning}

## Output Format
1. **Explanation**: 1-2 sentences explaining what you changed and why
2. **Code**: The complete corrected test code with self-contained PASS/FAIL judgments.
   - Use `printf("[CASE-N] PASS\\n");` for success
   - Use `printf("[CASE-N] FAIL: reason\\n");` for failure
   - Verify BOTH return value AND pointer outputs (use memcmp if needed)"""
    
    def _parse_feedback_response(self, response: str) -> Optional[dict]:
        """
        Parse JSON feedback from RedTeam's response
        
        Returns:
            Dict with keys: verdict, feedback_type, reasoning, confidence, evidence
            None if parsing fails
        """
        try:
            # Extract JSON block
            json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object without markdown
                json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: check for old format
                    if "CODE_BUG" in response:
                        return {
                            'verdict': 'CODE_BUG',
                            'feedback_type': 'disagree',
                            'reasoning': 'RedTeam believes this is a code issue (legacy format)',
                            'confidence': 0.7
                        }
                    return None
            
            feedback = json.loads(json_str)
            
            # Validate required fields
            if 'verdict' not in feedback:
                feedback['verdict'] = 'TB_ERROR'
            
            return feedback
            
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"  Failed to parse feedback JSON: {e}")
            # Try legacy format
            if "CODE_BUG" in response:
                return {
                    'verdict': 'CODE_BUG',
                    'feedback_type': 'disagree',
                    'reasoning': 'RedTeam believes this is a code issue',
                    'confidence': 0.7
                }
            return None
