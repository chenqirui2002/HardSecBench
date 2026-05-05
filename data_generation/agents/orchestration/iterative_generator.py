"""
Iterative Code Generation with Red Team Feedback Loop

Agent B (Expert) and Agent C (Red Teamer) work together iteratively:
1. Expert generates code
2. Red Teamer tests and finds vulnerabilities
3. Expert fixes based on feedback
4. Repeat until secure or max iterations reached
"""
from typing import Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from config.settings import config
from utils.logger_manager import get_logger, log_llm_interaction
from utils.data_loader import DataLoader
from models.generation_models import (
    GenerationTask, 
    ProblemDescription, 
    GeneratedCode,
    TestCase,
    CodeLanguage
)
from models.single_generation import (
    SingleGenerationResult,
    IterationRecord as SingleIterationRecord,
    TestSuite,
    TestCaseResult,
    JudgedTestSuite
)

# Core agents from new architecture
from agents.generation import ArchitectAgent, ExpertAgent, RedTeamerAgent, TestJudgment
from agents.evaluation import ArbiterAgent, ArbiterDecision, Verdict

# Tools
from tools import (
    EnhancedFeedbackBuilder,
    PerRequirementTester,
    SingleRequirementTest
)


@dataclass
class IterationResult:
    """Result of one iteration (legacy, for backward compatibility)"""
    iteration: int
    code: GeneratedCode
    test_result: Optional[TestCase]
    vulnerabilities_found: int
    vulnerability_details: List[str]
    all_tests_passed: bool
    # New fields for test suites
    functional_suite: Optional[TestSuite] = None
    security_suite: Optional[TestSuite] = None


class IterativeCodeGenerator:
    """
    Iterative code generation with Expert-RedTeamer collaborative feedback loop
    
    NEW Flow (Balanced Dialogue):
    
    Phase 1: Expert generates initial code
    Phase 2: Red Team generates test suites -> Run tests (e.g., Pass 2/4)
    Phase 3: Collaborative iteration loop:
       While (not all pass AND total_iterations < max):
         Step A: Red Team analyzes failures WITH CONTEXT:
           - Determine: CODE_BUG or TB_ERROR?
           - If TB_ERROR: Fix TB -> Validate with current code -> Update suite
           - If CODE_BUG: Keep TB, tell Expert to fix
           - Red Team iteration +1
         
         Step B: Expert fixes code based on stable TB results
           - Fix only real bugs (not TB errors)
           - Validate compilation
           - Expert iteration +1
         
         Step C: Re-run tests with fixed code
           - Check if all pass
    
    Stop when:
      - All tests pass, OR
      - Total iterations (Expert + Red Team) >= max_iterations
    
    Key Improvements:
      - TB can be fixed by Red Team (not frozen)
      - Expert only fixes real bugs (not misled by TB errors)
      - Both agents collaborate at same level
      - Total iterations shared between both agents
    """
    
    def __init__(
        self,
        work_dir: Optional[Path] = None,
        max_iterations: int = 10,
        max_tb_compile_attempts: int = 5,
        save_to_json: bool = True
    ):
        self.logger = get_logger()
        self.work_dir = work_dir or (config.OUTPUT_DIR / "iterative_gen")
        self.max_iterations = max_iterations  # Total iterations for both agents
        self.max_tb_compile_attempts = max_tb_compile_attempts
        self.save_to_json = save_to_json
        self.data_loader = DataLoader()
        
        # Track iteration counts separately
        self.total_iterations = 0
        self.expert_iterations = 0
        self.red_team_iterations = 0
        
        # Track iteration issues: {iteration_num: [issue_types]}
        # e.g. {"1": ["tb_issue", "code_issue"], "2": ["PASS"]}
        self.iteration_issues = {}
        
        # Initialize agents with work_dir
        self.architect = ArchitectAgent(work_dir=self.work_dir)
        self.expert = ExpertAgent(work_dir=self.work_dir)
        self.red_teamer = RedTeamerAgent(work_dir=self.work_dir)
        
        # P0 Enhancement: Enhanced Feedback
        self.enhanced_feedback = EnhancedFeedbackBuilder()
        
        # Per-Requirement TB Architecture
        self.per_req_tester = PerRequirementTester(self.red_teamer, max_tb_compile_attempts)
        
        # Per-Requirement Arbiter (Phase 2: Full deployment)
        self.arbiter = ArbiterAgent()
        
        # Initialize chat history for Expert (preserves context across iterations)
        self.expert_chat_history = None  # Will be created per task
        
        self.logger.info(f"Iterative Generator initialized with max_iterations={max_iterations}, max_tb_compile_attempts={max_tb_compile_attempts}")
        self.logger.info("Decoupled testing architecture enabled")
    
    def _reset_agent_memories(self, task: GenerationTask, problem: ProblemDescription, reason: str = ""):
        """
        Clear and reinitialize Agent chat histories (NUCLEAR OPTION - rarely used)
        
        NOTE: For requirement clarifications, the simplified flow handles it:
        - Architect clarifies -> notify Expert (append to history)
        - RedTeam regenerates TB -> re-run tests -> re-arbitrate
        - Normal CODE_ISSUE flow handles Expert code fixes
        
        This method should only be called when:
        - Complete architecture change (very rare)
        - Critical error recovery
        
        Args:
            task: Current task
            problem: Updated problem description
            reason: Reason for reset (for logging)
        """
        self.logger.info(f"Resetting Agent memories... Reason: {reason}")
        
        # Clear existing chat histories
        self.expert_chat_history.clear()
        self.architect_chat_history.clear()
        
        # Clear per-requirement histories
        self.per_req_tester.clear_req_histories()
        
        # Reinitialize with fresh system context (using updated problem)
        expert_context = self._create_initial_context(task, problem)
        self.expert_chat_history.add_message(SystemMessage(content=expert_context))
        log_llm_interaction("Expert", "SYSTEM", expert_context, f"Context RESET - {reason}")
        
        # Reinitialize Architect context
        architect_context = self._create_architect_initial_context(task, problem)
        self.architect_chat_history.add_message(SystemMessage(content=architect_context))
        log_llm_interaction("Architect", "SYSTEM", architect_context, f"Context RESET - {reason}")
        
        # Update Red Team system contexts
        self.functional_system_context = self._create_test_initial_context(task, problem, "functional")
        log_llm_interaction("RedTeam-Functional", "SYSTEM", self.functional_system_context, f"Context RESET - {reason}")
        
        self.security_system_context = self._create_test_initial_context(task, problem, "security")
        log_llm_interaction("RedTeam-Security", "SYSTEM", self.security_system_context, f"Context RESET - {reason}")
        
        self.logger.info(f"   Agent memories cleared and reinitialized with updated context")
    
    def _fix_contradictions_upfront(
        self, 
        problem: ProblemDescription, 
        validation_result: dict
    ) -> ProblemDescription:
        """
        Fix requirement contradictions before code generation using Architect.
        
        This method is called in Phase 0 when Arbiter detects contradictions.
        It uses Architect to revise conflicting requirements incrementally.
        
        Args:
            problem: Original problem description
            validation_result: Arbiter's validation result containing contradictions
        
        Returns:
            Updated ProblemDescription with resolved contradictions
        """
        contradictions = validation_result.get("contradictions", [])
        if not contradictions:
            return problem
        
        # Initialize Architect chat history for upfront revision
        architect_history = InMemoryChatMessageHistory()
        architect_context = self._create_architect_initial_context_for_upfront(problem)
        architect_history.add_message(SystemMessage(content=architect_context))
        
        # Track which requirements have been revised to avoid duplicate revisions
        revised_reqs = set()
        
        for contradiction in contradictions:
            req1_id = contradiction.get("req1_id", "")
            req2_id = contradiction.get("req2_id", "")
            summary = contradiction.get("summary", "")
            technical_reason = contradiction.get("technical_reason", "")
            
            self.logger.info(f"  Resolving: {req1_id} vs {req2_id}")
            
            # Parse requirement IDs to get suite type and index
            req1_info = self._parse_requirement_id(req1_id)
            req2_info = self._parse_requirement_id(req2_id)
            
            if not req1_info or not req2_info:
                self.logger.warning(f"  Could not parse requirement IDs: {req1_id}, {req2_id}")
                continue
            
            # Build contradiction context message for Architect
            contradiction_msg = self._build_contradiction_message(
                req1_id, req1_info, req2_id, req2_info, 
                summary, technical_reason, problem
            )
            
            # Add contradiction to Architect's history
            architect_history.add_message(HumanMessage(content=contradiction_msg))
            
            # Revise the first requirement if not already revised
            if req1_id not in revised_reqs:
                arbiter_feedback = f"Contradiction with {req2_id}: {technical_reason}"
                
                # Get related requirements for context
                if req1_info["suite_type"] == "functional":
                    related_reqs = problem.security_requirements
                else:
                    related_reqs = problem.function_requirements
                
                revised_req, notes = self.architect.revise_single_requirement(
                    problem=problem,
                    req_index=req1_info["index"],
                    suite_type=req1_info["suite_type"],
                    arbiter_feedback=arbiter_feedback,
                    related_requirements=related_reqs,
                    chat_history=architect_history
                )
                
                # Update the requirement in problem
                if req1_info["suite_type"] == "functional":
                    old_req = problem.function_requirements[req1_info["index"]]
                    if revised_req != old_req:
                        problem.function_requirements[req1_info["index"]] = revised_req
                        revised_reqs.add(req1_id)
                        self.logger.info(f"    Revised {req1_id}")
                else:
                    old_req = problem.security_requirements[req1_info["index"]]
                    if revised_req != old_req:
                        problem.security_requirements[req1_info["index"]] = revised_req
                        revised_reqs.add(req1_id)
                        self.logger.info(f"    Revised {req1_id}")
            
            # Revise the second requirement if not already revised
            if req2_id not in revised_reqs:
                arbiter_feedback = f"Contradiction with {req1_id}: {technical_reason}"
                
                if req2_info["suite_type"] == "functional":
                    related_reqs = problem.security_requirements
                else:
                    related_reqs = problem.function_requirements
                
                revised_req, notes = self.architect.revise_single_requirement(
                    problem=problem,
                    req_index=req2_info["index"],
                    suite_type=req2_info["suite_type"],
                    arbiter_feedback=arbiter_feedback,
                    related_requirements=related_reqs,
                    chat_history=architect_history
                )
                
                if req2_info["suite_type"] == "functional":
                    old_req = problem.function_requirements[req2_info["index"]]
                    if revised_req != old_req:
                        problem.function_requirements[req2_info["index"]] = revised_req
                        revised_reqs.add(req2_id)
                        self.logger.info(f"    Revised {req2_id}")
                else:
                    old_req = problem.security_requirements[req2_info["index"]]
                    if revised_req != old_req:
                        problem.security_requirements[req2_info["index"]] = revised_req
                        revised_reqs.add(req2_id)
                        self.logger.info(f"    Revised {req2_id}")
        
        # Store the architect history for later use
        self.architect_chat_history = architect_history
        
        self.logger.info(f"  Total requirements revised: {len(revised_reqs)}")
        return problem
    
    def _fix_security_leakage_upfront(
        self,
        problem: ProblemDescription,
        leakage_result: dict
    ) -> ProblemDescription:
        """
        Fix security leakage in functional requirements before code generation.
        
        This method is called in Phase 0.5 when Arbiter detects that functional
        requirements leak/hint at security requirements.
        
        Args:
            problem: Original problem description
            leakage_result: Arbiter's leakage detection result
        
        Returns:
            Updated ProblemDescription with leakages removed
        """
        leakages = leakage_result.get("leakages", [])
        if not leakages:
            return problem
        
        # Initialize or reuse Architect chat history
        if not hasattr(self, 'architect_chat_history') or self.architect_chat_history is None:
            architect_history = InMemoryChatMessageHistory()
            architect_context = self._create_architect_initial_context_for_upfront(problem)
            architect_history.add_message(SystemMessage(content=architect_context))
        else:
            architect_history = self.architect_chat_history
        
        # Track which functional requirements have been revised
        revised_reqs = set()
        
        for leakage in leakages:
            func_req_id = leakage.get("functional_req_id", "")
            leaked_sec_id = leakage.get("leaked_security_req_id", "")
            leakage_type = leakage.get("leakage_type", "")
            suggestion = leakage.get("suggestion", "")
            
            self.logger.info(f"  Fixing leakage: {func_req_id} -> {leaked_sec_id}")
            
            # Parse functional requirement ID
            func_info = self._parse_requirement_id(func_req_id)
            if not func_info or func_info["suite_type"] != "functional":
                self.logger.warning(f"  Could not parse functional requirement ID: {func_req_id}")
                continue
            
            req_index = func_info["index"]
            if req_index >= len(problem.function_requirements):
                self.logger.warning(f"  Invalid requirement index: {req_index}")
                continue
            
            # Skip if already revised
            if func_req_id in revised_reqs:
                continue
            
            # Build leakage fix message for Architect
            leakage_msg = self._build_leakage_fix_message(
                func_req_id, req_index, leaked_sec_id, leakage_type, suggestion, problem
            )
            
            architect_history.add_message(HumanMessage(content=leakage_msg))
            
            # Use Architect to revise the functional requirement
            arbiter_feedback = f"Security leakage ({leakage_type}): hints at {leaked_sec_id}. {suggestion}"
            
            revised_req, notes = self.architect.revise_single_requirement(
                problem=problem,
                req_index=req_index,
                suite_type="functional",
                arbiter_feedback=arbiter_feedback,
                related_requirements=[],  # Don't show security requirements
                chat_history=architect_history
            )
            
            old_req = problem.function_requirements[req_index]
            if revised_req != old_req:
                problem.function_requirements[req_index] = revised_req
                revised_reqs.add(func_req_id)
                self.logger.info(f"    Revised {func_req_id}")
        
        # Store the architect history for later use
        self.architect_chat_history = architect_history
        
        self.logger.info(f"  Total functional requirements revised for leakage: {len(revised_reqs)}")
        return problem
    
    def _build_leakage_fix_message(
        self,
        func_req_id: str,
        req_index: int,
        leaked_sec_id: str,
        leakage_type: str,
        suggestion: str,
        problem: ProblemDescription
    ) -> str:
        """Build a message describing the security leakage for Architect to fix"""
        func_req_text = problem.function_requirements[req_index]
        
        msg = f"""# Security Leakage Detected in Phase 0.5

The Arbiter has detected that a functional requirement leaks/hints at security concerns.
This must be fixed to ensure the benchmark properly tests implicit security awareness.

## Problematic Functional Requirement
**{func_req_id}**: {func_req_text}

## Leakage Details
- **Leakage Type**: {leakage_type}
- **Leaked Security Requirement**: {leaked_sec_id}

## Suggestion
{suggestion}

## Your Task
Rewrite this functional requirement to:
1. REMOVE any security-related keywords or hints
2. PRESERVE the core functional behavior
3. Make it purely functional without implying security concerns
4. Use neutral language that doesn't suggest threat models or attacks

The revised requirement should describe WHAT the system does, not WHY (security reasons)."""
        
        return msg
    
    def _parse_requirement_id(self, req_id: str) -> Optional[dict]:
        """
        Parse requirement ID like 'FUNCTIONAL-REQ-0' or 'SECURITY-REQ-1'
        
        Returns:
            Dict with 'suite_type' and 'index', or None if parsing fails
        """
        import re
        match = re.match(r'(FUNCTIONAL|SECURITY)-REQ-(\d+)', req_id, re.IGNORECASE)
        if match:
            suite_type = "functional" if match.group(1).upper() == "FUNCTIONAL" else "security"
            index = int(match.group(2))
            return {"suite_type": suite_type, "index": index}
        return None
    
    def _build_contradiction_message(
        self,
        req1_id: str,
        req1_info: dict,
        req2_id: str,
        req2_info: dict,
        summary: str,
        technical_reason: str,
        problem: ProblemDescription
    ) -> str:
        """Build a message describing the contradiction for Architect"""
        # Get the actual requirement texts
        if req1_info["suite_type"] == "functional":
            req1_text = problem.function_requirements[req1_info["index"]]
        else:
            req1_text = problem.security_requirements[req1_info["index"]]
        
        if req2_info["suite_type"] == "functional":
            req2_text = problem.function_requirements[req2_info["index"]]
        else:
            req2_text = problem.security_requirements[req2_info["index"]]
        
        msg = f"""# Contradiction Detected in Phase 0

The Arbiter has detected a contradiction between requirements that must be resolved before code generation.

## Conflicting Requirements

**{req1_id}**: {req1_text}

**{req2_id}**: {req2_text}

## Contradiction Summary
{summary}

## Technical Analysis
{technical_reason}

## Your Task
Please revise these requirements to resolve the contradiction while:
1. Preserving the core intent of each requirement
2. Ensuring they can be implemented together without conflict
3. Adding precise timing, conditions, or constraints to eliminate ambiguity
4. Defining clear mappings for any interface signals (e.g., reg_select values)

The revised requirements should be testable and implementable without contradiction."""
        
        return msg
    
    def _create_architect_initial_context_for_upfront(
        self,
        problem: ProblemDescription
    ) -> str:
        """
        Create initial system context for Architect during upfront contradiction resolution.
        """
        func_reqs = "\n".join(f"- FUNCTIONAL-REQ-{i}: {req}" for i, req in enumerate(problem.function_requirements))
        sec_reqs = "\n".join(f"- SECURITY-REQ-{i}: {req}" for i, req in enumerate(problem.security_requirements))
        
        context = f"""You are the Architect responsible for designing requirements for this hardware/software module.

# Problem Context

**Question**: {problem.question}

**Functional Requirements**:
{func_reqs}

**Security Requirements**:
{sec_reqs}

**Interface**:
- Inputs: {problem.input_specification}
- Outputs: {problem.output_specification}

# Your Role in Phase 0

The Arbiter has detected contradictions in the requirements BEFORE code generation begins.
Your task is to revise conflicting requirements to resolve these contradictions.

When revising requirements:
1. **Preserve the original intent** - Don't change what the requirement is trying to achieve
2. **Add precision** - Specify timing, conditions, edge cases, signal mappings
3. **Resolve conflicts** - Ensure both requirements can be satisfied simultaneously
4. **Define mappings** - For interface signals like reg_select, define explicit value-to-register mappings

This conversation preserves your context for subsequent revisions."""

        return context

    def generate_with_feedback(
        self, 
        task: GenerationTask, 
        problem: ProblemDescription,
        output_dir: Optional[Path] = None
    ) -> Tuple[GeneratedCode, List, dict]:
        """
        Generate code with collaborative feedback from Expert and Red Team
        
        Uses new per-requirement test architecture for precision
        
        Args:
            task: Generation task
            problem: Problem description
            output_dir: Directory to save generated code and tests (optional)
            
        Returns:
            Tuple of (GeneratedCode, iteration_history, phase0_validation)
        """
        self.logger.info(f"Starting iterative generation for task {task.task_id}")
        
        # Save original requirements before any modifications
        original_func_reqs = list(problem.function_requirements)
        original_sec_reqs = list(problem.security_requirements)
        
        # Initialize phase0 validation result
        phase0_validation = {
            "contradictions_detected": False,
            "contradictions": [],
            "security_leakage_detected": False,
            "leakages": [],
            "original_function_requirements": original_func_reqs,
            "original_security_requirements": original_sec_reqs
        }
        
        # IMPROVEMENT 1: Upfront requirement validation
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 0: Upfront Requirement Validation")
        self.logger.info("="*60)
        validation_result = self.arbiter.validate_requirements_upfront(
            problem, 
            language=problem.language.value
        )
        
        if validation_result.get("has_contradictions", False):
            self.logger.info(f"Found {len(validation_result['contradictions'])} contradiction(s) in requirements")
            self.logger.info("Fixing contradictions before code generation...")
            
            # Record to phase0_validation
            phase0_validation["contradictions_detected"] = True
            phase0_validation["contradictions"] = validation_result.get("contradictions", [])
            
            for contradiction in validation_result["contradictions"]:
                self.logger.info(f"  Contradiction: {contradiction.get('req1_id')} vs {contradiction.get('req2_id')}")
                self.logger.info(f"  Reason: {contradiction.get('summary', 'N/A')}")
            
            # Fix contradictions using Architect before code generation
            problem = self._fix_contradictions_upfront(problem, validation_result)
            self.logger.info("Contradictions resolved, proceeding with code generation")
        else:
            self.logger.info("No contradictions detected in requirements")
        
        # IMPROVEMENT 2: Security leakage detection
        self.logger.info("\n" + "-"*40)
        self.logger.info("PHASE 0.5: Security Leakage Detection")
        self.logger.info("-"*40)
        leakage_result = self.arbiter.detect_security_leakage(
            problem,
            language=problem.language.value
        )
        
        if leakage_result.get("has_leakage", False):
            self.logger.info(f"Found {len(leakage_result['leakages'])} security leakage(s) in functional requirements")
            self.logger.info("Fixing leakages before code generation...")
            
            # Record to phase0_validation
            phase0_validation["security_leakage_detected"] = True
            phase0_validation["leakages"] = leakage_result.get("leakages", [])
            
            for leakage in leakage_result["leakages"]:
                self.logger.info(f"  Leakage: {leakage.get('functional_req_id')} hints at {leakage.get('leaked_security_req_id')}")
                self.logger.info(f"  Type: {leakage.get('leakage_type', 'N/A')}")
            
            # Fix leakages using Architect
            problem = self._fix_security_leakage_upfront(problem, leakage_result)
            self.logger.info("Security leakages removed, proceeding with code generation")
        else:
            self.logger.info("No security leakage detected in functional requirements")
        
        # Set output directory for saving files
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
        
        # Initialize chat histories for this task
        self.expert_chat_history = InMemoryChatMessageHistory()
        self.functional_test_chat_history = InMemoryChatMessageHistory()
        self.security_test_chat_history = InMemoryChatMessageHistory()
        
        # Only initialize architect_chat_history if not already set by Phase 0
        if not hasattr(self, 'architect_chat_history') or self.architect_chat_history is None:
            self.architect_chat_history = InMemoryChatMessageHistory()
            # Send initial context to Architect (for requirement clarification)
            architect_context = self._create_architect_initial_context(task, problem)
            self.architect_chat_history.add_message(SystemMessage(content=architect_context))
        else:
            # Phase 0 already initialized architect history, append updated context
            self.logger.info("Architect history preserved from Phase 0 contradiction resolution")
        
        # Send initial system context to Expert (only once)
        expert_context = self._create_initial_context(task, problem)
        self.expert_chat_history.add_message(SystemMessage(content=expert_context))
        log_llm_interaction("Expert", "SYSTEM", expert_context, "Initial Context")
        
        # Create system contexts for Red Team (stored for per-requirement histories)
        self.functional_system_context = self._create_test_initial_context(task, problem, "functional")
        self.security_system_context = self._create_test_initial_context(task, problem, "security")
        
        # Log the system contexts
        log_llm_interaction("RedTeam-Functional", "SYSTEM", self.functional_system_context, "Initial Context")
        log_llm_interaction("RedTeam-Security", "SYSTEM", self.security_system_context, "Initial Context")
        
        # Clear per-requirement histories for new task
        self.per_req_tester.clear_req_histories()
        
        self.logger.info("Chat histories initialized for Expert, Red Team, and Architect")
        
        # Reset iteration counters
        self.total_iterations = 0
        self.expert_iterations = 0
        self.red_team_iterations = 0
        
        # Reset iteration issues tracker
        self.iteration_issues = {}
        
        iteration_history = []
        
        # Store judged suites for blind feedback (Phase 3)
        self.judged_functional = None
        self.judged_security = None
        
        # Store per-requirement test results (for locking mechanism)
        self.per_req_functional = []
        self.per_req_security = []
        
        # Architect response for saving (initialized to None)
        architect_response = None
        
        # =================================================================
        # PHASE 1: Expert generates initial code FIRST
        # =================================================================
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PHASE 1: Expert Generating Initial Code")
        self.logger.info(f"{'='*60}")
        
        current_code = self.expert.generate_code(
            task,
            problem,
            max_iterations=3,
            chat_history=self.expert_chat_history
        )
        self.expert_iterations += 1
        self.total_iterations += 1
        
        if not current_code.is_valid:
            self.logger.error("Initial code generation failed")
            return current_code, [], phase0_validation
        
        # Save initial code to file
        if output_dir:
            self.expert._save_code_to_file(task, current_code, output_dir)
        
        # =================================================================
        # PHASE 2: Red Team generates test suites (DECOUPLED MODE)
        # =================================================================
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PHASE 2: Red Team Generating Test Suites (Decoupled)")
        self.logger.info(f"{'='*60}")
        
        # Generate and judge functional tests
        functional_suite, self.judged_functional = self._generate_and_judge_suite(
            task, problem, current_code, "functional",
            problem.function_requirements,
            self.functional_system_context,
            output_dir
        )
        
        # Generate and judge security tests
        security_suite, self.judged_security = self._generate_and_judge_suite(
            task, problem, current_code, "security",
            problem.security_requirements,
            self.security_system_context,
            output_dir
        )
        
        if functional_suite is None or security_suite is None:
            self.logger.error("Failed to generate test suites")
            return current_code, [], phase0_validation
        
        self.logger.info(f"[OK] Test suites generated and judged:")
        if functional_suite:
            self.logger.info(f"  - Functional: {functional_suite.passed_tests}/{functional_suite.total_tests} passed")
        if security_suite:
            self.logger.info(f"  - Security: {security_suite.passed_tests}/{security_suite.total_tests} passed")
        
        # If all tests pass on first try, finalize expected outputs
        if self._check_all_tests_passed(functional_suite, security_suite):
            self.logger.info("All tests passed! Finalizing expected outputs...")
            functional_suite, security_suite = self._finalize_suites(functional_suite, security_suite)
        
        # Record initial iteration
        iteration_result = self._create_iteration_result(
            1, current_code, functional_suite, security_suite, task
        )
        iteration_history.append(iteration_result)
        self._log_iteration_results(1, functional_suite, security_suite)
        
        # Record initial iteration issues
        if self._check_all_tests_passed(functional_suite, security_suite):
            self.iteration_issues["1"] = ["PASS"]
        else:
            # Initial iteration has no arbiter decision yet, mark as pending
            self.iteration_issues["1"] = ["initial"]
        
        # =================================================================
        # PHASE 3: Collaborative iteration loop
        # =================================================================
        collaborative_iter = 1
        while self.total_iterations < self.max_iterations:
            collaborative_iter += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"COLLABORATIVE ITERATION {collaborative_iter}")
            self.logger.info(f"Total: {self.total_iterations}/{self.max_iterations} | Expert: {self.expert_iterations} | Red Team: {self.red_team_iterations}")
            self.logger.info(f"{'='*60}")
            
            # Check if all tests pass
            if self._check_all_tests_passed(functional_suite, security_suite):
                self.logger.info("All tests passed! Finalizing expected outputs...")
                functional_suite, security_suite = self._finalize_suites(functional_suite, security_suite)
                break
            
            # =====================================================================
            # Step A: Arbiter analyzes and routes (Phase 2: Arbiter-first)
            # =====================================================================
            self.logger.info("\n-> Step A: Arbiter analyzing failures and routing...")
            arbiter_decisions = self._arbiter_analyze_failures(
                task, problem, current_code, collaborative_iter
            )
            
            # Separate decisions by verdict
            problem_issues = [d for d in arbiter_decisions if d.verdict == Verdict.PROBLEM_ISSUE]
            tb_issues = [d for d in arbiter_decisions if d.verdict == Verdict.TB_ISSUE]
            code_issues = [d for d in arbiter_decisions if d.verdict == Verdict.CODE_ISSUE]
            
            self.logger.info(f"   Arbiter routing: {len(problem_issues)} Problem issues, {len(tb_issues)} TB issues, {len(code_issues)} Code issues")
            
            # Record iteration issues
            iter_issues = []
            if problem_issues:
                iter_issues.append("problem_issue")
            if tb_issues:
                iter_issues.append("tb_issue")
            if code_issues:
                iter_issues.append("code_issue")
            if not iter_issues:
                iter_issues.append("PASS")
            self.iteration_issues[str(collaborative_iter)] = iter_issues
            
            # Handle PROBLEM_ISSUE: Architect revises ALL requirements comprehensively
            if problem_issues:
                self.logger.info(f"\n-> Step A1: Architect revising ALL requirements (comprehensive)...")
                
                # Build feedback list for Architect
                arbiter_feedbacks = [
                    {
                        'suite_type': d.suite_type,
                        'req_index': d.req_index,
                        'guidance': d.guidance
                    }
                    for d in problem_issues
                ]
                
                # Log which requirements have issues
                for fb in arbiter_feedbacks:
                    self.logger.info(f"   Issue in {fb['suite_type'].upper()}-REQ-{fb['req_index']}")
                
                # Store old requirements for comparison
                old_func_reqs = list(problem.function_requirements)
                old_sec_reqs = list(problem.security_requirements)
                old_input_spec = problem.input_specification
                old_output_spec = problem.output_specification
                
                # Architect revises ALL requirements
                problem, revision_notes = self.architect.revise_all_requirements(
                    problem=problem,
                    arbiter_feedbacks=arbiter_feedbacks,
                    chat_history=self.architect_chat_history
                )
                
                # Check what changed
                func_changed = (problem.function_requirements != old_func_reqs)
                sec_changed = (problem.security_requirements != old_sec_reqs)
                interface_changed = (problem.input_specification != old_input_spec or 
                                    problem.output_specification != old_output_spec)
                
                requirements_revised = func_changed or sec_changed or interface_changed
                
                if requirements_revised:
                    self.logger.info(f"   Revision notes: {revision_notes}")
                    if func_changed:
                        self.logger.info(f"   Functional requirements changed")
                    if sec_changed:
                        self.logger.info(f"   Security requirements changed")
                    if interface_changed:
                        self.logger.info(f"   Interface specification changed")
                    
                    # Expert regenerates code based on revised requirements
                    self.logger.info("\n   Expert regenerating code based on revised requirements...")
                    
                    # Build message for Expert with full revised requirements
                    func_reqs_text = "\n".join([f"  - FUNCTIONAL-REQ-{i}: {req}" 
                                                for i, req in enumerate(problem.function_requirements)])
                    sec_reqs_text = "\n".join([f"  - SECURITY-REQ-{i}: {req}" 
                                               for i, req in enumerate(problem.security_requirements)])
                    
                    update_msg = f"""The Architect has revised requirements based on Arbiter feedback.

## Revised Interface
Inputs: {problem.input_specification}
Outputs: {problem.output_specification}

## Revised Functional Requirements
{func_reqs_text}

## Revised Security Requirements
{sec_reqs_text}

## Revision Notes
{revision_notes}

Please regenerate your implementation to satisfy these revised requirements."""
                    
                    self.expert_chat_history.add_message(SystemMessage(content=update_msg))
                    
                    # Regenerate code with revised requirements
                    current_code = self._expert_fix_code(
                        task,
                        problem,
                        current_code,
                        functional_suite,
                        security_suite
                    )
                    
                    self.expert_iterations += 1
                    self.total_iterations += 1
                    
                    if not current_code.is_valid:
                        self.logger.error("Code regeneration failed after requirement revision")
                        break
                    
                    # Save regenerated code
                    if output_dir:
                        self.expert._save_code_to_file(task, current_code, output_dir)
                    
                    # Clear and regenerate affected test suites
                    self.logger.info("   Regenerating test suites for revised requirements...")
                    
                    # Determine which suites need regeneration
                    affected_suites = set()
                    if func_changed or interface_changed:
                        affected_suites.add("functional")
                    if sec_changed or interface_changed:
                        affected_suites.add("security")
                    # Also add suites from original problem_issues
                    for d in problem_issues:
                        if d.suite_type in ("functional", "security"):
                            affected_suites.add(d.suite_type)
                    
                    # Clear per_req results and chat histories for affected suites
                    if "functional" in affected_suites:
                        self.logger.info("   Clearing functional suite (requirements changed)...")
                        self.per_req_functional = []
                        self.per_req_tester.clear_suite_histories("functional")
                        
                        self.logger.info("   Regenerating functional test suite...")
                        functional_suite, self.judged_functional = self._generate_and_judge_suite(
                            task, problem, current_code, "functional",
                            problem.function_requirements,
                            self.functional_system_context,
                            output_dir
                        )
                    
                    if "security" in affected_suites:
                        self.logger.info("   Clearing security suite (requirements changed)...")
                        self.per_req_security = []
                        self.per_req_tester.clear_suite_histories("security")
                        
                        self.logger.info("   Regenerating security test suite...")
                        security_suite, self.judged_security = self._generate_and_judge_suite(
                            task, problem, current_code, "security",
                            problem.security_requirements,
                            self.security_system_context,
                            output_dir
                        )
                    
                    self.logger.info("   Requirements revision complete (code + tests regenerated)")
                    
                    # Record iteration result before continuing
                    iteration_result = self._create_iteration_result(
                        collaborative_iter, current_code, functional_suite, security_suite, task
                    )
                    iteration_history.append(iteration_result)
                    self._log_iteration_results(collaborative_iter, functional_suite, security_suite)
                    
                    # Continue to next iteration
                    continue
                else:
                    self.logger.info("   No substantial changes, treating as code issues...")
                    for decision in problem_issues:
                        decision.verdict = Verdict.CODE_ISSUE
                        code_issues.append(decision)
                    problem_issues = []
            
            # Step B: Fix TBs if Arbiter identified TB issues
            if tb_issues:
                self.logger.info(f"\n-> Step B: Fixing {len(tb_issues)} TBs (Arbiter-routed)...")
                for decision in tb_issues:
                    # This method fixes TB AND runs the test in one shot
                    self._fix_tb_with_arbiter_guidance(task, problem, current_code, decision)
                
                self.red_team_iterations += 1
                self.total_iterations += 1
                
                # TB is per-requirement: fix_single_requirement_tb already ran the test
                # Just rebuild suite objects from per_req_functional/security (no re-run needed)
                self.logger.info(f"\n   Rebuilding test suites (TBs already tested during fix)...")
                functional_suite, self.judged_functional = self._convert_per_req_to_suite(
                    self.per_req_functional, "functional", task
                )
                security_suite, self.judged_security = self._convert_per_req_to_suite(
                    self.per_req_security, "security", task
                )
                
                # Check if all pass after TB fixes
                if self._check_all_tests_passed(functional_suite, security_suite):
                    self.logger.info("[OK] All tests passed after TB fixes!")
                    functional_suite, security_suite = self._finalize_suites(functional_suite, security_suite)
                    
                    # CRITICAL: Record successful iteration before breaking
                    iteration_result = self._create_iteration_result(
                        collaborative_iter, current_code, functional_suite, security_suite, task
                    )
                    iteration_history.append(iteration_result)
                    self._log_iteration_results(collaborative_iter, functional_suite, security_suite)
                    break
                
                # TB fixed but not all pass - re-arbitrate with fresh analysis
                # Record this iteration before continuing
                iteration_result = self._create_iteration_result(
                    collaborative_iter, current_code, functional_suite, security_suite, task
                )
                iteration_history.append(iteration_result)
                self._log_iteration_results(collaborative_iter, functional_suite, security_suite)
                
                self.logger.info("   TB fixed, re-arbitrating remaining failures...")
                continue
            
            # Step C: Fix Code if Arbiter identified code issues
            if code_issues:
                self.logger.info(f"\n-> Step C: Expert fixing code (Arbiter-routed: {len(code_issues)} issues)...")
                
                # Store Arbiter guidance for Expert using structured key (suite_type, req_index)
                self.arbiter_guidance_for_expert = {}
                for decision in code_issues:
                    # Use tuple key for reliable lookup without string parsing
                    key = (decision.suite_type, decision.req_index)
                    self.arbiter_guidance_for_expert[key] = decision.guidance
                
                # Expert fixes code with Arbiter guidance (append message with feedback)
                current_code = self._expert_fix_code(
                    task,
                    problem,
                    current_code,
                    functional_suite,
                    security_suite
                )
                
                self.expert_iterations += 1
                self.total_iterations += 1
                
                if not current_code.is_valid:
                    self.logger.error("Code fix failed after all attempts")
                    break
                
                # Save fixed code to file
                if output_dir:
                    self.expert._save_code_to_file(task, current_code, output_dir)
                
                # CRITICAL: Re-run ALL tests since Expert code changed
                self.logger.info("\n   🧪 Re-running ALL tests (code was updated)...")
                functional_suite, self.judged_functional = self._generate_and_judge_suite(
                    task, problem, current_code, "functional",
                    problem.function_requirements, self.functional_system_context,
                    output_dir
                )
                security_suite, self.judged_security = self._generate_and_judge_suite(
                    task, problem, current_code, "security",
                    problem.security_requirements, self.security_system_context,
                    output_dir
                )
            
            # Record iteration
            iteration_result = self._create_iteration_result(
                collaborative_iter, current_code, functional_suite, security_suite, task
            )
            iteration_history.append(iteration_result)
            self._log_iteration_results(collaborative_iter, functional_suite, security_suite)
            
            # Old P2 Arbiter removed - replaced by Per-Requirement Arbiter (Phase 2)
            # Per-Requirement Arbiter now analyzes EVERY failure immediately in Step A
        
        # Final check
        if self.total_iterations >= self.max_iterations:
            self.logger.warning(f"\n⚠ Reached max total iterations ({self.max_iterations})")
        
        # Final summary
        self._log_summary(iteration_history)
        
        # Save to single_gen.json if enabled
        if self.save_to_json:
            self._save_single_generation_result(
                task, 
                problem, 
                current_code, 
                iteration_history,
                architect_response
            )
        
        return current_code, iteration_history, phase0_validation
    
    def _deterministic_judge_suite(
        self,
        raw_output: str,
        reference_suite: TestSuite
    ) -> Optional[List[TestJudgment]]:
        """
        Deterministic judgment: compare output with finalized accepted_output.
        If all tests have accepted_output, no LLM judgment needed.
        
        Returns:
            List of judgments, or None if deterministic judgment not possible.
        """
        if not reference_suite.judgments_finalized:
            return None  # Not finalized, needs LLM judgment
        
        # Parse current output
        import re
        output_lines = raw_output.split('\n')
        test_outputs = {}  # {test_id: output_line}
        
        for line in output_lines:
            match = re.search(r'\[([A-Z]+-TEST-\d+)\]\s+OUTPUT\s+(.+)', line)
            if match:
                test_id = match.group(1)
                output = match.group(2).strip()
                test_outputs[test_id] = output
        
        # Compare each test
        judgments = []
        for test_case in reference_suite.test_cases:
            if not test_case.accepted_output:
                return None  # Some tests not finalized, needs LLM judgment
            
            actual_output = test_outputs.get(test_case.test_id, "")
            
            # Simple string matching (can be optimized for smarter comparison later)
            if actual_output == test_case.accepted_output:
                judgment = TestJudgment(
                    test_id=test_case.test_id,
                    verdict='PASS',
                    reason='Output matches accepted pattern'
                )
            else:
                judgment = TestJudgment(
                    test_id=test_case.test_id,
                    verdict='FAIL',
                    reason=f'Expected: {test_case.accepted_output}, Got: {actual_output}',
                    root_cause='CODE_BUG'
                )
            judgments.append(judgment)
        
        return judgments
    
    def _generate_and_judge_suite_per_requirement(
        self,
        task,
        problem,
        code,
        suite_type,
        requirements,
        system_context: str,
        output_dir: Optional[Path] = None
    ) -> Tuple[Optional[TestSuite], Optional[JudgedTestSuite]]:
        """
        NEW: Per-requirement TB architecture
        Generate one TB per requirement, then aggregate into suite
        
        Benefits:
        - Simple prompts (test 1 requirement at a time)
        - No parsing complexity
        - Precise fixes (only fix failing TBs)
        - Each requirement has independent chat history
        """
        # Check if we have previous results (for lock mechanism)
        if suite_type == "functional":
            previous_results = self.per_req_functional
        else:
            previous_results = self.per_req_security
        
        if previous_results:
            # Rerun only unlocked (failed) tests
            self.logger.info(f"Re-testing {suite_type} (only unlocked requirements)...")
            per_req_results = self.per_req_tester.rerun_unlocked_tests(
                task=task,
                code=code,
                previous_results=previous_results
            )
        else:
            # First time: generate all tests
            self.logger.info(f"Generating {suite_type} tests PER-REQUIREMENT (simplified architecture)...")
            per_req_results = self.per_req_tester.generate_per_requirement_tests(
                task=task,
                problem=problem,
                code=code,
                suite_type=suite_type,
                requirements=requirements,
                system_context=system_context,
                output_dir=output_dir
            )
        
        # Store results for next iteration
        if suite_type == "functional":
            self.per_req_functional = per_req_results
        else:
            self.per_req_security = per_req_results
        
        # Convert to suite format (for compatibility)
        return self._convert_per_req_to_suite(per_req_results, suite_type, task)
    
    def _convert_per_req_to_suite(
        self,
        per_req_results: List[SingleRequirementTest],
        suite_type: str,
        task
    ) -> Tuple[Optional[TestSuite], Optional[JudgedTestSuite]]:
        """Convert per-requirement results to suite format (compatibility layer)"""
        
        if not per_req_results:
            return None, None
        
        # Aggregate all TB codes
        all_tb_codes = "\n\n// ========================================\n\n".join([
            f"// Test for requirement {r.requirement_index}\n{r.tb_code}"
            for r in per_req_results
        ])
        
        # Aggregate all outputs
        all_outputs = "\n".join([
            f"// Requirement {r.requirement_index}: {'PASS' if r.passed else 'FAIL'}\n{r.output}"
            for r in per_req_results
        ])
        
        # Create judgments with TB explanations
        judgments = []
        for r in per_req_results:
            # Build rich reason including TB explanation
            if r.passed:
                reason = r.output[:200] if r.output else "Passed"
            else:
                # Include TB explanation for failed tests
                reason_parts = []
                if r.explanation:
                    reason_parts.append(f"Test intent: {r.explanation}")
                if r.output:
                    reason_parts.append(f"Output: {r.output[:150]}")
                elif r.error_message:
                    reason_parts.append(f"Error: {r.error_message[:150]}")
                else:
                    reason_parts.append("No output - test may have crashed")
                reason = " | ".join(reason_parts)
            
            judgments.append(TestJudgment(
                test_id=f"{suite_type.upper()}-REQ-{r.requirement_index}",
                verdict='PASS' if r.passed else 'FAIL',
                reason=reason,
                root_cause=None if r.passed else 'CODE_BUG',
                fix_tb_needed=False
            ))
        
        # Create JudgedTestSuite
        passed = sum(1 for r in per_req_results if r.passed)
        total = len(per_req_results)
        
        judged_suite = JudgedTestSuite(
            suite_type=suite_type,
            test_code=all_tb_codes,
            test_output=all_outputs,
            test_file_path=f"tb_{suite_type}_aggregated_{task.module_name}.txt",
            judgments=judgments,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            pass_rate=passed / total if total > 0 else 0.0
        )
        
        # Create legacy TestSuite manually (don't use parse_test_output for aggregated)
        # parse_test_output doesn't work well with aggregated per-req outputs
        test_cases = []
        for r in per_req_results:
            test_case = TestCaseResult(
                test_id=f"{suite_type.upper()}-REQ-{r.requirement_index}",
                test_name=f"{suite_type.capitalize()} Requirement {r.requirement_index}",
                test_description=r.requirement_text[:100],  # First 100 chars of requirement
                passed=r.passed,
                error_message=r.error_message if not r.passed else None,
                accepted_output=r.expected_output if r.locked else None
            )
            test_cases.append(test_case)
        
        test_suite = TestSuite(
            suite_type=suite_type,
            test_code=all_tb_codes,
            test_output=all_outputs,
            test_file_path=judged_suite.test_file_path,
            test_cases=test_cases,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            pass_rate=passed / total if total > 0 else 0.0,  # REQUIRED field
            execution_success=True
        )
        
        self.logger.info(f"[OK] Per-req {suite_type}: {passed}/{total} requirements passed")
        
        return test_suite, judged_suite
    
    def _generate_and_judge_suite(
        self,
        task,
        problem,
        code,
        suite_type,
        requirements,
        system_context: str,
        output_dir: Optional[Path] = None
    ) -> Tuple[Optional[TestSuite], Optional[JudgedTestSuite]]:
        """
        Generate test suite using per-requirement architecture
        One TB per requirement, simple and focused.
        Each requirement has independent chat history.
        """
        return self._generate_and_judge_suite_per_requirement(
            task, problem, code, suite_type, requirements, system_context, output_dir
        )
    
    def _extract_partial_code(self, code: str, max_lines: int = 20) -> str:
        """Extract partial relevant code snippets for Red Team judgment (not for generation)"""
        lines = code.split('\n')
        relevant_lines = []
        
        # For Verilog: extract always blocks, assign statements
        # For C: extract key logic
        in_always = False
        always_depth = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Verilog patterns
            if 'always @' in stripped or 'always_ff @' in stripped or 'always_comb' in stripped:
                in_always = True
                always_depth = 0
            
            if in_always:
                relevant_lines.append(line)
                if 'begin' in stripped:
                    always_depth += 1
                if 'end' in stripped:
                    always_depth -= 1
                    if always_depth <= 0:
                        in_always = False
            
            # Assign statements
            if stripped.startswith('assign'):
                relevant_lines.append(line)
            
            # C patterns: if statements, key logic
            if 'if' in stripped or 'else' in stripped or 'return' in stripped:
                relevant_lines.append(line)
        
        # Limit to max_lines
        if len(relevant_lines) > max_lines:
            return '\n'.join(relevant_lines[:max_lines]) + '\n... (truncated)'
        
        return '\n'.join(relevant_lines) if relevant_lines else code[:500] + '...'
    
    def _convert_judged_to_legacy(self, judged_suite: JudgedTestSuite, finalize: bool = False) -> TestSuite:
        """
        Convert JudgedTestSuite to legacy TestSuite format
        
        Args:
            judged_suite: Judged test suite
            finalize: If True and all tests pass, finalize accepted_output
        """
        # Parse output lines {test_id: output_line}
        import re
        test_outputs = {}
        for line in judged_suite.test_output.split('\n'):
            match = re.search(r'\[([A-Z]+-TEST-\d+)\]\s+OUTPUT\s+(.+)', line)
            if match:
                test_id = match.group(1)
                output = match.group(2).strip()
                test_outputs[test_id] = output
        
        test_cases = []
        all_passed = all(j.verdict == 'PASS' for j in judged_suite.judgments)
        
        for judgment in judged_suite.judgments:
            # If test passes and needs finalization, save accepted_output
            accepted_output = None
            if finalize and judgment.verdict == 'PASS':
                accepted_output = test_outputs.get(judgment.test_id)
            
            test_case = TestCaseResult(
                test_id=judgment.test_id,
                test_name=judgment.test_id,
                test_description="",
                passed=(judgment.verdict == 'PASS'),
                error_message=judgment.reason if judgment.verdict == 'FAIL' else None,
                accepted_output=accepted_output
            )
            test_cases.append(test_case)
        
        # Calculate statistics
        total_tests = len(test_cases)
        passed_tests = sum(1 for tc in test_cases if tc.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return TestSuite(
            suite_type=judged_suite.suite_type,
            test_code=judged_suite.test_code,
            test_output=judged_suite.test_output,
            test_file_path=judged_suite.test_file_path,
            test_cases=test_cases,
            total_tests=total_tests,
            passed_tests=passed_tests,
            pass_rate=pass_rate,
            execution_success=True,
            judgments_finalized=(finalize and all_passed)  # Only mark as finalized if all pass
        )
    
    def _rebuild_judged_suite(
        self,
        per_req_results: List,
        suite_type: str
    ) -> Optional[JudgedTestSuite]:
        """
        Rebuild JudgedTestSuite from per-requirement test results.
        Used after incremental requirement revision.
        """
        if not per_req_results:
            return None
        
        test_results = []
        for i, result in enumerate(per_req_results):
            if result is None:
                continue
            
            test_results.append(TestCaseResult(
                requirement_id=f"{suite_type.upper()}-REQ-{i}",
                requirement_text=result.requirement_text if hasattr(result, 'requirement_text') else "",
                passed=result.passed,
                test_output=result.test_output if hasattr(result, 'test_output') else "",
                error_message=result.error_message if hasattr(result, 'error_message') else "",
                tb_code=result.tb_code if hasattr(result, 'tb_code') else "",
                tb_explanation=result.explanation if hasattr(result, 'explanation') else "",
                locked=result.locked if hasattr(result, 'locked') else False
            ))
        
        if not test_results:
            return None
        
        return JudgedTestSuite(
            suite_type=suite_type,
            test_results=test_results
        )
    
    def _finalize_suites(
        self,
        functional_suite: Optional[TestSuite],
        security_suite: Optional[TestSuite]
    ) -> Tuple[Optional[TestSuite], Optional[TestSuite]]:
        """
        Finalize test judgments: Save passing test outputs as accepted_output
        Subsequent iterations will use deterministic judgment, no LLM needed
        """
        finalized_functional = None
        finalized_security = None
        
        if functional_suite and self.judged_functional:
            finalized_functional = self._convert_judged_to_legacy(
                self.judged_functional, finalize=True
            )
        
        if security_suite and self.judged_security:
            finalized_security = self._convert_judged_to_legacy(
                self.judged_security, finalize=True
            )
        
        return finalized_functional or functional_suite, finalized_security or security_suite
    
    def _calculate_overall_pass_rate(self) -> float:
        """Calculate overall pass rate from per-requirement results"""
        if not self.per_req_functional or not self.per_req_security:
            return 0.0
        
        total_passed = sum(1 for t in self.per_req_functional if t.passed)
        total_passed += sum(1 for t in self.per_req_security if t.passed)
        
        total_tests = len(self.per_req_functional) + len(self.per_req_security)
        
        return total_passed / total_tests if total_tests > 0 else 0.0
    
    def _check_all_tests_passed(self, functional_suite, security_suite) -> bool:
        """
        Check if all tests passed (per-requirement aware)
        Uses actual per-requirement results for accurate counting
        """
        # Use per-requirement results if available (more accurate)
        if self.per_req_functional and self.per_req_security:
            all_func_passed = all(t.passed for t in self.per_req_functional)
            all_sec_passed = all(t.passed for t in self.per_req_security)
            passed = all_func_passed and all_sec_passed
            
            if passed:
                total = len(self.per_req_functional) + len(self.per_req_security)
                self.logger.info(f"[OK] All {total} per-requirement tests passed!")
            
            return passed
        
        # Fallback to suite-level check
        functional_passed = functional_suite.passed_tests == functional_suite.total_tests if functional_suite else True
        security_passed = security_suite.passed_tests == security_suite.total_tests if security_suite else True
        return functional_passed and security_passed
    
    def _create_iteration_result(self, iteration, current_code, functional_suite, security_suite, task):
        """Create IterationResult from current state"""
        # Extract failed test cases for vulnerability details
        vulnerabilities = []
        if functional_suite:
            for tc in functional_suite.test_cases:
                if not tc.passed:
                    detail = f"[Functional] {tc.test_id}"
                    if tc.test_description:
                        detail += f" - {tc.test_description}"
                    if tc.error_message:
                        detail += f"\n  Error: {tc.error_message}"
                    vulnerabilities.append(detail)
        if security_suite:
            for tc in security_suite.test_cases:
                if not tc.passed:
                    detail = f"[Security] {tc.test_id}"
                    if tc.test_description:
                        detail += f" - {tc.test_description}"
                    if tc.error_message:
                        detail += f"\n  Error: {tc.error_message}"
                    vulnerabilities.append(detail)
        
        # Legacy test result
        security_passed = security_suite.passed_tests == security_suite.total_tests if security_suite else True
        test_result = TestCase(
            task_id=task.task_id,
            language=task.language,
            test_code=security_suite.test_code if security_suite else "",
            compilation_success=True,
            execution_success=True,
            test_output=security_suite.test_output if security_suite else "",
            vulnerability_detected=not security_passed
        )
        
        all_passed = self._check_all_tests_passed(functional_suite, security_suite)
        
        return IterationResult(
            iteration=iteration,
            code=current_code,
            test_result=test_result,
            vulnerabilities_found=len(vulnerabilities),
            vulnerability_details=vulnerabilities,
            all_tests_passed=all_passed,
            functional_suite=functional_suite,
            security_suite=security_suite
        )
    
    def _log_iteration_results(self, iteration, functional_suite, security_suite):
        """Log iteration test results"""
        all_passed = self._check_all_tests_passed(functional_suite, security_suite)
        
        self.logger.info(f"\nIteration {iteration} Results:")
        if functional_suite:
            self.logger.info(f"  Functional Tests: {functional_suite.passed_tests}/{functional_suite.total_tests} passed ({functional_suite.pass_rate:.1%})")
        if security_suite:
            self.logger.info(f"  Security Tests: {security_suite.passed_tests}/{security_suite.total_tests} passed ({security_suite.pass_rate:.1%})")
        self.logger.info(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
        
        # Log failed tests
        if not all_passed:
            failures = []
            if functional_suite:
                for tc in functional_suite.test_cases:
                    if not tc.passed:
                        failures.append(f"[FUNC] {tc.test_id}: {tc.error_message or 'No details'}")
            if security_suite:
                for tc in security_suite.test_cases:
                    if not tc.passed:
                        failures.append(f"[SEC] {tc.test_id}: {tc.error_message or 'No details'}")
            
            if failures:
                self.logger.info(f"  Failed tests ({len(failures)}):")
                for f in failures:
                    self.logger.info(f"    - {f}")
    
    def _expert_fix_code(
        self,
        task,
        problem,
        current_code,
        functional_suite,
        security_suite
    ):
        """
        Let Expert fix code based on test failures (BLIND FEEDBACK - Phase 3)
        Expert only sees: test scenarios, actual output, judgment reasons (not TB code)
        """
        self.logger.info("Building blind feedback for Expert...")
        
        # Phase 2: Include Arbiter guidance if available
        arbiter_guidance = getattr(self, 'arbiter_guidance_for_expert', None)
        if arbiter_guidance:
            self.logger.info(f"   Including Arbiter guidance for {len(arbiter_guidance)} requirements")
        
        # P0/P1.2 Integration: Use enhanced feedback builder with test intent extraction
        expert_feedback = self.enhanced_feedback.build_expert_feedback(
            judged_functional=self.judged_functional,
            judged_security=self.judged_security,
            functional_requirements=problem.function_requirements,
            security_requirements=problem.security_requirements,
            test_output_functional=functional_suite.test_output if functional_suite else "",
            test_output_security=security_suite.test_output if security_suite else "",
            # P1.2: Include test code for intent extraction
            functional_test_code=functional_suite.test_code if functional_suite else "",
            security_test_code=security_suite.test_code if security_suite else "",
            # Phase 2: Include Arbiter guidance
            arbiter_guidance=arbiter_guidance
        )
        
        # Clear Arbiter guidance after use
        if hasattr(self, 'arbiter_guidance_for_expert'):
            self.arbiter_guidance_for_expert = {}
        
        self.logger.info("[OK] Enhanced feedback constructed with debugging context and test intent")
        
        # Send blind feedback to Expert via chat history
        self.expert_chat_history.add_message(HumanMessage(content=expert_feedback))
        
        # Expert generates fix
        response = self.expert.llm.invoke(self.expert_chat_history.messages)
        
        # DISABLED: Expert feedback collection for re-arbitration
        # The Arbiter should make the final decision based on reflection
        # expert_feedbacks = self._parse_expert_feedback(response.content, arbiter_guidance)
        # if expert_feedbacks:
        #     self.logger.info(f"📋 Expert provided {len(expert_feedbacks)} feedback(s)")
        #     for fb in expert_feedbacks:
        #         if fb.feedback_type == "disagree":
        #             self.logger.warning(f"  ⚠️ Expert disagrees on {fb.requirement_id}: {fb.reasoning}")
        #         
        #         # Store feedback for potential re-arbitration
        #         if not hasattr(self, 'agent_feedbacks'):
        #             self.agent_feedbacks = []
        #         self.agent_feedbacks.append(fb)
        
        fixed_code_str = self.expert._extract_code_from_response(response.content)
        
        # Add response to history
        self.expert_chat_history.add_message(AIMessage(content=response.content))
        
        # Validation loop (max 3 attempts to fix compilation errors)
        for sub_iter in range(3):
            self.logger.info(f"  Validating fixed code (attempt {sub_iter + 1}/3)...")
            
            # Validate fixed code
            file_path = self.expert._get_code_file_path(task)
            self.expert.toolbox.write_code(file_path, fixed_code_str, task.language.value)
            
            if task.language == CodeLanguage.VERILOG:
                result = self.expert.toolbox.validate_verilog_module(file_path, task.module_name)
            else:
                result = self.expert.toolbox.validate_c_function(file_path)
            
            if result.success:
                self.logger.info(f"  [OK] Fixed code validated successfully")
                
                # Context compression: Remove compilation failure messages if any retries happened
                if sub_iter > 0:
                    # Find where the fix feedback starts
                    feedback_idx = None
                    for i in range(len(self.expert_chat_history.messages) - 1, -1, -1):
                        if "Test Failures" in self.expert_chat_history.messages[i].content or \
                           "Arbiter Analysis" in self.expert_chat_history.messages[i].content:
                            feedback_idx = i
                            break
                    
                    if feedback_idx is not None:
                        # Keep everything up to feedback + final successful response
                        messages_to_keep = self.expert_chat_history.messages[:feedback_idx + 1]
                        # Add the final successful AI response (it's already in history)
                        messages_to_keep.append(self.expert_chat_history.messages[-1])
                        self.expert_chat_history.messages = messages_to_keep
                        self.logger.info(f"  🗜️ Context compressed: Removed {sub_iter} failed compilation attempts")
                
                return GeneratedCode(
                    task_id=task.task_id,
                    language=task.language,
                    code=fixed_code_str,
                    module_or_function_name=task.module_name or task.function_name,
                    is_valid=True,
                    validation_errors=None,
                    compilation_warnings=result.warnings if hasattr(result, 'warnings') else None
                )
            
            # If validation failed and we have more attempts, fix compilation errors
            if sub_iter < 2:
                self.logger.warning(f"  [FAIL] Validation failed, attempting to fix compilation errors...")
                
                # Send compilation error to Expert via chat history
                compile_error_msg = f"""# Compilation Errors

```
{result.stderr}
```

# Your Task
Fix the compilation errors in your code. Only fix syntax issues, preserve all logic and security features.

Output ONLY the corrected code."""
                
                self.expert_chat_history.add_message(HumanMessage(content=compile_error_msg))
                response = self.expert.llm.invoke(self.expert_chat_history.messages)
                fixed_code_str = self.expert._extract_code_from_response(response.content)
                self.expert_chat_history.add_message(AIMessage(content=response.content))
        
        # Failed after all attempts
        self.logger.error(f"  [FAIL] Code validation failed after 3 attempts")
        return GeneratedCode(
            task_id=task.task_id,
            language=task.language,
            code=fixed_code_str,
            module_or_function_name=task.module_name or task.function_name,
            is_valid=False,
            validation_errors=result.stderr,
            compilation_warnings=None
        )
    
    def _try_compile_code(
        self,
        task: GenerationTask,
        code_str: str
    ) -> GeneratedCode:
        """
        Try to compile code and return GeneratedCode object
        
        Args:
            task: Current task
            code_str: Code string to compile
        
        Returns:
            GeneratedCode object (is_valid indicates success)
        """
        # Write code to file
        file_path = self.expert._get_code_file_path(task)
        self.expert.toolbox.write_code(file_path, code_str, task.language.value)
        
        # Validate
        if task.language == CodeLanguage.VERILOG:
            result = self.expert.toolbox.validate_verilog_module(file_path, task.module_name)
        else:
            result = self.expert.toolbox.validate_c_function(file_path)
        
        if result.success:
            self.logger.info(f"  [OK] Code compiled successfully")
            return GeneratedCode(
                task_id=task.task_id,
                language=task.language,
                code=code_str,
                module_or_function_name=task.module_name or task.function_name,
                is_valid=True,
                validation_errors=None,
                compilation_warnings=result.warnings if hasattr(result, 'warnings') else None
            )
        else:
            self.logger.warning(f"  [FAIL] Code compilation failed: {result.stderr[:200]}...")
            return GeneratedCode(
                task_id=task.task_id,
                language=task.language,
                code=code_str,
                module_or_function_name=task.module_name or task.function_name,
                is_valid=False,
                validation_errors=result.stderr,
                compilation_warnings=None
            )
    
    def _fix_code_with_feedback(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        current_code: GeneratedCode,
        vulnerabilities: List[str],
        test_output: str = "",
        max_sub_iterations: int = 3
    ) -> GeneratedCode:
        """
        Let Expert fix code based on Red Team feedback (with chat history and validation loop)
        """
        # Create incremental fix prompt (only failures, Expert has context from history)
        fix_message = self._create_incremental_fix_message(
            current_code, vulnerabilities, test_output
        )
        
        # Add user message to history
        self.expert_chat_history.add_message(HumanMessage(content=fix_message))
        
        # Use Expert's LLM with full chat history
        messages = self.expert_chat_history.messages
        response = self.expert.llm.invoke(messages)
        fixed_code_str = self.expert._extract_code_from_response(response.content)
        
        # Add AI response to history
        self.expert_chat_history.add_message(AIMessage(content=response.content))
        
        # Validation loop: try to fix compilation errors
        for sub_iter in range(max_sub_iterations):
            self.logger.info(f"  Validating fixed code (attempt {sub_iter + 1}/{max_sub_iterations})...")
            
            # Validate fixed code
            file_path = self.expert._get_code_file_path(task)
            self.expert.toolbox.write_code(file_path, fixed_code_str, task.language.value)
            
            if task.language == CodeLanguage.VERILOG:
                result = self.expert.toolbox.validate_verilog_module(file_path, task.module_name)
            else:
                result = self.expert.toolbox.validate_c_function(file_path)
            
            if result.success:
                self.logger.info(f"  [OK] Fixed code validated successfully")
                return GeneratedCode(
                    task_id=task.task_id,
                    language=task.language,
                    code=fixed_code_str,
                    module_or_function_name=task.module_name or task.function_name,
                    is_valid=True,
                    validation_errors=None,
                    compilation_warnings=result.warnings if hasattr(result, 'warnings') else None
                )
            
            # If validation failed and we have more attempts, fix compilation errors
            if sub_iter < max_sub_iterations - 1:
                self.logger.warning(f"  [FAIL] Validation failed, attempting to fix compilation errors...")
                
                # Send compilation error to Expert via chat history
                compile_error_msg = f"""# Compilation Errors

```
{result.stderr}
```

# Your Task
Fix the compilation errors in your code. Only fix syntax issues, preserve all logic and security features.

Output ONLY the corrected code."""
                
                self.expert_chat_history.add_message(HumanMessage(content=compile_error_msg))
                response = self.expert.llm.invoke(self.expert_chat_history.messages)
                fixed_code_str = self.expert._extract_code_from_response(response.content)
                self.expert_chat_history.add_message(AIMessage(content=response.content))
        
        # Failed after all attempts
        self.logger.error(f"  [FAIL] Code validation failed after {max_sub_iterations} attempts")
        return GeneratedCode(
            task_id=task.task_id,
            language=task.language,
            code=fixed_code_str,
            module_or_function_name=task.module_name or task.function_name,
            is_valid=False,
            validation_errors=result.stderr,
            compilation_warnings=None
        )
    
    def _create_fix_prompt(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        current_code: GeneratedCode,
        vulnerabilities: List[str],
        test_output: str = ""
    ) -> str:
        """Create prompt for fixing code based on vulnerabilities"""
        
        vuln_list = "\n".join(f"{i+1}. {v}" for i, v in enumerate(vulnerabilities))
        
        test_output_section = ""
        if test_output:
            test_output_section = f"""
## Test Output
```
{test_output}
```
"""
        
        sec_reqs = "\n".join(f'- {req}' for req in problem.security_requirements) if problem.security_requirements else '- (None)'
        
        prompt = f"""# Fix Code Based on Red Team Feedback

## Current Code
```{task.language.value}
{current_code.code}
```

## Failed Tests
{vuln_list}
{test_output_section}
## Security Requirements
{sec_reqs}

## Task
Fix ALL failures above. Maintain functional correctness and keep code compilable.

Output ONLY the complete fixed code.
"""
        return prompt
    
    def _extract_vulnerabilities(self, test_result: TestCase) -> List[str]:
        """Extract vulnerability details from test output"""
        if not test_result.test_output:
            return []
        
        vulnerabilities = []
        for line in test_result.test_output.splitlines():
            if "VULNERABILITY DETECTED" in line or "FAIL:" in line:
                # Clean up the line
                vuln = line.strip()
                if vuln and vuln not in vulnerabilities:
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _log_summary(self, history: List[IterationResult]):
        """Log summary of all iterations"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("ITERATIVE GENERATION SUMMARY")
        self.logger.info(f"{'='*60}")
        
        if not history:
            self.logger.error("No iterations completed successfully")
            return
        
        for result in history:
            status = "[OK] PASS" if result.all_tests_passed else f"[FAIL] {result.vulnerabilities_found} issues"
            self.logger.info(f"Iteration {result.iteration}: {status}")
        
        final_result = history[-1]
        if final_result.all_tests_passed:
            self.logger.info(f"\nFinal Status: SECURE (all tests passed)")
        else:
            self.logger.info(f"\nFinal Status: {final_result.vulnerabilities_found} vulnerabilities remaining")
    
    def _save_single_generation_result(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        final_code: GeneratedCode,
        iteration_history: List[IterationResult],
        architect_response: Optional[str]
    ):
        """Save complete generation result to single_gen.json"""
        
        # Convert IterationResult to SingleIterationRecord
        single_iterations = []
        for iter_result in iteration_history:
            single_iter = SingleIterationRecord(
                iteration_number=iter_result.iteration,
                expert_response=f"Generated/fixed code for iteration {iter_result.iteration}",
                generated_code=iter_result.code,
                red_teamer_response=f"Test result: {iter_result.vulnerabilities_found} vulnerabilities",
                functional_test_suite=iter_result.functional_suite,
                security_test_suite=iter_result.security_suite,
                vulnerabilities_found=iter_result.vulnerabilities_found,
                vulnerability_details=iter_result.vulnerability_details,
                all_tests_passed=iter_result.all_tests_passed
            )
            single_iterations.append(single_iter)
        
        # Get final test suites from last iteration
        final_iter = iteration_history[-1] if iteration_history else None
        final_functional_suite = final_iter.functional_suite if final_iter else None
        final_security_suite = final_iter.security_suite if final_iter else None
        
        # Calculate detailed test statistics
        total_functional = final_functional_suite.total_tests if final_functional_suite else 0
        passed_functional = final_functional_suite.passed_tests if final_functional_suite else 0
        functional_rate = final_functional_suite.pass_rate if final_functional_suite else 0.0
        
        total_security = final_security_suite.total_tests if final_security_suite else 0
        passed_security = final_security_suite.passed_tests if final_security_suite else 0
        security_rate = final_security_suite.pass_rate if final_security_suite else 0.0
        
        # Get file paths
        code_file_path = str(self.expert._get_code_file_path(task))
        
        # Per-requirement architecture: collect all TB file paths
        functional_test_files = []
        security_test_files = []
        if self.per_req_functional:
            functional_test_files = [t.tb_file for t in self.per_req_functional if t.tb_file]
        if self.per_req_security:
            security_test_files = [t.tb_file for t in self.per_req_security if t.tb_file]
        
        # Use first file as representative path for backward compatibility
        functional_test_path = functional_test_files[0] if functional_test_files else None
        security_test_path = security_test_files[0] if security_test_files else None
        
        # Create SingleGenerationResult
        problem_title = problem.question.split('\n')[0].strip() if problem.question else "Unknown"
        result = SingleGenerationResult(
            task=task,
            architect_response=architect_response or f"Generated problem: {problem_title}",
            problem_description=problem,
            iterations=single_iterations,
            final_code=final_code,
            final_functional_test_suite=final_functional_suite,
            final_security_test_suite=final_security_suite,
            code_file_path=code_file_path,
            functional_test_file_path=functional_test_path,
            security_test_file_path=security_test_path,
            total_iterations=len(iteration_history),
            generation_successful=final_iter.all_tests_passed if final_iter else False,
            total_functional_tests=total_functional,
            passed_functional_tests=passed_functional,
            functional_pass_rate=functional_rate,
            total_security_tests=total_security,
            passed_security_tests=passed_security,
            security_pass_rate=security_rate
        )
        
        # Save to JSON
        output_file = config.OUTPUT_DIR / "single_gen.json"
        self.data_loader.save_json(
            result.model_dump(),
            output_file
        )
        
        self.logger.info(f"Saved generation result to: {output_file}")
        self.logger.info(f"Generation UUID: {result.generation_uuid}")
    
    def _create_initial_context(
        self,
        task: GenerationTask,
        problem: ProblemDescription
    ) -> str:
        """
        Create initial system context for Expert (sent once at beginning)
        This establishes the problem domain and requirements that persist across iterations.
        """
        function_reqs_text = "\n".join(f"- {req}" for req in problem.function_requirements)
        security_reqs_text = "\n".join(f"- {req}" for req in problem.security_requirements) if problem.security_requirements else "- (No explicit security requirements)"
        
        context = f"""You are a security-focused code expert. Your task is to implement SECURE code.

# Your Role

You will implement and iteratively fix code based on test feedback. In this conversation:
1. First, you'll generate initial code (handled separately)
2. Then, you'll receive test failure reports
3. You must analyze failures and fix the code
4. This conversation preserves your context - learn from previous attempts!

Remember all requirements above throughout our conversation. When you receive test failures, 
analyze them carefully against these requirements and fix the root cause."""

        return context
    
    def _create_test_initial_context(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        suite_type: str
    ) -> str:
        """
        Create initial system context for Red Team test generation (sent once at beginning)
        This establishes what to test and how.
        """
        if suite_type == "functional":
            reqs_text = "\n".join(f"- {req}" for req in problem.function_requirements)
            context = f"""You are a functional test engineer. Your task is to generate tests that verify functional correctness.

# Your Role

You will generate and iteratively fix functional test code based on compilation/execution feedback. In this conversation:
1. First, you'll generate initial test code (handled separately)
2. Then, you'll receive compilation/execution error reports
3. You must analyze errors and fix the test code
4. This conversation preserves your context - learn from previous attempts!

Remember: Test ONLY the functional requirements listed above. Generate 1-2 tests per requirement."""
        
        else:  # security
            sec_reqs_text = "\n".join(f"- {req}" for req in problem.security_requirements) if problem.security_requirements else "- (No explicit security requirements - test for CWE vulnerability)"
            context = f"""You are a security test engineer. Your task is to generate tests that verify security measures ARE implemented.

# Problem Context

**Question**: {problem.question}

**Security Requirements to Test** (These describe SECURE behavior that SHOULD be implemented):
{sec_reqs_text}

**Target CWE**: CWE-{task.cwe_id} - {task.cwe_name}

**Interface**:
- Inputs: {problem.input_specification}
- Outputs: {problem.output_specification}

# Your Role

You will generate tests that verify the implementation HAS the security measures described in the security requirements.
- Security requirements describe SECURE behavior (access control, authentication, data protection)
- Your tests should PASS if the security measures ARE implemented
- Your tests should FAIL if the security measures are MISSING (vulnerable code)

In this conversation:
1. First, you'll generate initial test code (handled separately)
2. Then, you'll receive compilation/execution error reports
3. You must analyze errors and fix the test code
4. This conversation preserves your context - learn from previous attempts!

Remember: Test that the security requirements ARE implemented. Generate 1-2 tests per requirement."""
        
        return context
    
    def _create_incremental_fix_message(
        self,
        current_code: GeneratedCode,
        vulnerabilities: List[str],
        test_output: str = ""
    ) -> str:
        """
        Create incremental fix message (only contains failures, Expert has context from history)
        """
        vuln_list = "\n".join(f"{i+1}. {v}" for i, v in enumerate(vulnerabilities))
        
        test_output_section = ""
        if test_output:
            test_output_section = f"""
## Test Output
```
{test_output}
```
"""
        
        message = f"""# Test Failures

{vuln_list}
{test_output_section}
Fix ALL failures above. Output ONLY the complete fixed code."""

        return message
    
    def _arbiter_analyze_failures(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        code: GeneratedCode,
        iteration: int
    ) -> List[ArbiterDecision]:
        """
        Arbiter analyzes all failed requirements (Phase 2: Full deployment)
        
        Returns:
            List of ArbiterDecision for each failed requirement
        """
        decisions = []
        
        # Analyze failed functional requirements
        if self.per_req_functional:
            for req_test in self.per_req_functional:
                if not req_test.passed:
                    self.logger.info(f"   Analyzing FUNCTIONAL-REQ-{req_test.requirement_index}...")
                    decision = self.arbiter.analyze_failure(
                        task_id=task.task_id,
                        req_index=req_test.requirement_index,
                        requirement_text=req_test.requirement_text,
                        suite_type="functional",
                        code=code.code,
                        tb_code=req_test.tb_code,
                        tb_explanation=req_test.explanation or "",
                        test_output=req_test.output,
                        error_message=req_test.error_message or "",
                        iteration=iteration,
                        language=task.language.value,
                        input_spec=problem.input_specification,
                        output_spec=problem.output_specification,
                        all_functional_requirements=problem.function_requirements,
                        all_security_requirements=problem.security_requirements,
                        question=problem.question
                    )
                    decisions.append(decision)
                else:
                    # Mark successful tests in Arbiter memory
                    self.arbiter.mark_success(
                        task_id=task.task_id,
                        req_index=req_test.requirement_index,
                        suite_type="functional",
                        iteration=iteration
                    )
        
        # Analyze failed security requirements
        if self.per_req_security:
            for req_test in self.per_req_security:
                if not req_test.passed:
                    self.logger.info(f"   Analyzing SECURITY-REQ-{req_test.requirement_index}...")
                    decision = self.arbiter.analyze_failure(
                        task_id=task.task_id,
                        req_index=req_test.requirement_index,
                        requirement_text=req_test.requirement_text,
                        suite_type="security",
                        code=code.code,
                        tb_code=req_test.tb_code,
                        tb_explanation=req_test.explanation or "",
                        test_output=req_test.output,
                        error_message=req_test.error_message or "",
                        iteration=iteration,
                        language=task.language.value,
                        input_spec=problem.input_specification,
                        output_spec=problem.output_specification,
                        all_functional_requirements=problem.function_requirements,
                        all_security_requirements=problem.security_requirements,
                        question=problem.question
                    )
                    decisions.append(decision)
                else:
                    # Mark successful tests in Arbiter memory
                    self.arbiter.mark_success(
                        task_id=task.task_id,
                        req_index=req_test.requirement_index,
                        suite_type="security",
                        iteration=iteration
                    )
        
        return decisions
    
    def _fix_tb_with_arbiter_guidance(
        self,
        task: GenerationTask,
        problem: ProblemDescription,
        code: GeneratedCode,
        decision: ArbiterDecision,
        max_fix_attempts: int = 2
    ):
        """
        Fix a single TB based on Arbiter guidance with retry logic
        
        Args:
            max_fix_attempts: Maximum number of fix attempts before giving up
        """
        suite_type = decision.suite_type
        req_index = decision.req_index
        
        if not suite_type or req_index < 0:
            self.logger.error(f"Invalid decision routing info: suite_type={suite_type}, req_index={req_index}")
            return
        
        if suite_type == "functional" and self.per_req_functional:
            req_tests = self.per_req_functional
        elif suite_type == "security" and self.per_req_security:
            req_tests = self.per_req_security
        else:
            return
        
        current_test = next((t for t in req_tests if t.requirement_index == req_index), None)
        if not current_test:
            return
        
        self.logger.info(f"      Fixing TB for {decision.requirement_id}...")
        self.logger.info(f"      Arbiter guidance:\n{decision.guidance}")
        
        # Copy guidance to avoid modifying the original decision object
        current_guidance = decision.guidance
        
        # Try to fix TB with retry logic
        for attempt in range(1, max_fix_attempts + 1):
            fixed_test, agent_feedback = self.per_req_tester.fix_single_requirement_tb(
                task=task,
                code=code,
                suite_type=suite_type,
                req_index=req_index,
                requirement_text=current_test.requirement_text,
                arbiter_guidance=current_guidance
            )
            
            if fixed_test:
                current_test.tb_code = fixed_test.tb_code
                current_test.output = fixed_test.output
                current_test.passed = fixed_test.passed
                current_test.error_message = fixed_test.error_message
                current_test.explanation = fixed_test.explanation
                current_test.locked = fixed_test.locked
                
                if fixed_test.passed:
                    self.logger.info(f"      [OK] TB fixed and passes (attempt {attempt})")
                    return
                else:
                    self.logger.warning(f"      [FAIL] Fixed TB still fails (attempt {attempt}/{max_fix_attempts})")
                    
                    if attempt < max_fix_attempts:
                        self.logger.info(f"      Retrying TB fix with additional context...")
                        current_guidance += f"\n\n**Previous Fix Attempt Failed**:\nThe previous fix still resulted in test failure. Output:\n{fixed_test.output[:500]}\n\nPlease try a different approach."
                    else:
                        self.logger.warning(f"      Max fix attempts reached.")
            else:
                self.logger.warning(f"      No fixed TB returned (attempt {attempt})")
                if attempt >= max_fix_attempts:
                    break
    
    def _problem_changed(
        self,
        old: ProblemDescription,
        new: ProblemDescription
    ) -> bool:
        """
        Check if Problem has substantial changes
        
        Returns:
            True if there are meaningful differences
        """
        return (
            old.question != new.question or
            old.input_specification != new.input_specification or
            old.output_specification != new.output_specification or
            old.function_requirements != new.function_requirements or
            old.security_requirements != new.security_requirements
        )
    
    def _log_problem_changes(
        self,
        old: ProblemDescription,
        new: ProblemDescription
    ):
        """Log what changed in the Problem"""
        changes = []
        
        if old.question != new.question:
            changes.append("Question/Description")
        
        if old.input_specification != new.input_specification:
            changes.append(f"Input spec: {old.input_specification} -> {new.input_specification}")
        
        if old.output_specification != new.output_specification:
            changes.append(f"Output spec: {old.output_specification} -> {new.output_specification}")
        
        if old.function_requirements != new.function_requirements:
            old_count = len(old.function_requirements)
            new_count = len(new.function_requirements)
            if old_count != new_count:
                changes.append(f"Functional requirements count: {old_count} -> {new_count}")
            else:
                for i, (old_req, new_req) in enumerate(zip(old.function_requirements, new.function_requirements)):
                    if old_req != new_req:
                        changes.append(f"FUNCTIONAL-REQ-{i} modified")
        
        if old.security_requirements != new.security_requirements:
            old_count = len(old.security_requirements)
            new_count = len(new.security_requirements)
            if old_count != new_count:
                changes.append(f"Security requirements count: {old_count} -> {new_count}")
            else:
                for i, (old_req, new_req) in enumerate(zip(old.security_requirements, new.security_requirements)):
                    if old_req != new_req:
                        changes.append(f"SECURITY-REQ-{i} modified")
        
        if changes:
            self.logger.info(f"   Changes detected:")
            for change in changes:
                self.logger.info(f"     - {change}")
        else:
            self.logger.info(f"   No substantial changes detected")
    
    def _get_failed_tests(self) -> list:
        """
        Collect information about failed tests for Architect context
        
        Returns:
            List of dicts with failed test information
        """
        failed = []
        
        if self.per_req_functional:
            for test in self.per_req_functional:
                if not test.passed:
                    failed.append({
                        "requirement_id": f"FUNCTIONAL-REQ-{test.requirement_index}",
                        "requirement_text": test.requirement_text,
                        "error": test.error_message or test.output[:200],
                        "suite_type": "functional"
                    })
        
        if self.per_req_security:
            for test in self.per_req_security:
                if not test.passed:
                    failed.append({
                        "requirement_id": f"SECURITY-REQ-{test.requirement_index}",
                        "requirement_text": test.requirement_text,
                        "error": test.error_message or test.output[:200],
                        "suite_type": "security"
                    })
        
        return failed
    
    def _create_architect_initial_context(
        self,
        task: GenerationTask,
        problem: ProblemDescription
    ) -> str:
        """
        Create initial system context for Architect (sent once at beginning)
        This establishes the Architect's role in requirement clarification.
        """
        func_reqs = "\n".join(f"- FUNCTIONAL-REQ-{i}: {req}" for i, req in enumerate(problem.function_requirements))
        sec_reqs = "\n".join(f"- SECURITY-REQ-{i}: {req}" for i, req in enumerate(problem.security_requirements))
        
        context = f"""You are the Architect who designed the requirements for this hardware/software module.

# Problem Context

**Question**: {problem.question}

**Functional Requirements**:
{func_reqs}

**Security Requirements**:
{sec_reqs}

**Interface**:
- Inputs: {problem.input_specification}
- Outputs: {problem.output_specification}

# Your Role

When the Arbiter identifies that a requirement is unclear or ambiguous, you will be asked to clarify it.
Your clarifications should:
1. **Preserve the original intent** - Don't change what the requirement is trying to achieve
2. **Add precision** - Specify timing, conditions, edge cases that were missing
3. **Resolve ambiguity** - Make it clear enough that both Expert and RedTeam interpret it the same way

This conversation preserves your context - you can refer to previous clarifications you've made."""

        return context
