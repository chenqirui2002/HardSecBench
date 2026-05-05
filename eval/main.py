#!/usr/bin/env python3
"""
HardSecBench Evaluation System
Evaluate LLM models on hardware security benchmark
"""
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add eval directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from core.data_loader import DataLoader
from core.eval_orchestrator import EvalOrchestrator
from utils.logger import setup_logging, get_logger
from utils.output_manager import OutputManager
from utils.code_deduplication import (
    group_runs_by_code,
    select_representative_runs,
    get_deduplication_stats
)


# Thread-safe counter for progress tracking
class ProgressCounter:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self._lock = Lock()
    
    def increment(self, success: bool):
        with self._lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            return self.completed, self.successful, self.failed


def log_pass_at_k_progress(summary, k, completed_seeds, realtime=False):
    """
    Log Pass@k progress during evaluation
    
    Args:
        summary: Summary dict from compute_pass_at_k_statistics
        k: Requested k value
        completed_seeds: Number of seeds completed so far
        realtime: Whether this is a real-time update
    """
    from utils.logger import get_logger
    progress_logger = get_logger()
    
    pass_at_1 = summary.get('pass_at_1', {})
    pass_at_k = summary.get('pass_at_k', {})
    
    mode_str = "[REALTIME]" if realtime else "[FINAL]"
    progress_logger.info("\n" + "="*60)
    progress_logger.info(f"{mode_str} Pass@k Progress (Completed Seeds: {completed_seeds})")
    progress_logger.info("="*60)
    progress_logger.info(f"Pass@{k} Functional: {pass_at_k.get('seed_functional_pass_rate', 0):.1%}")
    progress_logger.info(f"Pass@{k} Security: {pass_at_k.get('seed_security_pass_rate', 0):.1%}")
    progress_logger.info(f"Pass@1 Functional: {pass_at_1.get('seed_functional_pass_rate', 0):.1%}")
    progress_logger.info(f"Pass@1 Security: {pass_at_1.get('seed_security_pass_rate', 0):.1%}")
    progress_logger.info(f"Avg Functional Rate (Pass@{k}): {pass_at_k.get('avg_functional_pass_rate', 0):.1%}")
    progress_logger.info(f"Avg Security Rate (Pass@{k}): {pass_at_k.get('avg_security_pass_rate', 0):.1%}")
    progress_logger.info("="*60)


def evaluate_single_case(test_case, args, output_manager, counter, logger, cwe_info):
    """
    Evaluate a single test case (for thread pool, quiet mode)
    """
    try:
        orchestrator = EvalOrchestrator(
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            output_manager=output_manager,
            api_key=args.api_key or config.OPENAI_API_KEY,
            base_url=args.base_url or config.OPENAI_BASE_URL,
            quiet=True,
            security_hint_level=args.security_hint,
            cwe_info=cwe_info,
            max_timeout=args.max_timeout,
            generate_only=args.generate_only
        )
        
        result = orchestrator.evaluate_test_case(test_case)
        completed, successful, failed = counter.increment(result.success)
        
        logger.info(f"[{completed}/{counter.total}] {test_case.uuid}: {'FUNC PASS' if result.success else 'FUNC FAIL'}")
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating {test_case.uuid}: {e}")
        import traceback
        traceback.print_exc()
        counter.increment(False)
        return None


def evaluate_single_case_with_run_index(
    test_case, run_index, args, output_manager, counter, logger, cwe_info
):
    """
    Evaluate a single test case with run index (for Pass@k mode)
    
    Args:
        test_case: Test case to evaluate
        run_index: Index of this run (0-based)
        args: Command line arguments
        output_manager: Output manager
        counter: Progress counter
        logger: Logger
        cwe_info: CWE information dict
    """
    try:
        orchestrator = EvalOrchestrator(
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            output_manager=output_manager,
            api_key=args.api_key or config.OPENAI_API_KEY,
            base_url=args.base_url or config.OPENAI_BASE_URL,
            quiet=True,
            security_hint_level=args.security_hint,
            cwe_info=cwe_info,
            max_timeout=args.max_timeout,
            generate_only=args.generate_only
        )
        
        result = orchestrator.evaluate_test_case(test_case, run_index=run_index)
        completed, successful, failed = counter.increment(result.success)
        
        logger.info(f"[{completed}/{counter.total}] {test_case.uuid} (run {run_index + 1}): {'FUNC PASS' if result.success else 'FUNC FAIL'}")
        
        # Real-time Pass@k statistics: check if this seed just completed all k runs
        # This is safe because we only include seeds with all runs completed (including run_0)
        if args.pass_at_k > 1:
            # Check if this seed just completed all k runs
            if (result.uuid in output_manager._completed_seeds and
                len(output_manager._completed_runs_by_seed.get(result.uuid, set())) == args.pass_at_k):
                # This seed just completed all k runs, trigger real-time statistics
                output_manager.compute_pass_at_k_statistics(args.pass_at_k, log_pass_at_k_progress, realtime=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating {test_case.uuid} (run {run_index + 1}): {e}")
        import traceback
        traceback.print_exc()
        counter.increment(False)
        
        # CRITICAL FIX: Create and save a failed result instead of returning None
        # This ensures the run appears in results even if it failed with an exception
        from models.eval_models import EvalResult
        from datetime import datetime
        
        failed_result = EvalResult(
            uuid=test_case.uuid,
            cwe_id=test_case.cwe_id,
            language=test_case.language,
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            run_index=run_index,
            iterations=[],
            total_iterations=0,
            success=False,
            end_time=datetime.now().isoformat(),
            error=str(e)
        )
        
        # Save the failed result
        output_manager.add_result(failed_result)
        
        return failed_result


def evaluate_single_case_with_run_index_preloaded(
    test_case, run_index, preloaded_context, args, output_manager, counter, logger, cwe_info
):
    """
    Evaluate a single test case with run index and preloaded context (for Pass@k with deduplication)
    
    Args:
        test_case: Test case to evaluate
        run_index: Index of this run (0-based)
        preloaded_context: Preloaded initial context (chat history)
        args: Command line arguments
        output_manager: Output manager
        counter: Progress counter
        logger: Logger
        cwe_info: CWE information dict
    """
    try:
        orchestrator = EvalOrchestrator(
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            output_manager=output_manager,
            api_key=args.api_key or config.OPENAI_API_KEY,
            base_url=args.base_url or config.OPENAI_BASE_URL,
            quiet=True,
            security_hint_level=args.security_hint,
            cwe_info=cwe_info,
            max_timeout=args.max_timeout,
            generate_only=args.generate_only
        )
        
        result = orchestrator.evaluate_test_case(test_case, run_index=run_index, preloaded_context=preloaded_context)
        completed, successful, failed = counter.increment(result.success)
        
        logger.info(f"[{completed}/{counter.total}] {test_case.uuid} (run {run_index + 1}): {'FUNC PASS' if result.success else 'FUNC FAIL'}")
        
        # CRITICAL FIX: Immediately copy result to duplicate runs (for code deduplication)
        # This ensures real-time statistics include all runs, not just representatives
        if args.deduplicate_code and args.pass_at_k > 1:
            result_dict = result.to_dict()
            output_manager.copy_result_to_duplicates_immediately(result_dict)
        
        # Real-time Pass@k statistics: check if this seed just completed all k runs
        # This is safe because we only include seeds with all runs completed (including run_0)
        if args.pass_at_k > 1:
            # Check if this seed just completed all k runs
            if (result.uuid in output_manager._completed_seeds and
                len(output_manager._completed_runs_by_seed.get(result.uuid, set())) == args.pass_at_k):
                # This seed just completed all k runs, trigger real-time statistics
                output_manager.compute_pass_at_k_statistics(args.pass_at_k, log_pass_at_k_progress, realtime=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating {test_case.uuid} (run {run_index + 1}): {e}")
        import traceback
        traceback.print_exc()
        counter.increment(False)
        
        # CRITICAL FIX: Create and save a failed result instead of returning None
        # This ensures the run appears in results even if it failed with an exception
        from models.eval_models import EvalResult
        from datetime import datetime
        
        failed_result = EvalResult(
            uuid=test_case.uuid,
            cwe_id=test_case.cwe_id,
            language=test_case.language,
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            run_index=run_index,
            iterations=[],
            total_iterations=0,
            success=False,
            end_time=datetime.now().isoformat(),
            error=str(e)
        )
        
        # Save the failed result
        output_manager.add_result(failed_result)
        
        return failed_result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HardSecBench Evaluation System"
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of the model to evaluate (e.g., gpt-4o, claude-sonnet-4-5-20250929)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to test data directory (containing result.json)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Temperature for the target LLM (default: 0.2)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations per test case (default: 5)'
    )
    
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        help='Maximum attempts for TB fixing per iteration (default: 3)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for target LLM (optional, uses env if not provided)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Base URL for target LLM API (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: eval/outputs)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of test cases to evaluate (for testing)'
    )
    
    parser.add_argument(
        '--batchall',
        action='store_true',
        help='Use batchall format result.json (with cwes structure)'
    )
    
    parser.add_argument(
        '--filter-report',
        type=str,
        default=None,
        help='Path to filter_report.json for testbench quality filtering (only for batchall mode)'
    )
    
    parser.add_argument(
        '--gen-workers',
        type=int,
        default=4,
        help='Number of parallel workers for code generation phase (default: 4)'
    )
    
    parser.add_argument(
        '--eval-workers',
        type=int,
        default=4,
        help='Number of parallel workers for evaluation phase (default: 4)'
    )
    
    # Keep --workers for backward compatibility
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='(Deprecated) Use --gen-workers and --eval-workers instead. If set, applies to both phases.'
    )
    
    parser.add_argument(
        '--security-hint',
        type=int,
        choices=[0, 1, 2],
        default=0,
        help='Security hint level: 0=none, 1=general, 2=cwe (default: 0)'
    )
    
    parser.add_argument(
        '--max-timeout',
        type=int,
        default=120,
        help='Max timeout for API requests in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--pass-at-k',
        type=int,
        default=1,
        help='Number of independent runs per test case for Pass@k evaluation (default: 1)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        choices=['c', 'verilog'],
        help='Filter test cases by language (c or verilog). If not specified, all languages are included.'
    )
    
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Generate only mode: only generate code once without iteration. Useful for small models to avoid context explosion.'
    )
    
    parser.add_argument(
        '--deduplicate-code',
        action='store_true',
        help='Enable code deduplication: group identical code (ignoring whitespace) and only evaluate once per group. Significantly improves efficiency for Pass@k evaluation.'
    )
    
    args = parser.parse_args()
    
    # Handle backward compatibility for --workers
    if args.workers is not None:
        if args.gen_workers == 4:  # Default value not changed
            args.gen_workers = args.workers
        if args.eval_workers == 4:  # Default value not changed
            args.eval_workers = args.workers
    
    # Setup logging
    # Use separate directory for Pass@k mode
    if args.pass_at_k > 1:
        base_output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
        output_dir = base_output_dir.parent / "outputs_passk"
    else:
        output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    
    setup_logging(output_dir / "logs")
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("HardSecBench Evaluation System")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Max Iterations: {args.max_iterations}")
    logger.info(f"Max Attempts: {args.max_attempts}")
    logger.info(f"Security Hint Level: {args.security_hint}")
    logger.info(f"Generate Only Mode: {args.generate_only}")
    if args.pass_at_k > 1:
        logger.info(f"Pass@k Mode: k={args.pass_at_k}")
    
    # Load test cases
    logger.info("\nLoading test cases...")
    data_loader = DataLoader()
    
    try:
        if args.batchall:
            test_cases, cwe_info = data_loader.load_batchall_test_cases(
                args.data_path,
                filter_report_path=args.filter_report
            )
        else:
            test_cases, cwe_info = data_loader.load_test_cases(args.data_path)
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        sys.exit(1)
    
    if not test_cases:
        logger.error("No test cases found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    # Apply language filter if specified
    if args.language:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.language.lower() == args.language.lower()]
        logger.info(f"Language filter '{args.language}': {len(test_cases)}/{original_count} test cases")
    
    # Apply limit if specified
    if args.limit:
        test_cases = test_cases[:args.limit]
        logger.info(f"Limited to {len(test_cases)} test cases")
    
    # Initialize output manager
    output_manager = OutputManager(
        output_dir, 
        eval_name=args.model,
        security_hint=args.security_hint
    )
    output_manager.set_eval_info(
        model_name=args.model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        security_hint=args.security_hint,
        data_path=args.data_path,
        batchall=args.batchall,
        gen_workers=args.gen_workers,
        eval_workers=args.eval_workers,
        limit=args.limit,
        api_key="***" if args.api_key else None,
        base_url=args.base_url,
        total_cases=len(test_cases),
        pass_at_k=args.pass_at_k
    )
    
    logger.info(f"Output directory: {output_manager.get_eval_dir()}")
    
    # Evaluate test cases
    logger.info("\nStarting evaluation...")
    logger.info("="*60)
    
    # Pass@k mode: run each test case k times
    if args.pass_at_k > 1:
        logger.info(f"Pass@k mode: Running each test case {args.pass_at_k} times")
        if args.deduplicate_code:
            logger.info("Code deduplication enabled: identical code will only be evaluated once")
        
        reused_runs_dict = {}  # Empty dict for compatibility with deduplication code
        
        # Step 1: Generate all code (for deduplication)
        if args.deduplicate_code:
            logger.info("\nStep 1: Generating code for all runs...")
            all_runs_with_code = []  # List of (test_case, run_index, code)
            
            # CRITICAL FIX (Bug 15): Separate run_0 generation from other runs
            # Run 0 should be generated sequentially (like Pass@1) to ensure consistency
            # Other runs can be generated in parallel
            runs_to_generate_run0 = []  # Run 0 only
            runs_to_generate_others = []  # Run 1, 2, 3, ...
            
            for test_case in test_cases:
                runs_needed = args.pass_at_k
                
                for i in range(runs_needed):
                    run_idx = i
                    if run_idx == 0:
                        runs_to_generate_run0.append((test_case, run_idx))
                    else:
                        runs_to_generate_others.append((test_case, run_idx))
            
            logger.info(f"Need to generate code for {len(runs_to_generate_run0) + len(runs_to_generate_others)} runs")
            logger.info(f"  Run 0: {len(runs_to_generate_run0)} (sequential for Pass@1 consistency)")
            logger.info(f"  Other runs: {len(runs_to_generate_others)} (parallel)")
            
            # CRITICAL FIX (Bug 15): Generate run_0 sequentially first
            # This ensures run_0 has the same generation environment as Pass@1 mode
            def generate_code_for_run(test_case, run_idx):
                    """Generate code for a single run and save initial context"""
                    try:
                        # Create orchestrator for this run
                        temp_orchestrator = EvalOrchestrator(
                            model_name=args.model,
                            temperature=args.temperature,
                            max_iterations=args.max_iterations,
                            max_attempts=args.max_attempts,
                            output_manager=output_manager,
                            api_key=args.api_key or config.OPENAI_API_KEY,
                            base_url=args.base_url or config.OPENAI_BASE_URL,
                            quiet=True,
                            security_hint_level=args.security_hint,
                            cwe_info=cwe_info,
                            max_timeout=args.max_timeout,
                            generate_only=args.generate_only
                        )
                        
                        # CRITICAL FIX (Bug 20): Initialize test_runner for compilation testing
                        # test_runner is normally initialized in evaluate_test_case(), but we need it here
                        from core.test_runner import TestRunner
                        work_dir = output_manager.get_eval_dir() / test_case.cwe_id / f"run_{run_idx}"
                        temp_orchestrator.test_runner = TestRunner(work_dir=work_dir)
                        
                        # CRITICAL FIX (Bug 20): Use _generate_code_with_compile_check() like Pass@1 mode
                        # This ensures the code generation process is identical to Pass@1
                        # including compilation verification and automatic fixing
                        code = temp_orchestrator._generate_code_with_compile_check(
                            test_case=test_case,
                            is_initial=True
                        )
                        
                        logger.info(f"  ✓ Generated compilable code for {test_case.uuid} run {run_idx + 1}")
                        
                        # Save generated code (final version after compilation fixes)
                        output_manager.save_code(
                            test_case.uuid, test_case.cwe_id, 0, code, test_case.language, run_index=run_idx
                        )
                        
                        # Save initial context for later reuse
                        initial_context = temp_orchestrator.target_llm.get_messages()
                        output_manager.save_initial_context(
                            test_case.uuid, test_case.cwe_id, initial_context, run_index=run_idx
                        )
                        
                        logger.info(f"  Generated code for {test_case.uuid} run {run_idx + 1}")
                        return (test_case, run_idx, code)
                    except Exception as e:
                        logger.error(f"Error generating code for {test_case.uuid} run {run_idx + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # CRITICAL FIX: Save empty initial_context to avoid missing file errors
                        # This ensures the file exists even if code generation failed
                        try:
                            output_manager.save_initial_context(
                                test_case.uuid, test_case.cwe_id, [], run_index=run_idx
                            )
                            logger.warning(f"⚠️  Saved empty initial_context for {test_case.uuid} run {run_idx + 1} due to generation failure")
                        except Exception as save_error:
                            logger.error(f"Failed to save empty context: {save_error}")
                        
                        return (test_case, run_idx, "")
            
            # CRITICAL FIX (Bug 15): Generate run_0 first (before other runs) for Pass@1 consistency
            # Run 0 can be generated in parallel (like Pass@1 mode), but must complete before other runs
            if runs_to_generate_run0:
                logger.info("\nGenerating Run 0 first (Pass@1 consistency)...")
                if args.batchall and args.gen_workers > 1:
                    logger.info(f"  Using {args.gen_workers} parallel workers for Run 0")
                    with ThreadPoolExecutor(max_workers=args.gen_workers) as executor:
                        futures = [
                            executor.submit(generate_code_for_run, test_case, run_idx)
                            for test_case, run_idx in runs_to_generate_run0
                        ]
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                all_runs_with_code.append(result)
                            except Exception as e:
                                logger.error(f"Run 0 generation task exception: {e}")
                else:
                    # Sequential fallback
                    for test_case, run_idx in runs_to_generate_run0:
                        logger.info(f"  Generating code for {test_case.uuid} run {run_idx}")
                        result = generate_code_for_run(test_case, run_idx)
                        all_runs_with_code.append(result)
            
            # Generate other runs in parallel
            if args.batchall and args.gen_workers > 1 and len(runs_to_generate_others) > 0:
                logger.info(f"\nGenerating other runs in parallel with {args.gen_workers} workers...")
                
                # Submit all code generation tasks for non-run_0
                with ThreadPoolExecutor(max_workers=args.gen_workers) as executor:
                    futures = [
                        executor.submit(generate_code_for_run, test_case, run_idx)
                        for test_case, run_idx in runs_to_generate_others
                    ]
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            all_runs_with_code.append(result)
                        except Exception as e:
                            logger.error(f"Code generation task exception: {e}")
            elif runs_to_generate_others:
                # Sequential code generation for other runs (fallback)
                logger.info("\nGenerating other runs sequentially (fallback)...")
                for test_case, run_idx in runs_to_generate_others:
                    logger.info(f"  Generating code for {test_case.uuid} run {run_idx + 1}...")
                    result = generate_code_for_run(test_case, run_idx)
                    all_runs_with_code.append(result)
            
            logger.info(f"Generated code for {len(all_runs_with_code)} total runs")
            
            # Step 2: Group by code content
            logger.info("\nStep 2: Grouping identical code...")
            code_groups = group_runs_by_code(all_runs_with_code)
            
            # Step 3: Select representatives and save run_mapping
            logger.info("Step 3: Selecting representative runs...")
            representatives, run_mapping = select_representative_runs(code_groups)
            
            # CRITICAL FIX: Save run_mapping to output_manager for later use in finalize()
            # This avoids race conditions in concurrent execution
            output_manager.set_run_mapping(run_mapping)
            
            # Get statistics
            stats = get_deduplication_stats(code_groups)
            logger.info(f"\nDeduplication Statistics:")
            logger.info(f"  Total runs: {stats['total_runs']}")
            logger.info(f"  Unique codes: {stats['unique_codes']}")
            logger.info(f"  Duplicate runs: {stats['duplicate_runs']} ({stats['deduplication_rate']:.1%})")
            logger.info(f"  Runs to evaluate: {stats['runs_to_evaluate']}")
            logger.info(f"  Max group size: {stats['max_group_size']}")
            logger.info(f"  Avg group size: {stats['avg_group_size']:.2f}")
            
            # Step 4: Prepare evaluation list (only representatives that need evaluation)
            # Load initial context for each run
            expanded_cases = []
            for test_case, run_idx, code in representatives:
                # Load initial context
                initial_context = output_manager.load_initial_context(
                    test_case.uuid, test_case.cwe_id, run_index=run_idx
                )
                expanded_cases.append((test_case, run_idx, code, run_mapping, initial_context))
            
            logger.info(f"\nStep 4: Evaluating {len(expanded_cases)} representative runs...")
            
        else:
            # Original behavior: no deduplication
            expanded_cases = []
            for test_case in test_cases:
                runs_needed = args.pass_at_k
                
                # Add new runs (only the ones we need to execute)
                # Use same structure as deduplication mode for consistency
                for i in range(runs_needed):
                    run_idx = i
                    expanded_cases.append((test_case, run_idx, None, None, None))
            
            logger.info(f"Total runs to execute: {len(expanded_cases)}")
        
        if args.batchall and args.eval_workers > 1:
            # Parallel evaluation
            logger.info(f"Using {args.eval_workers} parallel workers for evaluation")
            counter = ProgressCounter(len(expanded_cases))
            
            # Choose evaluation function based on deduplication mode
            if args.deduplicate_code:
                with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
                    futures = [
                        executor.submit(
                            evaluate_single_case_with_run_index_preloaded,
                            test_case, run_idx, initial_context, args, output_manager,
                            counter, logger, cwe_info
                        )
                        for test_case, run_idx, code, run_mapping, initial_context in expanded_cases
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Task exception: {e}")
            else:
                with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
                    futures = [
                        executor.submit(
                            evaluate_single_case_with_run_index,
                            test_case, run_idx, args, output_manager,
                            counter, logger, cwe_info
                        )
                        for test_case, run_idx, _, _, _ in expanded_cases
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Task exception: {e}")
                
            successful = counter.successful
            failed = counter.failed
            
            # Note: Result copying to duplicates is now handled in output_manager.finalize()
            # to avoid race conditions in concurrent execution
        else:
            # Sequential evaluation
            successful = 0
            failed = 0
            
            for i, (test_case, run_idx, code, run_mapping, initial_context) in enumerate(expanded_cases, 1):
                logger.info(f"\n[{i}/{len(expanded_cases)}] Evaluating: {test_case.uuid} (run {run_idx + 1})")
                
                try:
                    orchestrator = EvalOrchestrator(
                        model_name=args.model,
                        temperature=args.temperature,
                        max_iterations=args.max_iterations,
                        max_attempts=args.max_attempts,
                        output_manager=output_manager,
                        api_key=args.api_key or config.OPENAI_API_KEY,
                        base_url=args.base_url or config.OPENAI_BASE_URL,
                        security_hint_level=args.security_hint,
                        cwe_info=cwe_info,
                        max_timeout=args.max_timeout,
                        generate_only=args.generate_only
                    )
                    
                    # Use preloaded context if available (deduplication mode)
                    # In non-deduplication mode, initial_context is None, which is fine
                    result = orchestrator.evaluate_test_case(test_case, run_index=run_idx, preloaded_context=initial_context)
                    
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Real-time Pass@k statistics: check if this seed just completed all k runs
                    if (result.uuid in output_manager._completed_seeds and
                        len(output_manager._completed_runs_by_seed.get(result.uuid, set())) == args.pass_at_k):
                        output_manager.compute_pass_at_k_statistics(args.pass_at_k, log_pass_at_k_progress, realtime=True)
                        
                except Exception as e:
                    logger.error(f"Error evaluating {test_case.uuid}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
            
            # Note: Result copying to duplicates is now handled in output_manager.finalize()
            # to avoid race conditions in concurrent execution
        
        # Compute final Pass@k statistics
        logger.info("\nComputing final Pass@k statistics...")
        output_manager.compute_pass_at_k_statistics(args.pass_at_k, log_pass_at_k_progress, realtime=False)
        
    elif args.batchall and args.eval_workers > 1:
        # Original parallel evaluation (Pass@1)
        logger.info(f"Using {args.eval_workers} parallel workers")
        counter = ProgressCounter(len(test_cases))
        
        with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_single_case,
                    test_case, args, output_manager, counter, logger, cwe_info
                )
                for test_case in test_cases
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Task exception: {e}")
        
        successful = counter.successful
        failed = counter.failed
    else:
        # Sequential evaluation
        successful = 0
        failed = 0
        
        # Create orchestrator once and reuse (important for transformers to avoid reloading model)
        orchestrator = EvalOrchestrator(
            model_name=args.model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            output_manager=output_manager,
            api_key=args.api_key or config.OPENAI_API_KEY,
            base_url=args.base_url or config.OPENAI_BASE_URL,
            quiet=True,
            security_hint_level=args.security_hint,
            cwe_info=cwe_info,
            max_timeout=args.max_timeout,
            generate_only=args.generate_only
        )
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"[{i}/{len(test_cases)}] {test_case.uuid}: Evaluating...")
            
            try:
                result = orchestrator.evaluate_test_case(test_case)
                
                if result.success:
                    successful += 1
                    logger.info(f"[{i}/{len(test_cases)}] {test_case.uuid}: FUNC PASS")
                else:
                    failed += 1
                    logger.info(f"[{i}/{len(test_cases)}] {test_case.uuid}: FUNC FAIL")
                
                # Clear GPU cache after each test case (for transformers engine)
                orchestrator.target_llm.clear_cache()
                    
            except Exception as e:
                logger.error(f"Error evaluating {test_case.uuid}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    # Finalize
    output_manager.finalize()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete")
    logger.info("="*60)
    if args.pass_at_k > 1:
        logger.info(f"Total Test Cases: {len(test_cases)}")
        logger.info(f"Total Runs: {len(test_cases) * args.pass_at_k}")
        logger.info(f"Successful Runs: {successful}")
        logger.info(f"Failed Runs: {failed}")
        logger.info(f"\nPass@{args.pass_at_k} statistics computed and saved")
    else:
        logger.info(f"Total Test Cases: {len(test_cases)}")
        logger.info(f"Successful (all functional passed): {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {successful/len(test_cases)*100:.1f}%")
    logger.info(f"\nResults saved to: {output_manager.get_result_file()}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
