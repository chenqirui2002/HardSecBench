#!/usr/bin/env python3
"""
HardSecBench Data Generation System
Unified entry point for the automated data factory
"""
import argparse
import shutil
import traceback
from pathlib import Path
from typing import Optional, List
from config.settings import config
from utils.data_loader import DataLoader
from utils.logger_manager import configure_logging, get_logger
from utils.batch_output_manager import BatchOutputManager
from agents.generation import SeedGeneratorAgent, ArchitectAgent
from agents.orchestration import IterativeCodeGenerator
from models import CodeLanguage, GenerationTask


class DataGenerationPipeline:
    """
    Main pipeline for hardware security benchmark data generation
    
    Architecture:
    1. Agent 0 (Seed Generator): Generate diverse seed questions from CWE specifications
    2. Agent A (Architect): Expand seed questions into detailed problem descriptions
    3. Agent B (Expert): Generate golden code
    4. Agent C (Red Teamer): Generate adversarial testbenches
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.logger = get_logger()
        self.data_loader = DataLoader()
        
        self.logger.info("="*60)
        self.logger.info("[Pipeline] HardSecBench Data Generation Pipeline")
        self.logger.info("="*60)
        
        # Load data
        self.logger.info("[Pipeline] Loading CWE database...")
        self.cwes = self.data_loader.load_cwes(config.CWE_DATABASE_PATH)
        self.logger.info(f"[Pipeline] Loaded {len(self.cwes)} CWEs")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HardSecBench Data Generation System"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Seed generation command (PRIMARY COMMAND)
    seed_parser = subparsers.add_parser(
        'seed',
        help='Generate seed questions for each CWE'
    )
    seed_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of CWEs to process (default: all)'
    )
    seed_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: outputs/cwe_seed_questions.json)'
    )
    seed_parser.add_argument(
        '--test-cwe',
        type=str,
        default=None,
        help='Test seed generation for a single CWE ID (e.g., CWE-1191)'
    )
    
    # Generate command (Single seed generation)
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate a complete dataset entry from a single seed question'
    )
    generate_parser.add_argument(
        '--cwe',
        type=str,
        required=True,
        help='CWE ID to generate (e.g., CWE-798)'
    )
    generate_parser.add_argument(
        '--seed-index',
        type=int,
        required=True,
        help='Index of seed question to use (0-based, use seed command first to see available seeds)'
    )
    generate_parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations for iterative generation (default: 5)'
    )
    generate_parser.add_argument(
        '--max-tb-compile-attempts',
        type=int,
        default=5,
        help='Maximum attempts for TB compilation per requirement (default: 5)'
    )
    
    # Batch generate command (NEW: Generate multiple tasks for a single CWE)
    batch_parser = subparsers.add_parser(
        'batch-generate',
        help='Generate multiple dataset entries for a single CWE with different seeds'
    )
    batch_parser.add_argument(
        '--cwe',
        type=str,
        required=True,
        help='CWE ID to generate (e.g., CWE-798)'
    )
    batch_parser.add_argument(
        '--max-seeds',
        type=int,
        default=None,
        help='Maximum number of seeds to generate (default: all suitable seeds)'
    )
    batch_parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations per task (default: 5)'
    )
    batch_parser.add_argument(
        '--max-tb-compile-attempts',
        type=int,
        default=5,
        help='Maximum attempts for TB compilation per requirement (default: 5)'
    )
    batch_parser.add_argument(
        '--language-priority',
        type=str,
        choices=['c', 'verilog', 'none'],
        default='none',
        help='Language priority when limiting seeds: c (C first), verilog (Verilog first), none (no preference, default)'
    )
    batch_parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for batch results (default: outputs/batch_<cwe>)'
    )
    
    # Batchall command (Generate for ALL CWEs with concurrency)
    batchall_parser = subparsers.add_parser(
        'batchall',
        help='Generate dataset entries for ALL CWEs with concurrent processing'
    )
    batchall_parser.add_argument(
        '--max-seeds-per-cwe',
        type=int,
        default=None,
        help='Maximum number of seeds per CWE (default: all suitable seeds)'
    )
    batchall_parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations per task (default: 5)'
    )
    batchall_parser.add_argument(
        '--max-tb-compile-attempts',
        type=int,
        default=5,
        help='Maximum attempts for TB compilation per requirement (default: 5)'
    )
    batchall_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of concurrent workers (default: 4)'
    )
    batchall_parser.add_argument(
        '--language-priority',
        type=str,
        choices=['c', 'verilog', 'none'],
        default='none',
        help='Language priority when limiting seeds'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Display system information'
    )
    
    args = parser.parse_args()
    
    # Configure logging system
    configure_logging(config.LOGS_DIR)
    
    # Initialize pipeline
    pipeline = DataGenerationPipeline()
    
    if args.command == 'seed':
        # Seed question generation
        pipeline.logger.info("\n" + "="*60)
        pipeline.logger.info("Stage: Seed Question Generation")
        pipeline.logger.info("="*60)
        
        # Initialize Seed Generator Agent
        seed_generator = SeedGeneratorAgent()
        
        # Test single CWE or process all
        if args.test_cwe:
            # Find the CWE
            cwe = next((c for c in pipeline.cwes if c.cwe_id == args.test_cwe), None)
            if not cwe:
                pipeline.logger.error(f"CWE {args.test_cwe} not found in database")
                return
            
            pipeline.logger.info(f"Testing seed generation for {args.test_cwe}")
            result = seed_generator.generate_seeds_for_cwe(cwe)
            
            # Display result
            print("\n" + "="*60)
            print(f"Seed Generation Result: {result.cwe_id} - {result.cwe_name}")
            print("="*60)
            print(f"Total Seeds: {result.total_seeds}")
            print(f"\nGenerated Seed Questions:")
            for i, seed in enumerate(result.seeds, 1):
                print(f"\n{i}. [{seed.language.upper()}] {seed.question}")
            
            print(f"\nReasoning:\n{result.reasoning}")
            print("="*60)
        else:
            # Process all CWEs
            cwes_to_process = pipeline.cwes[:args.limit] if args.limit else pipeline.cwes
            pipeline.logger.info(f"Generating seeds for {len(cwes_to_process)} CWEs")
            
            batch_result = seed_generator.generate_seeds_for_all_cwes(cwes_to_process)
            
            # Save results
            output_path = args.output or (config.OUTPUT_DIR / "cwe_seed_questions.json")
            pipeline.data_loader.save_json(
                batch_result.model_dump(),
                output_path
            )
            pipeline.logger.info(f"Saved seed questions to: {output_path}")
            
            # Display statistics
            pipeline.logger.info("\n" + "="*60)
            pipeline.logger.info("Seed Generation Statistics")
            pipeline.logger.info("="*60)
            pipeline.logger.info(f"Total CWEs processed: {batch_result.total_cwes}")
            pipeline.logger.info(f"Total seed questions: {batch_result.total_seeds}")
            pipeline.logger.info(f"Average seeds per CWE: {batch_result.total_seeds / batch_result.total_cwes:.1f}")
            
            # Show CWEs with most seeds
            top_cwes = sorted(batch_result.results, key=lambda x: x.total_seeds, reverse=True)[:10]
            pipeline.logger.info("\nTop 10 CWEs by seed count:")
            for result in top_cwes:
                pipeline.logger.info(f"  {result.cwe_id}: {result.total_seeds} seeds - {result.cwe_name[:60]}...")
            pipeline.logger.info("="*60)
    
    elif args.command == 'generate':
        # Single seed generation
        pipeline.logger.info("\n" + "="*60)
        pipeline.logger.info("Stage: Single Seed Generation")
        pipeline.logger.info("="*60)
        
        # Find CWE
        cwe = next((c for c in pipeline.cwes if c.cwe_id == args.cwe), None)
        if not cwe:
            pipeline.logger.error(f"CWE {args.cwe} not found in database")
            return
        
        # Generate seeds for this CWE
        pipeline.logger.info(f"Generating seeds for {args.cwe}...")
        seed_generator = SeedGeneratorAgent()
        seed_result = seed_generator.generate_seeds_for_cwe(cwe)
        
        if seed_result.total_seeds == 0:
            pipeline.logger.error(f"No seeds generated for {args.cwe}")
            return
        
        # Check seed index
        if args.seed_index < 0 or args.seed_index >= seed_result.total_seeds:
            pipeline.logger.error(f"Invalid seed index {args.seed_index}. Available: 0-{seed_result.total_seeds-1}")
            pipeline.logger.info(f"\nAvailable seeds:")
            for i, seed in enumerate(seed_result.seeds):
                pipeline.logger.info(f"  [{i}] ({seed.language}) {seed.question[:80]}...")
            return
        
        # Select seed
        selected_seed = seed_result.seeds[args.seed_index]
        pipeline.logger.info(f"Selected seed [{args.seed_index}]: ({selected_seed.language}) {selected_seed.question}")
        
        # Initialize Architect
        architect = ArchitectAgent()
        
        # Create task ID
        task_id = f"TASK-{cwe.cwe_id}-SEED{args.seed_index}"
        
        # Initialize generator
        pipeline.logger.info(f"Generating for CWE: {cwe.cwe_id} - {cwe.name}")
        pipeline.logger.info(f"Language: {selected_seed.language}")
        
        generator = IterativeCodeGenerator(
            max_iterations=args.max_iterations,
            max_tb_compile_attempts=args.max_tb_compile_attempts,
            save_to_json=True
        )
        
        # Step 1: Generate problem description from seed
        pipeline.logger.info("\nStep 1: Generating problem description from seed...")
        problem = architect.generate_problem_from_seed(selected_seed, cwe, task_id)
        problem_title = problem.question.split('\n')[0].strip() if problem.question else "Unknown"
        pipeline.logger.info(f"Problem generated: {problem_title}")
        
        # Create GenerationTask (with placeholder component/trigger info)
        language = problem.language
        
        task = GenerationTask(
            task_id=task_id,
            cwe_id=cwe.cwe_id,
            cwe_name=cwe.name,
            cwe_description=cwe.description,
            cwe_extended_description=getattr(cwe, 'extended_description', None),
            component_id=f"SEED-{args.seed_index}",
            component_name=f"Seed Question {args.seed_index}",
            component_description=selected_seed.question[:100],
            trigger_id=f"SEED-{args.seed_index}",
            trigger_name=f"Scenario {args.seed_index}",
            trigger_description=selected_seed.question,
            language=language,
            module_name=problem.module_or_function_name if language == CodeLanguage.VERILOG else None,
            function_name=problem.module_or_function_name if language == CodeLanguage.C else None
        )
        
        # Step 2: Iterative code generation
        pipeline.logger.info("\nStep 2: Starting iterative code generation...")
        final_code, iterations, phase0_validation = generator.generate_with_feedback(
            task,
            problem,
            output_dir=generator.work_dir
        )
        
        # Display summary
        print("\n" + "="*60)
        print("Generation Complete")
        print("="*60)
        print(f"CWE: {cwe.cwe_id} - {cwe.name}")
        print(f"Seed Index: {args.seed_index}")
        print(f"Language: {selected_seed.language}")
        print(f"\nProblem: {problem_title}")
        print(f"\nIterations: {len(iterations)}")
        print(f"Successful: {iterations[-1].all_tests_passed if iterations else False}")
        print(f"Final Vulnerabilities: {iterations[-1].vulnerabilities_found if iterations else 0}")
        print(f"\nResult saved to: {config.OUTPUT_DIR / 'single_gen.json'}")
        print("="*60)
    
    elif args.command == 'batch-generate':
        # Batch generation for a single CWE
        pipeline.logger.info("\n" + "="*60)
        pipeline.logger.info("Stage: Batch CWE Generation")
        pipeline.logger.info("="*60)
        
        # Find CWE
        cwe = next((c for c in pipeline.cwes if c.cwe_id == args.cwe), None)
        if not cwe:
            pipeline.logger.error(f"CWE {args.cwe} not found in database")
            return
        
        # Step 1: Generate seed questions for this CWE
        pipeline.logger.info(f"\nStep 1: Generating seed questions for {cwe.cwe_id}...")
        seed_generator = SeedGeneratorAgent()
        seed_result = seed_generator.generate_seeds_for_cwe(cwe)
        
        if seed_result.total_seeds == 0:
            pipeline.logger.error(f"No seeds generated for {cwe.cwe_id}")
            return
        
        pipeline.logger.info(f"Generated {seed_result.total_seeds} seed questions")
        
        # Step 2: Select seeds to process
        pipeline.logger.info(f"\nStep 2: Selecting seeds to process...")
        
        seeds = seed_result.seeds
        
        # Apply language priority if specified
        if args.language_priority != 'none':
            pipeline.logger.info(f"Applying language priority: {args.language_priority}")
            if args.language_priority == 'c':
                # C language first, then Verilog
                seeds = sorted(seeds, key=lambda s: (0 if s.language == 'c' else 1))
            elif args.language_priority == 'verilog':
                # Verilog first, then C
                seeds = sorted(seeds, key=lambda s: (0 if s.language == 'verilog' else 1))
        
        # Limit seeds if requested
        if args.max_seeds and len(seeds) > args.max_seeds:
            pipeline.logger.info(f"Limiting to {args.max_seeds} seeds (out of {len(seeds)})")
            seeds = seeds[:args.max_seeds]
        
        pipeline.logger.info(f"Processing {len(seeds)} seeds")
        
        # Show language distribution
        lang_counts = {}
        for seed in seeds:
            lang_counts[seed.language] = lang_counts.get(seed.language, 0) + 1
        pipeline.logger.info(f"Language distribution: {', '.join(f'{lang}: {count}' for lang, count in sorted(lang_counts.items()))}")
        
        if len(seeds) == 0:
            pipeline.logger.error("No suitable seeds found for this CWE")
            return
        
        # Step 3: Initialize BatchOutputManager
        batch_manager = BatchOutputManager(
            base_output_dir=config.OUTPUT_DIR,
            batch_name=cwe.cwe_id.lower()
        )
        
        batch_dir = batch_manager.get_batch_dir()
        pipeline.logger.info(f"\nBatch directory: {batch_dir}")
        pipeline.logger.info(f"Result file: {batch_manager.get_result_file_path()}")
        
        # Initialize Architect
        architect = ArchitectAgent()
        
        # Step 4: Generate for each seed
        pipeline.logger.info(f"\nStep 4: Generating {len(seeds)} tasks...")
        
        for idx, seed in enumerate(seeds, 1):
            # Create seed UUID and directory
            seed_uuid, seed_dir = batch_manager.create_seed_directory(seed_index=idx-1)
            task_id = f"TASK-{cwe.cwe_id}-SEED{idx-1}"
            
            pipeline.logger.info(f"\n{'='*60}")
            pipeline.logger.info(f"Seed {idx}/{len(seeds)}")
            pipeline.logger.info(f"UUID: {seed_uuid}")
            pipeline.logger.info(f"Task ID: {task_id}")
            pipeline.logger.info(f"{'='*60}")
            pipeline.logger.info(f"Question: ({seed.language}) {seed.question[:80]}...")
            
            try:
                # Generate problem from seed
                problem = architect.generate_problem_from_seed(seed, cwe, task_id)
                problem_title = problem.question.split('\n')[0].strip() if problem.question else "Unknown"
                pipeline.logger.info(f"Problem: {problem_title}")
                
                # Add seed entry to result.json with full requirements
                batch_manager.add_seed_entry(
                    seed_uuid=seed_uuid,
                    cwe_id=cwe.cwe_id,
                    task_id=task_id,
                    problem_description=problem.question,
                    language=seed.language,
                    function_requirements=problem.function_requirements,
                    security_requirements=problem.security_requirements,
                    input_specification=problem.input_specification,
                    output_specification=problem.output_specification,
                    module_or_function_name=problem.module_or_function_name
                )
                
                # Initialize generator with seed directory
                generator = IterativeCodeGenerator(
                    work_dir=seed_dir,
                    max_iterations=args.max_iterations,
                    max_tb_compile_attempts=args.max_tb_compile_attempts,
                    save_to_json=False
                )
                
                # Create GenerationTask
                language = problem.language
                
                task = GenerationTask(
                    task_id=task_id,
                    cwe_id=cwe.cwe_id,
                    cwe_name=cwe.name,
                    cwe_description=cwe.description,
                    cwe_extended_description=getattr(cwe, 'extended_description', None),
                    component_id=f"SEED-{idx-1}",
                    component_name=f"Seed Question {idx-1}",
                    component_description=seed.question[:100],
                    trigger_id=f"SEED-{idx-1}",
                    trigger_name=f"Scenario {idx-1}",
                    trigger_description=seed.question,
                    language=language,
                    module_name=problem.module_or_function_name if language == CodeLanguage.VERILOG else None,
                    function_name=problem.module_or_function_name if language == CodeLanguage.C else None
                )
                
                # Generate code
                final_code, iterations, phase0_validation = generator.generate_with_feedback(
                    task,
                    problem,
                    output_dir=seed_dir
                )
                
                # Get final test suites
                final_iter = iterations[-1] if iterations else None
                final_functional_suite = final_iter.functional_suite if final_iter else None
                final_security_suite = final_iter.security_suite if final_iter else None
                
                # Calculate statistics
                total_functional = final_functional_suite.total_tests if final_functional_suite else 0
                passed_functional = final_functional_suite.passed_tests if final_functional_suite else 0
                functional_rate = final_functional_suite.pass_rate if final_functional_suite else 0.0
                
                total_security = final_security_suite.total_tests if final_security_suite else 0
                passed_security = final_security_suite.passed_tests if final_security_suite else 0
                security_rate = final_security_suite.pass_rate if final_security_suite else 0.0
                
                # Get file paths (relative to seed directory)
                code_file = generator.expert._get_code_file_path(task) if generator.expert else None
                
                # Scan work directory for actual test files generated
                work_dir = seed_dir
                
                # Build test files dict with explanation from per-requirement results
                functional_test_files = {}
                security_test_files = {}
                
                # Get per-requirement test results from generator (contains explanation)
                if generator.per_req_functional:
                    for req_test in generator.per_req_functional:
                        functional_test_files[req_test.tb_file] = req_test.explanation or "No explanation"
                
                if generator.per_req_security:
                    for req_test in generator.per_req_security:
                        security_test_files[req_test.tb_file] = req_test.explanation or "No explanation"
                
                # Update result.json with final results (including potentially revised problem)
                batch_manager.update_seed_result(
                    seed_uuid=seed_uuid,
                    iterations=generator.iteration_issues,
                    passed=final_iter.all_tests_passed if final_iter else False,
                    golden_file=str(code_file),
                    functional_test_files=functional_test_files,
                    security_test_files=security_test_files,
                    functional_pass_rate=functional_rate,
                    security_pass_rate=security_rate,
                    status="completed",
                    problem_description=problem.question,
                    function_requirements=problem.function_requirements,
                    security_requirements=problem.security_requirements,
                    input_specification=problem.input_specification,
                    output_specification=problem.output_specification
                )
                
                pipeline.logger.info(f"✓ Seed {idx} completed successfully")
                pipeline.logger.info(f"  Iterations: {generator.total_iterations}")
                pipeline.logger.info(f"  Functional: {passed_functional}/{total_functional} ({functional_rate:.1%})")
                pipeline.logger.info(f"  Security: {passed_security}/{total_security} ({security_rate:.1%})")
                
            except Exception as e:
                pipeline.logger.error(f"Seed {idx} failed: {str(e)}")
                traceback.print_exc()
                
                # Update result.json with error
                try:
                    batch_manager.update_seed_result(
                        seed_uuid=seed_uuid,
                        error=str(e),
                        status="failed"
                    )
                except Exception:
                    pipeline.logger.warning(f"Failed to update result.json for seed {seed_uuid}")
                
                continue
        
        # Finalize batch
        batch_manager.finalize_batch()
        
        # Display summary
        print("\n" + "="*60)
        print("Batch Generation Complete")
        print("="*60)
        print(f"CWE: {cwe.cwe_id} - {cwe.name}")
        print(f"Total Seeds: {len(seeds)}")
        print(f"\nBatch Directory: {batch_dir}")
        print(f"Result File: {batch_manager.get_result_file_path()}")
        print("="*60)
    
    elif args.command == 'batchall':
        # Batchall generation for ALL CWEs with concurrency
        import uuid
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.batch_output_manager import BatchAllOutputManager
        
        pipeline.logger.info("\n" + "="*60)
        pipeline.logger.info("Stage: Batchall CWE Generation (Concurrent)")
        pipeline.logger.info("="*60)
        pipeline.logger.info(f"Total CWEs: {len(pipeline.cwes)}")
        pipeline.logger.info(f"Max workers: {args.max_workers}")
        
        # Initialize BatchAllOutputManager
        batchall_manager = BatchAllOutputManager(base_output_dir=config.OUTPUT_DIR)
        batchall_dir = batchall_manager.get_batchall_dir()
        pipeline.logger.info(f"Batchall directory: {batchall_dir}")
        
        def process_single_cwe(cwe):
            """Process a single CWE (runs in thread pool)"""
            cwe_logger = get_logger()
            try:
                cwe_logger.info(f"\n[{cwe.cwe_id}] Starting generation...")
                
                # Generate seeds for this CWE
                seed_generator = SeedGeneratorAgent()
                seed_result = seed_generator.generate_seeds_for_cwe(cwe)
                
                if seed_result.total_seeds == 0:
                    cwe_logger.warning(f"[{cwe.cwe_id}] No seeds generated, skipping")
                    return {"cwe_id": cwe.cwe_id, "status": "skipped", "reason": "no_seeds"}
                
                seeds = seed_result.seeds
                
                # Apply language priority
                if args.language_priority != 'none':
                    if args.language_priority == 'c':
                        seeds = sorted(seeds, key=lambda s: (0 if s.language == 'c' else 1))
                    elif args.language_priority == 'verilog':
                        seeds = sorted(seeds, key=lambda s: (0 if s.language == 'verilog' else 1))
                
                # Limit seeds
                if args.max_seeds_per_cwe and len(seeds) > args.max_seeds_per_cwe:
                    seeds = seeds[:args.max_seeds_per_cwe]
                
                # Create CWE directory
                cwe_dir = batchall_manager.create_cwe_directory(cwe.cwe_id)
                
                # Add CWE entry
                batchall_manager.add_cwe_entry(cwe.cwe_id, cwe.name, len(seeds))
                
                # Initialize Architect
                architect = ArchitectAgent()
                
                # Process each seed
                for idx, seed in enumerate(seeds, 1):
                    seed_uuid = str(uuid.uuid4())
                    seed_dir = cwe_dir / seed_uuid
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    task_id = f"TASK-{cwe.cwe_id}-SEED{idx-1}"
                    
                    try:
                        # Generate problem
                        problem = architect.generate_problem_from_seed(seed, cwe, task_id)
                        
                        # Add seed entry
                        batchall_manager.add_seed_to_cwe(
                            cwe_id=cwe.cwe_id,
                            seed_uuid=seed_uuid,
                            task_id=task_id,
                            problem_description=problem.question,
                            language=seed.language,
                            function_requirements=problem.function_requirements,
                            security_requirements=problem.security_requirements,
                            input_specification=problem.input_specification,
                            output_specification=problem.output_specification,
                            module_or_function_name=problem.module_or_function_name
                        )
                        
                        # Initialize generator
                        generator = IterativeCodeGenerator(
                            work_dir=seed_dir,
                            max_iterations=args.max_iterations,
                            max_tb_compile_attempts=args.max_tb_compile_attempts,
                            save_to_json=False
                        )
                        
                        # Create task
                        language = problem.language
                        task = GenerationTask(
                            task_id=task_id,
                            cwe_id=cwe.cwe_id,
                            cwe_name=cwe.name,
                            cwe_description=cwe.description,
                            cwe_extended_description=getattr(cwe, 'extended_description', None),
                            component_id=f"SEED-{idx-1}",
                            component_name=f"Seed Question {idx-1}",
                            component_description=seed.question[:100],
                            trigger_id=f"SEED-{idx-1}",
                            trigger_name=f"Scenario {idx-1}",
                            trigger_description=seed.question,
                            language=language,
                            module_name=problem.module_or_function_name if language == CodeLanguage.VERILOG else None,
                            function_name=problem.module_or_function_name if language == CodeLanguage.C else None
                        )
                        
                        # Generate code
                        final_code, iterations, phase0_validation = generator.generate_with_feedback(
                            task, problem, output_dir=seed_dir
                        )
                        
                        # Get results
                        final_iter = iterations[-1] if iterations else None
                        final_functional_suite = final_iter.functional_suite if final_iter else None
                        final_security_suite = final_iter.security_suite if final_iter else None
                        
                        functional_rate = final_functional_suite.pass_rate if final_functional_suite else 0.0
                        security_rate = final_security_suite.pass_rate if final_security_suite else 0.0
                        
                        code_file = generator.expert._get_code_file_path(task) if generator.expert else None
                        
                        # Build test files dict
                        functional_test_files = {}
                        security_test_files = {}
                        if generator.per_req_functional:
                            for req_test in generator.per_req_functional:
                                functional_test_files[req_test.tb_file] = req_test.explanation or "No explanation"
                        if generator.per_req_security:
                            for req_test in generator.per_req_security:
                                security_test_files[req_test.tb_file] = req_test.explanation or "No explanation"
                        
                        # Update result
                        batchall_manager.update_seed_result(
                            cwe_id=cwe.cwe_id,
                            seed_uuid=seed_uuid,
                            iterations=generator.iteration_issues,
                            passed=final_iter.all_tests_passed if final_iter else False,
                            golden_file=str(code_file),
                            functional_test_files=functional_test_files,
                            security_test_files=security_test_files,
                            functional_pass_rate=functional_rate,
                            security_pass_rate=security_rate,
                            status="completed",
                            problem_description=problem.question,
                            function_requirements=problem.function_requirements,
                            security_requirements=problem.security_requirements,
                            input_specification=problem.input_specification,
                            output_specification=problem.output_specification
                        )
                        
                        cwe_logger.info(f"[{cwe.cwe_id}] Seed {idx}/{len(seeds)} completed")
                        passed = final_iter.all_tests_passed if final_iter else False
                        if passed:
                            print(f"✓ {cwe.cwe_id} seed {idx}/{len(seeds)} PASS", flush=True)

                    except Exception as e:
                        cwe_logger.error(f"[{cwe.cwe_id}] Seed {idx} failed: {str(e)}")
                        batchall_manager.update_seed_result(
                            cwe_id=cwe.cwe_id,
                            seed_uuid=seed_uuid,
                            error=str(e),
                            status="failed"
                        )
                
                # Finalize CWE
                batchall_manager.finalize_cwe(cwe.cwe_id)
                cwe_logger.info(f"[{cwe.cwe_id}] Completed")
                return {"cwe_id": cwe.cwe_id, "status": "completed"}
                
            except Exception as e:
                cwe_logger.error(f"[{cwe.cwe_id}] Failed: {str(e)}")
                batchall_manager.finalize_cwe(cwe.cwe_id, status="failed")
                return {"cwe_id": cwe.cwe_id, "status": "failed", "error": str(e)}
        
        # Process all CWEs with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_cwe = {executor.submit(process_single_cwe, cwe): cwe for cwe in pipeline.cwes}
            
            for future in as_completed(future_to_cwe):
                cwe = future_to_cwe[future]
                try:
                    result = future.result()
                    results.append(result)
                    pipeline.logger.info(f"[{cwe.cwe_id}] Result: {result['status']}")
                except Exception as e:
                    pipeline.logger.error(f"[{cwe.cwe_id}] Exception: {str(e)}")
                    results.append({"cwe_id": cwe.cwe_id, "status": "error", "error": str(e)})
        
        # Finalize batchall
        batchall_manager.finalize_batchall()
        
        # Summary
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] in ["failed", "error"])
        skipped = sum(1 for r in results if r["status"] == "skipped")
        
        print("\n" + "="*60)
        print("Batchall Generation Complete")
        print("="*60)
        print(f"Total CWEs: {len(pipeline.cwes)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"\nBatchall Directory: {batchall_dir}")
        print(f"Result File: {batchall_manager.get_result_file_path()}")
        print("="*60)
    
    elif args.command == 'info':
        # Display system info
        print("\n" + "="*60)
        print("HardSecBench System Information")
        print("="*60)
        print(f"Project Root: {config.PROJECT_ROOT}")
        print(f"CWE Database: {len(pipeline.cwes)} CWEs")
        print(f"\nLLM Configuration:")
        print(f"  Default Model: {config.LLM_MODEL}")
        if config.OPENAI_BASE_URL:
            print(f"  API Base URL: {config.OPENAI_BASE_URL}")
        print(f"\nAgent Models:")
        print(f"  Agent 0 (Seed Generator): {config.ARCHITECT_MODEL or config.LLM_MODEL}")
        print(f"  Agent A (Architect): {config.ARCHITECT_MODEL or config.LLM_MODEL}")
        print(f"  Agent B (Expert): {config.EXPERT_MODEL or config.LLM_MODEL}")
        print(f"  Agent C (Red Teamer): {config.RED_TEAMER_MODEL or config.LLM_MODEL}")
        print(f"\nLangSmith Monitoring:")
        if config.LANGCHAIN_TRACING_V2:
            print(f"  Status: ✓ ENABLED")
            print(f"  Project: {config.LANGCHAIN_PROJECT}")
            print(f"  Endpoint: {config.LANGCHAIN_ENDPOINT}")
        else:
            print(f"  Status: ✗ DISABLED")
            print(f"  (Set LANGCHAIN_TRACING_V2=true in .env to enable)")
        print(f"\nOutput Directory: {config.OUTPUT_DIR}")
        print("="*60)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
