#!/usr/bin/env python3
"""
Filter Seeds by Testbench Quality

This script validates testbench effectiveness via mutation testing and coverage,
then filters seeds based on quality thresholds. It updates result.json with
a 'tb_validated' field for each seed.

Usage:
    python filter_seeds_by_tb_quality.py <batch_folder_path> [options]

Options:
    --mutations N          Number of mutations per seed (default: 5)
    --mutation-threshold   Minimum mutation detection rate (default: 0.5)
    --coverage-threshold   Minimum code coverage rate (default: 0.8)
    --no-coverage          Disable coverage measurement
    --dry-run              Don't modify result.json, just show statistics
"""

import os
import sys
import json
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from existing validation script
from validate_tb_effectiveness import (
    TBEffectivenessValidator,
    SeedValidationResult
)


@dataclass
class FilterResult:
    """Result of filtering operation"""
    total_seeds: int = 0
    validated_seeds: int = 0
    filtered_seeds: int = 0
    
    # Breakdown of filter reasons
    low_mutation_rate: int = 0
    low_coverage: int = 0
    both_low: int = 0
    no_data: int = 0


class SeedFilter:
    """Filter seeds based on TB quality metrics"""
    
    def __init__(
        self,
        batch_dir: Path,
        mutation_threshold: float = 0.5,
        coverage_threshold: float = 0.8,
        max_mutations: int = 5,
        enable_coverage: bool = True,
        output_suffix: str = "",
        num_threads: int = 1
    ):
        self.batch_dir = batch_dir
        self.mutation_threshold = mutation_threshold
        self.coverage_threshold = coverage_threshold
        self.max_mutations = max_mutations
        self.enable_coverage = enable_coverage
        self.output_suffix = output_suffix
        self.num_threads = num_threads
        
        self.validator = TBEffectivenessValidator(
            batch_dir,
            max_mutations=max_mutations,
            enable_coverage=enable_coverage
        )
        
        self.validation_results: Dict[str, SeedValidationResult] = {}
        self.filter_result = FilterResult()
        self.lock = threading.Lock()
    
    def _process_single_seed(self, cwe_id, seed_info):
        """Process a single seed (thread-safe)"""
        seed_uuid = seed_info["seed_uuid"]
        
        if "_cwe_id" in seed_info:
            seed_dir = self.batch_dir / seed_info["_cwe_id"] / seed_uuid
        else:
            seed_dir = self.batch_dir / seed_uuid
        
        if not seed_dir.exists():
            with self.lock:
                print(f"[SKIP] {seed_uuid}: Directory not found")
                seed_info["tb_validated"] = False
                seed_info["tb_filter_reason"] = "directory_not_found"
                self.filter_result.no_data += 1
            return
        
        with self.lock:
            print(f"[TEST] {seed_uuid}...", end=" ", flush=True)
        
        result = self.validator.validate_seed(seed_dir, seed_info)
        
        # Determine if seed passes filters
        mutation_ok = result.detection_rate >= self.mutation_threshold
        coverage_ok = True
        if self.enable_coverage and result.coverage:
            coverage_ok = result.coverage.line_coverage >= self.coverage_threshold
        
        tb_validated = mutation_ok and coverage_ok
        
        # Record filter reason
        filter_reason = None
        if not tb_validated:
            if not mutation_ok and not coverage_ok:
                filter_reason = "low_mutation_and_coverage"
                filter_type = "both_low"
            elif not mutation_ok:
                filter_reason = "low_mutation_rate"
                filter_type = "low_mutation_rate"
            else:
                filter_reason = "low_coverage"
                filter_type = "low_coverage"
        else:
            filter_type = "validated"
        
        # Update seed info
        seed_info["tb_validated"] = tb_validated
        seed_info["tb_mutation_rate"] = result.detection_rate
        if result.coverage:
            seed_info["tb_coverage"] = result.coverage.line_coverage
        if filter_reason:
            seed_info["tb_filter_reason"] = filter_reason
        
        # Update results (thread-safe)
        with self.lock:
            self.validation_results[seed_uuid] = result
            
            if filter_type == "validated":
                self.filter_result.validated_seeds += 1
            elif filter_type == "both_low":
                self.filter_result.both_low += 1
                self.filter_result.filtered_seeds += 1
            elif filter_type == "low_mutation_rate":
                self.filter_result.low_mutation_rate += 1
                self.filter_result.filtered_seeds += 1
            elif filter_type == "low_coverage":
                self.filter_result.low_coverage += 1
                self.filter_result.filtered_seeds += 1
            
            # Print status
            status = "PASS" if tb_validated else "FILTERED"
            cov_str = f", cov: {result.coverage.line_coverage:.0%}" if result.coverage else ""
            print(f"{status} (mut: {result.detection_rate:.0%}{cov_str})")
    
    def validate_and_filter(self, dry_run: bool = False) -> FilterResult:
        """Run validation and filter seeds"""
        result_json_path = self.batch_dir / "result.json"
        
        if not result_json_path.exists():
            print(f"Error: result.json not found in {self.batch_dir}")
            return self.filter_result
        
        with open(result_json_path, 'r') as f:
            batch_data = json.load(f)
        
        # Store batch_data for report generation
        self.batch_data = batch_data
        
        # Collect only PASSED seeds for testing
        all_seeds_info = []
        if "cwes" in batch_data:
            for cwe_id, cwe_data in batch_data["cwes"].items():
                for seed_info in cwe_data.get("seeds", []):
                    if seed_info.get("passed", False):
                        seed_info["_cwe_id"] = cwe_id.lower()
                        all_seeds_info.append((cwe_id, seed_info))
        else:
            for seed_info in batch_data.get("seeds", []):
                if seed_info.get("passed", False):
                    all_seeds_info.append((None, seed_info))
        
        self.filter_result.total_seeds = len(all_seeds_info)
        
        print(f"Validating and filtering {len(all_seeds_info)} seeds with {self.num_threads} threads...")
        print(f"Thresholds: mutation >= {self.mutation_threshold:.0%}, coverage >= {self.coverage_threshold:.0%}")
        print("-" * 70)
        
        # Process seeds in parallel
        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(self._process_single_seed, cwe_id, seed_info)
                    for cwe_id, seed_info in all_seeds_info
                ]
                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing seed: {e}")
        else:
            # Single-threaded processing
            for cwe_id, seed_info in all_seeds_info:
                self._process_single_seed(cwe_id, seed_info)
        
        # Save to new file result_filtered.json
        if not dry_run:
            # Add suffix if provided
            if self.output_suffix:
                output_filename = f"result_filtered_{self.output_suffix}.json"
            else:
                output_filename = "result_filtered.json"
            
            output_path = self.batch_dir / output_filename
            with open(output_path, 'w') as f:
                json.dump(batch_data, f, indent=2)
            print(f"\nSaved to {output_path}")
            
            # Save filter report
            self._save_report()
        else:
            print("\n[DRY RUN] No file created")
        
        return self.filter_result
    
    def _save_report(self):
        """Save detailed filter report for paper writing - includes ALL seeds"""
        r = self.filter_result
        
        # Compute statistics for tested seeds
        mutation_rates = []
        coverage_rates = []
        for result in self.validation_results.values():
            mutation_rates.append(result.detection_rate)
            if result.coverage:
                coverage_rates.append(result.coverage.line_coverage)
        
        avg_mutation = sum(mutation_rates) / len(mutation_rates) if mutation_rates else 0
        avg_coverage = sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0
        
        # Collect ALL seeds from batch_data (including non-passed ones)
        all_seeds_details = []
        total_all_seeds = 0
        
        if "cwes" in self.batch_data:
            for cwe_id, cwe_data in self.batch_data["cwes"].items():
                for seed_info in cwe_data.get("seeds", []):
                    total_all_seeds += 1
                    seed_uuid = seed_info["seed_uuid"]
                    
                    # Check if this seed was tested
                    if seed_uuid in self.validation_results:
                        result = self.validation_results[seed_uuid]
                        all_seeds_details.append({
                            "seed_uuid": seed_uuid,
                            "mutation_rate": result.detection_rate,
                            "coverage": result.coverage.line_coverage if result.coverage else None,
                            "is_effective": result.is_effective,
                            "language": result.language,
                            "passed": seed_info.get("passed", False),
                            "tb_validated": seed_info.get("tb_validated", False)
                        })
                    else:
                        # Seed was not tested (didn't pass initial validation)
                        all_seeds_details.append({
                            "seed_uuid": seed_uuid,
                            "mutation_rate": None,
                            "coverage": None,
                            "is_effective": False,
                            "language": seed_info.get("language", "unknown"),
                            "passed": seed_info.get("passed", False),
                            "tb_validated": False
                        })
        else:
            for seed_info in self.batch_data.get("seeds", []):
                total_all_seeds += 1
                seed_uuid = seed_info["seed_uuid"]
                
                if seed_uuid in self.validation_results:
                    result = self.validation_results[seed_uuid]
                    all_seeds_details.append({
                        "seed_uuid": seed_uuid,
                        "mutation_rate": result.detection_rate,
                        "coverage": result.coverage.line_coverage if result.coverage else None,
                        "is_effective": result.is_effective,
                        "language": result.language,
                        "passed": seed_info.get("passed", False),
                        "tb_validated": seed_info.get("tb_validated", False)
                    })
                else:
                    all_seeds_details.append({
                        "seed_uuid": seed_uuid,
                        "mutation_rate": None,
                        "coverage": None,
                        "is_effective": False,
                        "language": seed_info.get("language", "unknown"),
                        "passed": seed_info.get("passed", False),
                        "tb_validated": False
                    })
        
        report = {
            "filter_config": {
                "mutation_threshold": self.mutation_threshold,
                "coverage_threshold": self.coverage_threshold,
                "mutations_per_seed": self.max_mutations
            },
            "summary": {
                "total_all_seeds": total_all_seeds,
                "total_tested_seeds": r.total_seeds,
                "validated_seeds": r.validated_seeds,
                "filtered_seeds": r.filtered_seeds,
                "validation_rate": r.validated_seeds / r.total_seeds if r.total_seeds > 0 else 0
            },
            "filter_breakdown": {
                "low_mutation_rate": r.low_mutation_rate,
                "low_coverage": r.low_coverage,
                "both_low": r.both_low,
                "no_data": r.no_data
            },
            "statistics": {
                "avg_mutation_detection_rate": avg_mutation,
                "avg_code_coverage": avg_coverage,
                "mutation_rates": {
                    "min": min(mutation_rates) if mutation_rates else 0,
                    "max": max(mutation_rates) if mutation_rates else 0,
                    "median": sorted(mutation_rates)[len(mutation_rates)//2] if mutation_rates else 0
                },
                "coverage_rates": {
                    "min": min(coverage_rates) if coverage_rates else 0,
                    "max": max(coverage_rates) if coverage_rates else 0,
                    "median": sorted(coverage_rates)[len(coverage_rates)//2] if coverage_rates else 0
                }
            },
            "seed_details": all_seeds_details
        }
        
        # Add suffix if provided
        if self.output_suffix:
            report_filename = f"filter_report_{self.output_suffix}.json"
        else:
            report_filename = "filter_report.json"
        
        report_path = self.batch_dir / report_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")
    
    def print_summary(self):
        """Print filtering summary"""
        r = self.filter_result
        
        print("\n" + "=" * 70)
        print("FILTERING SUMMARY")
        print("=" * 70)
        print(f"Total seeds:              {r.total_seeds}")
        print(f"Validated (passed):       {r.validated_seeds} ({r.validated_seeds/r.total_seeds:.1%})")
        print(f"Filtered (removed):       {r.filtered_seeds} ({r.filtered_seeds/r.total_seeds:.1%})")
        
        if r.filtered_seeds > 0:
            print(f"\nFilter breakdown:")
            print(f"  - Low mutation rate:    {r.low_mutation_rate}")
            print(f"  - Low coverage:         {r.low_coverage}")
            print(f"  - Both low:             {r.both_low}")
            print(f"  - No data:              {r.no_data}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter seeds by testbench quality metrics"
    )
    parser.add_argument("batch_path", help="Path to batch folder")
    parser.add_argument(
        "--mutations", "-m", type=int, default=5,
        help="Number of mutations per seed (default: 5)"
    )
    parser.add_argument(
        "--mutation-threshold", type=float, default=0.5,
        help="Minimum mutation detection rate (default: 0.5)"
    )
    parser.add_argument(
        "--coverage-threshold", type=float, default=0.8,
        help="Minimum code coverage rate (default: 0.8)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Disable coverage measurement"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't modify result.json, just show statistics"
    )
    parser.add_argument(
        "--suffix", "-s", type=str, default="",
        help="Suffix to add to output filenames (e.g., 'v2' -> result_filtered_v2.json)"
    )
    parser.add_argument(
        "--threads", "-j", type=int, default=1,
        help="Number of parallel threads (default: 1)"
    )
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_path)
    if not batch_dir.exists():
        print(f"Error: {batch_dir} does not exist")
        sys.exit(1)
    
    filter_obj = SeedFilter(
        batch_dir=batch_dir,
        mutation_threshold=args.mutation_threshold,
        coverage_threshold=args.coverage_threshold,
        max_mutations=args.mutations,
        enable_coverage=not args.no_coverage,
        output_suffix=args.suffix,
        num_threads=args.threads
    )
    
    filter_obj.validate_and_filter(dry_run=args.dry_run)
    filter_obj.print_summary()


if __name__ == "__main__":
    main()
