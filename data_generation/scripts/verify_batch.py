#!/usr/bin/env python3
"""
Verify all golden code passes their testbenches in a batch output folder.
Usage: python verify_batch.py <batch_folder_path>
"""

import os
import sys
import subprocess
import signal
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def safe_run_command(cmd, timeout=30):
    """
    Run command with proper timeout and process cleanup.
    Returns (returncode, stdout, stderr)
    """
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return -1, "", "Timeout"
    except Exception as e:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            process.wait()
        return -1, "", str(e)


def run_verilog_test(golden_file: Path, tb_file: Path, work_dir: Path) -> dict:
    """Run a single Verilog testbench using iverilog."""
    result = {
        "golden": golden_file.name,
        "testbench": tb_file.name,
        "passed": False,
        "error": None
    }
    
    sim_out = work_dir / f"sim_{tb_file.stem}"
    
    try:
        compile_cmd = ["iverilog", "-o", str(sim_out), str(golden_file), str(tb_file)]
        returncode, stdout, stderr = safe_run_command(compile_cmd, timeout=30)
        
        if returncode != 0:
            result["error"] = f"Compile error: {stderr}"
            return result
        
        returncode, stdout, stderr = safe_run_command(
            ["vvp", str(sim_out)], timeout=15
        )
        
        output = stdout + stderr
        result["passed"] = check_test_output(output, returncode)
        if not result["passed"]:
            result["error"] = f"Test failed: {output[-500:]}"
            
    except Exception as e:
        result["error"] = str(e)
    finally:
        if sim_out.exists():
            sim_out.unlink()
        # Clean up VCD files
        for vcd_file in work_dir.glob("*.vcd"):
            try:
                vcd_file.unlink()
            except Exception:
                pass
    
    return result


def run_c_test(golden_file: Path, tb_file: Path, work_dir: Path) -> dict:
    """Run a single C testbench using gcc."""
    result = {
        "golden": golden_file.name,
        "testbench": tb_file.name,
        "passed": False,
        "error": None
    }
    
    exec_out = work_dir / f"test_{tb_file.stem}"
    
    try:
        compile_cmd = ["gcc", "-o", str(exec_out), str(golden_file), str(tb_file)]
        returncode, stdout, stderr = safe_run_command(compile_cmd, timeout=30)
        
        if returncode != 0:
            result["error"] = f"Compile error: {stderr}"
            return result
        
        returncode, stdout, stderr = safe_run_command(
            [str(exec_out)], timeout=15
        )
        
        output = stdout + stderr
        result["passed"] = check_test_output(output, returncode)
        if not result["passed"]:
            result["error"] = f"Test failed: {output[-500:]}"
            
    except Exception as e:
        result["error"] = str(e)
    finally:
        if exec_out.exists():
            exec_out.unlink()
    
    return result


def check_test_output(output: str, returncode: int) -> bool:
    """Check if test output indicates pass or fail."""
    output_upper = output.upper()
    if "FAIL" in output_upper:
        return False
    if "PASS" in output_upper:
        return True
    return returncode == 0


def verify_seed_folder(seed_dir: Path) -> dict:
    """Verify all tests in a seed folder."""
    results = {
        "seed_uuid": seed_dir.name,
        "tests": [],
        "total": 0,
        "passed": 0,
        "failed": 0,
        "language": None
    }
    
    # Detect language: Verilog (.v) or C (.c)
    verilog_golden = [f for f in seed_dir.glob("*.v") if not f.name.startswith("tb_")]
    c_golden = [f for f in seed_dir.glob("*.c") if not f.name.startswith("tb_")]
    
    if verilog_golden:
        results["language"] = "verilog"
        golden_file = verilog_golden[0]
        tb_files = list(seed_dir.glob("tb_*.v"))
        run_test = run_verilog_test
    elif c_golden:
        results["language"] = "c"
        golden_file = c_golden[0]
        tb_files = list(seed_dir.glob("tb_*.c"))
        run_test = run_c_test
    else:
        results["error"] = "No golden file found"
        return results
    
    results["total"] = len(tb_files)
    
    for tb_file in tb_files:
        test_result = run_test(golden_file, tb_file, seed_dir)
        results["tests"].append(test_result)
        if test_result["passed"]:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_batch.py <batch_folder_path>")
        sys.exit(1)
    
    batch_dir = Path(sys.argv[1])
    if not batch_dir.exists():
        print(f"Error: {batch_dir} does not exist")
        sys.exit(1)
    
    # Load result.json to get claimed passed seeds
    result_json_path = batch_dir / "result.json"
    claimed_passed = {}  # seed_uuid -> claimed passed status
    if result_json_path.exists():
        with open(result_json_path) as f:
            result_data = json.load(f)
            for seed in result_data.get("seeds", []):
                claimed_passed[seed["seed_uuid"]] = seed.get("passed", False)
    
    # Find all seed folders (UUID format directories)
    seed_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(seed_dirs)} seed folders in {batch_dir}")
    print(f"Claimed passed in result.json: {sum(claimed_passed.values())}")
    print("-" * 60)
    
    all_results = []
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    # Track consistency with result.json
    consistent = []
    inconsistent = []
    
    for seed_dir in sorted(seed_dirs):
        result = verify_seed_folder(seed_dir)
        all_results.append(result)
        
        total_tests += result["total"]
        total_passed += result["passed"]
        total_failed += result["failed"]
        
        actual_passed = result["failed"] == 0
        claimed = claimed_passed.get(seed_dir.name, None)
        
        status = "PASS" if actual_passed else "FAIL"
        claimed_str = f"(claimed: {'PASS' if claimed else 'FAIL'})" if claimed is not None else ""
        print(f"[{status}] {seed_dir.name}: {result['passed']}/{result['total']} tests passed {claimed_str}")
        
        # Check consistency
        if claimed is not None:
            if claimed == actual_passed:
                consistent.append(seed_dir.name)
            else:
                inconsistent.append({
                    "seed_uuid": seed_dir.name,
                    "claimed": claimed,
                    "actual": actual_passed
                })
        
        # Print failed tests
        for test in result["tests"]:
            if not test["passed"]:
                print(f"       FAILED: {test['testbench']}")
                if test["error"]:
                    print(f"         Error: {test['error'][:200]}")
    
    print("-" * 60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print(f"Seeds with all tests passing: {sum(1 for r in all_results if r['failed'] == 0)}/{len(seed_dirs)}")
    
    # Report consistency with result.json
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK WITH result.json")
    print("=" * 60)
    print(f"Consistent: {len(consistent)}/{len(consistent) + len(inconsistent)}")
    
    if inconsistent:
        print(f"\nINCONSISTENT SEEDS ({len(inconsistent)}):")
        for item in inconsistent:
            claimed_str = "PASS" if item["claimed"] else "FAIL"
            actual_str = "PASS" if item["actual"] else "FAIL"
            print(f"  {item['seed_uuid']}: claimed {claimed_str}, actual {actual_str}")
    else:
        print("\nAll 'passed: true' seeds verified successfully!")
    
    # Save detailed results
    output_file = batch_dir / "verification_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_seeds": len(seed_dirs),
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "consistent_with_result_json": len(inconsistent) == 0
            },
            "inconsistent_seeds": inconsistent,
            "results": all_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    sys.exit(0 if len(inconsistent) == 0 else 1)


if __name__ == "__main__":
    main()
