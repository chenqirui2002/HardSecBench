#!/usr/bin/env python3
"""
Validate Testbench Effectiveness via Mutation Testing

This script validates that testbenches are not "empty tests" by:
1. Loading golden code that passed all tests
2. Generating mutant code with injected bugs
3. Running the same TBs against mutant code
4. If TB detects the bug (FAIL), the TB is effective
5. If TB still passes (PASS), the TB is ineffective ("empty test")

Usage:
    python validate_tb_effectiveness.py <batch_folder_path> [--mutations N]
"""

import os
import sys
import json
import re
import subprocess
import signal
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


def safe_run_command(cmd, timeout=30, cwd=None):
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
            cwd=cwd,
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


class MutationType(Enum):
    """Types of mutations we can inject"""
    CONSTANT_CHANGE = "constant_change"      # Change constant values
    OPERATOR_SWAP = "operator_swap"          # Swap operators (+ <-> -, & <-> |)
    CONDITION_NEGATE = "condition_negate"    # Negate conditions
    SIGNAL_STUCK = "signal_stuck"            # Stuck-at-0 or stuck-at-1
    ASSIGNMENT_REMOVE = "assignment_remove"  # Remove an assignment


@dataclass
class MutationResult:
    """Result of a single mutation test"""
    mutation_type: str
    mutation_desc: str
    original_line: str
    mutated_line: str
    tb_file: str
    detected: bool  # True if TB detected the mutation (FAIL)
    output: str = ""


@dataclass
class CoverageResult:
    """Code coverage result for a single seed"""
    lines_total: int = 0
    lines_covered: int = 0
    branches_total: int = 0
    branches_covered: int = 0
    
    @property
    def line_coverage(self) -> float:
        if self.lines_total == 0:
            return 0.0
        return self.lines_covered / self.lines_total
    
    @property
    def branch_coverage(self) -> float:
        if self.branches_total == 0:
            return 0.0
        return self.branches_covered / self.branches_total


@dataclass
class MutationAggregateResult:
    """Aggregated result for a single mutation across all TBs"""
    mutation_type: str
    mutation_desc: str
    tb_results: Dict[str, bool] = field(default_factory=dict)  # tb_file -> detected
    
    @property
    def detected_by_any(self) -> bool:
        """True if ANY TB detected this mutation"""
        return any(self.tb_results.values())
    
    @property
    def detected_count(self) -> int:
        return sum(1 for v in self.tb_results.values() if v)


@dataclass
class SeedValidationResult:
    """Validation result for a single seed"""
    seed_uuid: str
    language: str
    golden_file: str
    total_mutations: int = 0
    detected_mutations: int = 0  # Mutations detected by ANY TB (aggregated)
    mutation_results: List[MutationResult] = field(default_factory=list)
    aggregated_results: List[MutationAggregateResult] = field(default_factory=list)
    
    # Per-TB metrics (for reference)
    per_tb_total: int = 0
    per_tb_detected: int = 0
    
    # Coverage metrics
    coverage: Optional[CoverageResult] = None
    
    @property
    def detection_rate(self) -> float:
        """Aggregated detection rate: mutation detected by ANY TB"""
        if self.total_mutations == 0:
            return 0.0
        return self.detected_mutations / self.total_mutations
    
    @property
    def per_tb_detection_rate(self) -> float:
        """Per-TB detection rate (old metric, for comparison)"""
        if self.per_tb_total == 0:
            return 0.0
        return self.per_tb_detected / self.per_tb_total
    
    @property
    def is_effective(self) -> bool:
        # Consider effective if aggregated detection rate >= 50%
        return self.detection_rate >= 0.5


class MutationGenerator:
    """Generate mutations for Verilog and C code"""
    
    def __init__(self):
        self.mutations_applied = []
    
    def generate_mutations(self, code: str, language: str, max_mutations: int = 3) -> List[Tuple[str, str, str]]:
        """
        Generate mutated versions of the code
        Collects multiple mutations per type for better coverage
        
        Returns: List of (mutated_code, mutation_type, mutation_description)
        """
        if language == "verilog":
            all_mutations = self._verilog_mutations(code, max_mutations * 5)
        elif language == "c":
            all_mutations = self._c_mutations(code, max_mutations * 5)
        else:
            return []
        
        # Group mutations by type
        mutations_by_type = {}
        for mutation in all_mutations:
            mut_type = mutation[1]
            if mut_type not in mutations_by_type:
                mutations_by_type[mut_type] = []
            mutations_by_type[mut_type].append(mutation)
        
        # Select mutations: prioritize diversity, then quantity
        # Take up to 2 mutations per type to ensure diversity
        selected = []
        mutations_per_type = max(1, max_mutations // max(1, len(mutations_by_type)))
        
        for mut_type, muts in mutations_by_type.items():
            selected.extend(muts[:mutations_per_type])
        
        return selected[:max_mutations]
    
    def _verilog_mutations(self, code: str, max_mutations: int) -> List[Tuple[str, str, str]]:
        """Generate Verilog-specific mutations with multiple types"""
        mutations = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if len(mutations) >= max_mutations:
                break
                
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            
            # Mutation 1: Change hex constants
            hex_match = re.search(r"(\d+'h)([0-9A-Fa-f]+)", line)
            if hex_match:
                original_val = hex_match.group(2)
                try:
                    val = int(original_val, 16)
                    # Try different bit flips
                    for flip in [0xFF, 0x01, 0x80]:
                        mutated_val = format(val ^ flip, 'x').upper()
                        if mutated_val != original_val.upper():
                            mutated_line = line.replace(
                                hex_match.group(0),
                                f"{hex_match.group(1)}{mutated_val}"
                            )
                            new_lines = lines.copy()
                            new_lines[i] = mutated_line
                            mutations.append((
                                '\n'.join(new_lines),
                                MutationType.CONSTANT_CHANGE.value,
                                f"Line {i+1}: Changed {hex_match.group(0)} to {hex_match.group(1)}{mutated_val}"
                            ))
                            break
                except:
                    pass
            
            # Mutation 2: Change binary constants
            bin_match = re.search(r"(\d+'b)([01]+)", line)
            if bin_match:
                original_val = bin_match.group(2)
                # Flip first bit
                flipped = ('0' if original_val[0] == '1' else '1') + original_val[1:]
                mutated_line = line.replace(bin_match.group(0), f"{bin_match.group(1)}{flipped}")
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.CONSTANT_CHANGE.value,
                    f"Line {i+1}: Flipped bit in {bin_match.group(0)}"
                ))
            
            # Mutation 3: Negate simple conditions
            if_match = re.search(r'if\s*\(\s*(\w+)\s*\)', line)
            if if_match:
                signal = if_match.group(1)
                mutated_line = line.replace(f'if ({signal})', f'if (!{signal})')
                if mutated_line != line:
                    new_lines = lines.copy()
                    new_lines[i] = mutated_line
                    mutations.append((
                        '\n'.join(new_lines),
                        MutationType.CONDITION_NEGATE.value,
                        f"Line {i+1}: Negated condition '{signal}'"
                    ))
            
            # Mutation 4: Negate negated conditions (!signal -> signal)
            neg_match = re.search(r'if\s*\(\s*!(\w+)\s*\)', line)
            if neg_match:
                signal = neg_match.group(1)
                mutated_line = line.replace(f'if (!{signal})', f'if ({signal})')
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.CONDITION_NEGATE.value,
                    f"Line {i+1}: Removed negation from '!{signal}'"
                ))
            
            # Mutation 5: Swap && with ||
            if '&&' in line:
                mutated_line = line.replace('&&', '||', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '&&' to '||'"
                ))
            
            # Mutation 6: Swap || with &&
            if '||' in line and '&&' not in line:
                mutated_line = line.replace('||', '&&', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '||' to '&&'"
                ))
            
            # Mutation 7: Stuck-at-0 for register assignments
            reg_assign = re.search(r'(\w+)\s*<=\s*([^;]+);', line)
            if reg_assign:
                reg_name = reg_assign.group(1)
                if "'h0" not in line and "'b0" not in line and "= 0" not in line:
                    mutated_line = re.sub(r'(\w+)\s*<=\s*[^;]+;', f"{reg_name} <= 0;", line)
                    new_lines = lines.copy()
                    new_lines[i] = mutated_line
                    mutations.append((
                        '\n'.join(new_lines),
                        MutationType.SIGNAL_STUCK.value,
                        f"Line {i+1}: Stuck {reg_name} at 0"
                    ))
            
            # Mutation 8: Stuck-at-1 for register assignments
            if reg_assign:
                reg_name = reg_assign.group(1)
                if "'hF" not in line.upper() and "= 1" not in line:
                    mutated_line = re.sub(r'(\w+)\s*<=\s*[^;]+;', f"{reg_name} <= ~0;", line)
                    new_lines = lines.copy()
                    new_lines[i] = mutated_line
                    mutations.append((
                        '\n'.join(new_lines),
                        MutationType.SIGNAL_STUCK.value,
                        f"Line {i+1}: Stuck {reg_name} at all 1s"
                    ))
            
            # Mutation 9: Remove else branch (comment out)
            if stripped.startswith('else') and 'if' not in stripped:
                mutated_line = '//' + line
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.ASSIGNMENT_REMOVE.value,
                    f"Line {i+1}: Commented out else branch"
                ))
        
        return mutations[:max_mutations]
    
    def _c_mutations(self, code: str, max_mutations: int) -> List[Tuple[str, str, str]]:
        """Generate C-specific mutations with multiple types"""
        mutations = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if len(mutations) >= max_mutations:
                break
            
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            
            # Mutation 1: Change numeric constants
            num_match = re.search(r'(?<![0-9a-fA-FxX])(\d+)(?![0-9a-fA-FxX])', line)
            if num_match:
                original_val = int(num_match.group(1))
                if original_val > 0:
                    for flip in [0xFF, 0x01, original_val + 1]:
                        mutated_val = original_val ^ flip if flip != original_val + 1 else flip
                        if mutated_val != original_val:
                            mutated_line = line[:num_match.start(1)] + str(mutated_val) + line[num_match.end(1):]
                            new_lines = lines.copy()
                            new_lines[i] = mutated_line
                            mutations.append((
                                '\n'.join(new_lines),
                                MutationType.CONSTANT_CHANGE.value,
                                f"Line {i+1}: Changed {original_val} to {mutated_val}"
                            ))
                            break
            
            # Mutation 2: Change hex constants
            hex_match = re.search(r'0x([0-9A-Fa-f]+)', line)
            if hex_match:
                original_val = hex_match.group(1)
                try:
                    val = int(original_val, 16)
                    mutated_val = format(val ^ 0xFF, 'x')
                    mutated_line = line.replace(f"0x{original_val}", f"0x{mutated_val}")
                    new_lines = lines.copy()
                    new_lines[i] = mutated_line
                    mutations.append((
                        '\n'.join(new_lines),
                        MutationType.CONSTANT_CHANGE.value,
                        f"Line {i+1}: Changed 0x{original_val} to 0x{mutated_val}"
                    ))
                except:
                    pass
            
            # Mutation 3: Swap == with !=
            if '==' in line:
                mutated_line = line.replace('==', '!=', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '==' to '!='"
                ))
            
            # Mutation 4: Swap != with ==
            if '!=' in line and '==' not in line:
                mutated_line = line.replace('!=', '==', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '!=' to '=='"
                ))
            
            # Mutation 5: Swap + with -
            if ' + ' in line:
                mutated_line = line.replace(' + ', ' - ', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '+' to '-'"
                ))
            
            # Mutation 6: Negate simple conditions
            if_match = re.search(r'if\s*\(\s*(\w+)\s*\)', line)
            if if_match:
                signal = if_match.group(1)
                mutated_line = line.replace(f'if ({signal})', f'if (!{signal})')
                if mutated_line != line:
                    new_lines = lines.copy()
                    new_lines[i] = mutated_line
                    mutations.append((
                        '\n'.join(new_lines),
                        MutationType.CONDITION_NEGATE.value,
                        f"Line {i+1}: Negated condition '{signal}'"
                    ))
            
            # Mutation 7: Swap && with ||
            if '&&' in line:
                mutated_line = line.replace('&&', '||', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '&&' to '||'"
                ))
            
            # Mutation 8: Swap || with &&
            if '||' in line and '&&' not in line:
                mutated_line = line.replace('||', '&&', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '||' to '&&'"
                ))
            
            # Mutation 9: Swap < with <=
            if ' < ' in line and '<=' not in line:
                mutated_line = line.replace(' < ', ' <= ', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '<' to '<='"
                ))
            
            # Mutation 10: Swap > with >=
            if ' > ' in line and '>=' not in line:
                mutated_line = line.replace(' > ', ' >= ', 1)
                new_lines = lines.copy()
                new_lines[i] = mutated_line
                mutations.append((
                    '\n'.join(new_lines),
                    MutationType.OPERATOR_SWAP.value,
                    f"Line {i+1}: Changed '>' to '>='"
                ))
        
        return mutations[:max_mutations]


class TBEffectivenessValidator:
    """Main validator class"""
    
    def __init__(self, batch_dir: Path, max_mutations: int = 3, enable_coverage: bool = True):
        self.batch_dir = batch_dir
        self.max_mutations = max_mutations
        self.enable_coverage = enable_coverage
        self.mutation_generator = MutationGenerator()
        self.results: List[SeedValidationResult] = []
    
    def run_verilog_test(self, golden_file: Path, tb_file: Path, work_dir: Path) -> Tuple[bool, str]:
        """Run Verilog test, return (passed, output)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sim_out = tmpdir_path / f"sim_{tb_file.stem}"
            
            try:
                # Compile
                compile_cmd = ["iverilog", "-o", str(sim_out), str(golden_file), str(tb_file)]
                returncode, stdout, stderr = safe_run_command(compile_cmd, timeout=30)
                
                if returncode != 0:
                    return True, f"Compile error: {stderr}"  # Compile error = mutation detected
                
                # Run
                returncode, stdout, stderr = safe_run_command(
                    ["vvp", str(sim_out)], timeout=15
                )
                
                output = stdout + stderr
                passed = self._check_pass_fail(output)
                return passed, output
                
            except Exception as e:
                return True, str(e)  # Error = mutation detected
    
    def run_c_test(self, golden_file: Path, tb_file: Path, work_dir: Path) -> Tuple[bool, str]:
        """Run C test, return (passed, output)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exec_out = tmpdir_path / f"test_{tb_file.stem}"
            
            try:
                # Compile
                compile_cmd = ["gcc", "-o", str(exec_out), str(golden_file), str(tb_file)]
                returncode, stdout, stderr = safe_run_command(compile_cmd, timeout=30)
                
                if returncode != 0:
                    return True, f"Compile error: {stderr}"  # Compile error = mutation detected
                
                # Run
                returncode, stdout, stderr = safe_run_command(
                    [str(exec_out)], timeout=15
                )
                
                output = stdout + stderr
                passed = self._check_pass_fail(output)
                return passed, output
                
            except Exception as e:
                return True, str(e)
    
    def _check_pass_fail(self, output: str) -> bool:
        """Check if test passed (True) or failed (False)"""
        output_upper = output.upper()
        if "FAIL" in output_upper:
            return False
        if "PASS" in output_upper:
            return True
        return False
    
    def measure_c_coverage(self, golden_file: Path, tb_files: List[Path], work_dir: Path) -> Optional[CoverageResult]:
        """
        Measure code coverage for C using gcov/llvm-cov
        Runs all TBs and aggregates coverage
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Copy golden file to temp dir
            temp_golden = tmpdir_path / golden_file.name
            temp_golden.write_text(golden_file.read_text())
            
            total_lines = set()
            covered_lines = set()
            
            for tb_file in tb_files:
                try:
                    # Copy TB file to temp dir
                    temp_tb = tmpdir_path / tb_file.name
                    temp_tb.write_text(tb_file.read_text())
                    
                    # Compile with coverage flags, use -fprofile-arcs -ftest-coverage
                    # and specify object file location to control gcno/gcda naming
                    obj_golden = tmpdir_path / f"{golden_file.stem}.o"
                    obj_tb = tmpdir_path / f"{tb_file.stem}.o"
                    exec_out = tmpdir_path / "test_exe"
                    
                    # Compile golden file to object
                    returncode, _, _ = safe_run_command(
                        ["gcc", "-fprofile-arcs", "-ftest-coverage", "-c",
                         "-o", str(obj_golden), str(temp_golden)],
                        timeout=30, cwd=str(tmpdir_path)
                    )
                    if returncode != 0:
                        continue
                    
                    # Compile TB file to object
                    returncode, _, _ = safe_run_command(
                        ["gcc", "-fprofile-arcs", "-ftest-coverage", "-c",
                         "-o", str(obj_tb), str(temp_tb)],
                        timeout=30, cwd=str(tmpdir_path)
                    )
                    if returncode != 0:
                        continue
                    
                    # Link
                    returncode, _, _ = safe_run_command(
                        ["gcc", "-fprofile-arcs", "-ftest-coverage",
                         "-o", str(exec_out), str(obj_golden), str(obj_tb)],
                        timeout=30, cwd=str(tmpdir_path)
                    )
                    if returncode != 0:
                        continue
                    
                    # Run test
                    safe_run_command(
                        [str(exec_out)], timeout=15, cwd=str(tmpdir_path)
                    )
                    
                    # Run gcov on the golden file
                    gcov_commands = [
                        ["xcrun", "llvm-cov", "gcov", temp_golden.name],
                        ["llvm-cov", "gcov", temp_golden.name],
                        ["gcov", temp_golden.name],
                    ]
                    
                    for gcov_cmd in gcov_commands:
                        returncode, _, _ = safe_run_command(
                            gcov_cmd, timeout=30, cwd=str(tmpdir_path)
                        )
                        if returncode == 0:
                            break
                    
                    # Parse gcov output file
                    gcov_file = tmpdir_path / f"{golden_file.name}.gcov"
                    if gcov_file.exists():
                        self._parse_gcov_file(gcov_file, total_lines, covered_lines)
                    
                    # Clean up for next TB
                    for pattern in ["*.gcda", "*.gcno", "*.gcov", "*.o"]:
                        for f in tmpdir_path.glob(pattern):
                            f.unlink()
                    if exec_out.exists():
                        exec_out.unlink()
                        
                except Exception:
                    continue
            
            if not total_lines:
                return None
            
            return CoverageResult(
                lines_total=len(total_lines),
                lines_covered=len(covered_lines)
            )
    
    def measure_verilog_coverage(self, golden_file: Path, tb_files: List[Path], work_dir: Path) -> Optional[CoverageResult]:
        """
        Estimate code coverage for Verilog by analyzing TB signal coverage
        This is a static analysis approach - checks which signals/conditions TBs exercise
        """
        try:
            golden_code = golden_file.read_text()
            
            # Extract executable lines from golden code (assignments, conditions)
            executable_lines = set()
            covered_lines = set()
            
            lines = golden_code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip comments, empty lines, module/endmodule declarations
                if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                    continue
                if stripped.startswith('module') or stripped.startswith('endmodule'):
                    continue
                if stripped.startswith('input') or stripped.startswith('output') or stripped.startswith('reg') or stripped.startswith('wire'):
                    continue
                    
                # Count assignments and conditions as executable
                if '<=' in stripped or '=' in stripped or 'if' in stripped or 'else' in stripped or 'case' in stripped:
                    executable_lines.add(i + 1)
            
            # Analyze TB files to see which signals they test
            tested_signals = set()
            for tb_file in tb_files:
                tb_code = tb_file.read_text()
                # Find signal assignments in TB (driving DUT inputs)
                signal_matches = re.findall(r'(\w+)\s*[<]?=', tb_code)
                tested_signals.update(signal_matches)
                # Find signal checks in TB (reading DUT outputs)
                check_matches = re.findall(r'if\s*\([^)]*(\w+)', tb_code)
                tested_signals.update(check_matches)
            
            # Estimate coverage: lines that reference tested signals
            for i, line in enumerate(lines):
                if i + 1 not in executable_lines:
                    continue
                for signal in tested_signals:
                    if signal in line:
                        covered_lines.add(i + 1)
                        break
            
            if not executable_lines:
                return None
            
            return CoverageResult(
                lines_total=len(executable_lines),
                lines_covered=len(covered_lines)
            )
        except Exception:
            return None
    
    def _parse_gcov_file(self, gcov_file: Path, total_lines: set, covered_lines: set):
        """Parse gcov output file to extract coverage info"""
        try:
            content = gcov_file.read_text()
            for line in content.split('\n'):
                # gcov format: "count:line_number:source"
                # -: not executable, #####: not covered, number: covered
                parts = line.split(':', 2)
                if len(parts) < 2:
                    continue
                
                count_str = parts[0].strip()
                try:
                    line_num = int(parts[1].strip())
                except ValueError:
                    continue
                
                if count_str == '-':
                    # Not executable line
                    continue
                elif count_str == '#####':
                    # Executable but not covered
                    total_lines.add(line_num)
                else:
                    # Covered (has execution count)
                    try:
                        int(count_str)
                        total_lines.add(line_num)
                        covered_lines.add(line_num)
                    except ValueError:
                        pass
        except Exception:
            pass
    
    def validate_seed(self, seed_dir: Path, seed_info: dict) -> SeedValidationResult:
        """Validate a single seed's testbenches with aggregated metrics"""
        language = seed_info.get("language", "verilog")
        golden_filename = seed_info.get("golden_file", "")
        
        result = SeedValidationResult(
            seed_uuid=seed_dir.name,
            language=language,
            golden_file=golden_filename
        )
        
        # Find golden file
        golden_file = seed_dir / golden_filename
        if not golden_file.exists():
            print(f"  Warning: Golden file not found: {golden_file}")
            return result
        
        # Read golden code
        golden_code = golden_file.read_text()
        
        # Find TB files
        ext = ".v" if language == "verilog" else ".c"
        tb_files = list(seed_dir.glob(f"tb_*{ext}"))
        
        if not tb_files:
            print(f"  Warning: No TB files found in {seed_dir}")
            return result
        
        # Generate mutations
        mutations = self.mutation_generator.generate_mutations(
            golden_code, language, self.max_mutations
        )
        
        if not mutations:
            print(f"  Warning: Could not generate mutations for {golden_filename}")
            return result
        
        # Test each mutation against ALL TBs
        run_test = self.run_verilog_test if language == "verilog" else self.run_c_test
        
        for mutated_code, mutation_type, mutation_desc in mutations:
            # Write mutated code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix=ext, delete=False
            ) as tmp_file:
                tmp_file.write(mutated_code)
                mutant_file = Path(tmp_file.name)
            
            # Aggregate result for this mutation
            agg_result = MutationAggregateResult(
                mutation_type=mutation_type,
                mutation_desc=mutation_desc
            )
            
            try:
                # Test against each TB
                for tb_file in tb_files:
                    result.per_tb_total += 1
                    
                    passed, output = run_test(mutant_file, tb_file, seed_dir)
                    
                    # If TB fails on mutant, it detected the mutation
                    detected = not passed
                    if detected:
                        result.per_tb_detected += 1
                    
                    agg_result.tb_results[tb_file.name] = detected
                    
                    result.mutation_results.append(MutationResult(
                        mutation_type=mutation_type,
                        mutation_desc=mutation_desc,
                        original_line="",
                        mutated_line="",
                        tb_file=tb_file.name,
                        detected=detected,
                        output=output[:200] if output else ""
                    ))
            finally:
                mutant_file.unlink()
            
            # Add aggregated result
            result.aggregated_results.append(agg_result)
        
        # Calculate aggregated metrics
        result.total_mutations = len(result.aggregated_results)
        result.detected_mutations = sum(1 for a in result.aggregated_results if a.detected_by_any)
        
        # Measure coverage
        if self.enable_coverage:
            if language == "c":
                result.coverage = self.measure_c_coverage(golden_file, tb_files, seed_dir)
            elif language == "verilog":
                result.coverage = self.measure_verilog_coverage(golden_file, tb_files, seed_dir)
        
        return result
    
    def validate_batch(self) -> Dict:
        """Validate all seeds in the batch"""
        # Load result.json
        result_json = self.batch_dir / "result.json"
        if not result_json.exists():
            print(f"Error: result.json not found in {self.batch_dir}")
            return {}
        
        with open(result_json) as f:
            batch_data = json.load(f)
        
        # Support both old format (seeds at root) and new format (cwes -> seeds)
        all_seeds = []
        if "cwes" in batch_data:
            # New format: cwes/CWE-xxx/seeds
            for cwe_id, cwe_data in batch_data["cwes"].items():
                for seed_info in cwe_data.get("seeds", []):
                    if seed_info.get("passed", False):
                        seed_info["_cwe_id"] = cwe_id.lower()
                        all_seeds.append(seed_info)
        else:
            # Old format: seeds at root level
            all_seeds = [s for s in batch_data.get("seeds", []) if s.get("passed", False)]
        
        print(f"Validating TB effectiveness for {len(all_seeds)} passed seeds...")
        print(f"Mutations per seed: {self.max_mutations}")
        print("-" * 60)
        
        for seed_info in all_seeds:
            seed_uuid = seed_info["seed_uuid"]
            
            # Determine seed directory path based on format
            if "_cwe_id" in seed_info:
                # New format: cwe-xxx/<uuid>/
                seed_dir = self.batch_dir / seed_info["_cwe_id"] / seed_uuid
            else:
                # Old format: <uuid>/
                seed_dir = self.batch_dir / seed_uuid
            
            if not seed_dir.exists():
                print(f"[SKIP] {seed_uuid}: Directory not found")
                continue
            
            print(f"[TEST] {seed_uuid}...", end=" ", flush=True)
            
            result = self.validate_seed(seed_dir, seed_info)
            self.results.append(result)
            
            status = "EFFECTIVE" if result.is_effective else "WEAK"
            cov_str = ""
            if result.coverage:
                cov_str = f", coverage: {result.coverage.line_coverage:.1%}"
            print(f"{status} (aggregated: {result.detection_rate:.1%} [{result.detected_mutations}/{result.total_mutations}], "
                  f"per-tb: {result.per_tb_detection_rate:.1%} [{result.per_tb_detected}/{result.per_tb_total}]{cov_str})")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate validation report with both aggregated and per-TB metrics"""
        total_seeds = len(self.results)
        effective_seeds = sum(1 for r in self.results if r.is_effective)
        
        # Aggregated metrics (mutation detected by ANY TB)
        total_mutations = sum(r.total_mutations for r in self.results)
        total_detected = sum(r.detected_mutations for r in self.results)
        
        # Per-TB metrics (for comparison)
        per_tb_total = sum(r.per_tb_total for r in self.results)
        per_tb_detected = sum(r.per_tb_detected for r in self.results)
        
        # Coverage metrics
        seeds_with_coverage = [r for r in self.results if r.coverage]
        avg_line_coverage = 0.0
        if seeds_with_coverage:
            avg_line_coverage = sum(r.coverage.line_coverage for r in seeds_with_coverage) / len(seeds_with_coverage)
        
        report = {
            "summary": {
                "total_seeds_validated": total_seeds,
                "effective_seeds": effective_seeds,
                "weak_seeds": total_seeds - effective_seeds,
                "effectiveness_rate": effective_seeds / total_seeds if total_seeds > 0 else 0,
                # Aggregated metrics (main metric)
                "total_mutations": total_mutations,
                "mutations_detected_by_any_tb": total_detected,
                "aggregated_detection_rate": total_detected / total_mutations if total_mutations > 0 else 0,
                # Per-TB metrics (for reference)
                "per_tb_total_tests": per_tb_total,
                "per_tb_detected": per_tb_detected,
                "per_tb_detection_rate": per_tb_detected / per_tb_total if per_tb_total > 0 else 0,
                # Coverage metrics
                "seeds_with_coverage": len(seeds_with_coverage),
                "average_line_coverage": avg_line_coverage
            },
            "seed_results": []
        }
        
        for result in self.results:
            seed_report = {
                "seed_uuid": result.seed_uuid,
                "language": result.language,
                "golden_file": result.golden_file,
                # Aggregated metrics
                "total_mutations": result.total_mutations,
                "detected_by_any_tb": result.detected_mutations,
                "aggregated_detection_rate": result.detection_rate,
                "is_effective": result.is_effective,
                # Per-TB metrics
                "per_tb_total": result.per_tb_total,
                "per_tb_detected": result.per_tb_detected,
                "per_tb_detection_rate": result.per_tb_detection_rate,
                # Coverage metrics
                "coverage": {
                    "lines_total": result.coverage.lines_total,
                    "lines_covered": result.coverage.lines_covered,
                    "line_coverage": result.coverage.line_coverage
                } if result.coverage else None,
                # Aggregated mutation details
                "mutations": [
                    {
                        "type": a.mutation_type,
                        "description": a.mutation_desc,
                        "detected_by_any": a.detected_by_any,
                        "detected_count": a.detected_count,
                        "total_tbs": len(a.tb_results),
                        "tb_results": a.tb_results
                    }
                    for a in result.aggregated_results
                ]
            }
            report["seed_results"].append(seed_report)
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate TB effectiveness via mutation testing"
    )
    parser.add_argument("batch_path", help="Path to batch folder")
    parser.add_argument(
        "--mutations", "-m", type=int, default=3,
        help="Number of mutations per seed (default: 3)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Disable coverage measurement"
    )
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_path)
    if not batch_dir.exists():
        print(f"Error: {batch_dir} does not exist")
        sys.exit(1)
    
    validator = TBEffectivenessValidator(
        batch_dir, 
        args.mutations,
        enable_coverage=not args.no_coverage
    )
    report = validator.validate_batch()
    
    if not report:
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TB EFFECTIVENESS VALIDATION REPORT")
    print("=" * 60)
    
    summary = report["summary"]
    print(f"Seeds Validated:     {summary['total_seeds_validated']}")
    print(f"Effective Seeds:     {summary['effective_seeds']} ({summary['effectiveness_rate']:.1%})")
    print(f"Weak Seeds:          {summary['weak_seeds']}")
    
    print("\n--- Aggregated Metrics (Per-Requirement TBs Combined) ---")
    print(f"Total Mutations:     {summary['total_mutations']}")
    print(f"Detected by ANY TB:  {summary['mutations_detected_by_any_tb']} ({summary['aggregated_detection_rate']:.1%})")
    
    print("\n--- Per-TB Metrics (For Reference) ---")
    print(f"Total TB Tests:      {summary['per_tb_total_tests']}")
    print(f"Individual Detected: {summary['per_tb_detected']} ({summary['per_tb_detection_rate']:.1%})")
    
    if summary['seeds_with_coverage'] > 0:
        print("\n--- Coverage Metrics (C Language Only) ---")
        print(f"Seeds with Coverage: {summary['seeds_with_coverage']}")
        print(f"Average Line Coverage: {summary['average_line_coverage']:.1%}")
    
    # List weak seeds
    weak_seeds = [r for r in report["seed_results"] if not r["is_effective"]]
    if weak_seeds:
        print(f"\nWEAK SEEDS ({len(weak_seeds)}):")
        for seed in weak_seeds:
            print(f"  {seed['seed_uuid']}: {seed['aggregated_detection_rate']:.1%} aggregated rate")
    
    # Save report
    output_file = batch_dir / "tb_effectiveness_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {output_file}")
    
    # Exit code based on effectiveness
    if summary['effectiveness_rate'] < 0.5:
        print("\nWARNING: Overall TB effectiveness is below 50%!")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
