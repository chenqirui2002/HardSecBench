"""
Compiler and verification tools for code generation
"""
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(str, Enum):
    """Supported programming languages"""
    VERILOG = "verilog"
    C = "c"


@dataclass
class CompileResult:
    """Result of compilation attempt"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    
    @property
    def has_errors(self) -> bool:
        """Check if compilation had errors"""
        return not self.success or self.exit_code != 0
    
    @property
    def errors(self) -> str:
        """Get error messages"""
        return self.stderr if self.stderr else ""
    
    @property
    def warnings(self) -> str:
        """Get warning messages"""
        # Simple heuristic: lines with 'warning' in stderr
        if not self.stderr:
            return ""
        lines = [l for l in self.stderr.split('\n') if 'warning' in l.lower()]
        return '\n'.join(lines)


@dataclass
class SimulationResult:
    """Result of simulation/execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float = 0.0


class CompilerTools:
    """Tools for compiling and verifying generated code"""
    
    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize compiler tools
        
        Args:
            work_dir: Working directory for compilation (default: current dir)
        """
        self.work_dir = work_dir or Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_command(
        self,
        cmd: str,
        cwd: Optional[Path] = None,
        timeout: int = 30
    ) -> Tuple[int, str, str]:
        """
        Run shell command and capture output with proper process cleanup
        
        Args:
            cmd: Command to run
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            (exit_code, stdout, stderr)
        """
        import os
        import signal
        
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd or self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            if process is not None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                process.wait()
            return -1, "", f"Command failed: {str(e)}"
    
    def syntax_check_verilog(
        self,
        file_path: Path,
        top_module: Optional[str] = None
    ) -> CompileResult:
        """
        Check Verilog syntax without generating output
        Useful for validating standalone modules
        
        Args:
            file_path: Path to Verilog file
            top_module: Top module name (optional)
            
        Returns:
            CompileResult with syntax check status
        """
        # Convert to Path and make absolute
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.work_dir / file_path
        
        if not file_path.exists():
            return CompileResult(
                success=False,
                stdout="",
                stderr=f"File not found: {file_path}",
                exit_code=-1
            )
        
        # Use iverilog with -t null for syntax check only
        # Add -g2012 for SystemVerilog support (typedef enum, etc.)
        cmd = f"iverilog -g2012 -t null -Wall"
        if top_module:
            cmd += f" -s {top_module}"
        cmd += f" '{file_path.absolute()}'"
        
        exit_code, stdout, stderr = self._run_command(cmd)
        
        return CompileResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def compile_verilog(
        self,
        file_path: Path,
        top_module: Optional[str] = None
    ) -> CompileResult:
        """
        Compile Verilog code using Icarus Verilog
        Alias for syntax_check_verilog for compatibility
        
        Args:
            file_path: Path to Verilog file
            top_module: Top module name (optional)
            
        Returns:
            CompileResult with compilation status
        """
        return self.syntax_check_verilog(file_path, top_module)
    
    def compile_c(
        self,
        file_path: Path,
        output_path: Optional[Path] = None,
        extra_flags: Optional[str] = None
    ) -> CompileResult:
        """
        Compile C code using GCC
        
        Args:
            file_path: Path to C source file
            output_path: Output executable path (optional)
            extra_flags: Additional compiler flags
            
        Returns:
            CompileResult with compilation status
        """
        # Convert to Path and make absolute
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.work_dir / file_path
        
        if not file_path.exists():
            return CompileResult(
                success=False,
                stdout="",
                stderr=f"File not found: {file_path}",
                exit_code=-1
            )
        
        # Default output path
        if output_path is None:
            output_path = file_path.with_suffix('.out')
        
        # Make output path absolute
        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = self.work_dir / output_path
        
        # Build gcc command
        cmd = f"gcc -Wall -Wextra"
        if extra_flags:
            cmd += f" {extra_flags}"
        cmd += f" -o '{output_path.absolute()}' '{file_path.absolute()}'"
        
        exit_code, stdout, stderr = self._run_command(cmd)
        
        return CompileResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def compile_c_to_object(
        self,
        file_path: Path,
        output_path: Optional[Path] = None
    ) -> CompileResult:
        """
        Compile C code to object file (.o) without linking
        Useful for validating standalone functions without main()
        
        Args:
            file_path: Path to C source file
            output_path: Output object file path (optional)
            
        Returns:
            CompileResult with compilation status
        """
        # Convert to Path and make absolute
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.work_dir / file_path
        
        if not file_path.exists():
            return CompileResult(
                success=False,
                stdout="",
                stderr=f"File not found: {file_path}",
                exit_code=-1
            )
        
        # Default output path (same dir as source)
        if output_path is None:
            output_path = file_path.with_suffix('.o')
        else:
            # Make output path absolute
            if isinstance(output_path, str):
                output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.work_dir / output_path
        
        # Compile to object file only (-c flag)
        cmd = f"gcc -Wall -Wextra -c -o '{output_path.absolute()}' '{file_path.absolute()}'"
        exit_code, stdout, stderr = self._run_command(cmd)
        
        return CompileResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def simulate_verilog(
        self,
        design_file: Path,
        testbench_file: Path,
        output_vcd: Optional[Path] = None
    ) -> SimulationResult:
        """
        Run Verilog simulation using Icarus Verilog
        
        Args:
            design_file: Design file path
            testbench_file: Testbench file path
            output_vcd: VCD output file (optional)
            
        Returns:
            SimulationResult with simulation status
        """
        import time
        
        # Convert to Path and make absolute
        if isinstance(design_file, str):
            design_file = Path(design_file)
        if not design_file.is_absolute():
            design_file = self.work_dir / design_file
        
        if isinstance(testbench_file, str):
            testbench_file = Path(testbench_file)
        if not testbench_file.is_absolute():
            testbench_file = self.work_dir / testbench_file
        
        if not design_file.exists() or not testbench_file.exists():
            return SimulationResult(
                success=False,
                stdout="",
                stderr="Design or testbench file not found",
                exit_code=-1
            )
        
        # Compile with SystemVerilog support
        sim_out = self.work_dir / "sim.out"
        cmd = f"iverilog -g2012 -o '{sim_out.absolute()}' '{design_file.absolute()}' '{testbench_file.absolute()}'"
        exit_code, stdout, stderr = self._run_command(cmd)
        
        if exit_code != 0:
            return SimulationResult(
                success=False,
                stdout=stdout,
                stderr=f"Compilation failed:\n{stderr}",
                exit_code=exit_code
            )
        
        # Run simulation
        start_time = time.time()
        cmd = f"vvp '{sim_out.absolute()}'"
        if output_vcd:
            cmd += f" -vcd '{output_vcd.absolute()}'"
        
        exit_code, stdout, stderr = self._run_command(cmd, timeout=60)
        execution_time = time.time() - start_time
        
        # Clean up
        if sim_out.exists():
            sim_out.unlink()
        
        # Clean up VCD files generated by testbench (prevent disk space issues)
        if not output_vcd:  # Only clean if user didn't explicitly request VCD
            for vcd_file in self.work_dir.glob("*.vcd"):
                try:
                    vcd_file.unlink()
                except Exception:
                    pass
        
        return SimulationResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=execution_time
        )
    
    def run_c_executable(
        self,
        exe_path: Path,
        args: Optional[str] = None
    ) -> SimulationResult:
        """
        Run compiled C executable
        
        Args:
            exe_path: Path to executable
            args: Command line arguments
            
        Returns:
            SimulationResult with execution status
        """
        import time
        
        # Convert to Path and make absolute
        if isinstance(exe_path, str):
            exe_path = Path(exe_path)
        if not exe_path.is_absolute():
            exe_path = self.work_dir / exe_path
        
        if not exe_path.exists():
            return SimulationResult(
                success=False,
                stdout="",
                stderr=f"Executable not found: {exe_path}",
                exit_code=-1
            )
        
        cmd = str(exe_path.absolute())
        if args:
            cmd += f" {args}"
        
        start_time = time.time()
        exit_code, stdout, stderr = self._run_command(cmd, timeout=30)
        execution_time = time.time() - start_time
        
        return SimulationResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=execution_time
        )
