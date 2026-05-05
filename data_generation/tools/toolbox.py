"""
Unified toolbox for code generation agents
"""
from pathlib import Path
from typing import Optional
from .compiler_tools import CompilerTools, CompileResult, SimulationResult, Language
from .file_tools import FileTools


class AgentToolbox:
    """
    Unified toolbox providing all tools for code generation agents
    """
    
    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize toolbox
        
        Args:
            work_dir: Working directory for all operations
        """
        self.work_dir = work_dir or Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tool modules
        self.compiler = CompilerTools(work_dir=self.work_dir)
        self.files = FileTools(base_dir=self.work_dir)
    
    # ============ File Operations ============
    
    def write_code(
        self,
        file_path: Path,
        content: str,
        language: str,
        overwrite: bool = True
    ) -> bool:
        """Write code to file"""
        return self.files.write_code(file_path, content, language, overwrite)
    
    def read_code(self, file_path: Path) -> str:
        """Read code from file"""
        return self.files.read_code(file_path)
    
    # ============ Compilation & Validation ============
    
    # Agent B methods: Validate standalone code
    def validate_verilog_module(
        self,
        file_path: Path,
        module_name: Optional[str] = None
    ) -> CompileResult:
        """
        Validate standalone Verilog module (Agent B)
        Checks syntax and synthesizability
        """
        return self.compiler.syntax_check_verilog(file_path, module_name)
    
    def validate_c_function(
        self,
        file_path: Path
    ) -> CompileResult:
        """
        Validate standalone C function (Agent B)
        Compiles to object file without requiring main()
        """
        return self.compiler.compile_c_to_object(file_path)
    
    # Legacy methods for compatibility
    def compile_verilog(
        self,
        file_path: Path,
        top_module: Optional[str] = None
    ) -> CompileResult:
        """Compile Verilog code"""
        return self.compiler.compile_verilog(file_path, top_module)
    
    def syntax_check_verilog(
        self,
        file_path: Path,
        top_module: Optional[str] = None
    ) -> CompileResult:
        """Check Verilog syntax"""
        return self.compiler.syntax_check_verilog(file_path, top_module)
    
    def compile_c(
        self,
        file_path: Path,
        output_path: Optional[Path] = None,
        extra_flags: Optional[str] = None
    ) -> CompileResult:
        """Compile C code"""
        return self.compiler.compile_c(file_path, output_path, extra_flags)
    
    def compile_c_to_object(
        self,
        file_path: Path,
        output_path: Optional[Path] = None
    ) -> CompileResult:
        """Compile C to object file"""
        return self.compiler.compile_c_to_object(file_path, output_path)
    
    # ============ Simulation/Execution (Agent C) ============
    
    def test_verilog_with_testbench(
        self,
        design_file: Path,
        testbench_file: Path,
        output_vcd: Optional[Path] = None
    ) -> SimulationResult:
        """
        Test Verilog design with testbench (Agent C)
        Runs simulation and returns results
        """
        return self.compiler.simulate_verilog(design_file, testbench_file, output_vcd)
    
    def test_c_function_with_main(
        self,
        function_file: Path,
        test_main_file: Path,
        exe_path: Optional[Path] = None
    ) -> tuple[CompileResult, Optional[SimulationResult]]:
        """
        Test C function with test main (Agent C)
        Compiles function + test together and runs
        
        Args:
            function_file: C file with function(s) to test
            test_main_file: C file with main() and test cases
            exe_path: Output executable path
            
        Returns:
            (compile_result, execution_result)
        """
        # Make paths absolute
        if not function_file.is_absolute():
            function_file = self.work_dir / function_file
        if not test_main_file.is_absolute():
            test_main_file = self.work_dir / test_main_file
        
        if exe_path is None:
            exe_path = self.work_dir / "test_program.out"
        elif isinstance(exe_path, str):
            exe_path = Path(exe_path)
            if not exe_path.is_absolute():
                exe_path = self.work_dir / exe_path
        
        # Ensure exe_path is absolute
        exe_path = exe_path.absolute()
        
        # Compile both files together
        cmd = f"gcc -Wall -Wextra -o '{exe_path}' '{function_file.absolute()}' '{test_main_file.absolute()}'"
        exit_code, stdout, stderr = self.compiler._run_command(cmd)
        
        compile_result = CompileResult(
            success=(exit_code == 0),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
        
        if not compile_result.success:
            return compile_result, None
        
        # Run the executable (use compiler directly to avoid double path conversion)
        run_result = self.compiler.run_c_executable(exe_path)
        return compile_result, run_result
    
    # Legacy methods
    def simulate_verilog(
        self,
        design_file: Path,
        testbench_file: Path,
        output_vcd: Optional[Path] = None
    ) -> SimulationResult:
        """Run Verilog simulation"""
        return self.compiler.simulate_verilog(design_file, testbench_file, output_vcd)
    
    def run_c_executable(
        self,
        exe_path: Path,
        args: Optional[str] = None
    ) -> SimulationResult:
        """Run C executable"""
        return self.compiler.run_c_executable(exe_path, args)
    
