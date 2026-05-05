"""
Agent C: The Red Teamer (Simplified)

This is a minimal version that only contains methods actually used by the system.
The main test generation logic is now in PerRequirementTester.
"""
import re
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from agents.base_agent import GenerationAgent
from config.settings import config
from models.generation_models import GenerationTask, GeneratedCode, CodeLanguage
from tools import AgentToolbox, SimulationResult


@dataclass
class TestJudgment:
    """Result of judging a single test by Red Team"""
    test_id: str
    verdict: str  # PASS or FAIL
    reason: str
    root_cause: Optional[str] = None  # CODE_BUG, TB_ERROR, or None (if PASS)
    fix_tb_needed: bool = False  # Whether Red Team should fix testbench


class RedTeamerAgent(GenerationAgent):
    """
    Agent C: The Red Teamer (Simplified)
    
    This agent provides:
    1. LLM access for test generation (via self.llm)
    2. Toolbox for file operations and test execution
    3. Utility methods for code extraction and test running
    
    Note: Main test generation logic is in PerRequirementTester.
    """
    
    def __init__(
        self, 
        temperature: Optional[float] = None, 
        model: Optional[str] = None,
        work_dir: Optional[Path] = None
    ):
        temp = temperature if temperature is not None else config.RED_TEAMER_TEMPERATURE
        mdl = model or config.RED_TEAMER_MODEL or config.LLM_MODEL
        
        super().__init__(
            agent_name="RedTeamer",
            temperature=temp,
            model=mdl,
            work_dir=work_dir
        )
        
        # Initialize toolbox
        self.toolbox = AgentToolbox(work_dir=self.work_dir)
        
        self.log_info(f"Initialized with model={self.model}, temp={self.temperature}")
    
    def _get_suite_test_file_path(self, task: GenerationTask, suite_type: str) -> Path:
        """Get file path for suite test code"""
        if task.language == CodeLanguage.VERILOG:
            return Path(f"tb_{suite_type}_topmodule.v")
        else:
            return Path(f"test_{suite_type}_module.c")
    
    def _save_test_to_file(self, task: GenerationTask, test_code: str, test_file: str, output_dir: Path = None) -> Path:
        """
        Save test code to file
        
        Args:
            task: Generation task
            test_code: Test code string
            test_file: Test file name
            output_dir: Directory to save test (default: work_dir)
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.work_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / test_file
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        self.log_info(f"Saved test to: {file_path}")
        return file_path
    
    def _run_verilog_test(
        self,
        task: GenerationTask,
        golden_code: GeneratedCode,
        test_file_path: Path
    ) -> tuple[bool, Optional[SimulationResult]]:
        """Run Verilog testbench"""
        
        design_file = Path("topmodule.v")
        
        sim_result = self.toolbox.test_verilog_with_testbench(
            design_file,
            test_file_path
        )
        
        return (sim_result.success, sim_result)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response"""
        
        # Try to extract from code block
        code_match = re.search(r'```(?:verilog|c|systemverilog)?\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code block, assume entire response is code
        return response.strip()
