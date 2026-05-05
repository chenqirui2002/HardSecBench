"""
Seed Generator Agent
Generates diverse seed questions from CWE specifications
"""
from typing import Optional, List
from pathlib import Path

from agents.base_agent import GenerationAgent
from config.settings import config
from models.cwe_models import CWEInfo
from models.seed_generation import SeedGenerationResult, BatchSeedGenerationResult
from models.seed_models import SeedQuestion
from utils.output_cleaner import CleanJsonOutputParser


class SeedGeneratorAgent(GenerationAgent):
    """
    Seed Generator Agent: Large-Scale Architecture Agent
    
    Responsibilities:
    1. Read complete CWE information
    2. Design 5-30 seed questions from different scenarios and vulnerability perspectives
    3. Each seed question: 1-2 sentences + language specification (C/Verilog)
    4. Questions focus on scenarios, not specific implementation details
    """
    
    def __init__(
        self,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        work_dir: Optional[Path] = None
    ):
        """
        Initialize Seed Generator Agent
        
        Args:
            temperature: Override default temperature
            model: Override default model
            work_dir: Working directory
        """
        temp = temperature if temperature is not None else config.ARCHITECT_TEMPERATURE
        mdl = model or config.ARCHITECT_MODEL or config.LLM_MODEL
        
        super().__init__(
            agent_name="SeedGenerator",
            temperature=temp,
            model=mdl,
            work_dir=work_dir
        )
        
        self.log_info(f"Initialized with model={self.model}, temp={self.temperature}")
    
    def generate_seeds_for_cwe(self, cwe: CWEInfo) -> SeedGenerationResult:
        """
        Generate seed questions for a specific CWE
        
        Args:
            cwe: CWE information
            
        Returns:
            SeedGenerationResult with generated seed questions
        """
        self.log_info(f"Generating seed questions for {cwe.cwe_id}")
        
        prompt = self._create_seed_generation_prompt(cwe)
        
        try:
            # Call LLM directly (not via chain) to ensure logging works
            response = self.llm.invoke(prompt)
            
            # Parse JSON from response
            from utils.output_cleaner import parse_json_output
            result = parse_json_output(response.content)
            
            self.log_debug(f"Raw LLM output parsed to: {result}")
            
            seeds_raw = result.get("seeds", [])
            
            seeds = []
            for seed_data in seeds_raw:
                if isinstance(seed_data, dict) and "question" in seed_data and "language" in seed_data:
                    lang = seed_data["language"].lower()
                    if lang in ["c", "verilog"]:
                        seeds.append(SeedQuestion(
                            question=seed_data["question"],
                            language=lang
                        ))
                    else:
                        self.logger.warning(f"Invalid language '{lang}', skipping seed")
                else:
                    self.logger.warning(f"Invalid seed format: {seed_data}, skipping")
            
            seed_result = SeedGenerationResult(
                cwe_id=cwe.cwe_id,
                cwe_name=cwe.name,
                seeds=seeds,
                reasoning=result.get("reasoning", ""),
                total_seeds=len(seeds)
            )
            
            self.log_info(f"Generated {len(seeds)} seed questions for {cwe.cwe_id}")
            for i, seed in enumerate(seeds, 1):
                self.log_debug(f"  Seed {i} ({seed.language}): {seed.question[:80]}...")
            
            return seed_result
            
        except Exception as e:
            self.log_error(f"Error generating seeds for {cwe.cwe_id}: {e}")
            return SeedGenerationResult(
                cwe_id=cwe.cwe_id,
                cwe_name=cwe.name,
                reasoning=f"Error during generation: {str(e)}",
                total_seeds=0
            )
    
    def generate_seeds_for_all_cwes(
        self,
        cwes: List[CWEInfo]
    ) -> BatchSeedGenerationResult:
        """
        Generate seeds for all CWEs
        
        Args:
            cwes: List of CWEs to process
            
        Returns:
            BatchSeedGenerationResult with all results
        """
        self.log_info(f"Generating seeds for {len(cwes)} CWEs")
        
        results = []
        for i, cwe in enumerate(cwes, 1):
            self.log_info(f"Processing CWE {i}/{len(cwes)}: {cwe.cwe_id}")
            result = self.generate_seeds_for_cwe(cwe)
            results.append(result)
        
        total_seeds = sum(r.total_seeds for r in results)
        
        batch_result = BatchSeedGenerationResult(
            total_cwes=len(cwes),
            results=results,
            total_seeds=total_seeds
        )
        
        self.log_info(
            f"Seed generation complete: {len(cwes)} CWEs processed, "
            f"{total_seeds} total seed questions generated"
        )
        
        return batch_result
    
    def _create_seed_generation_prompt(self, cwe: CWEInfo) -> str:
        """Create prompt for seed question generation"""
        
        cwe_info_sections = []
        cwe_info_sections.append(f"- **CWE ID**: {cwe.cwe_id}")
        cwe_info_sections.append(f"- **Name**: {cwe.name}")
        cwe_info_sections.append(f"\n**Description**:\n{cwe.description}")
        
        if cwe.extended_description:
            cwe_info_sections.append(f"\n**Extended Description**:\n{cwe.extended_description}")
        
        if cwe.demonstrative_examples:
            examples_preview = cwe.demonstrative_examples[:1000] + "..." if len(cwe.demonstrative_examples) > 1000 else cwe.demonstrative_examples
            cwe_info_sections.append(f"\n**Demonstrative Examples**:\n{examples_preview}")
        
        cwe_info_block = "\n".join(cwe_info_sections)
        
        prompt = f"""You are a hardware security architect designing diverse test scenarios.

# CWE Information
{cwe_info_block}

# Task
Generate **5-30 seed questions** covering different scenarios and attack vectors for this CWE.

## Seed Question Requirements
- 1-2 sentences, focus on scenario (not implementation details)
- Diverse: different components, contexts, triggering conditions
- Realistic: based on real hardware/firmware design scenarios
- Specify language: C or Verilog

## Language Selection
- **Verilog**: Hardware timing, FSM, signal control, register access, debug interfaces, side-channels
- **C**: Firmware logic, memory management, authentication, boot/crypto algorithms

# Output (JSON only)
{{
  "seeds": [
    {{"question": "Brief scenario description", "language": "verilog"}},
    {{"question": "Another scenario", "language": "c"}}
  ],
  "reasoning": "Diversity strategy explanation"
}}
"""
        return prompt
