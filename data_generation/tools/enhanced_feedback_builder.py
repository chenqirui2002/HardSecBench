"""
Enhanced feedback builder with rich debugging context
Provides actionable information for Expert agent to fix issues
"""
from typing import Dict, List, Optional
from models.single_generation import JudgedTestSuite


class EnhancedFeedbackBuilder:
    """
    Builds rich, actionable feedback for Expert agent
    Includes debugging context and specific guidance
    
    P1.2 Enhancement: Breaks blind feedback limitation by providing test intent
    """
    
    def build_expert_feedback(
        self,
        judged_functional: Optional[JudgedTestSuite],
        judged_security: Optional[JudgedTestSuite],
        functional_requirements: List[str],
        security_requirements: List[str],
        test_output_functional: str = "",
        test_output_security: str = "",
        # P1.2: Optional test code for intent extraction (DEPRECATED - not used in simplified version)
        functional_test_code: str = "",
        security_test_code: str = "",
        # Phase 2: Arbiter guidance
        arbiter_guidance: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Build SIMPLIFIED feedback for Expert.
        
        Principle: Only provide requirement + Arbiter's fix guidance.
        Expert doesn't need verbose debug info - Arbiter already analyzed.
        
        Args:
            arbiter_guidance: Dict mapping requirement_id to Arbiter's guidance
        """
        # Count failures
        failed_functional = []
        failed_security = []
        
        if judged_functional:
            failed_functional = [j for j in judged_functional.judgments if j.verdict == "FAIL"]
        if judged_security:
            failed_security = [j for j in judged_security.judgments if j.verdict == "FAIL"]
        
        total_failed = len(failed_functional) + len(failed_security)
        
        if total_failed == 0:
            return "All tests passed."
        
        feedback_parts = [f"# Code Issues ({total_failed} failed)\n"]
        
        # Build simplified feedback for each failed requirement
        for judgment in failed_functional:
            req_index = self._extract_req_index(judgment.test_id)
            req_text = functional_requirements[req_index] if 0 <= req_index < len(functional_requirements) else "Requirement not found"
            # Use tuple key (suite_type, req_index) for direct lookup
            guidance = arbiter_guidance.get(("functional", req_index), "") if arbiter_guidance else ""
            feedback_parts.append(self._build_simple_feedback(judgment.test_id, req_text, guidance))
        
        for judgment in failed_security:
            req_index = self._extract_req_index(judgment.test_id)
            req_text = security_requirements[req_index] if 0 <= req_index < len(security_requirements) else "Requirement not found"
            # Use tuple key (suite_type, req_index) for direct lookup
            guidance = arbiter_guidance.get(("security", req_index), "") if arbiter_guidance else ""
            feedback_parts.append(self._build_simple_feedback(judgment.test_id, req_text, guidance))
        
        # Add action instruction
        feedback_parts.append("---")
        feedback_parts.append("Apply the fixes above and output the complete corrected code.")
        
        return "\n".join(feedback_parts)
    
    def _extract_req_index(self, test_id: str) -> int:
        """
        Extract requirement index from test_id.
        Supports: FUNCTIONAL-REQ-0, SECURITY-REQ-1, FUNC-TEST-1, SEC-TEST-2
        """
        import re
        
        # Per-requirement format: REQ-N (0-indexed)
        match = re.search(r'REQ-(\d+)', test_id)
        if match:
            return int(match.group(1))
        
        # Legacy format: TEST-N (1-indexed)
        match = re.search(r'TEST-(\d+)', test_id)
        if match:
            return int(match.group(1)) - 1
        
        return -1
    
    def _build_simple_feedback(self, req_id: str, requirement: str, arbiter_guidance: str) -> str:
        """
        Build minimal feedback for one requirement.
        Only: requirement description + Arbiter's fix guidance.
        """
        lines = [
            f"## {req_id}",
            f"**Requirement**: {requirement}",
            ""
        ]
        
        if arbiter_guidance:
            lines.append(f"**Fix**: {arbiter_guidance}")
        else:
            lines.append("**Fix**: Review the implementation against the requirement.")
        
        lines.append("")
        return "\n".join(lines)
    
    
    def _extract_test_intent(self, test_code: str, test_id: str) -> Dict[str, str]:
        """
        P1.2: Extract test intent from test code
        Provides Expert with understanding of what test is trying to verify
        """
        intent = {
            'approach': '',
            'key_signals': [],
            'timing_info': '',
            'expected_behavior': ''
        }
        
        if not test_code:
            return intent
        
        import re
        
        # Find the test block for this specific test_id
        # Look for comments near test_id definition
        lines = test_code.split('\n')
        test_section = []
        in_test_section = False
        
        for i, line in enumerate(lines):
            if test_id in line and ('START' in line or 'initial begin' in line or f'// [{test_id}]' in line):
                in_test_section = True
                # Look back for comments
                for j in range(max(0, i-5), i):
                    if '//' in lines[j]:
                        test_section.append(lines[j])
            
            if in_test_section:
                test_section.append(line)
                # Stop at next test or end
                if (i > 0 and re.search(r'\[(FUNC|SEC)-TEST-\d+\]', line) and test_id not in line):
                    break
                if 'end_of_test' in line or '$finish' in line:
                    break
        
        test_code_block = '\n'.join(test_section)
        
        # Extract approach from comments
        comment_lines = [l.strip() for l in test_section if '//' in l]
        if comment_lines:
            intent['approach'] = ' '.join([c.replace('//', '').strip() for c in comment_lines[:3]])
        
        # Extract key signals being set
        signal_patterns = [
            (r'(\w+)\s*=\s*([01xb\d]+)', 'set'),
            (r'write_(\w+)', 'write operation'),
            (r'read_(\w+)', 'read operation'),
            (r'lock\s*=\s*([01])', 'lock control'),
            (r'rst_n\s*=\s*([01])', 'reset')
        ]
        
        for pattern, desc in signal_patterns:
            matches = re.findall(pattern, test_code_block)
            if matches:
                if desc == 'set':
                    intent['key_signals'].extend([f"{m[0]}={m[1]}" for m in matches[:5]])
                else:
                    intent['key_signals'].append(desc)
        
        # Extract timing information
        posedge_count = test_code_block.count('@(posedge clk)')
        delay_count = test_code_block.count('#')
        
        if posedge_count > 0:
            intent['timing_info'] = f"{posedge_count} clock cycles"
        elif delay_count > 0:
            intent['timing_info'] = f"time delays present"
        
        # Extract expected behavior from comments
        expect_pattern = r'[Ee]xpect.*?(?:should|must|will).*?([^\n.]+)'
        expect_matches = re.findall(expect_pattern, test_code_block)
        if expect_matches:
            intent['expected_behavior'] = expect_matches[0].strip()
        
        return intent
