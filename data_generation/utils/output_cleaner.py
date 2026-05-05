"""
Output cleaner utilities for LLM responses
Handles various output formats and extracts clean JSON
"""
import re
import json
from typing import Any
from langchain_core.output_parsers import JsonOutputParser as BaseJsonOutputParser


def extract_text_content(response: Any) -> str:
    """
    Normalize different LLM response shapes into plain text.

    Handles raw strings, LangChain messages/generations, and content blocks.
    """
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if isinstance(response, list):
        parts = []
        for item in response:
            text = extract_text_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts)

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    message = getattr(response, "message", None)
    if message is not None:
        return extract_text_content(message)

    return str(response)


def clean_llm_output(text: str) -> str:
    """
    Clean LLM output to extract pure JSON.
    
    Removes:
    - <think>...</think> tags and their content
    - Markdown code blocks (```json, ```)
    - JavaScript-style comments (// and /* */)
    - Leading/trailing whitespace and explanatory text
    
    Args:
        text: Raw LLM output text
        
    Returns:
        Cleaned text containing only JSON
    """
    text = extract_text_content(text)

    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object or array in the text
    # Look for the first { or [ and last } or ]
    json_match = re.search(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Remove JavaScript-style comments
    # Remove // comments
    text = re.sub(r'//[^\n]*\n', '\n', text)
    # Remove /* */ comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Remove trailing commas (common JSON error)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def parse_json_output(text: Any) -> Any:
    """
    Parse JSON from LLM output with automatic cleaning.
    
    Args:
        text: Raw LLM output text or response object
        
    Returns:
        Parsed JSON content
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails after cleaning
    """
    raw_text = extract_text_content(text)
    cleaned_text = clean_llm_output(raw_text)

    if not cleaned_text:
        raw_preview = raw_text.strip().replace("\n", "\\n")[:200]
        raise ValueError(
            "Empty LLM response after cleaning; cannot parse JSON. "
            f"Raw response preview: {raw_preview or '<empty>'}"
        )

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raw_preview = raw_text.strip().replace("\n", "\\n")[:200]
        cleaned_preview = cleaned_text.replace("\n", "\\n")[:200]
        raise ValueError(
            "Failed to parse JSON from LLM response. "
            f"Cleaned preview: {cleaned_preview or '<empty>'}. "
            f"Raw preview: {raw_preview or '<empty>'}. "
            f"Original error: {e}"
        ) from e


class CleanJsonOutputParser(BaseJsonOutputParser):
    """
    Custom JSON output parser that cleans LLM output before parsing.
    
    Automatically removes:
    - <think> tags and content
    - Markdown code blocks
    - Extra whitespace and text
    """
    
    def parse_result(self, result, *, partial: bool = False):
        """Parse the result of an LLM call to a JSON object with cleaning."""
        # Clean and parse
        try:
            return parse_json_output(result)
        except ValueError:
            # If still fails, try the parent class parser as fallback
            return super().parse_result(result, partial=partial)
