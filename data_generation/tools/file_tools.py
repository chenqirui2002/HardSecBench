"""
File operation tools for code generation
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information about a file"""
    path: Path
    exists: bool
    size: int = 0
    lines: int = 0


class FileTools:
    """Tools for file operations"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize file tools
        
        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = base_dir or Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def write_code(
        self,
        file_path,  # Can be str or Path
        content: str,
        language: str,
        overwrite: bool = True
    ) -> bool:
        """
        Write code to file
        
        Args:
            file_path: Target file path (str or Path)
            content: Code content
            language: Programming language (for validation)
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path
        
        # Check if file exists
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate extension matches language
        expected_ext = {
            'verilog': ['.v', '.sv'],
            'c': ['.c'],
            'cpp': ['.cpp', '.cc'],
            'python': ['.py']
        }.get(language.lower(), [])
        
        if expected_ext and file_path.suffix not in expected_ext:
            raise ValueError(
                f"File extension {file_path.suffix} doesn't match language {language}. "
                f"Expected: {expected_ext}"
            )
        
        # Write file
        try:
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            raise IOError(f"Failed to write file {file_path}: {e}")
    
    def read_code(self, file_path: Path) -> str:
        """
        Read code from file
        
        Args:
            file_path: File path to read
            
        Returns:
            File content as string
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")
    
    def get_file_info(self, file_path: Path) -> FileInfo:
        """
        Get information about a file
        
        Args:
            file_path: File path
            
        Returns:
            FileInfo object
        """
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path
        
        if not file_path.exists():
            return FileInfo(path=file_path, exists=False)
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return FileInfo(
                path=file_path,
                exists=True,
                size=file_path.stat().st_size,
                lines=len(content.splitlines())
            )
        except Exception:
            return FileInfo(
                path=file_path,
                exists=True,
                size=file_path.stat().st_size
            )
