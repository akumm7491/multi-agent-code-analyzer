from typing import Dict, Any, List
from pathlib import Path
import subprocess
import ast

class TestRunner:
    """Runs tests to validate code changes."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_cache = {}
        
    async def run_tests(self, modified_files: List[str]) -> Dict[str, Any]:
        """Run tests relevant to modified files."""
        results = {
            "success": True,
            "test_runs": [],
            "coverage": {},
            "failures": []
        }
        
        # Find relevant tests
        test_files = await self._find_relevant_tests(modified_files)
        
        # Run each test file
        for test_file in test_files:
            test_result = await self._run_single_test(test_file)
            results["test_runs"].append(test_result)
            
            if not test_result["success"]:
                results["success"] = False
                results["failures"].extend(test_result["failures"])
                
        return results
        
    async def _find_relevant_tests(self, modified_files: List[str]) -> List[str]:
        """Find tests related to modified files."""
        test_files = set()
        
        for file_path in modified_files:
            # Direct test file
            test_path = self._get_test_path(file_path)
            if test_path.exists():
                test_files.add(str(test_path))
                
            # Find tests that import this module
            test_files.update(
                await self._find_importing_tests(file_path)
            )
            
        return list(test_files)
        
    async def _run_single_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            return {
                "file": test_file,
                "success": result.returncode == 0,
                "output": result.stdout,
                "failures": self._parse_test_failures(result.stdout)
            }
            
        except Exception as e:
            return {
                "file": test_file,
                "success": False,
                "output": str(e),
                "failures": [str(e)]
            }