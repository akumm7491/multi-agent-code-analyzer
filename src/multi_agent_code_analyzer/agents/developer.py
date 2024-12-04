from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import json
import logging
import ast
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram
from ..knowledge.graph import KnowledgeGraph
from ..verification.verifier import VerificationService
from ..mcp.client import MCPClient
from ..tools.manager import ToolManager
from ..tools.github import GithubService
from .base import BaseAgent

# Metrics
CODE_GENERATION = Counter('code_generation_total',
                          'Code generation attempts', ['type', 'status'])
CODE_QUALITY = Gauge('code_quality', 'Code quality score', ['component'])
TEST_COVERAGE = Gauge(
    'test_coverage', 'Test coverage percentage', ['component'])
IMPROVEMENT_SUGGESTIONS = Counter(
    'improvement_suggestions_total', 'Improvement suggestions', ['type'])


@dataclass
class CodeGenerationResult:
    """Result of code generation"""
    code: str
    tests: Optional[str]
    documentation: Optional[str]
    quality_score: float
    coverage: float
    metrics: Dict[str, Any]
    improvements: List[Dict[str, Any]]


@dataclass
class CodeImprovementSuggestion:
    """Suggestion for code improvement"""
    type: str
    description: str
    priority: str
    effort: str
    benefits: List[str]
    implementation_guide: Dict[str, Any]
    before_code: Optional[str] = None
    after_code: Optional[str] = None


class DeveloperAgent(BaseAgent):
    """Agent specialized in code development and improvement"""

    def __init__(
        self,
        name: str,
        knowledge_graph: KnowledgeGraph,
        verifier: VerificationService,
        mcp_client: MCPClient,
        tool_manager: ToolManager,
        github_service: GithubService,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, specialty="development")
        self.knowledge_graph = knowledge_graph
        self.verifier = verifier
        self.mcp_client = mcp_client
        self.tool_manager = tool_manager
        self.github_service = github_service
        self.config = config or {}

        # Development tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.code_patterns: Dict[str, Dict[str, Any]] = {}
        self.quality_thresholds: Dict[str, float] = {}
        self.improvement_strategies: Dict[str, Dict[str, Any]] = {}

        # Learning
        self.learned_patterns: Set[str] = set()
        self.successful_strategies: Dict[str, int] = {}
        self.failed_attempts: Dict[str, List[Dict[str, Any]]] = {}

        self.logger = logging.getLogger(__name__)

    async def generate_code(
        self,
        description: str,
        context: Dict[str, Any],
        requirements: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None
    ) -> CodeGenerationResult:
        """Generate code based on description and context"""
        try:
            CODE_GENERATION.labels(type="initial", status="started").inc()

            # Create MCP context
            mcp_context = await self.mcp_client.create_context(
                model_id=self.config["model_id"],
                task_type="code_generation",
                metadata={
                    "description": description,
                    "requirements": requirements,
                    "constraints": constraints
                }
            )

            # Find similar patterns
            similar_patterns = await self.knowledge_graph.find_similar_nodes(
                description,
                node_type="code_pattern",
                min_similarity=0.7
            )

            # Generate initial code
            initial_code = await self._generate_initial_code(
                description,
                context,
                similar_patterns
            )

            # Generate tests
            tests = await self._generate_tests(initial_code, requirements)

            # Generate documentation
            documentation = await self._generate_documentation(
                initial_code,
                tests,
                description
            )

            # Evaluate quality
            quality_result = await self._evaluate_code_quality(
                initial_code,
                tests,
                requirements
            )

            # Generate improvements
            improvements = await self._generate_improvements(
                initial_code,
                quality_result
            )

            # Store in knowledge graph
            await self._store_generation_result(
                initial_code,
                tests,
                documentation,
                quality_result,
                improvements,
                mcp_context
            )

            result = CodeGenerationResult(
                code=initial_code,
                tests=tests,
                documentation=documentation,
                quality_score=quality_result["score"],
                coverage=quality_result["coverage"],
                metrics=quality_result["metrics"],
                improvements=improvements
            )

            CODE_GENERATION.labels(type="initial", status="completed").inc()
            return result

        except Exception as e:
            CODE_GENERATION.labels(type="initial", status="failed").inc()
            self.logger.error(f"Error generating code: {str(e)}")
            raise

    async def improve_code(
        self,
        code: str,
        context: Dict[str, Any],
        quality_threshold: float = 0.8
    ) -> CodeGenerationResult:
        """Improve existing code"""
        try:
            CODE_GENERATION.labels(type="improvement", status="started").inc()

            # Create MCP context
            mcp_context = await self.mcp_client.create_context(
                model_id=self.config["model_id"],
                task_type="code_improvement",
                metadata={
                    "quality_threshold": quality_threshold,
                    **context
                }
            )

            # Analyze current code
            analysis = await self._analyze_code(code)

            # Generate improvements
            improvements = await self._generate_improvements(
                code,
                analysis
            )

            # Apply improvements
            improved_code = await self._apply_improvements(
                code,
                improvements
            )

            # Generate tests for improvements
            tests = await self._generate_tests(
                improved_code,
                context.get("requirements")
            )

            # Update documentation
            documentation = await self._generate_documentation(
                improved_code,
                tests,
                context.get("description", "")
            )

            # Evaluate new quality
            quality_result = await self._evaluate_code_quality(
                improved_code,
                tests,
                context.get("requirements")
            )

            # Store improvement results
            await self._store_improvement_result(
                code,
                improved_code,
                improvements,
                quality_result,
                mcp_context
            )

            result = CodeGenerationResult(
                code=improved_code,
                tests=tests,
                documentation=documentation,
                quality_score=quality_result["score"],
                coverage=quality_result["coverage"],
                metrics=quality_result["metrics"],
                improvements=improvements
            )

            CODE_GENERATION.labels(
                type="improvement", status="completed").inc()
            return result

        except Exception as e:
            CODE_GENERATION.labels(type="improvement", status="failed").inc()
            self.logger.error(f"Error improving code: {str(e)}")
            raise

    async def _generate_initial_code(
        self,
        description: str,
        context: Dict[str, Any],
        similar_patterns: List[Dict[str, Any]]
    ) -> str:
        """Generate initial code based on description and patterns"""
        try:
            # Extract patterns from similar code
            patterns = await self._extract_patterns(similar_patterns)

            # Generate code structure
            structure = await self._generate_code_structure(
                description,
                patterns
            )

            # Generate implementation
            implementation = await self._generate_implementation(
                structure,
                context
            )

            # Verify implementation
            await self.verifier.verify_code(
                implementation,
                context.get("requirements", [])
            )

            return implementation

        except Exception as e:
            self.logger.error(f"Error generating initial code: {str(e)}")
            raise

    async def _generate_tests(
        self,
        code: str,
        requirements: Optional[List[str]] = None
    ) -> str:
        """Generate tests for code"""
        try:
            # Parse code
            tree = ast.parse(code)

            # Extract testable components
            components = await self._extract_testable_components(tree)

            # Generate test cases
            test_cases = await self._generate_test_cases(
                components,
                requirements
            )

            # Generate test implementation
            test_code = await self._generate_test_implementation(test_cases)

            # Verify test coverage
            coverage = await self._verify_test_coverage(code, test_code)
            TEST_COVERAGE.labels(component="generated").set(coverage)

            return test_code

        except Exception as e:
            self.logger.error(f"Error generating tests: {str(e)}")
            raise

    async def _evaluate_code_quality(
        self,
        code: str,
        tests: Optional[str],
        requirements: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Evaluate code quality"""
        try:
            # Run static analysis
            static_analysis = await self.tool_manager.execute_tool(
                "StaticAnalysisTool",
                "analyze_code",
                code=code,
                tests=tests
            )

            # Check requirements coverage
            req_coverage = await self._check_requirements_coverage(
                code,
                requirements
            )

            # Calculate metrics
            metrics = await self._calculate_code_metrics(code)

            # Calculate overall score
            score = await self._calculate_quality_score(
                static_analysis,
                req_coverage,
                metrics
            )

            CODE_QUALITY.labels(component="generated").set(score)

            return {
                "score": score,
                "static_analysis": static_analysis,
                "requirements_coverage": req_coverage,
                "metrics": metrics,
                "coverage": metrics.get("test_coverage", 0.0)
            }

        except Exception as e:
            self.logger.error(f"Error evaluating code quality: {str(e)}")
            raise

    async def _generate_improvements(
        self,
        code: str,
        analysis: Dict[str, Any]
    ) -> List[CodeImprovementSuggestion]:
        """Generate improvement suggestions"""
        try:
            suggestions = []

            # Check for pattern improvements
            pattern_improvements = await self._suggest_pattern_improvements(
                code,
                analysis
            )
            suggestions.extend(pattern_improvements)

            # Check for quality improvements
            quality_improvements = await self._suggest_quality_improvements(
                code,
                analysis
            )
            suggestions.extend(quality_improvements)

            # Check for performance improvements
            perf_improvements = await self._suggest_performance_improvements(
                code,
                analysis
            )
            suggestions.extend(perf_improvements)

            # Track suggestions
            for suggestion in suggestions:
                IMPROVEMENT_SUGGESTIONS.labels(type=suggestion.type).inc()

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating improvements: {str(e)}")
            raise

    async def _store_generation_result(
        self,
        code: str,
        tests: str,
        documentation: str,
        quality_result: Dict[str, Any],
        improvements: List[CodeImprovementSuggestion],
        mcp_context: Dict[str, Any]
    ):
        """Store code generation results"""
        try:
            # Store code node
            await self.knowledge_graph.add_node(
                f"code_{mcp_context['id']}",
                {
                    "type": "generated_code",
                    "content": code,
                    "metadata": {
                        "quality_score": quality_result["score"],
                        "coverage": quality_result["coverage"]
                    }
                },
                "Code"
            )

            # Store tests
            if tests:
                await self.knowledge_graph.add_node(
                    f"tests_{mcp_context['id']}",
                    {
                        "type": "tests",
                        "content": tests,
                        "metadata": {
                            "coverage": quality_result["coverage"]
                        }
                    },
                    "Tests"
                )

            # Store documentation
            if documentation:
                await self.knowledge_graph.add_node(
                    f"docs_{mcp_context['id']}",
                    {
                        "type": "documentation",
                        "content": documentation,
                        "metadata": {}
                    },
                    "Documentation"
                )

            # Store improvements
            if improvements:
                await self.knowledge_graph.add_node(
                    f"improvements_{mcp_context['id']}",
                    {
                        "type": "improvements",
                        "content": [asdict(imp) for imp in improvements],
                        "metadata": {
                            "count": len(improvements)
                        }
                    },
                    "Improvements"
                )

        except Exception as e:
            self.logger.error(f"Error storing generation results: {str(e)}")
            raise

    async def _store_improvement_result(
        self,
        original_code: str,
        improved_code: str,
        improvements: List[CodeImprovementSuggestion],
        quality_result: Dict[str, Any],
        mcp_context: Dict[str, Any]
    ):
        """Store code improvement results"""
        try:
            # Store improvement node
            await self.knowledge_graph.add_node(
                f"improvement_{mcp_context['id']}",
                {
                    "type": "code_improvement",
                    "content": {
                        "original": original_code,
                        "improved": improved_code,
                        "improvements": [asdict(imp) for imp in improvements]
                    },
                    "metadata": {
                        "quality_score": quality_result["score"],
                        "coverage": quality_result["coverage"]
                    }
                },
                "Improvement"
            )

            # Update patterns
            for improvement in improvements:
                if improvement.type == "pattern":
                    await self._update_pattern_knowledge(improvement)

        except Exception as e:
            self.logger.error(f"Error storing improvement results: {str(e)}")
            raise

    async def _update_pattern_knowledge(self, improvement: CodeImprovementSuggestion):
        """Update knowledge about successful patterns"""
        try:
            pattern_id = f"pattern_{improvement.type}_{len(self.learned_patterns)}"

            # Store pattern
            await self.knowledge_graph.add_node(
                pattern_id,
                {
                    "type": "code_pattern",
                    "content": {
                        "before": improvement.before_code,
                        "after": improvement.after_code,
                        "description": improvement.description,
                        "benefits": improvement.benefits
                    },
                    "metadata": {
                        "success_count": self.successful_strategies.get(
                            improvement.type,
                            0
                        )
                    }
                },
                "Pattern"
            )

            self.learned_patterns.add(pattern_id)

        except Exception as e:
            self.logger.error(f"Error updating pattern knowledge: {str(e)}")
            raise
