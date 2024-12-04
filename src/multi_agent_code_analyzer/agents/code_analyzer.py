from typing import Dict, Any, List, Optional
import os
import logging
import json
from prometheus_client import Counter, Gauge, Histogram
from .base import BaseAgent

# Define Prometheus metrics
TASK_COUNTER = Counter('agent_tasks_total',
                       'Total number of analysis tasks', ['agent_id'])
TASK_DURATION = Histogram('agent_task_duration_seconds',
                          'Task duration in seconds', ['agent_id'])
TASK_SUCCESS = Counter('agent_task_success_total',
                       'Total number of successful tasks', ['agent_id'])
TASK_FAILURE = Counter('agent_task_failure_total',
                       'Total number of failed tasks', ['agent_id'])
MEMORY_USAGE = Gauge('agent_memory_usage_bytes',
                     'Memory usage in bytes', ['agent_id'])
PATTERN_CONFIDENCE = Gauge('agent_pattern_confidence', 'Confidence in pattern detection', [
                           'agent_id', 'pattern_type'])

# Code quality metrics
CODE_COVERAGE = Gauge('code_coverage_percent',
                      'Code coverage percentage', ['package'])
CODE_TEST_COUNT = Gauge('code_test_count', 'Number of tests', ['package'])
CODE_COMPLEXITY = Gauge('code_complexity_score',
                        'Code complexity score', ['package'])
CODE_ISSUES = Counter('code_issues_total', 'Number of code issues', [
                      'severity', 'package'])
CODE_DEPENDENCIES = Gauge('code_dependencies_total',
                          'Number of dependencies', ['type', 'package'])
CODE_PATTERNS = Counter('code_patterns_detected_total',
                        'Number of design patterns detected', ['pattern_type', 'package'])


class CodeAnalyzerAgent(BaseAgent):
    """Agent for analyzing code repositories"""

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id)
        self.logger = logging.getLogger(__name__)
        TASK_COUNTER.labels(agent_id=self.agent_id)
        MEMORY_USAGE.labels(agent_id=self.agent_id)
        PATTERN_CONFIDENCE.labels(agent_id=self.agent_id, pattern_type="ddd")
        PATTERN_CONFIDENCE.labels(
            agent_id=self.agent_id, pattern_type="microservices")
        PATTERN_CONFIDENCE.labels(
            agent_id=self.agent_id, pattern_type="clean_architecture")

    async def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a repository"""
        try:
            TASK_COUNTER.labels(agent_id=self.agent_id).inc()
            with TASK_DURATION.labels(agent_id=self.agent_id).time():
                repo_path = context.get("repo_path")
                analysis_type = context.get("analysis_type", "full")

                if not repo_path or not os.path.exists(repo_path):
                    raise ValueError(f"Invalid repository path: {repo_path}")

                # Perform analysis based on type
                if analysis_type == "full":
                    result = await self._full_analysis(repo_path)
                elif analysis_type == "quick":
                    result = await self._quick_analysis(repo_path)
                else:
                    raise ValueError(f"Unknown analysis type: {analysis_type}")

                TASK_SUCCESS.labels(agent_id=self.agent_id).inc()
                return result

        except Exception as e:
            TASK_FAILURE.labels(agent_id=self.agent_id).inc()
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    async def _implement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Not supported for code analyzer agent"""
        raise NotImplementedError(
            "CodeAnalyzerAgent does not support implementation tasks")

    async def _custom_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom task"""
        try:
            description = context.get("description")
            task_context = context.get("context", {})

            # For now, just return a simple analysis
            return {
                "task": description,
                "context": task_context,
                "result": "Custom task executed successfully"
            }

        except Exception as e:
            self.logger.error(f"Custom task failed: {str(e)}")
            raise

    async def _full_analysis(self, repo_path: str) -> Dict[str, Any]:
        """Perform a full repository analysis"""
        try:
            # Analyze architecture
            architecture = await self._analyze_architecture(repo_path)

            # Analyze patterns
            patterns = await self._analyze_patterns(repo_path)

            # Analyze dependencies
            dependencies = await self._analyze_dependencies(repo_path)

            # Analyze API endpoints
            api_endpoints = await self._analyze_api_endpoints(repo_path)

            # Analyze data models
            data_models = await self._analyze_data_models(repo_path)

            # Analyze business logic
            business_logic = await self._analyze_business_logic(repo_path)

            return {
                "analysis": {
                    "architecture": architecture,
                    "patterns": patterns,
                    "dependencies": dependencies,
                    "api_endpoints": api_endpoints,
                    "data_models": data_models,
                    "business_logic": business_logic
                }
            }

        except Exception as e:
            self.logger.error(f"Full analysis failed: {str(e)}")
            raise

    async def _quick_analysis(self, repo_path: str) -> Dict[str, Any]:
        """Perform a quick repository analysis"""
        try:
            # Basic structure analysis
            structure = await self._analyze_structure(repo_path)

            # Basic code quality analysis
            code_quality = await self._analyze_code_quality(repo_path)

            return {
                "analysis": {
                    "structure": structure,
                    "code_quality": code_quality
                }
            }

        except Exception as e:
            self.logger.error(f"Quick analysis failed: {str(e)}")
            raise

    async def _analyze_architecture(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository architecture"""
        try:
            architecture = {
                "layers": [],
                "components": [],
                "patterns_detected": [],
                "dependencies": [],
                "summary": {}
            }

            # Common architectural patterns to detect
            patterns = {
                "ddd": ["domain", "application", "infrastructure", "interfaces", "entities", "repositories", "services", "valueobjects"],
                "clean_architecture": ["entities", "usecases", "interfaces", "frameworks"],
                "mvc": ["models", "views", "controllers"],
                "hexagonal": ["adapters", "ports", "domain"],
                "microservices": ["services", "api", "gateway", "registry"]
            }

            # Scan directory structure
            for root, dirs, files in os.walk(repo_path):
                rel_path = os.path.relpath(root, repo_path)
                if rel_path == ".":
                    continue

                path_parts = rel_path.lower().split(os.sep)

                # Detect layers
                for part in path_parts:
                    if part in ["domain", "application", "infrastructure", "presentation", "persistence", "api"]:
                        if part not in [layer["name"] for layer in architecture["layers"]]:
                            architecture["layers"].append({
                                "name": part,
                                "path": rel_path
                            })

                # Detect components
                if any(f.endswith(('.py', '.js', '.ts', '.java', '.cs', '.go')) for f in files):
                    component = {
                        "name": os.path.basename(rel_path),
                        "path": rel_path,
                        "type": "unknown",
                        "files": len(files)
                    }

                    # Determine component type
                    if "test" in rel_path.lower():
                        component["type"] = "test"
                    elif "controller" in rel_path.lower():
                        component["type"] = "controller"
                    elif "service" in rel_path.lower():
                        component["type"] = "service"
                    elif "repository" in rel_path.lower():
                        component["type"] = "repository"
                    elif "model" in rel_path.lower() or "entity" in rel_path.lower():
                        component["type"] = "model"

                    architecture["components"].append(component)

            # Detect architectural patterns
            for pattern_name, pattern_dirs in patterns.items():
                matches = sum(1 for dir_name in pattern_dirs if any(
                    dir_name in component["path"].lower() for component in architecture["components"]
                ))
                # If at least half of the pattern directories are found
                if matches >= len(pattern_dirs) // 2:
                    architecture["patterns_detected"].append(pattern_name)

            # Generate summary
            architecture["summary"] = {
                "total_layers": len(architecture["layers"]),
                "total_components": len(architecture["components"]),
                "components_by_type": {},
                "detected_patterns": architecture["patterns_detected"],
                "architectural_style": "unknown"
            }

            # Count components by type
            for component in architecture["components"]:
                comp_type = component["type"]
                architecture["summary"]["components_by_type"][comp_type] = \
                    architecture["summary"]["components_by_type"].get(
                        comp_type, 0) + 1

            # Determine overall architectural style
            if "ddd" in architecture["patterns_detected"]:
                architecture["summary"]["architectural_style"] = "Domain-Driven Design"
            elif "clean_architecture" in architecture["patterns_detected"]:
                architecture["summary"]["architectural_style"] = "Clean Architecture"
            elif "mvc" in architecture["patterns_detected"]:
                architecture["summary"]["architectural_style"] = "Model-View-Controller"
            elif "hexagonal" in architecture["patterns_detected"]:
                architecture["summary"]["architectural_style"] = "Hexagonal Architecture"
            elif "microservices" in architecture["patterns_detected"]:
                architecture["summary"]["architectural_style"] = "Microservices"

            return architecture

        except Exception as e:
            self.logger.error(f"Architecture analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_patterns(self, repo_path: str) -> Dict[str, Any]:
        """Analyze design patterns used in the repository"""
        try:
            patterns = {
                "detected": [],
                "potential": [],
                "by_category": {
                    "creational": [],
                    "structural": [],
                    "behavioral": [],
                    "architectural": [],
                    "enterprise": []
                }
            }

            # Common pattern indicators
            pattern_indicators = {
                "creational": {
                    "factory": ["factory", "create", "builder"],
                    "singleton": ["singleton", "instance"],
                    "prototype": ["clone", "prototype"],
                    "builder": ["builder", "director"],
                    "abstract_factory": ["abstract", "factory"]
                },
                "structural": {
                    "adapter": ["adapter", "wrapper"],
                    "bridge": ["bridge", "implementation"],
                    "composite": ["composite", "component"],
                    "decorator": ["decorator", "wrapper"],
                    "facade": ["facade"],
                    "proxy": ["proxy"]
                },
                "behavioral": {
                    "observer": ["observer", "subscriber", "event"],
                    "strategy": ["strategy", "algorithm"],
                    "command": ["command", "invoker"],
                    "state": ["state", "context"],
                    "mediator": ["mediator", "coordinator"],
                    "chain_of_responsibility": ["chain", "handler"]
                },
                "architectural": {
                    "repository": ["repository", "repo"],
                    "unit_of_work": ["unitofwork", "unit-of-work"],
                    "service": ["service"],
                    "controller": ["controller"],
                    "middleware": ["middleware"]
                },
                "enterprise": {
                    "specification": ["specification", "spec"],
                    "aggregate": ["aggregate", "root"],
                    "entity": ["entity"],
                    "value_object": ["valueobject", "value-object"],
                    "domain_event": ["domain-event", "domainevent"]
                }
            }

            package_name = os.path.basename(repo_path)

            # Scan all TypeScript/JavaScript files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(('.ts', '.js')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()

                                # Check for pattern indicators
                                for category, category_patterns in pattern_indicators.items():
                                    for pattern, indicators in category_patterns.items():
                                        if any(indicator in content for indicator in indicators):
                                            if pattern not in patterns["by_category"][category]:
                                                patterns["by_category"][category].append(
                                                    pattern)
                                                patterns["detected"].append({
                                                    "name": pattern,
                                                    "category": category,
                                                    "file": os.path.relpath(file_path, repo_path)
                                                })
                                                # Update Prometheus metrics
                                                CODE_PATTERNS.labels(
                                                    pattern_type=pattern,
                                                    package=package_name
                                                ).inc()

                                # Check for potential patterns based on file structure
                                if "test" in file_path:
                                    patterns["potential"].append({
                                        "name": "test_fixture",
                                        "category": "testing",
                                        "file": os.path.relpath(file_path, repo_path)
                                    })
                                elif "mock" in file_path or "stub" in file_path:
                                    patterns["potential"].append({
                                        "name": "test_double",
                                        "category": "testing",
                                        "file": os.path.relpath(file_path, repo_path)
                                    })

                        except Exception as e:
                            self.logger.warning(
                                f"Error analyzing file {file_path}: {str(e)}")

            # Add summary
            patterns["summary"] = {
                "total_patterns": len(patterns["detected"]),
                "patterns_by_category": {
                    category: len(patterns["by_category"][category])
                    for category in patterns["by_category"]
                },
                "potential_patterns": len(patterns["potential"])
            }

            # Update pattern confidence metrics
            total_patterns = len(patterns["detected"])
            if total_patterns > 0:
                ddd_patterns = len(
                    [p for p in patterns["detected"] if p["category"] == "enterprise"])
                ms_patterns = len(
                    [p for p in patterns["detected"] if p["category"] == "architectural"])
                clean_patterns = len([p for p in patterns["detected"] if p["category"] in [
                                     "creational", "structural"]])

                PATTERN_CONFIDENCE.labels(
                    agent_id=self.agent_id,
                    pattern_type="ddd"
                ).set(ddd_patterns / total_patterns)
                PATTERN_CONFIDENCE.labels(
                    agent_id=self.agent_id,
                    pattern_type="microservices"
                ).set(ms_patterns / total_patterns)
                PATTERN_CONFIDENCE.labels(
                    agent_id=self.agent_id,
                    pattern_type="clean_architecture"
                ).set(clean_patterns / total_patterns)

            return patterns

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_dependencies(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository dependencies"""
        try:
            dependencies = {
                "packages": {},
                "internal_dependencies": [],
                "external_dependencies": {},
                "dev_dependencies": {},
                "peer_dependencies": {},
                "summary": {}
            }

            # Find all package.json files
            for root, _, files in os.walk(repo_path):
                if "package.json" in files:
                    package_path = os.path.join(root, "package.json")
                    try:
                        with open(package_path, 'r', encoding='utf-8') as f:
                            package_data = json.loads(f.read())
                            package_name = package_data.get(
                                "name", os.path.basename(root))

                            # Store package info
                            dependencies["packages"][package_name] = {
                                "version": package_data.get("version", "unknown"),
                                "description": package_data.get("description", ""),
                                "main": package_data.get("main", ""),
                                "dependencies": package_data.get("dependencies", {}),
                                "devDependencies": package_data.get("devDependencies", {}),
                                "peerDependencies": package_data.get("peerDependencies", {}),
                                "path": os.path.relpath(root, repo_path)
                            }

                            # Track dependencies
                            for dep_name, dep_version in package_data.get("dependencies", {}).items():
                                if dep_name.startswith("@" + package_name):
                                    # Internal dependency
                                    dependencies["internal_dependencies"].append({
                                        "from": package_name,
                                        "to": dep_name,
                                        "version": dep_version
                                    })
                                else:
                                    # External dependency
                                    if dep_name not in dependencies["external_dependencies"]:
                                        dependencies["external_dependencies"][dep_name] = {
                                            "version": dep_version,
                                            "used_by": []
                                        }
                                    dependencies["external_dependencies"][dep_name]["used_by"].append(
                                        package_name)

                            # Track dev dependencies
                            for dep_name, dep_version in package_data.get("devDependencies", {}).items():
                                if dep_name not in dependencies["dev_dependencies"]:
                                    dependencies["dev_dependencies"][dep_name] = {
                                        "version": dep_version,
                                        "used_by": []
                                    }
                                dependencies["dev_dependencies"][dep_name]["used_by"].append(
                                    package_name)

                            # Track peer dependencies
                            for dep_name, dep_version in package_data.get("peerDependencies", {}).items():
                                if dep_name not in dependencies["peer_dependencies"]:
                                    dependencies["peer_dependencies"][dep_name] = {
                                        "version": dep_version,
                                        "used_by": []
                                    }
                                dependencies["peer_dependencies"][dep_name]["used_by"].append(
                                    package_name)

                    except Exception as e:
                        self.logger.warning(
                            f"Error analyzing package.json at {package_path}: {str(e)}")

            # Generate summary
            dependencies["summary"] = {
                "total_packages": len(dependencies["packages"]),
                "internal_dependencies": len(dependencies["internal_dependencies"]),
                "external_dependencies": len(dependencies["external_dependencies"]),
                "dev_dependencies": len(dependencies["dev_dependencies"]),
                "peer_dependencies": len(dependencies["peer_dependencies"]),
                "most_used_dependencies": sorted(
                    [
                        {"name": name, "usage_count": len(info["used_by"])}
                        for name, info in dependencies["external_dependencies"].items()
                    ],
                    key=lambda x: x["usage_count"],
                    reverse=True
                )[:5]
            }

            return dependencies

        except Exception as e:
            self.logger.error(f"Dependencies analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_api_endpoints(self, repo_path: str) -> Dict[str, Any]:
        """Analyze API endpoints in the repository"""
        try:
            api_info = {
                "endpoints": [],
                "controllers": [],
                "middleware": [],
                "routes": [],
                "summary": {}
            }

            # Common API decorators and patterns
            api_patterns = {
                "decorators": [
                    "@get", "@post", "@put", "@delete", "@patch",
                    "@controller", "@route", "@middleware",
                    "@api", "@rest", "@graphql"
                ],
                "methods": [
                    "get", "post", "put", "delete", "patch",
                    "options", "head", "trace", "connect"
                ],
                "frameworks": [
                    "express", "koa", "fastify", "nest",
                    "apollo", "graphql", "rest", "openapi"
                ]
            }

            # Scan TypeScript/JavaScript files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(('.ts', '.js', '.tsx', '.jsx')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.split('\n')

                                # Track current context
                                current_controller = None
                                current_route = None

                                for i, line in enumerate(lines):
                                    line_lower = line.lower()

                                    # Detect controllers
                                    if "@controller" in line_lower or "controller" in file.lower():
                                        controller_info = {
                                            "name": file.replace(".ts", "").replace(".js", ""),
                                            "path": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "endpoints": []
                                        }
                                        current_controller = controller_info
                                        api_info["controllers"].append(
                                            controller_info)

                                    # Detect middleware
                                    if "@middleware" in line_lower or "middleware" in file.lower():
                                        api_info["middleware"].append({
                                            "name": file.replace(".ts", "").replace(".js", ""),
                                            "path": os.path.relpath(file_path, repo_path),
                                            "line": i + 1
                                        })

                                    # Detect routes and endpoints
                                    for method in api_patterns["methods"]:
                                        method_pattern = f"@{method}"
                                        if method_pattern in line_lower or f".{method}(" in line_lower:
                                            # Extract route path if present
                                            route_match = None
                                            if "(" in line:
                                                route_match = line[line.find(
                                                    "(")+1:line.find(")")].strip("'\"")

                                            endpoint_info = {
                                                "method": method.upper(),
                                                "path": route_match or "unknown",
                                                "controller": current_controller["name"] if current_controller else None,
                                                "file": os.path.relpath(file_path, repo_path),
                                                "line": i + 1
                                            }

                                            api_info["endpoints"].append(
                                                endpoint_info)
                                            if current_controller:
                                                current_controller["endpoints"].append(
                                                    endpoint_info)

                                    # Detect route definitions
                                    if ".route" in line_lower or "@route" in line_lower:
                                        route_info = {
                                            "path": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "definition": line.strip()
                                        }
                                        api_info["routes"].append(route_info)

                        except Exception as e:
                            self.logger.warning(
                                f"Error analyzing file {file_path}: {str(e)}")

            # Generate summary
            api_info["summary"] = {
                "total_endpoints": len(api_info["endpoints"]),
                "total_controllers": len(api_info["controllers"]),
                "total_middleware": len(api_info["middleware"]),
                "endpoints_by_method": {},
                "endpoints_by_controller": {}
            }

            # Count endpoints by method
            for endpoint in api_info["endpoints"]:
                method = endpoint["method"]
                api_info["summary"]["endpoints_by_method"][method] = \
                    api_info["summary"]["endpoints_by_method"].get(
                        method, 0) + 1

            # Count endpoints by controller
            for controller in api_info["controllers"]:
                controller_name = controller["name"]
                api_info["summary"]["endpoints_by_controller"][controller_name] = \
                    len(controller["endpoints"])

            return api_info

        except Exception as e:
            self.logger.error(f"API endpoints analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_data_models(self, repo_path: str) -> Dict[str, Any]:
        """Analyze data models in the repository"""
        try:
            models = {
                "entities": [],
                "value_objects": [],
                "aggregates": [],
                "data_transfer_objects": [],
                "interfaces": [],
                "enums": [],
                "summary": {}
            }

            # Common model patterns
            model_patterns = {
                "entity": ["class", "extends", "entity", "model"],
                "value_object": ["valueobject", "value-object", "readonly", "immutable"],
                "aggregate": ["aggregate", "root", "aggregate-root"],
                "dto": ["dto", "interface", "type", "data transfer"],
                "enum": ["enum", "enumeration"]
            }

            # Scan TypeScript files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.ts'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.split('\n')

                                # Track current context
                                current_class = None
                                in_interface = False
                                in_enum = False
                                properties = []

                                for i, line in enumerate(lines):
                                    line_lower = line.lower().strip()

                                    # Detect class definitions
                                    if "class" in line_lower and "{" in line:
                                        class_info = {
                                            "name": line.split("class")[1].split("{")[0].strip().split(" ")[0],
                                            "type": "unknown",
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "properties": [],
                                            "methods": []
                                        }

                                        # Determine class type
                                        if any(p in line_lower for p in model_patterns["entity"]):
                                            class_info["type"] = "entity"
                                            models["entities"].append(
                                                class_info)
                                        elif any(p in line_lower for p in model_patterns["value_object"]):
                                            class_info["type"] = "value_object"
                                            models["value_objects"].append(
                                                class_info)
                                        elif any(p in line_lower for p in model_patterns["aggregate"]):
                                            class_info["type"] = "aggregate"
                                            models["aggregates"].append(
                                                class_info)

                                        current_class = class_info

                                    # Detect interface definitions
                                    elif "interface" in line_lower and "{" in line:
                                        interface_info = {
                                            "name": line.split("interface")[1].split("{")[0].strip(),
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "properties": []
                                        }
                                        models["interfaces"].append(
                                            interface_info)
                                        in_interface = True
                                        current_class = interface_info

                                    # Detect enum definitions
                                    elif "enum" in line_lower and "{" in line:
                                        enum_info = {
                                            "name": line.split("enum")[1].split("{")[0].strip(),
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "values": []
                                        }
                                        models["enums"].append(enum_info)
                                        in_enum = True
                                        current_class = enum_info

                                    # Track properties and methods
                                    elif current_class and not line_lower.startswith("//"):
                                        if ":" in line and "(" not in line:  # Property
                                            prop = line.split(":")[0].strip()
                                            if prop and not prop.startswith(("constructor", "private", "protected")):
                                                if in_enum:
                                                    current_class["values"].append(
                                                        prop)
                                                else:
                                                    current_class["properties"].append(
                                                        prop)
                                        # Method
                                        elif "(" in line and ")" in line and "{" in line:
                                            if hasattr(current_class, "methods"):
                                                method = line.split(
                                                    "(")[0].strip()
                                                if method and not method.startswith(("constructor", "private", "protected")):
                                                    current_class["methods"].append(
                                                        method)

                                    # End of definition
                                    if "}" in line:
                                        if current_class:
                                            current_class = None
                                        in_interface = False
                                        in_enum = False

                        except Exception as e:
                            self.logger.warning(
                                f"Error analyzing file {file_path}: {str(e)}")

            # Generate summary
            models["summary"] = {
                "total_entities": len(models["entities"]),
                "total_value_objects": len(models["value_objects"]),
                "total_aggregates": len(models["aggregates"]),
                "total_interfaces": len(models["interfaces"]),
                "total_enums": len(models["enums"]),
                "models_by_type": {
                    "entities": len(models["entities"]),
                    "value_objects": len(models["value_objects"]),
                    "aggregates": len(models["aggregates"]),
                    "interfaces": len(models["interfaces"]),
                    "enums": len(models["enums"])
                }
            }

            return models

        except Exception as e:
            self.logger.error(f"Data models analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_business_logic(self, repo_path: str) -> Dict[str, Any]:
        """Analyze business logic in the repository"""
        try:
            business_logic = {
                "services": [],
                "commands": [],
                "queries": [],
                "domain_rules": [],
                "behaviors": [],
                "validations": [],
                "summary": {}
            }

            # Common business logic patterns
            logic_patterns = {
                "service": ["service", "manager", "handler", "processor"],
                "command": ["command", "commandhandler", "usecase"],
                "query": ["query", "queryhandler", "finder", "reader"],
                "domain_rule": ["rule", "policy", "specification", "constraint"],
                "behavior": ["behavior", "behaviour", "trait", "aspect"],
                "validation": ["validate", "validator", "check", "assert"]
            }

            # Scan TypeScript files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.ts'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.split('\n')

                                # Track current context
                                current_item = None

                                for i, line in enumerate(lines):
                                    line_lower = line.lower().strip()

                                    # Detect service classes
                                    if any(p in line_lower for p in logic_patterns["service"]) and "class" in line_lower:
                                        service_info = {
                                            "name": line.split("class")[1].split("{")[0].strip().split(" ")[0],
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "methods": [],
                                            "dependencies": []
                                        }
                                        business_logic["services"].append(
                                            service_info)
                                        current_item = service_info

                                    # Detect command handlers
                                    elif any(p in line_lower for p in logic_patterns["command"]) and ("class" in line_lower or "interface" in line_lower):
                                        command_info = {
                                            "name": line.split(("class" if "class" in line_lower else "interface"))[1].split("{")[0].strip().split(" ")[0],
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "type": "command",
                                            "parameters": []
                                        }
                                        business_logic["commands"].append(
                                            command_info)
                                        current_item = command_info

                                    # Detect query handlers
                                    elif any(p in line_lower for p in logic_patterns["query"]) and ("class" in line_lower or "interface" in line_lower):
                                        query_info = {
                                            "name": line.split(("class" if "class" in line_lower else "interface"))[1].split("{")[0].strip().split(" ")[0],
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "type": "query",
                                            "parameters": []
                                        }
                                        business_logic["queries"].append(
                                            query_info)
                                        current_item = query_info

                                    # Detect domain rules and specifications
                                    elif any(p in line_lower for p in logic_patterns["domain_rule"]):
                                        rule_info = {
                                            "name": line.split(("class" if "class" in line_lower else "interface"))[1].split("{")[0].strip().split(" ")[0] if "class" in line_lower or "interface" in line_lower else "unknown",
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "type": "domain_rule",
                                            "description": line.strip()
                                        }
                                        business_logic["domain_rules"].append(
                                            rule_info)

                                    # Detect behaviors and traits
                                    elif any(p in line_lower for p in logic_patterns["behavior"]):
                                        behavior_info = {
                                            "name": line.split(("class" if "class" in line_lower else "interface"))[1].split("{")[0].strip().split(" ")[0] if "class" in line_lower or "interface" in line_lower else "unknown",
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "type": "behavior",
                                            "description": line.strip()
                                        }
                                        business_logic["behaviors"].append(
                                            behavior_info)

                                    # Detect validations
                                    elif any(p in line_lower for p in logic_patterns["validation"]):
                                        validation_info = {
                                            "file": os.path.relpath(file_path, repo_path),
                                            "line": i + 1,
                                            "type": "validation",
                                            "description": line.strip()
                                        }
                                        business_logic["validations"].append(
                                            validation_info)

                                    # Track methods and parameters
                                    elif current_item and "(" in line and ")" in line:
                                        if hasattr(current_item, "methods"):
                                            method = line.split("(")[0].strip()
                                            if method and not method.startswith(("constructor", "private", "protected")):
                                                current_item["methods"].append(
                                                    method)

                                        if hasattr(current_item, "parameters"):
                                            params = line[line.find(
                                                "(")+1:line.find(")")].strip()
                                            if params:
                                                current_item["parameters"].extend(
                                                    [p.strip() for p in params.split(",")])

                                    # Track dependencies (imports and injections)
                                    elif current_item and ("import" in line_lower or "@inject" in line_lower):
                                        if hasattr(current_item, "dependencies"):
                                            current_item["dependencies"].append(
                                                line.strip())

                                    # End of definition
                                    if "}" in line and current_item:
                                        current_item = None

                        except Exception as e:
                            self.logger.warning(
                                f"Error analyzing file {file_path}: {str(e)}")

            # Generate summary
            business_logic["summary"] = {
                "total_services": len(business_logic["services"]),
                "total_commands": len(business_logic["commands"]),
                "total_queries": len(business_logic["queries"]),
                "total_domain_rules": len(business_logic["domain_rules"]),
                "total_behaviors": len(business_logic["behaviors"]),
                "total_validations": len(business_logic["validations"]),
                "components_by_type": {
                    "services": len(business_logic["services"]),
                    "commands": len(business_logic["commands"]),
                    "queries": len(business_logic["queries"]),
                    "domain_rules": len(business_logic["domain_rules"]),
                    "behaviors": len(business_logic["behaviors"]),
                    "validations": len(business_logic["validations"])
                }
            }

            return business_logic

        except Exception as e:
            self.logger.error(f"Business logic analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure"""
        try:
            structure = {
                "directories": [],
                "files": [],
                "summary": {}
            }

            for root, dirs, files in os.walk(repo_path):
                rel_path = os.path.relpath(root, repo_path)
                if rel_path == ".":
                    rel_path = ""

                # Skip .git and other hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                # Add directory info
                if rel_path:
                    structure["directories"].append({
                        "path": rel_path,
                        "name": os.path.basename(rel_path)
                    })

                # Add file info
                for file in files:
                    if not file.startswith('.'):
                        file_path = os.path.join(rel_path, file)
                        structure["files"].append({
                            "path": file_path,
                            "name": file,
                            "extension": os.path.splitext(file)[1][1:] if os.path.splitext(file)[1] else None
                        })

            # Generate summary
            extensions = {}
            for file in structure["files"]:
                ext = file["extension"] or "no_extension"
                extensions[ext] = extensions.get(ext, 0) + 1

            structure["summary"] = {
                "total_directories": len(structure["directories"]),
                "total_files": len(structure["files"]),
                "file_types": extensions
            }

            return structure
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_code_quality(self, repo_path: str) -> Dict[str, Any]:
        """Analyze code quality"""
        try:
            quality_metrics = {
                "files_analyzed": 0,
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
                "files_by_language": {},
                "issues": []
            }

            package_name = os.path.basename(repo_path)

            # File extensions to analyze
            code_extensions = {
                'py': 'Python',
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'java': 'Java',
                'cs': 'C#',
                'go': 'Go',
                'rs': 'Rust',
                'cpp': 'C++',
                'c': 'C',
                'php': 'PHP',
                'rb': 'Ruby'
            }

            for root, _, files in os.walk(repo_path):
                for file in files:
                    ext = os.path.splitext(file)[1][1:].lower()
                    if ext in code_extensions:
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()

                                quality_metrics['files_analyzed'] += 1
                                quality_metrics['total_lines'] += len(lines)

                                code_lines = 0
                                comment_lines = 0
                                blank_lines = 0

                                in_multiline_comment = False

                                for line in lines:
                                    line = line.strip()

                                    if not line:
                                        blank_lines += 1
                                    elif line.startswith('#') or line.startswith('//'):
                                        comment_lines += 1
                                    elif '"""' in line or "'''" in line:
                                        comment_lines += 1
                                        in_multiline_comment = not in_multiline_comment
                                    elif in_multiline_comment:
                                        comment_lines += 1
                                    else:
                                        code_lines += 1

                                quality_metrics['code_lines'] += code_lines
                                quality_metrics['comment_lines'] += comment_lines
                                quality_metrics['blank_lines'] += blank_lines

                                lang = code_extensions[ext]
                                if lang not in quality_metrics['files_by_language']:
                                    quality_metrics['files_by_language'][lang] = {
                                        'files': 0,
                                        'total_lines': 0,
                                        'code_lines': 0,
                                        'comment_lines': 0,
                                        'blank_lines': 0
                                    }

                                quality_metrics['files_by_language'][lang]['files'] += 1
                                quality_metrics['files_by_language'][lang]['total_lines'] += len(
                                    lines)
                                quality_metrics['files_by_language'][lang]['code_lines'] += code_lines
                                quality_metrics['files_by_language'][lang]['comment_lines'] += comment_lines
                                quality_metrics['files_by_language'][lang]['blank_lines'] += blank_lines

                        except Exception as e:
                            quality_metrics['issues'].append({
                                'file': file_path,
                                'error': str(e)
                            })
                            CODE_ISSUES.labels(
                                severity="error",
                                package=package_name
                            ).inc()

            # Calculate some basic metrics
            if quality_metrics['code_lines'] > 0:
                quality_metrics['comment_ratio'] = round(
                    quality_metrics['comment_lines'] / quality_metrics['code_lines'] * 100, 2)
            else:
                quality_metrics['comment_ratio'] = 0

            # Update Prometheus metrics
            CODE_COVERAGE.labels(package=package_name).set(
                quality_metrics['comment_ratio'])
            CODE_TEST_COUNT.labels(package=package_name).set(
                len([f for f in os.listdir(repo_path) if f.endswith(
                    ('test.ts', 'spec.ts', 'test.js', 'spec.js'))])
            )
            CODE_COMPLEXITY.labels(package=package_name).set(
                quality_metrics['code_lines'] /
                max(quality_metrics['files_analyzed'], 1)
            )

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {str(e)}")
            return {"error": str(e)}
