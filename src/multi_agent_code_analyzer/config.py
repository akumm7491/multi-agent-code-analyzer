from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    name: str
    specialty: str
    enabled: bool = True
    confidence_threshold: float = 0.7
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 30
    additional_params: Dict[str, Any] = None

@dataclass
class NetworkConfig:
    """Configuration for the agent network."""
    knowledge_graph_path: Optional[str] = None
    max_agents_per_query: int = 3
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"

class ConfigManager:
    """Manages configuration for the multi-agent system."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.network_config = NetworkConfig()
        
        if Path(self.config_path).exists():
            self.load_config()
        else:
            self._set_default_configs()

    def load_config(self):
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)

        # Load network configuration
        network_data = config_data.get('network', {})
        self.network_config = NetworkConfig(**network_data)

        # Load agent configurations
        agents_data = config_data.get('agents', {})
        for agent_name, agent_data in agents_data.items():
            self.agent_configs[agent_name] = AgentConfig(
                name=agent_name,
                **agent_data
            )

    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            'network': {
                'knowledge_graph_path': self.network_config.knowledge_graph_path,
                'max_agents_per_query': self.network_config.max_agents_per_query,
                'enable_caching': self.network_config.enable_caching,
                'cache_ttl_seconds': self.network_config.cache_ttl_seconds,
                'log_level': self.network_config.log_level
            },
            'agents': {}
        }

        for name, config in self.agent_configs.items():
            config_data['agents'][name] = {
                'specialty': config.specialty,
                'enabled': config.enabled,
                'confidence_threshold': config.confidence_threshold,
                'max_concurrent_tasks': config.max_concurrent_tasks,
                'timeout_seconds': config.timeout_seconds,
                'additional_params': config.additional_params or {}
            }

        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    def _set_default_configs(self):
        """Set default configurations for all agents."""
        default_agents = [
            ("architect", "architecture"),
            ("code", "implementation"),
            ("domain", "business_logic"),
            ("security", "security"),
            ("testing", "quality_assurance"),
            ("documentation", "documentation"),
            ("dependency", "dependencies")
        ]

        for name, specialty in default_agents:
            self.agent_configs[name] = AgentConfig(
                name=name,
                specialty=specialty
            )

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agent_configs.get(agent_name)

    def update_agent_config(self, agent_name: str, **updates):
        """Update configuration for a specific agent."""
        if agent_name in self.agent_configs:
            config = self.agent_configs[agent_name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            self.save_config()

    def update_network_config(self, **updates):
        """Update network configuration."""
        for key, value in updates.items():
            if hasattr(self.network_config, key):
                setattr(self.network_config, key, value)
        self.save_config()