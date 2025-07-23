#!/usr/bin/env python3
"""
Dynamic Configuration Manager
============================
Advanced configuration management with runtime updates and profile support.
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

class ConfigManager:
    """Advanced configuration manager with dynamic updates and profiles"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.current_profile = "default"
        self.config_cache = {}
        self.watchers = []  # For config change notifications
        
        # Setup logging
        self.logger = logging.getLogger("ConfigManager")
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configuration profiles"""
        default_configs = {
            "default": {
                "system": {
                    "name": "TERMINALIS-V.2",
                    "version": "2.0.0",
                    "max_agents": 10,
                    "debug_mode": False,
                    "performance_mode": "balanced"
                },
                "ui": {
                    "theme": "dark",
                    "animation_speed": "normal",
                    "auto_save": True,
                    "verbosity": False
                },
                "agents": {
                    "max_concurrent": 5,
                    "timeout_seconds": 300,
                    "retry_attempts": 3,
                    "load_balancing": True
                },
                "models": {
                    "cache_size_gb": 2,
                    "auto_download": False,
                    "preferred_format": "gguf"
                }
            },
            "performance": {
                "system": {
                    "name": "TERMINALIS-V.2-PERFORMANCE",
                    "version": "2.0.0",
                    "max_agents": 20,
                    "debug_mode": False,
                    "performance_mode": "maximum"
                },
                "ui": {
                    "theme": "minimal",
                    "animation_speed": "fast",
                    "auto_save": True,
                    "verbosity": False
                },
                "agents": {
                    "max_concurrent": 10,
                    "timeout_seconds": 600,
                    "retry_attempts": 5,
                    "load_balancing": True
                },
                "models": {
                    "cache_size_gb": 8,
                    "auto_download": True,
                    "preferred_format": "gguf"
                }
            },
            "development": {
                "system": {
                    "name": "TERMINALIS-V.2-DEV",
                    "version": "2.0.0-dev",
                    "max_agents": 5,
                    "debug_mode": True,
                    "performance_mode": "debug"
                },
                "ui": {
                    "theme": "light",
                    "animation_speed": "slow",
                    "auto_save": False,
                    "verbosity": True
                },
                "agents": {
                    "max_concurrent": 3,
                    "timeout_seconds": 120,
                    "retry_attempts": 1,
                    "load_balancing": False
                },
                "models": {
                    "cache_size_gb": 1,
                    "auto_download": False,
                    "preferred_format": "gguf"
                }
            }
        }
        
        # Save default configs if they don't exist
        for profile_name, config in default_configs.items():
            profile_path = self.config_dir / f"{profile_name}.yaml"
            if not profile_path.exists():
                self.save_profile(profile_name, config)
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load configuration from a specific profile"""
        profile_path = self.config_dir / f"{profile_name}.yaml"
        
        if not profile_path.exists():
            self.logger.warning(f"Profile '{profile_name}' not found, using default")
            profile_name = "default"
            profile_path = self.config_dir / f"{profile_name}.yaml"
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.config_cache[profile_name] = config
                self.logger.info(f"Loaded profile: {profile_name}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading profile {profile_name}: {e}")
            return {}
    
    def save_profile(self, profile_name: str, config: Dict[str, Any]):
        """Save configuration to a specific profile"""
        profile_path = self.config_dir / f"{profile_name}.yaml"
        
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                self.config_cache[profile_name] = config
                self.logger.info(f"Saved profile: {profile_name}")
                
                # Notify watchers of config change
                self._notify_watchers(profile_name, config)
                
        except Exception as e:
            self.logger.error(f"Error saving profile {profile_name}: {e}")
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different configuration profile"""
        try:
            config = self.load_profile(profile_name)
            if config:
                self.current_profile = profile_name
                self.logger.info(f"Switched to profile: {profile_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error switching to profile {profile_name}: {e}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current active configuration"""
        if self.current_profile not in self.config_cache:
            self.load_profile(self.current_profile)
        return self.config_cache.get(self.current_profile, {})
    
    def update_config(self, key_path: str, value: Any, save: bool = True):
        """Update a configuration value dynamically"""
        config = self.get_current_config()
        
        # Parse key path (e.g., "system.max_agents" -> ["system", "max_agents"])
        keys = key_path.split('.')
        
        # Navigate to the parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
        if save:
            self.save_profile(self.current_profile, config)
        
        self.logger.info(f"Updated {key_path} = {value}")
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by key path"""
        config = self.get_current_config()
        
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def list_profiles(self) -> List[str]:
        """List all available configuration profiles"""
        profiles = []
        for file_path in self.config_dir.glob("*.yaml"):
            profiles.append(file_path.stem)
        return sorted(profiles)
    
    def create_profile(self, profile_name: str, base_profile: str = "default") -> bool:
        """Create a new configuration profile based on an existing one"""
        try:
            base_config = self.load_profile(base_profile)
            if base_config:
                # Add metadata to the new profile
                base_config["_metadata"] = {
                    "created_from": base_profile,
                    "created_at": datetime.now().isoformat(),
                    "description": f"Profile created from {base_profile}"
                }
                
                self.save_profile(profile_name, base_config)
                self.logger.info(f"Created profile '{profile_name}' from '{base_profile}'")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating profile {profile_name}: {e}")
            return False
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a configuration profile"""
        if profile_name == "default":
            self.logger.warning("Cannot delete default profile")
            return False
        
        profile_path = self.config_dir / f"{profile_name}.yaml"
        
        try:
            if profile_path.exists():
                profile_path.unlink()
                if profile_name in self.config_cache:
                    del self.config_cache[profile_name]
                self.logger.info(f"Deleted profile: {profile_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting profile {profile_name}: {e}")
            return False
    
    def export_profile(self, profile_name: str, export_path: str) -> bool:
        """Export a profile to a specific location"""
        try:
            config = self.load_profile(profile_name)
            export_file = Path(export_path)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Exported profile '{profile_name}' to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting profile {profile_name}: {e}")
            return False
    
    def import_profile(self, profile_name: str, import_path: str) -> bool:
        """Import a profile from a file"""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_file, 'r', encoding='utf-8') as f:
                if import_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            self.save_profile(profile_name, config)
            self.logger.info(f"Imported profile '{profile_name}' from {import_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing profile {profile_name}: {e}")
            return False
    
    def register_watcher(self, callback):
        """Register a callback to be notified of configuration changes"""
        self.watchers.append(callback)
    
    def _notify_watchers(self, profile_name: str, config: Dict[str, Any]):
        """Notify all registered watchers of configuration changes"""
        for callback in self.watchers:
            try:
                callback(profile_name, config)
            except Exception as e:
                self.logger.error(f"Error in config watcher: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Define validation rules
        rules = {
            "system.max_agents": lambda x: isinstance(x, int) and 1 <= x <= 50,
            "agents.max_concurrent": lambda x: isinstance(x, int) and 1 <= x <= 20,
            "agents.timeout_seconds": lambda x: isinstance(x, int) and x > 0,
            "models.cache_size_gb": lambda x: isinstance(x, (int, float)) and x > 0,
        }
        
        for key_path, validator in rules.items():
            value = self._get_nested_value(config, key_path.split('.'))
            if value is not None and not validator(value):
                issues.append(f"Invalid value for {key_path}: {value}")
        
        return issues
    
    def _get_nested_value(self, config: Dict[str, Any], keys: List[str]):
        """Get nested value from config dictionary"""
        current = config
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None
    
    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """Get information about a specific profile"""
        profile_path = self.config_dir / f"{profile_name}.yaml"
        
        if not profile_path.exists():
            return {}
        
        try:
            config = self.load_profile(profile_name)
            metadata = config.get("_metadata", {})
            
            return {
                "name": profile_name,
                "path": str(profile_path),
                "size_bytes": profile_path.stat().st_size,
                "modified": datetime.fromtimestamp(profile_path.stat().st_mtime).isoformat(),
                "metadata": metadata,
                "is_current": profile_name == self.current_profile
            }
        except Exception as e:
            self.logger.error(f"Error getting profile info for {profile_name}: {e}")
            return {}

# Global config manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager
