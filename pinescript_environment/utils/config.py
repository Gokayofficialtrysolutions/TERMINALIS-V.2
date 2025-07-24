"""
Configuration Manager for PineScript Environment
Centralized configuration management

Author: Gokaytrysolutions Team
Version: 1.0.0
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

@dataclass
class AIConfig:
    """AI service configuration"""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_primary: str = "gpt-4"
    model_fallback: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.3
    top_p: float = 0.9

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "pinescript_env.db"
    connection_string: str = ""

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    workers: int = 4
    reload: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/pinescript_env.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    api_rate_limit: int = 100  # requests per minute

@dataclass
class PineScriptConfig:
    """PineScript specific configuration"""
    max_script_size: int = 1048576  # 1MB
    max_execution_time: int = 30  # seconds
    max_plots: int = 64
    max_variables: int = 1000
    supported_versions: list = None
    
    def __post_init__(self):
        if self.supported_versions is None:
            self.supported_versions = ["v1", "v2", "v3", "v4", "v5", "v6"]

class ConfigManager:
    """
    Centralized configuration manager for the PineScript environment
    
    Features:
    - Environment variable support
    - Multiple config file formats (JSON, YAML)
    - Default configurations
    - Validation and type checking
    - Hot reloading of configurations
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config_dir = Path(__file__).parent.parent
        
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration objects
        self.ai = AIConfig()
        self.database = DatabaseConfig()
        self.server = ServerConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.pinescript = PineScriptConfig()
        
        # Load configurations
        self.load_configurations()
    
    def load_configurations(self):
        """Load configurations from various sources"""
        try:
            # 1. Load from environment variables
            self._load_from_env()
            
            # 2. Load from config file if specified
            if self.config_file:
                self._load_from_file(self.config_file)
            else:
                # Try to find default config files
                self._load_default_configs()
            
            # 3. Validate configurations
            self._validate_configs()
            
            self.logger.info("✅ Configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configurations: {e}")
            raise
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # AI Configuration
        if os.getenv("OPENAI_API_KEY"):
            self.ai.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.ai.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if os.getenv("AI_MODEL_PRIMARY"):
            self.ai.model_primary = os.getenv("AI_MODEL_PRIMARY", "gpt-4")
        
        # Database Configuration
        if os.getenv("DATABASE_TYPE"):
            self.database.type = os.getenv("DATABASE_TYPE", "sqlite")
        if os.getenv("DATABASE_HOST"):
            self.database.host = os.getenv("DATABASE_HOST", "localhost")
        if os.getenv("DATABASE_PORT"):
            self.database.port = int(os.getenv("DATABASE_PORT", "5432"))
        if os.getenv("DATABASE_URL"):
            self.database.connection_string = os.getenv("DATABASE_URL", "")
        
        # Server Configuration
        if os.getenv("SERVER_HOST"):
            self.server.host = os.getenv("SERVER_HOST", "localhost")
        if os.getenv("SERVER_PORT"):
            self.server.port = int(os.getenv("SERVER_PORT", "8080"))
        if os.getenv("DEBUG"):
            self.server.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Security Configuration
        if os.getenv("SECRET_KEY"):
            self.security.secret_key = os.getenv("SECRET_KEY", "")
        
        # Logging Configuration
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL", "INFO")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {config_path.suffix}")
                    return
            
            self._apply_config_data(config_data)
            self.logger.info(f"Loaded configuration from: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _load_default_configs(self):
        """Load default configuration files"""
        config_files = [
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            self.config_dir / "config.json",
            self.config_dir / "settings.yaml",
            self.config_dir / "settings.yml",
            self.config_dir / "settings.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                self._load_from_file(str(config_file))
                break
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to config objects"""
        if "ai" in config_data:
            ai_config = config_data["ai"]
            for key, value in ai_config.items():
                if hasattr(self.ai, key):
                    setattr(self.ai, key, value)
        
        if "database" in config_data:
            db_config = config_data["database"]
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        if "server" in config_data:
            server_config = config_data["server"]
            for key, value in server_config.items():
                if hasattr(self.server, key):
                    setattr(self.server, key, value)
        
        if "logging" in config_data:
            logging_config = config_data["logging"]
            for key, value in logging_config.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        if "security" in config_data:
            security_config = config_data["security"]
            for key, value in security_config.items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)
        
        if "pinescript" in config_data:
            ps_config = config_data["pinescript"]
            for key, value in ps_config.items():
                if hasattr(self.pinescript, key):
                    setattr(self.pinescript, key, value)
    
    def _validate_configs(self):
        """Validate configuration values"""
        # Validate AI configuration
        if not self.ai.openai_api_key and not self.ai.anthropic_api_key:
            self.logger.warning("No AI API keys configured - AI features will be limited")
        
        # Validate server configuration
        if not (1 <= self.server.port <= 65535):
            raise ValueError(f"Invalid server port: {self.server.port}")
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.logging.level}")
        
        # Validate security configuration
        if not self.security.secret_key:
            import secrets
            self.security.secret_key = secrets.token_urlsafe(32)
            self.logger.warning("Generated random secret key - consider setting SECRET_KEY environment variable")
        
        # Validate PineScript configuration
        if self.pinescript.max_script_size <= 0:
            raise ValueError("max_script_size must be positive")
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        if self.database.connection_string:
            return self.database.connection_string
        
        if self.database.type == "sqlite":
            db_path = self.config_dir / self.database.database
            return f"sqlite:///{db_path}"
        elif self.database.type == "postgresql":
            return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
        elif self.database.type == "mysql":
            return f"mysql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            if not config_file:
                config_file = str(self.config_dir / "config.yaml")
            
            config_data = {
                "ai": asdict(self.ai),
                "database": asdict(self.database),
                "server": asdict(self.server),
                "logging": asdict(self.logging),
                "security": asdict(self.security),
                "pinescript": asdict(self.pinescript)
            }
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get all configurations as dictionary"""
        return {
            "ai": asdict(self.ai),
            "database": asdict(self.database),
            "server": asdict(self.server),
            "logging": asdict(self.logging),
            "security": asdict(self.security),
            "pinescript": asdict(self.pinescript)
        }
    
    def update_config(self, section: str, **kwargs):
        """Update configuration values"""
        config_obj = getattr(self, section, None)
        if not config_obj:
            raise ValueError(f"Unknown configuration section: {section}")
        
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {section}.{key}")
        
        self.logger.info(f"Updated {section} configuration")
    
    def reload_config(self):
        """Reload configuration from sources"""
        self.load_configurations()
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration for Python logging"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.logging.format
                },
            },
            'handlers': {
                'console': {
                    'level': self.logging.level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                },
                'file': {
                    'level': self.logging.level,
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.logging.file_path,
                    'maxBytes': self.logging.max_size,
                    'backupCount': self.logging.backup_count,
                    'formatter': 'standard',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': self.logging.level,
                    'propagate': False
                }
            }
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.server.debug or os.getenv("ENVIRONMENT", "").lower() in ["dev", "development"]
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.is_development()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(ai={bool(self.ai.openai_api_key)}, db={self.database.type}, server={self.server.host}:{self.server.port})"
    
    def __repr__(self) -> str:
        return self.__str__()
