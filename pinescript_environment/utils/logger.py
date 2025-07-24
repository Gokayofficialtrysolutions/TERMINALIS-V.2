"""
Logger Utility for PineScript Environment
Advanced logging with multiple handlers and formatters

Author: Gokaytrysolutions Team
Version: 1.0.0
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Format the message
        formatted_message = super().format(record)
        
        return formatted_message

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)

class PineScriptLogger:
    """
    Advanced logger for PineScript environment with multiple features:
    - Color-coded console output
    - File rotation
    - JSON structured logging
    - Performance metrics
    - Error tracking
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Performance tracking
        self.performance_metrics = {}
        self.error_count = 0
        self.warning_count = 0
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        log_level = self.config.get('level', 'INFO').upper()
        log_format = self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_dir = Path(self.config.get('log_dir', 'logs'))
        
        # Create log directory
        log_dir.mkdir(exist_ok=True)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(
            f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - "
            f"{Fore.MAGENTA}%(name)s{Style.RESET_ALL} - "
            f"%(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name.replace('.', '_')}.log",
            maxBytes=self.config.get('max_size', 10485760),  # 10MB
            backupCount=self.config.get('backup_count', 5)
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name.replace('.', '_')}_structured.json",
            maxBytes=self.config.get('max_size', 10485760),
            backupCount=self.config.get('backup_count', 5)
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name.replace('.', '_')}_errors.log",
            maxBytes=self.config.get('max_size', 10485760),
            backupCount=self.config.get('backup_count', 5)
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, extra_data: Optional[Dict] = None):
        """Log debug message"""
        self._log(logging.DEBUG, message, extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message"""
        self._log(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message"""
        self.warning_count += 1
        self._log(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict] = None, exc_info: bool = False):
        """Log error message"""
        self.error_count += 1
        self._log(logging.ERROR, message, extra_data, exc_info)
    
    def critical(self, message: str, extra_data: Optional[Dict] = None, exc_info: bool = False):
        """Log critical message"""
        self.error_count += 1
        self._log(logging.CRITICAL, message, extra_data, exc_info)
    
    def exception(self, message: str, extra_data: Optional[Dict] = None):
        """Log exception with traceback"""
        self.error_count += 1
        self._log(logging.ERROR, message, extra_data, exc_info=True)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict] = None, exc_info: bool = False):
        """Internal logging method"""
        if extra_data:
            # Create a new LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, __file__, 0, message, (), None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            self.logger.log(level, message, exc_info=exc_info)
    
    def log_performance(self, operation: str, duration: float, extra_data: Optional[Dict] = None):
        """Log performance metrics"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append(duration)
        
        perf_data = {
            'operation': operation,
            'duration': duration,
            'unit': 'seconds'
        }
        
        if extra_data:
            perf_data.update(extra_data)
        
        self.info(f"Performance: {operation} took {duration:.4f}s", perf_data)
    
    def log_code_analysis(self, code: str, result: Dict[str, Any]):
        """Log code analysis results"""
        analysis_data = {
            'code_length': len(code),
            'lines_count': len(code.split('\n')),
            'analysis_result': result
        }
        
        self.info("Code analysis completed", analysis_data)
    
    def log_ai_interaction(self, request_type: str, prompt: str, response: str, model: str, tokens_used: int):
        """Log AI interactions"""
        ai_data = {
            'request_type': request_type,
            'model': model,
            'tokens_used': tokens_used,
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
        
        self.info(f"AI interaction: {request_type} using {model}", ai_data)
    
    def log_user_action(self, user_id: str, action: str, details: Optional[Dict] = None):
        """Log user actions"""
        user_data = {
            'user_id': user_id,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if details:
            user_data.update(details)
        
        self.info(f"User action: {action}", user_data)
    
    def log_system_status(self, component: str, status: str, metrics: Optional[Dict] = None):
        """Log system status"""
        status_data = {
            'component': component,
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if metrics:
            status_data['metrics'] = metrics
        
        self.info(f"System status: {component} - {status}", status_data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for operation, durations in self.performance_metrics.items():
            if durations:
                stats[operation] = {
                    'count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        
        return stats
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }
    
    def reset_metrics(self):
        """Reset performance metrics and error counts"""
        self.performance_metrics.clear()
        self.error_count = 0
        self.warning_count = 0
    
    def create_child_logger(self, suffix: str) -> 'PineScriptLogger':
        """Create a child logger"""
        child_name = f"{self.name}.{suffix}"
        return PineScriptLogger(child_name, self.config)
    
    def set_level(self, level: str):
        """Set logging level"""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Update handler levels
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)
    
    def add_context(self, **kwargs):
        """Add context to all future log messages"""
        # This would be implemented using a context manager or filter
        pass

class PerformanceLogger:
    """Context manager for logging performance"""
    
    def __init__(self, logger: PineScriptLogger, operation: str, extra_data: Optional[Dict] = None):
        self.logger = logger
        self.operation = operation
        self.extra_data = extra_data or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Started: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            if exc_type:
                self.logger.error(f"Failed: {self.operation} - {exc_val}", self.extra_data, exc_info=True)
            else:
                self.logger.log_performance(self.operation, duration, self.extra_data)

def setup_logger(name: str, config: Optional[Dict[str, Any]] = None) -> PineScriptLogger:
    """Setup and return a configured logger"""
    return PineScriptLogger(name, config)

def get_logger(name: str) -> PineScriptLogger:
    """Get or create a logger instance"""
    return PineScriptLogger(name)

# Global logger instances for common use
main_logger = setup_logger("pinescript_env")
engine_logger = main_logger.create_child_logger("engine")
ai_logger = main_logger.create_child_logger("ai")
api_logger = main_logger.create_child_logger("api")
ui_logger = main_logger.create_child_logger("ui")

# Convenience functions
def log_startup():
    """Log application startup"""
    main_logger.info("🚀 PineScript XXXXLARGE Environment starting up...")

def log_shutdown():
    """Log application shutdown"""
    stats = main_logger.get_performance_stats()
    error_stats = main_logger.get_error_stats()
    
    main_logger.info("📊 Performance statistics:", stats)
    main_logger.info("⚠️ Error statistics:", error_stats)
    main_logger.info("👋 PineScript XXXXLARGE Environment shutting down...")

def log_critical_error(message: str, exc_info: bool = True):
    """Log critical system error"""
    main_logger.critical(f"💥 CRITICAL ERROR: {message}", exc_info=exc_info)

def log_user_session_start(user_id: str, session_id: str):
    """Log user session start"""
    main_logger.log_user_action(user_id, "session_start", {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    })

def log_user_session_end(user_id: str, session_id: str, duration: float):
    """Log user session end"""
    main_logger.log_user_action(user_id, "session_end", {
        "session_id": session_id,
        "duration": duration,
        "timestamp": datetime.utcnow().isoformat()
    })
