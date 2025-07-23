#!/usr/bin/env python3
"""
Performance Monitor and Resource Optimizer
==========================================
Advanced system monitoring and optimization for maximum performance.
"""

import psutil
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_agents: int
    completed_tasks: int
    avg_response_time: float
    error_rate: float

@dataclass
class SystemLimits:
    """System resource limits and thresholds"""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_agents: int = 10
    max_concurrent_tasks: int = 5
    response_time_threshold: float = 5.0
    error_rate_threshold: float = 0.05

class PerformanceMonitor:
    """Advanced performance monitoring and optimization system"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger("PerformanceMonitor")
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
        # Performance optimization
        self.system_limits = SystemLimits()
        self.optimization_callbacks: List[Callable] = []
        self.last_optimization = datetime.now()
        self.optimization_interval = 30  # seconds
        
        # Resource tracking
        self.process = psutil.Process()
        self.initial_io_counters = psutil.disk_io_counters()
        self.initial_net_counters = psutil.net_io_counters()
        self.start_time = time.time()
        
        # Agent performance tracking
        self.agent_stats = {}
        self.task_stats = {
            'completed': 0,
            'failed': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
        
        self.logger.info("Performance Monitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring in background thread"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Check if optimization is needed
                if self._should_optimize():
                    self._optimize_system()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # IO metrics
        current_io = psutil.disk_io_counters()
        io_read_mb = (current_io.read_bytes - self.initial_io_counters.read_bytes) / (1024 * 1024)
        io_write_mb = (current_io.write_bytes - self.initial_io_counters.write_bytes) / (1024 * 1024)
        
        # Network metrics
        current_net = psutil.net_io_counters()
        net_sent_mb = (current_net.bytes_sent - self.initial_net_counters.bytes_sent) / (1024 * 1024)
        net_recv_mb = (current_net.bytes_recv - self.initial_net_counters.bytes_recv) / (1024 * 1024)
        
        # Agent metrics
        active_agents = len(self.agent_stats)
        completed_tasks = self.task_stats['completed']
        avg_response_time = self.task_stats['average_time']
        error_rate = self.task_stats['failed'] / max(1, self.task_stats['completed'] + self.task_stats['failed'])
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_io_read_mb=io_read_mb,
            disk_io_write_mb=io_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            active_agents=active_agents,
            completed_tasks=completed_tasks,
            avg_response_time=avg_response_time,
            error_rate=error_rate
        )
    
    def _should_optimize(self) -> bool:
        """Check if system optimization is needed"""
        if not self.metrics_history:
            return False
        
        current_time = datetime.now()
        if (current_time - self.last_optimization).seconds < self.optimization_interval:
            return False
        
        latest = self.metrics_history[-1]
        
        # Check if any limits are exceeded
        if (latest.cpu_percent > self.system_limits.max_cpu_percent or
            latest.memory_percent > self.system_limits.max_memory_percent or
            latest.avg_response_time > self.system_limits.response_time_threshold or
            latest.error_rate > self.system_limits.error_rate_threshold):
            return True
        
        return False
    
    def _optimize_system(self):
        """Perform system optimization"""
        self.logger.info("Performing system optimization...")
        
        latest = self.metrics_history[-1]
        
        # CPU optimization
        if latest.cpu_percent > self.system_limits.max_cpu_percent:
            self._optimize_cpu_usage()
        
        # Memory optimization
        if latest.memory_percent > self.system_limits.max_memory_percent:
            self._optimize_memory_usage()
        
        # Response time optimization
        if latest.avg_response_time > self.system_limits.response_time_threshold:
            self._optimize_response_time()
        
        # Error rate optimization
        if latest.error_rate > self.system_limits.error_rate_threshold:
            self._optimize_error_rate()
        
        # Call registered optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(latest)
            except Exception as e:
                self.logger.error(f"Error in optimization callback: {e}")
        
        self.last_optimization = datetime.now()
        self.logger.info("System optimization completed")
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        self.logger.info("Optimizing CPU usage...")
        
        # Reduce max concurrent agents
        if self.system_limits.max_agents > 3:
            self.system_limits.max_agents -= 1
            if self.config_manager:
                self.config_manager.update_config("system.max_agents", self.system_limits.max_agents)
        
        # Reduce concurrent tasks
        if self.system_limits.max_concurrent_tasks > 2:
            self.system_limits.max_concurrent_tasks -= 1
            if self.config_manager:
                self.config_manager.update_config("agents.max_concurrent", self.system_limits.max_concurrent_tasks)
    
    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        self.logger.info("Optimizing memory usage...")
        
        # Clear old metrics
        if len(self.metrics_history) > 500:
            self.metrics_history = self.metrics_history[-500:]
        
        # Suggest garbage collection
        import gc
        gc.collect()
        
        # Reduce cache sizes if config manager is available
        if self.config_manager:
            current_cache = self.config_manager.get_config_value("models.cache_size_gb", 2)
            if current_cache > 1:
                self.config_manager.update_config("models.cache_size_gb", max(1, current_cache - 0.5))
    
    def _optimize_response_time(self):
        """Optimize response time"""
        self.logger.info("Optimizing response time...")
        
        # Increase timeout for agents to complete tasks
        if self.config_manager:
            current_timeout = self.config_manager.get_config_value("agents.timeout_seconds", 300)
            self.config_manager.update_config("agents.timeout_seconds", min(600, current_timeout + 60))
    
    def _optimize_error_rate(self):
        """Optimize error rate"""
        self.logger.info("Optimizing error rate...")
        
        # Increase retry attempts
        if self.config_manager:
            current_retries = self.config_manager.get_config_value("agents.retry_attempts", 3)
            self.config_manager.update_config("agents.retry_attempts", min(5, current_retries + 1))
    
    def track_agent_performance(self, agent_name: str, task_time: float, success: bool):
        """Track performance of individual agents"""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'last_used': datetime.now()
            }
        
        stats = self.agent_stats[agent_name]
        
        if success:
            stats['tasks_completed'] += 1
            self.task_stats['completed'] += 1
        else:
            stats['tasks_failed'] += 1
            self.task_stats['failed'] += 1
        
        stats['total_time'] += task_time
        stats['average_time'] = stats['total_time'] / max(1, stats['tasks_completed'])
        stats['last_used'] = datetime.now()
        
        # Update global task stats
        total_tasks = self.task_stats['completed'] + self.task_stats['failed']
        self.task_stats['total_time'] += task_time
        self.task_stats['average_time'] = self.task_stats['total_time'] / max(1, total_tasks)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for the specified duration"""
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return self.agent_stats.copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        if not self.metrics_history:
            return {"status": "unknown", "details": "No metrics available"}
        
        latest = self.metrics_history[-1]
        issues = []
        warnings = []
        
        # Check CPU
        if latest.cpu_percent > self.system_limits.max_cpu_percent:
            issues.append(f"High CPU usage: {latest.cpu_percent:.1f}%")
        elif latest.cpu_percent > self.system_limits.max_cpu_percent * 0.8:
            warnings.append(f"Elevated CPU usage: {latest.cpu_percent:.1f}%")
        
        # Check Memory
        if latest.memory_percent > self.system_limits.max_memory_percent:
            issues.append(f"High memory usage: {latest.memory_percent:.1f}%")
        elif latest.memory_percent > self.system_limits.max_memory_percent * 0.8:
            warnings.append(f"Elevated memory usage: {latest.memory_percent:.1f}%")
        
        # Check Response Time
        if latest.avg_response_time > self.system_limits.response_time_threshold:
            issues.append(f"Slow response time: {latest.avg_response_time:.2f}s")
        elif latest.avg_response_time > self.system_limits.response_time_threshold * 0.8:
            warnings.append(f"Elevated response time: {latest.avg_response_time:.2f}s")
        
        # Check Error Rate
        if latest.error_rate > self.system_limits.error_rate_threshold:
            issues.append(f"High error rate: {latest.error_rate:.2%}")
        elif latest.error_rate > self.system_limits.error_rate_threshold * 0.8:
            warnings.append(f"Elevated error rate: {latest.error_rate:.2%}")
        
        # Determine overall status
        if issues:
            status = "critical"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "active_agents": latest.active_agents,
                "avg_response_time": latest.avg_response_time,
                "error_rate": latest.error_rate
            },
            "uptime_seconds": time.time() - self.start_time
        }
    
    def register_optimization_callback(self, callback: Callable):
        """Register a callback for system optimization events"""
        self.optimization_callbacks.append(callback)
    
    def export_metrics(self, filepath: str, duration_minutes: int = 60):
        """Export metrics to a file"""
        metrics_data = []
        history = self.get_metrics_history(duration_minutes)
        
        for metric in history:
            metrics_data.append({
                "timestamp": metric.timestamp.isoformat(),
                "cpu_percent": metric.cpu_percent,
                "memory_percent": metric.memory_percent,
                "memory_used_mb": metric.memory_used_mb,
                "memory_available_mb": metric.memory_available_mb,
                "disk_io_read_mb": metric.disk_io_read_mb,
                "disk_io_write_mb": metric.disk_io_write_mb,
                "network_sent_mb": metric.network_sent_mb,
                "network_recv_mb": metric.network_recv_mb,
                "active_agents": metric.active_agents,
                "completed_tasks": metric.completed_tasks,
                "avg_response_time": metric.avg_response_time,
                "error_rate": metric.error_rate
            })
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "metrics_count": len(metrics_data),
            "agent_stats": self.agent_stats,
            "system_health": self.get_system_health(),
            "metrics": metrics_data
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return ["Start monitoring to get performance recommendations"]
        
        # Analyze recent metrics
        recent_metrics = self.get_metrics_history(10)  # Last 10 minutes
        if len(recent_metrics) < 5:
            return ["Not enough data for recommendations"]
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # CPU recommendations
        if avg_cpu > 70:
            recommendations.append("Consider reducing the number of concurrent agents")
            recommendations.append("Enable CPU optimization in performance mode")
        elif avg_cpu < 30:
            recommendations.append("System has spare CPU capacity - consider increasing concurrent agents")
        
        # Memory recommendations
        if avg_memory > 75:
            recommendations.append("Consider reducing model cache size")
            recommendations.append("Enable memory optimization and garbage collection")
        elif avg_memory < 40:
            recommendations.append("System has spare memory - consider increasing cache sizes")
        
        # Response time recommendations
        if avg_response > 3.0:
            recommendations.append("Consider optimizing agent selection algorithms")
            recommendations.append("Enable response time monitoring and optimization")
        
        # Error rate recommendations
        if avg_error_rate > 0.02:
            recommendations.append("Increase retry attempts for failed tasks")
            recommendations.append("Enable detailed error logging to identify issues")
        
        # Agent-specific recommendations
        for agent_name, stats in self.agent_stats.items():
            if stats['tasks_failed'] > stats['tasks_completed'] * 0.1:
                recommendations.append(f"Agent '{agent_name}' has high failure rate - consider investigation")
            
            if stats['average_time'] > 5.0:
                recommendations.append(f"Agent '{agent_name}' has slow response time - consider optimization")
        
        return recommendations if recommendations else ["System performance is optimal"]

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return performance_monitor
