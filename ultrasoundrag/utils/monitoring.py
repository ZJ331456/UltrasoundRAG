"""
UltrasoundRAG 监控和可观测性模块
提供系统监控、健康检查、指标收集和告警功能
"""

import time
import json
import threading
import psutil
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import structlog
from datetime import datetime, timezone
import traceback

from .exceptions import UltrasoundRAGException, ErrorCode


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """健康检查项"""
    name: str
    check_func: Callable[[], bool]
    description: str = ""
    timeout: float = 5.0
    required: bool = True  # 是否为必需的检查项
    interval: float = 60.0  # 检查间隔（秒）
    last_check: Optional[float] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    network_io: Dict[str, float] = field(default_factory=dict)
    disk_io: Dict[str, float] = field(default_factory=dict)


@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: float = field(default_factory=time.time)
    active_requests: int = 0
    total_requests: int = 0
    error_requests: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    model_memory_usage: float = 0.0
    active_models: int = 0


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = "system"
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.lock = threading.RLock()
        
        # 注册默认检查项
        self._register_default_checks()
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        description: str = "",
        timeout: float = 5.0,
        required: bool = True,
        interval: float = 60.0
    ):
        """注册健康检查项"""
        self.checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            description=description,
            timeout=timeout,
            required=required,
            interval=interval
        )
    
    def _register_default_checks(self):
        """注册默认检查项"""
        # 内存检查
        def check_memory():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # 内存使用率不超过90%
        
        # CPU检查
        def check_cpu():
            cpu_usage = psutil.cpu_percent(interval=1)
            return cpu_usage < 80  # CPU使用率不超过80%
        
        # 磁盘检查
        def check_disk():
            disk = psutil.disk_usage('/')
            return disk.percent < 90  # 磁盘使用率不超过90%
        
        # Milvus连接检查
        def check_milvus():
            try:
                from ..milvus.milvus_manager import MilvusManager
                manager = MilvusManager(collection_type="md")
                return manager.health_check()
            except Exception:
                return False
        
        self.register_check("memory", check_memory, "内存使用率检查", required=True)
        self.register_check("cpu", check_cpu, "CPU使用率检查", required=False)
        self.register_check("disk", check_disk, "磁盘使用率检查", required=True)
        self.register_check("milvus", check_milvus, "Milvus数据库连接检查", required=True, timeout=10.0)
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """运行单个检查"""
        if name not in self.checks:
            return {"status": "unknown", "error": f"检查项 {name} 不存在"}
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # 运行检查（带超时）
            result = self._run_with_timeout(check.check_func, check.timeout)
            duration = time.time() - start_time
            
            # 更新检查信息
            check.last_check = start_time
            check.last_result = result
            check.last_error = None
            
            return {
                "status": "healthy" if result else "unhealthy",
                "duration": duration,
                "description": check.description,
                "required": check.required,
                "last_check": check.last_check
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # 更新检查信息
            check.last_check = start_time
            check.last_result = False
            check.last_error = error_msg
            
            return {
                "status": "unhealthy",
                "duration": duration,
                "error": error_msg,
                "description": check.description,
                "required": check.required,
                "last_check": check.last_check
            }
    
    def _run_with_timeout(self, func: Callable, timeout: float):
        """带超时的函数执行"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"检查超时 ({timeout}s)")
        
        # 设置超时
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func()
            signal.alarm(0)  # 取消超时
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        with self.lock:
            results = {}
            healthy_count = 0
            required_count = 0
            
            for name, check in self.checks.items():
                result = self.run_check(name)
                results[name] = result
                
                if check.required:
                    required_count += 1
                    if result["status"] == "healthy":
                        healthy_count += 1
            
            # 计算整体状态
            if required_count == 0:
                self.overall_status = HealthStatus.UNKNOWN
            elif healthy_count == required_count:
                self.overall_status = HealthStatus.HEALTHY
            elif healthy_count > 0:
                self.overall_status = HealthStatus.DEGRADED
            else:
                self.overall_status = HealthStatus.UNHEALTHY
            
            self.results = results
            
            return {
                "overall_status": self.overall_status.value,
                "timestamp": time.time(),
                "checks": results,
                "summary": {
                    "total_checks": len(self.checks),
                    "required_checks": required_count,
                    "healthy_checks": healthy_count,
                    "health_ratio": healthy_count / required_count if required_count > 0 else 0
                }
            }
    
    def get_status(self) -> HealthStatus:
        """获取整体健康状态"""
        return self.overall_status


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.app_metrics: deque = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.RLock()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        metrics = SystemMetrics()
        
        # CPU使用率
        metrics.cpu_usage = psutil.cpu_percent(interval=None)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        metrics.memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        metrics.disk_usage = disk.percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        metrics.network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.disk_io = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            }
        
        # GPU指标（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                metrics.gpu_usage = gpu.load * 100
                metrics.gpu_memory_usage = gpu.memoryUtil * 100
        except ImportError:
            pass  # GPU监控不可用
        
        with self.lock:
            self.system_metrics.append(metrics)
        
        return metrics
    
    def collect_app_metrics(self, **kwargs) -> ApplicationMetrics:
        """收集应用指标"""
        metrics = ApplicationMetrics(**kwargs)
        
        with self.lock:
            self.app_metrics.append(metrics)
        
        return metrics
    
    def record_custom_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """记录自定义指标"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.custom_metrics[name].append({"value": value, "timestamp": timestamp})
    
    def get_recent_metrics(self, count: int = 10) -> Dict[str, Any]:
        """获取最近的指标"""
        with self.lock:
            return {
                "system_metrics": [asdict(m) for m in list(self.system_metrics)[-count:]],
                "app_metrics": [asdict(m) for m in list(self.app_metrics)[-count:]],
                "custom_metrics": {
                    name: list(values)[-count:]
                    for name, values in self.custom_metrics.items()
                }
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            current_system = self.system_metrics[-1] if self.system_metrics else None
            current_app = self.app_metrics[-1] if self.app_metrics else None
            
            return {
                "current_system_metrics": asdict(current_system) if current_system else None,
                "current_app_metrics": asdict(current_app) if current_app else None,
                "metrics_count": {
                    "system": len(self.system_metrics),
                    "app": len(self.app_metrics),
                    "custom": len(self.custom_metrics)
                }
            }


class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.lock = threading.RLock()
        self.alert_counter = 0
    
    def register_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """注册告警处理器"""
        self.alert_handlers[level].append(handler)
    
    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """创建告警"""
        with self.lock:
            self.alert_counter += 1
            alert = Alert(
                id=f"alert_{self.alert_counter}",
                level=level,
                title=title,
                message=message,
                source=source,
                metadata=metadata or {}
            )
            
            self.alerts.append(alert)
            
            # 触发处理器
            for handler in self.alert_handlers[level]:
                try:
                    handler(alert)
                except Exception as e:
                    # 避免处理器异常影响告警系统
                    print(f"告警处理器异常: {e}")
            
            return alert
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_recent_alerts(self, count: int = 50) -> List[Alert]:
        """获取最近告警"""
        with self.lock:
            return list(self.alerts)[-count:]


class StructuredLogger:
    """结构化日志器"""
    
    def __init__(self, service_name: str = "ultrasound_rag"):
        self.service_name = service_name
        
        # 配置structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(indent=2)
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(service_name)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """记录请求日志"""
        self.logger.info(
            "API请求",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            user_id=user_id,
            **kwargs
        )
    
    def log_retrieval(
        self,
        retrieval_type: str,
        query: str,
        results_count: int,
        duration: float,
        **kwargs
    ):
        """记录检索日志"""
        self.logger.info(
            "检索操作",
            retrieval_type=retrieval_type,
            query=query[:100],  # 截断长查询
            results_count=results_count,
            duration=duration,
            **kwargs
        )
    
    def log_model_operation(
        self,
        model_name: str,
        operation: str,
        duration: float,
        **kwargs
    ):
        """记录模型操作日志"""
        self.logger.info(
            "模型操作",
            model_name=model_name,
            operation=operation,
            duration=duration,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """记录错误日志"""
        self.logger.error(
            "系统错误",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            traceback=traceback.format_exc(),
            **kwargs
        )


class SystemMonitor:
    """系统监控器 - 统一监控入口"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.logger = StructuredLogger()
        
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitor_interval = 30.0  # 30秒监控间隔
        
        # 注册默认告警处理器
        self._register_default_alert_handlers()
    
    def _register_default_alert_handlers(self):
        """注册默认告警处理器"""
        def log_alert(alert: Alert):
            self.logger.logger.log(
                alert.level.value.upper(),
                "系统告警",
                alert_id=alert.id,
                title=alert.title,
                message=alert.message,
                source=alert.source,
                metadata=alert.metadata
            )
        
        # 为所有级别注册日志处理器
        for level in AlertLevel:
            self.alert_manager.register_handler(level, log_alert)
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.logger.info("系统监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # 检查是否需要告警
                self._check_system_alerts(system_metrics)
                
                # 运行健康检查
                health_status = self.health_checker.run_all_checks()
                
                # 记录监控日志
                self.logger.logger.debug(
                    "监控周期完成",
                    cpu_usage=system_metrics.cpu_usage,
                    memory_usage=system_metrics.memory_usage,
                    disk_usage=system_metrics.disk_usage,
                    health_status=health_status["overall_status"]
                )
                
            except Exception as e:
                self.logger.log_error(e, {"context": "监控循环"})
            
            time.sleep(self.monitor_interval)
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统告警"""
        # CPU告警
        if metrics.cpu_usage > 90:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "CPU使用率过高",
                f"CPU使用率达到 {metrics.cpu_usage:.1f}%",
                metadata={"cpu_usage": metrics.cpu_usage}
            )
        elif metrics.cpu_usage > 80:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "CPU使用率较高",
                f"CPU使用率达到 {metrics.cpu_usage:.1f}%",
                metadata={"cpu_usage": metrics.cpu_usage}
            )
        
        # 内存告警
        if metrics.memory_usage > 95:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "内存使用率过高",
                f"内存使用率达到 {metrics.memory_usage:.1f}%",
                metadata={"memory_usage": metrics.memory_usage}
            )
        elif metrics.memory_usage > 85:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "内存使用率较高",
                f"内存使用率达到 {metrics.memory_usage:.1f}%",
                metadata={"memory_usage": metrics.memory_usage}
            )
        
        # 磁盘告警
        if metrics.disk_usage > 95:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "磁盘空间不足",
                f"磁盘使用率达到 {metrics.disk_usage:.1f}%",
                metadata={"disk_usage": metrics.disk_usage}
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        health_status = self.health_checker.run_all_checks()
        metrics_summary = self.metrics_collector.get_metrics_summary()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": time.time(),
            "service": "UltrasoundRAG",
            "version": "1.0.0",
            "health": health_status,
            "metrics": metrics_summary,
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": [asdict(alert) for alert in active_alerts[-10:]]  # 最近10个
            },
            "monitoring": {
                "is_active": self.is_monitoring,
                "interval": self.monitor_interval
            }
        }


# 全局监控器实例
system_monitor = SystemMonitor()


# 监控装饰器
def monitor_operation(operation_name: str, log_args: bool = False):
    """操作监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功操作
                log_context = {"operation": operation_name, "duration": duration}
                if log_args:
                    log_context.update({"args": str(args)[:200], "kwargs": str(kwargs)[:200]})
                
                system_monitor.logger.logger.info("操作完成", **log_context)
                system_monitor.metrics_collector.record_custom_metric(f"{operation_name}_duration", duration)
                system_monitor.metrics_collector.record_custom_metric(f"{operation_name}_success", 1)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败操作
                system_monitor.logger.log_error(e, {"operation": operation_name, "duration": duration})
                system_monitor.metrics_collector.record_custom_metric(f"{operation_name}_error", 1)
                
                # 如果是严重错误，创建告警
                if isinstance(e, UltrasoundRAGException) and e.status_code >= 500:
                    system_monitor.alert_manager.create_alert(
                        AlertLevel.ERROR,
                        f"操作失败: {operation_name}",
                        str(e),
                        metadata={"operation": operation_name, "error_code": e.error_code.value}
                    )
                
                raise
        
        return wrapper
    return decorator
