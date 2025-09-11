"""
UltrasoundRAG 统一异常处理系统
提供标准化的异常定义、错误码和异常处理机制
"""

from enum import Enum
from typing import Dict, Any, Optional
import traceback
import time


class ErrorCode(Enum):
    """系统错误码枚举"""
    # 系统错误 (1000-1999)
    SYSTEM_ERROR = "SYS_1000"
    CONFIG_ERROR = "SYS_1001"
    INITIALIZATION_ERROR = "SYS_1002"
    
    # 数据库错误 (2000-2999)
    DATABASE_CONNECTION_ERROR = "DB_2000"
    DATABASE_QUERY_ERROR = "DB_2001"
    DATABASE_INSERT_ERROR = "DB_2002"
    MILVUS_ERROR = "DB_2100"
    
    # 模型错误 (3000-3999)
    MODEL_LOADING_ERROR = "MODEL_3000"
    MODEL_INFERENCE_ERROR = "MODEL_3001"
    EMBEDDING_ERROR = "MODEL_3002"
    CLIP_MODEL_ERROR = "MODEL_3100"
    
    # 检索错误 (4000-4999)
    RETRIEVAL_ERROR = "RETR_4000"
    RETRIEVAL_TIMEOUT = "RETR_4001"
    SEARCH_ERROR = "RETR_4002"
    RERANK_ERROR = "RETR_4003"
    
    # 输入验证错误 (5000-5999)
    INVALID_INPUT = "INPUT_5000"
    INVALID_QUERY = "INPUT_5001"
    INVALID_IMAGE = "INPUT_5002"
    FILE_NOT_FOUND = "INPUT_5003"
    
    # API错误 (6000-6999)
    AUTHENTICATION_ERROR = "API_6000"
    AUTHORIZATION_ERROR = "API_6001"
    RATE_LIMIT_ERROR = "API_6002"
    INVALID_REQUEST = "API_6003"
    
    # 业务逻辑错误 (7000-7999)
    BUSINESS_LOGIC_ERROR = "BIZ_7000"
    FUSION_ERROR = "BIZ_7001"
    RERANK_STRATEGY_ERROR = "BIZ_7002"


class UltrasoundRAGException(Exception):
    """系统基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = time.time()
        
        # 记录堆栈信息
        self.traceback_str = traceback.format_exc() if original_exception else None
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于API返回"""
        return {
            "error": self.error_code.value,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


class DatabaseError(UltrasoundRAGException):
    """数据库相关异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR, **kwargs):
        super().__init__(message, error_code, status_code=503, **kwargs)


class MilvusError(DatabaseError):
    """Milvus数据库异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCode.MILVUS_ERROR, **kwargs)


class ModelError(UltrasoundRAGException):
    """模型相关异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.MODEL_LOADING_ERROR, **kwargs):
        super().__init__(message, error_code, status_code=500, **kwargs)


class CLIPModelError(ModelError):
    """CLIP模型异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCode.CLIP_MODEL_ERROR, **kwargs)


class RetrievalError(UltrasoundRAGException):
    """检索相关异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.RETRIEVAL_ERROR, **kwargs):
        super().__init__(message, error_code, status_code=500, **kwargs)


class RetrievalTimeoutError(RetrievalError):
    """检索超时异常"""
    
    def __init__(self, timeout_seconds: float, **kwargs):
        message = f"检索超时，耗时 {timeout_seconds:.2f}s"
        super().__init__(message, ErrorCode.RETRIEVAL_TIMEOUT, status_code=408, **kwargs)


class ValidationError(UltrasoundRAGException):
    """输入验证异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_INPUT, **kwargs):
        super().__init__(message, error_code, status_code=400, **kwargs)


class AuthenticationError(UltrasoundRAGException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败", **kwargs):
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR, status_code=401, **kwargs)


class AuthorizationError(UltrasoundRAGException):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足", **kwargs):
        super().__init__(message, ErrorCode.AUTHORIZATION_ERROR, status_code=403, **kwargs)


class RateLimitError(UltrasoundRAGException):
    """频率限制异常"""
    
    def __init__(self, message: str = "请求过于频繁", retry_after: int = 60, **kwargs):
        super().__init__(message, ErrorCode.RATE_LIMIT_ERROR, status_code=429, **kwargs)
        self.details["retry_after"] = retry_after


# 异常处理装饰器
def handle_exceptions(
    default_error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
    reraise: bool = True,
    log_error: bool = True
):
    """
    异常处理装饰器
    
    Args:
        default_error_code: 默认错误码
        reraise: 是否重新抛出异常
        log_error: 是否记录错误日志
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UltrasoundRAGException:
                # 已经是系统异常，直接重新抛出
                if reraise:
                    raise
            except Exception as e:
                # 包装为系统异常
                wrapped_exception = UltrasoundRAGException(
                    message=f"函数 {func.__name__} 执行失败: {str(e)}",
                    error_code=default_error_code,
                    original_exception=e
                )
                
                if log_error:
                    # 这里可以添加日志记录
                    print(f"异常捕获: {wrapped_exception}")
                
                if reraise:
                    raise wrapped_exception
                else:
                    return None
        
        return wrapper
    return decorator


# 批量异常处理
class ExceptionCollector:
    """异常收集器，用于批量操作中的异常收集"""
    
    def __init__(self):
        self.exceptions: List[UltrasoundRAGException] = []
        self.success_count = 0
        self.total_count = 0
    
    def add_exception(self, exception: UltrasoundRAGException):
        """添加异常"""
        self.exceptions.append(exception)
        self.total_count += 1
    
    def add_success(self):
        """添加成功记录"""
        self.success_count += 1
        self.total_count += 1
    
    def has_exceptions(self) -> bool:
        """是否有异常"""
        return len(self.exceptions) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        return {
            "total_count": self.total_count,
            "success_count": self.success_count,
            "exception_count": len(self.exceptions),
            "success_rate": self.success_count / self.total_count if self.total_count > 0 else 0,
            "exceptions": [exc.to_dict() for exc in self.exceptions]
        }


# 错误恢复策略
class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    @staticmethod
    def with_retry(
        func, 
        max_retries: int = 3, 
        retry_delay: float = 1.0,
        retry_on: tuple = (DatabaseError, RetrievalError)
    ):
        """
        重试策略
        
        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            retry_on: 需要重试的异常类型
        """
        for attempt in range(max_retries + 1):
            try:
                return func()
            except retry_on as e:
                if attempt == max_retries:
                    raise e
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
    @staticmethod
    def with_fallback(primary_func, fallback_func, fallback_on: tuple = (Exception,)):
        """
        降级策略
        
        Args:
            primary_func: 主要函数
            fallback_func: 降级函数
            fallback_on: 需要降级的异常类型
        """
        try:
            return primary_func()
        except fallback_on:
            return fallback_func()
