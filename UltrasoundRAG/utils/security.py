"""
UltrasoundRAG API安全模块
提供认证、授权、限流和文件验证功能
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import mimetypes
import os
from PIL import Image
import io

from .exceptions import (
    AuthenticationError, 
    AuthorizationError, 
    RateLimitError, 
    ValidationError,
    ErrorCode
)
from fastapi import Request


class UserRole(Enum):
    """用户角色枚举"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class APIKey:
    """API密钥信息"""
    key_id: str
    key_secret: str
    user_id: str
    role: UserRole
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used_at: Optional[float] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # 每小时请求限制


class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self._init_default_keys()
    
    def _init_default_keys(self):
        """初始化默认API密钥"""
        # 管理员密钥
        admin_key = self.generate_api_key(
            user_id="admin",
            role=UserRole.ADMIN,
            permissions={"read", "write", "admin"},
            rate_limit=1000
        )
        
        # 用户密钥
        user_key = self.generate_api_key(
            user_id="user", 
            role=UserRole.USER,
            permissions={"read"},
            rate_limit=100
        )
        
        # 演示密钥（与现有API兼容）
        demo_key = APIKey(
            key_id="rag_demo_key",
            key_secret="demo_secret",
            user_id="demo_user",
            role=UserRole.USER,
            permissions={"read"},
            rate_limit=50
        )
        self.api_keys["rag_demo_key"] = demo_key
    
    def generate_api_key(
        self, 
        user_id: str, 
        role: UserRole,
        permissions: Set[str] = None,
        rate_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """生成新的API密钥"""
        key_id = f"rag_{secrets.token_hex(8)}"
        key_secret = secrets.token_hex(32)
        
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 24 * 3600)
        
        api_key = APIKey(
            key_id=key_id,
            key_secret=key_secret,
            user_id=user_id,
            role=role,
            permissions=permissions or set(),
            rate_limit=rate_limit,
            expires_at=expires_at
        )
        
        self.api_keys[key_id] = api_key
        return api_key
    
    def validate_api_key(self, key_id: str, key_secret: Optional[str] = None) -> APIKey:
        """验证API密钥"""
        if key_id not in self.api_keys:
            raise AuthenticationError("无效的API密钥")
        
        api_key = self.api_keys[key_id]
        
        # 检查密钥是否激活
        if not api_key.is_active:
            raise AuthenticationError("API密钥已被禁用")
        
        # 检查是否过期
        if api_key.expires_at and time.time() > api_key.expires_at:
            raise AuthenticationError("API密钥已过期")
        
        # 如果提供了secret，验证签名
        if key_secret and api_key.key_secret != key_secret:
            raise AuthenticationError("API密钥验证失败")
        
        # 更新使用信息
        api_key.last_used_at = time.time()
        api_key.usage_count += 1
        
        return api_key
    
    def revoke_api_key(self, key_id: str):
        """吊销API密钥"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False


class RateLimiter:
    """频率限制器"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = 3600  # 1小时清理一次
        self.last_cleanup = time.time()
    
    def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int = 3600
    ) -> bool:
        """
        检查频率限制
        
        Args:
            key: 限制键（通常是API key或IP）
            limit: 限制数量
            window_seconds: 时间窗口（秒）
        
        Returns:
            是否在限制范围内
        
        Raises:
            RateLimitError: 超出频率限制
        """
        current_time = time.time()
        
        # 清理过期记录
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time - window_seconds)
            self.last_cleanup = current_time
        
        # 获取时间窗口内的请求
        window_start = current_time - window_seconds
        recent_requests = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # 检查是否超出限制
        if len(recent_requests) >= limit:
            retry_after = int(recent_requests[0] + window_seconds - current_time) + 1
            raise RateLimitError(
                f"频率限制：{limit}次/{window_seconds}秒",
                retry_after=retry_after
            )
        
        # 记录本次请求
        self.requests[key] = recent_requests + [current_time]
        return True
    
    def _cleanup_old_requests(self, cutoff_time: float):
        """清理过期的请求记录"""
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > cutoff_time
            ]
            if not self.requests[key]:
                del self.requests[key]


class FileValidator:
    """文件验证器"""
    
    # 支持的文件类型
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp',
        'image/webp', 'image/tiff', 'image/svg+xml'
    }
    
    ALLOWED_DOCUMENT_TYPES = {
        'text/plain', 'text/markdown', 'application/pdf',
        'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    # 文件大小限制（字节）
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    
    # 图像尺寸限制
    MAX_IMAGE_DIMENSION = 8192  # 8K分辨率
    
    @classmethod
    def validate_image_file(
        cls, 
        file_content: bytes, 
        filename: str,
        check_content: bool = True
    ) -> Dict[str, any]:
        """
        验证图像文件
        
        Args:
            file_content: 文件内容
            filename: 文件名
            check_content: 是否检查文件内容
        
        Returns:
            验证结果字典
        
        Raises:
            ValidationError: 文件验证失败
        """
        result = {
            "is_valid": False,
            "file_type": None,
            "size": len(file_content),
            "dimensions": None,
            "format": None
        }
        
        # 检查文件大小
        if len(file_content) > cls.MAX_IMAGE_SIZE:
            raise ValidationError(
                f"图像文件过大，最大允许 {cls.MAX_IMAGE_SIZE // (1024*1024)}MB",
                ErrorCode.INVALID_INPUT,
                details={"max_size": cls.MAX_IMAGE_SIZE, "actual_size": len(file_content)}
            )
        
        # 检查文件扩展名
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}:
            raise ValidationError(
                f"不支持的图像格式: {file_ext}",
                ErrorCode.INVALID_INPUT
            )
        
        # 检查MIME类型
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type not in cls.ALLOWED_IMAGE_TYPES:
            raise ValidationError(
                f"不支持的MIME类型: {mime_type}",
                ErrorCode.INVALID_INPUT
            )
        
        result["file_type"] = mime_type
        
        # 检查文件内容
        if check_content:
            try:
                with Image.open(io.BytesIO(file_content)) as img:
                    # 检查图像尺寸
                    width, height = img.size
                    if width > cls.MAX_IMAGE_DIMENSION or height > cls.MAX_IMAGE_DIMENSION:
                        raise ValidationError(
                            f"图像尺寸过大，最大允许 {cls.MAX_IMAGE_DIMENSION}x{cls.MAX_IMAGE_DIMENSION}",
                            ErrorCode.INVALID_INPUT,
                            details={"max_dimension": cls.MAX_IMAGE_DIMENSION, "actual_size": (width, height)}
                        )
                    
                    result["dimensions"] = (width, height)
                    result["format"] = img.format
                    
            except Exception as e:
                raise ValidationError(
                    f"无法解析图像文件: {str(e)}",
                    ErrorCode.INVALID_INPUT,
                    original_exception=e
                )
        
        result["is_valid"] = True
        return result
    
    @classmethod
    def validate_document_file(
        cls, 
        file_content: bytes, 
        filename: str
    ) -> Dict[str, any]:
        """
        验证文档文件
        
        Args:
            file_content: 文件内容
            filename: 文件名
        
        Returns:
            验证结果字典
        """
        result = {
            "is_valid": False,
            "file_type": None,
            "size": len(file_content),
            "encoding": None
        }
        
        # 检查文件大小
        if len(file_content) > cls.MAX_DOCUMENT_SIZE:
            raise ValidationError(
                f"文档文件过大，最大允许 {cls.MAX_DOCUMENT_SIZE // (1024*1024)}MB",
                ErrorCode.INVALID_INPUT
            )
        
        # 检查MIME类型
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type not in cls.ALLOWED_DOCUMENT_TYPES:
            raise ValidationError(
                f"不支持的文档类型: {mime_type}",
                ErrorCode.INVALID_INPUT
            )
        
        result["file_type"] = mime_type
        result["is_valid"] = True
        return result
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """清理文件名，移除危险字符"""
        # 移除路径分隔符和特殊字符
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*', '\0']
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # 限制文件名长度
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized


class SecurityManager:
    """安全管理器 - 统一安全功能入口"""
    
    def __init__(self):
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
        self.file_validator = FileValidator()
    
    def authenticate_request(
        self, 
        auth_header: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> APIKey:
        """
        认证请求
        
        Args:
            auth_header: Authorization头部 (Bearer token)
            api_key: 直接的API密钥
        
        Returns:
            API密钥信息
        """
        # 从Authorization头部提取token
        if auth_header:
            if not auth_header.startswith("Bearer "):
                raise AuthenticationError("无效的认证头部格式")
            api_key = auth_header[7:]  # 移除"Bearer "前缀
        
        if not api_key:
            raise AuthenticationError("缺少API密钥")
        
        return self.api_key_manager.validate_api_key(api_key)
    
    def check_permission(
        self, 
        api_key_info: APIKey, 
        required_permission: str
    ):
        """检查权限"""
        if required_permission not in api_key_info.permissions:
            raise AuthorizationError(
                f"权限不足，需要权限: {required_permission}",
                details={"required": required_permission, "available": list(api_key_info.permissions)}
            )
    
    def enforce_rate_limit(
        self, 
        api_key_info: APIKey,
        client_ip: Optional[str] = None
    ):
        """强制执行频率限制"""
        # API key级别的限制
        if api_key_info.rate_limit:
            self.rate_limiter.check_rate_limit(
                f"api_key:{api_key_info.key_id}",
                api_key_info.rate_limit
            )
        
        # IP级别的限制（防止滥用）
        if client_ip:
            self.rate_limiter.check_rate_limit(
                f"ip:{client_ip}",
                200  # 每小时200次请求
            )
    
    def validate_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str = "image"
    ) -> Dict[str, any]:
        """验证上传的文件"""
        if file_type == "image":
            return self.file_validator.validate_image_file(file_content, filename)
        elif file_type == "document":
            return self.file_validator.validate_document_file(file_content, filename)
        else:
            raise ValidationError(f"不支持的文件类型: {file_type}")


# 全局安全管理器实例
security_manager = SecurityManager()


# FastAPI依赖项
def get_authenticated_user(request: Request) -> APIKey:
    """FastAPI依赖：获取认证用户（从请求头惰性提取）"""
    auth_header = request.headers.get("Authorization")
    return security_manager.authenticate_request(auth_header)


def require_permission(permission: str):
    """FastAPI依赖：要求特定权限（惰性解析请求）"""
    def dependency(request: Request) -> APIKey:
        api_key_info = get_authenticated_user(request)
        security_manager.check_permission(api_key_info, permission)
        return api_key_info
    return dependency


def enforce_rate_limits(client_ip: Optional[str] = None):
    """FastAPI依赖：强制执行频率限制（惰性解析请求）"""
    def dependency(request: Request) -> APIKey:
        api_key_info = get_authenticated_user(request)
        security_manager.enforce_rate_limit(api_key_info, client_ip)
        return api_key_info
    return dependency
