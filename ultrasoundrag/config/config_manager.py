"""
增强的配置管理系统
支持环境变量覆盖、配置验证、热更新等高级功能
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading


class ConfigEnvironment(Enum):
    """配置环境枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class ConfigValidationRule:
    """配置验证规则"""
    path: str  # 配置路径，如 'milvus.milvus_uri'
    required: bool = False
    type_check: Optional[type] = None
    allowed_values: Optional[List[Any]] = None
    validator: Optional[callable] = None
    default: Any = None


class ConfigManager:
    """增强的配置管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._config = {}
        self._config_path = None
        self._environment = ConfigEnvironment.DEVELOPMENT
        self._validation_rules = []
        self._watchers = []
        self._logger = logging.getLogger(__name__)
        self._initialized = True
        
        # 设置默认验证规则
        self._setup_default_validation_rules()
    
    def load_config(self, config_path: Optional[str] = None, 
                   environment: Optional[ConfigEnvironment] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if environment:
            self._environment = environment
        
        if config_path is None:
            config_path = self._find_config_file()
        
        self._config_path = config_path
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载基础配置
        with open(config_path, 'r', encoding='utf-8') as file:
            base_config = yaml.safe_load(file)
        
        # 应用环境覆盖
        config_with_env = self._apply_environment_overrides(base_config)
        
        # 应用环境变量覆盖
        final_config = self._apply_env_var_overrides(config_with_env)
        
        # 验证配置
        self._validate_config(final_config)
        
        # 解析路径
        final_config = self._resolve_paths(final_config)
        
        self._config = final_config
        self._logger.info(f"配置加载成功: {config_path} (环境: {self._environment.value})")
        
        return self._config
    
    def get(self, path: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径访问"""
        keys = path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> None:
        """设置配置值，支持点号路径"""
        keys = path.split('.')
        target = self._config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # 设置值
        target[keys[-1]] = value
        
        # 通知观察者
        self._notify_watchers(path, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新配置"""
        for path, value in updates.items():
            self.set(path, value)
    
    def reload(self) -> Dict[str, Any]:
        """重新加载配置"""
        if self._config_path:
            return self.load_config(self._config_path, self._environment)
        else:
            raise RuntimeError("没有配置文件路径，无法重新加载")
    
    def watch(self, callback: callable) -> None:
        """注册配置变更监听器"""
        self._watchers.append(callback)
    
    def add_validation_rule(self, rule: ConfigValidationRule) -> None:
        """添加配置验证规则"""
        self._validation_rules.append(rule)
    
    def validate(self) -> List[str]:
        """验证当前配置，返回错误列表"""
        return self._validate_config(self._config)
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'environment': self._environment.value,
            'config_path': self._config_path,
            'config_size': len(str(self._config)),
            'validation_rules': len(self._validation_rules),
            'watchers': len(self._watchers)
        }
    
    def export_config(self, output_path: str, include_sensitive: bool = False) -> None:
        """导出配置到文件"""
        config_to_export = self._config.copy()
        
        if not include_sensitive:
            config_to_export = self._remove_sensitive_data(config_to_export)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_export, f, default_flow_style=False, allow_unicode=True)
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config
    
    @property
    def environment(self) -> ConfigEnvironment:
        """获取当前环境"""
        return self._environment
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        # 优先级：环境变量 > 当前目录 > 默认位置
        env_path = os.getenv('ULTRASOUNDRAG_CONFIG')
        if env_path and os.path.exists(env_path):
            return env_path
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "config.yaml")
        
        if os.path.exists(config_file):
            return config_file
        
        raise FileNotFoundError("找不到配置文件")
    
    def _apply_environment_overrides(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境特定的配置覆盖"""
        env_config_path = os.path.join(
            os.path.dirname(self._config_path),
            f"config.{self._environment.value}.yaml"
        )
        
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_overrides = yaml.safe_load(f)
            
            return self._deep_merge(base_config, env_overrides)
        
        return base_config
    
    def _apply_env_var_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # 支持的环境变量格式: ULTRASOUNDRAG_MILVUS_URI, ULTRASOUNDRAG_EMBEDDING_PROVIDER 等
        prefix = "ULTRASOUNDRAG_"
        
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(prefix):
                continue
            
            # 转换环境变量名为配置路径
            config_path = env_name[len(prefix):].lower().replace('_', '.')
            
            # 尝试转换类型
            converted_value = self._convert_env_value(env_value)
            
            # 设置配置值
            self._set_nested_value(config, config_path, converted_value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值的类型"""
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 字符串
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """设置嵌套配置值"""
        keys = path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """验证配置"""
        errors = []
        
        for rule in self._validation_rules:
            try:
                value = self._get_nested_value(config, rule.path)
                
                # 检查必需字段
                if rule.required and value is None:
                    errors.append(f"必需的配置项缺失: {rule.path}")
                    continue
                
                # 如果值为None且有默认值，设置默认值
                if value is None and rule.default is not None:
                    self._set_nested_value(config, rule.path, rule.default)
                    continue
                
                # 类型检查
                if rule.type_check and value is not None and not isinstance(value, rule.type_check):
                    errors.append(f"配置项类型错误 {rule.path}: 期望 {rule.type_check.__name__}, 得到 {type(value).__name__}")
                
                # 允许值检查
                if rule.allowed_values and value not in rule.allowed_values:
                    errors.append(f"配置项值无效 {rule.path}: {value}, 允许的值: {rule.allowed_values}")
                
                # 自定义验证器
                if rule.validator and value is not None:
                    try:
                        if not rule.validator(value):
                            errors.append(f"配置项验证失败 {rule.path}: {value}")
                    except Exception as e:
                        errors.append(f"配置项验证异常 {rule.path}: {e}")
            
            except Exception as e:
                errors.append(f"验证配置项时出错 {rule.path}: {e}")
        
        if errors:
            for error in errors:
                self._logger.error(error)
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def _setup_default_validation_rules(self) -> None:
        """设置默认验证规则"""
        rules = [
            ConfigValidationRule("milvus.milvus_uri", required=True, type_check=str),
            ConfigValidationRule("milvus.db_name", required=True, type_check=str),
            ConfigValidationRule("embedding.provider", required=True, type_check=str),
            ConfigValidationRule("indexing.markdown_parse.max_chunk_size", type_check=int, default=1500),
            ConfigValidationRule("indexing.markdown_parse.overlap_ratio", type_check=float, default=0.15),
            ConfigValidationRule("indexing.image_parse.model_path", required=True, type_check=str),
        ]
        
        self._validation_rules.extend(rules)
    
    def _detect_project_root(self) -> Path:
        """智能检测项目根目录"""
        # 1. 优先使用环境变量
        env_root = os.environ.get('ULTRASOUNDRAG_PROJECT_ROOT')
        if env_root and os.path.exists(env_root):
            return Path(env_root)
        
        # 2. 从当前配置文件位置开始向上查找
        current_path = Path(__file__).resolve().parent
        project_indicators = [
            'requirements.txt',
            'setup.py', 
            'pyproject.toml',
            '.git',
            'UltrasoundRAG',  # 项目特有目录
            'README.md'
        ]
        
        # 向上查找直到找到项目根目录标识
        for parent in [current_path] + list(current_path.parents):
            for indicator in project_indicators:
                if (parent / indicator).exists():
                    self._logger.info(f"检测到项目根目录: {parent} (标识: {indicator})")
                    return parent
        
        # 3. 回退到当前工作目录
        cwd = Path(os.getcwd())
        self._logger.warning(f"无法自动检测项目根目录，使用当前工作目录: {cwd}")
        return cwd

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将配置中的相对路径转换为绝对路径"""
        # 获取项目根目录
        raw_project_root = config.get('paths', {}).get('project_root', 'auto')
        
        if raw_project_root == 'auto':
            project_root = self._detect_project_root()
        elif os.path.isabs(raw_project_root):
            project_root = Path(raw_project_root)
        else:
            # 相对路径相对于配置文件所在目录
            config_dir = Path(self._config_path).parent if self._config_path else Path.cwd()
            project_root = (config_dir / raw_project_root).resolve()
        
        # 更新配置中的项目根目录为实际检测到的路径
        if 'paths' not in config:
            config['paths'] = {}
        config['paths']['project_root'] = str(project_root)
        
        def resolve_path_in_dict(obj, current_path=""):
            if isinstance(obj, dict):
                resolved = {}
                for key, value in obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    if key.endswith('_path') or key.endswith('_dir') or key == 'base_path':
                        if isinstance(value, str) and value and not os.path.isabs(value):
                            resolved[key] = str(project_root / value)
                        else:
                            resolved[key] = value
                    else:
                        resolved[key] = resolve_path_in_dict(value, new_path)
                return resolved
            elif isinstance(obj, list):
                return [resolve_path_in_dict(item, current_path) for item in obj]
            else:
                return obj
        
        return resolve_path_in_dict(config)
    
    def _remove_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """移除敏感数据"""
        sensitive_keys = ['token', 'password', 'secret', 'key', 'api_key']
        
        def remove_sensitive(obj):
            if isinstance(obj, dict):
                return {
                    k: "***HIDDEN***" if any(sens in k.lower() for sens in sensitive_keys) else remove_sensitive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_sensitive(item) for item in obj]
            else:
                return obj
        
        return remove_sensitive(config)
    
    def _notify_watchers(self, path: str, value: Any) -> None:
        """通知配置变更监听器"""
        for watcher in self._watchers:
            try:
                watcher(path, value)
            except Exception as e:
                self._logger.error(f"配置监听器执行失败: {e}")


# 创建全局配置管理器实例
enhanced_config_manager = ConfigManager()


# 便捷函数
def get_config() -> Dict[str, Any]:
    """获取配置"""
    if not enhanced_config_manager.config:
        enhanced_config_manager.load_config()
    return enhanced_config_manager.config


def get_config_value(path: str, default: Any = None) -> Any:
    """获取配置值"""
    if not enhanced_config_manager.config:
        enhanced_config_manager.load_config()
    return enhanced_config_manager.get(path, default)


def set_config_value(path: str, value: Any) -> None:
    """设置配置值"""
    enhanced_config_manager.set(path, value)


def reload_config() -> Dict[str, Any]:
    """重新加载配置"""
    return enhanced_config_manager.reload()


def validate_config() -> List[str]:
    """验证配置"""
    return enhanced_config_manager.validate()


def set_environment(env: Union[str, ConfigEnvironment]) -> None:
    """设置环境"""
    if isinstance(env, str):
        env = ConfigEnvironment(env)
    enhanced_config_manager._environment = env
