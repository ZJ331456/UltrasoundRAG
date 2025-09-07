import logging
import logging.config
import yaml
import os
from pathlib import Path

_LOGGER_CONFIGURED = False

def setup_logger(name: str) -> logging.Logger:
    """
    设置日志器，根据 logger.yaml 配置初始化

    Args:
        name (str): 日志器名称（通常为模块名）

    Returns:
        logging.Logger: 配置好的日志器
    """
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # 从 utils/ 回到项目根目录
    
    # 确保日志目录存在
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 日志配置文件路径
    logger_config_path = project_root / "config" / "logger.yaml"
    
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        # 加载 YAML 配置
        try:
            if logger_config_path.exists():
                with open(logger_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 动态设置日志文件路径
                if 'handlers' in config and 'file' in config['handlers']:
                    config['handlers']['file']['filename'] = str(log_dir / "rag_engine.log")
                
                logging.config.dictConfig(config)
                print(f"成功加载日志配置: {logger_config_path}")
            else:
                print(f"日志配置文件不存在: {logger_config_path}，使用默认配置")
                raise FileNotFoundError("Logger config not found")
                
        except Exception as e:
            # 回退到基本配置
            print(f"日志配置加载失败: {str(e)}，使用默认配置")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(
                        str(log_dir / "rag_engine.log"), 
                        encoding="utf-8"
                    )
                ]
            )
        finally:
            _LOGGER_CONFIGURED = True

    return logging.getLogger(name)