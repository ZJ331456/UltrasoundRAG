import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    """
    设置日志器，根据 logger.yaml 配置初始化

    Args:
        name (str): 日志器名称（通常为模块名）

    Returns:
        logging.Logger: 配置好的日志器
    """
    # 获取项目根目录
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(os.path.dirname(current_dir))
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # 从 utils/ 回到项目根目录
    logger_config_path = project_root / "config" / "logger.yaml"
    
    # 确保日志目录存在
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 加载 YAML 配置
    config_path = os.path.join(project_root, "config", "logger.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        # 回退到基本配置
        print(f"Failed to load logger config: {str(e)}")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, "rag_engine.log"), encoding="utf-8")
            ]
        )

    return logging.getLogger(name)