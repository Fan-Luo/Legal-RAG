import logging
import sys
from typing import Optional

def get_logger(
    name: str,
    level: Optional[int] = None,
    log_to_file: Optional[str] = None,
) -> logging.Logger:
    """
    Args:
        name: Logger 名称
        level: 日志等级，默认从环境或 INFO
        log_to_file: 如果提供，日志将输出到文件

    Returns:
        logging.Logger 对象
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if not logger.handlers:
        # 设置日志等级
        log_level = level or logging.INFO
        logger.setLevel(log_level)

        # 控制台 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件 Handler 
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 避免日志传播到根 logger, 防止重复打印
        logger.propagate = False

    return logger
