import sys

from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="INFO")
