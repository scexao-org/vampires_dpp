import sys

from loguru import logger

logger.configure(
    handlers=[
        dict(sink=sys.stderr, level="INFO", format="[{time:hh:mm:ss.SSS}] {message}", colorize=True)
    ]
)
logger.enable("vampires_dpp")
