import sys
from pathlib import Path

from loguru import logger


def configure_logging():
    logger.configure(
        handlers=[
            dict(
                sink=sys.stderr,
                level="INFO",
                format="[{time:hh:mm:ss.SSS}] {message}",
                colorize=True,
            )
        ]
    )
    logger.enable("vampires_dpp")
    return logger


def add_logfile(outdir: Path, logger):
    logfile = outdir / "debug.log"
    logfile.unlink(missing_ok=True)
    logger.add(logfile, level="DEBUG", colorize=False, enqueue=True)
    return logger
