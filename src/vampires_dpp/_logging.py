from pathlib import Path

import tqdm
from loguru import logger

# File sink format includes level + module:line for debugging
_FILE_FMT = "{time:HH:mm:ss.SSS} | {level:<8} | {name}:{line} - {message}"
# Stderr format is compact; loguru colorizes by level automatically
_STDERR_FMT = "<dim>[{time:HH:mm:ss}]</dim> <level>{message}</level>"


def _tqdm_sink(message):
    """Write log records via tqdm.write so active progress bars are not corrupted."""
    tqdm.tqdm.write(message, end="")


def configure_logging(level: str = "INFO") -> logger:
    """Configure the main-process logger (stderr + optional level)."""
    logger.configure(
        handlers=[{"sink": _tqdm_sink, "level": level, "format": _STDERR_FMT, "colorize": True}]
    )
    logger.enable("vampires_dpp")
    return logger


def configure_subprocess_logging(workdir: Path) -> logger:
    """Configure logging for multiprocessing workers.

    Workers are silent on stderr — the main process owns stderr and the
    tqdm progress bars. All worker output goes to the shared log file only.
    """
    logfile = workdir / "debug.log"
    logger.configure(handlers=[])
    logger.add(logfile, level="DEBUG", colorize=False, enqueue=True, format=_FILE_FMT)
    logger.enable("vampires_dpp")
    return logger


def add_logfile(outdir: Path, logger) -> logger:
    """Add a debug-level file sink to the main-process logger.

    Call once from the main process before spawning workers. The file is
    cleared on each new run so stale output from a previous run is not mixed in.
    """
    logfile = outdir / "debug.log"
    logfile.unlink(missing_ok=True)
    logger.add(logfile, level="DEBUG", colorize=False, enqueue=True, format=_FILE_FMT)
    return logger
