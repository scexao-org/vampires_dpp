from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.markup import escape
from tqdm.auto import tqdm

_FILE_FMT = "{time:HH:mm:ss.SSS} | {level:<8} | {name}:{line} - {message}"

# Rich console used purely to format log lines. We never let it print directly
# to stderr while a tqdm bar may be active — every rendered line is routed
# through tqdm.write so the bar redraws cleanly underneath.
console = Console(stderr=True, highlight=False)

_LEVEL_STYLES = {
    "TRACE": "dim",
    "DEBUG": "dim cyan",
    "INFO": "green",
    "SUCCESS": "bold green",
    "WARNING": "bold yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}


def _rich_sink(message):
    record = message.record
    level = record["level"].name
    style = _LEVEL_STYLES.get(level, "white")
    time_str = record["time"].strftime("%H:%M:%S")
    with console.capture() as cap:
        console.print(
            f"[dim]\\[{time_str}][/dim] [{style}]{level:<8}[/{style}] {escape(record['message'])}",
            highlight=False,
        )
    tqdm.write(cap.get(), end="")


def configurelogging_utils(level: str = "INFO") -> logger:
    """Configure the main-process logger (stderr, tqdm-safe)."""
    logger.configure(
        handlers=[{"sink": _rich_sink, "level": level, "format": "{message}", "colorize": True}]
    )
    logger.enable("vampires_dpp")
    return logger


def configure_subprocesslogging_utils(workdir: Path) -> logger:
    """Configure logging for multiprocessing workers (file only, no stderr)."""
    logfile = workdir / "debug.log"
    logger.configure(handlers=[])
    logger.add(logfile, level="DEBUG", colorize=False, enqueue=True, format=_FILE_FMT)
    logger.enable("vampires_dpp")
    return logger


def add_logfile(outdir: Path, logger) -> logger:
    """Add a debug-level file sink to the main-process logger."""
    logfile = outdir / "debug.log"
    logfile.unlink(missing_ok=True)
    logger.add(logfile, level="DEBUG", colorize=False, enqueue=True, format=_FILE_FMT)
    return logger
