from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_FILE_FMT = "{time:HH:mm:ss.SSS} | {level:<8} | {name}:{line} - {message}"

# Shared console — all loguru output and Progress instances must use this object
# so the Live display correctly absorbs log messages while progress is rendering.
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
    console.print(
        f"[dim]\\[{time_str}][/dim] [{style}]{level:<8}[/{style}] {escape(record['message'])}",
        highlight=False,
    )


def make_progress(**kwargs) -> Progress:
    """Full progress bar with spinner, bar, count, and ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


def make_worker_progress(**kwargs) -> Progress:
    """Minimal spinner + description for per-worker stage lines."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        **kwargs,
    )


def configure_logging(level: str = "INFO") -> logger:
    """Configure the main-process logger (stderr via rich console)."""
    logger.configure(
        handlers=[{"sink": _rich_sink, "level": level, "format": "{message}", "colorize": True}]
    )
    logger.enable("vampires_dpp")
    return logger


def configure_subprocess_logging(workdir: Path) -> logger:
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
