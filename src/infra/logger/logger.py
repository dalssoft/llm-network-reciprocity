import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import warnings
from .per_message_handler import PerMessageFileHandler


# Silence all loggers by default IMMEDIATELY at import time
# This must happen before any other module imports and creates loggers
logging.root.setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", module="pydantic")


def delete_logs(log_path):
    """Delete existing log files in the log directory.

    Args:
        log_path: Path to the log directory
    """
    if not os.path.exists(log_path):
        return

    try:
        for file in os.listdir(log_path):
            file_path = os.path.join(log_path, file)
            # Deletar arquivos .log e .log.{id}, ignorando diretórios e outros arquivos
            if os.path.isfile(file_path) and (file.endswith(".log") or ".log." in file):
                try:
                    os.remove(file_path)
                except OSError as e:
                    # Log error but don't fail - might be permission issues
                    logging.warning(f"Failed to delete log file {file_path}: {e}")
    except OSError as e:
        logging.warning(f"Failed to list log directory {log_path}: {e}")


def setup_logger(project_name, log_level=logging.DEBUG, enable_console=False):
    """Set up logging configuration for the project.

    Args:
        project_name: Name of the project (used for log file naming)
        log_level: Logging level for PROJECT loggers (default: DEBUG)
        enable_console: If True, also log to console (default: False)

    Returns:
        Tuple of (app_logger, project_logger)
    """
    # Silence everything by default - opt-in approach
    # Only our project loggers will be enabled at the specified level
    logging.root.setLevel(logging.CRITICAL)

    log_path = os.getenv("LOG_PATH", "logs")
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create log directory {log_path}: {e}")
        # Fallback to current directory
        log_path = "logs"

    app_env = os.getenv("APP_ENV", "dev")
    if app_env == "dev":
        delete_logs(log_path)

    # Generate a timestamp for the log file names
    timestamp = datetime.now().strftime("%Y%m%d")
    app_log_file = f"{log_path}/{timestamp}_app.log"
    project_log_file = f"{log_path}/{timestamp}_{project_name}.log"

    # Use RotatingFileHandler instead of FileHandler for the base config
    root_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=20,
        mode="a",
    )
    root_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.root.addHandler(root_handler)

    # Create app logger - opt-in to our log level
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    # Add handlers directly to app_logger as well
    app_logger.addHandler(root_handler)  # Add the file handler for _app.log
    app_logger.propagate = False  # Prevent duplication if root logger also logs

    # Create project-specific logger - opt-in to our log level
    project_logger = logging.getLogger(project_name)
    project_logger.setLevel(log_level)

    # Create a rotating file handler for project-specific logs
    project_file_handler = RotatingFileHandler(
        project_log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=20,
        mode="a",
    )
    project_file_handler.setLevel(log_level)
    project_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_file_handler.setFormatter(project_formatter)
    project_logger.addHandler(project_file_handler)

    # Optionally add console handler for project logs
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        project_logger.addHandler(console_handler)
        # Also add console output to the 'app' logger used by controllers
        app_logger.addHandler(console_handler)

    return app_logger, project_logger


def setup_prompt_logger(
    logger_name: str,
    log_dir: str,
    log_level=logging.DEBUG,
    filename_template: str = "prompt_{timestamp}_{counter}",
    file_extension: str = ".txt",
):
    """Set up a logger that writes each message to a separate file.

    Useful for logging LLM prompts/responses where each interaction
    should be stored in its own file.

    Args:
        logger_name: Name of the logger (e.g., "coop_net.prompts")
        log_dir: Directory where individual log files will be written
        log_level: Logging level for this logger
        filename_template: Template for filename. Can use {timestamp}, {counter}, {session_id}
        file_extension: Extension for log files (default: .txt)

    Returns:
        The configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Add per-message file handler
    per_message_handler = PerMessageFileHandler(
        log_dir=log_dir,
        filename_template=filename_template,
        file_extension=file_extension,
    )
    per_message_handler.setLevel(log_level)
    # Use a simple formatter - just the message itself
    per_message_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(per_message_handler)
    logger.propagate = False  # Don't propagate to root

    return logger


app_logger, project_logger = setup_logger("coop_net")
