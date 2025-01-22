import logging

# ANSI escape codes for colors
COLOR_MAP = {
    "DEBUG": "\033[96m",  # Cyan
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


# Custom log formatter function
class ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds color to log messages based on their severity level.

    This class inherits from the `logging.Formatter` class and overrides the `format` method 
    to add ANSI escape codes for color formatting to the log messages. The color of the log 
    message is determined by the severity level of the log (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Colors are applied as follows:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Magenta

    Example:
        log_message = ColorFormatter().format(record)
    """
    def format(self, record):
        log_color = COLOR_MAP.get(record.levelname, COLOR_MAP["RESET"])
        log_message = super().format(record)
        return f"{log_color}{log_message}{COLOR_MAP['RESET']}"


# Function to configure logging
def setup_logger():
    """
    Configures the logger with colorized output and a custom log format.

    This function sets up the logging configuration to display log messages with different colors 
    based on their severity level. It also sets the default log level to `INFO` and defines a 
    custom format for log messages. The log messages are displayed on the console with a 
    colored output using ANSI escape codes.

    Steps:
        1. Set the log level to `INFO`.
        2. Set the log message format to show the log level and message.
        3. Add a `StreamHandler` to print log messages to the console.
        4. Apply a `ColorFormatter` to the log handler to colorize the log messages.

    Returns:
        logger: The configured logger instance.

    Example:
        logger = setup_logger()
        logger.info("This is an info message.")
        logger.error("This is an error message.")
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Apply colored formatter to all handlers
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))

    return logger
