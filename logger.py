import logging

# ANSI escape codes for colors
COLOR_MAP = {
    'DEBUG': '\033[96m',    # Cyan
    'INFO': '\033[92m',     # Green
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m', # Magenta
    'RESET': '\033[0m'      # Reset
}

# Custom log formatter function
class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLOR_MAP.get(record.levelname, COLOR_MAP['RESET'])
        log_message = super().format(record)
        return f"{log_color}{log_message}{COLOR_MAP['RESET']}"

# Function to configure logging
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Apply colored formatter to all handlers
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))

    return logger
