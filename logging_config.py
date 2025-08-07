import logging
import sys

def setup_logging():
    """
    Configures the root logger to output to both a file and the console
    using UTF-8 encoding to prevent Unicode errors.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a handler to write logs to a file using UTF-8
    ### CHANGED: Added encoding='utf-8' ###
    file_handler = logging.FileHandler('argopulse_ai.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Create a handler to write logs to the console using UTF-8
    ### CHANGED: Added encoding='utf-8' ###
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)