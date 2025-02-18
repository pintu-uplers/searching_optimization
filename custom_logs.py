import logging

# Create a custom logger for your application
logger = logging.getLogger("custom_logger")
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler('actions.log', mode='a')
file_handler.setLevel(logging.INFO)

# Define log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Function to log actions
def log_action(function_name, message, log_level="info"):
    log_message = f"Function: {function_name} - {message}"

    if log_level == "debug":
        logger.debug(log_message)
    elif log_level == "info":
        logger.info(log_message)
    elif log_level == "warning":
        logger.warning(log_message)
    elif log_level == "error":
        logger.error(log_message)
    elif log_level == "critical":
        logger.critical(log_message)
    else:
        logger.info(log_message)  # Default to INFO if an invalid level is provided
