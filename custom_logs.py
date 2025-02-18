import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, filename='actions.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

def log_action(function_name, message, log_level="info"):
    log_message = f"Function: {function_name} - {message}"
    
    if log_level == "debug":
        logging.debug(log_message)
    elif log_level == "info":
        logging.info(log_message)
    elif log_level == "warning":
        logging.warning(log_message)
    elif log_level == "error":
        logging.error(log_message)
    elif log_level == "critical":
        logging.critical(log_message)
    else:
        logging.info(log_message)  # Default to INFO if an invalid level is provided
