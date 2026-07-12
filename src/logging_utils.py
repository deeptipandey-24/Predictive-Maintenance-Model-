
import logging
import json
from datetime import datetime, timezone

def setup_logger(log_path: str):
    logger = logging.getLogger('predictive_maintenance')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger

def log_prediction(logger, machine_id, status, raw_input, result=None, error=None):
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'machine_id': machine_id,
        'status': status,
        'raw_input': raw_input
    }
    if result:
        entry.update(result)
    if error:
        entry['error'] = error
    logger.info(json.dumps(entry))
