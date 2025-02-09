"""Shared utility functions and definitions.
"""

from datetime import datetime
import logging
import numpy as np
import os
from pathlib import Path

import subprocess


DATE_FORMAT: str = '%Y_%m_%d_%H-%M-%S'
START_TIME = datetime.now().strftime(DATE_FORMAT) 
DATA_DIR = Path(os.path.abspath(__file__)).parent /'data'
OPENAI_SECRET_FILE = Path(os.path.abspath(__file__)).parent / 'SECRET_openai.txt'
LOG_DIR = Path('LOGS')
OUTPUT_DIR = Path('OUTPUTS')


def get_git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def setup_logging(logfile_prefix: str | Path, level=logging.INFO) -> logging.Logger:
    """ Setup logging at some prefix `logfile_prefix` with a timestamp and log extension 
    suffixed to it.
    """
    logger = logging.getLogger(logfile_prefix)
    logger.setLevel(level)
    
    logfile = LOG_DIR / logfile_prefix / f'{START_TIME}.log'
    logfile.parent.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(logfile)
    logger.addHandler(file_handler)
    
    logger.info(f'Script started at {START_TIME}')    
    logger.info(f'Current git hash: {get_git_hash()}')
    return logger

def get_openai_secret() -> str:
    """Read the OpenAI secret key from a file.
    """
    with open(OPENAI_SECRET_FILE, 'r') as f:
        return f.read().strip()

def log_time(logger, text: str=""):
    """Log the current time"""
    logger.info(f'{text} : {datetime.now().strftime(DATE_FORMAT)}')
