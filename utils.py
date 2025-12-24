import logging
import os

def init_log(output_dir, file_level=logging.DEBUG, console_level=logging.INFO):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')

    # Create a dedicated logger
    logger = logging.getLogger('MobileFaceNet')
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers control filtering
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(file_level)
        fh_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y%m%d-%H:%M:%S'
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    pass
