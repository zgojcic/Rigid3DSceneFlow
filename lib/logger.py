import os
import sys
import numpy as np
import logging
import coloredlogs


_logger = logging.getLogger()


def print_info(config, log_dir=None):
    """ Logs source code configuration
        Code adapted from RPMNet repository: https://github.com/yewzijian/RPMNet/
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Arguments
    arg_str = []

    for k_id, k_val in config.items():
        if isinstance(k_val, dict):
            for key in k_val:
                arg_str.append("{}_{}: {}".format(k_id, key, k_val[key]))
        else:
            arg_str.append("{}: {}".format(k_id, k_val))

    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def prepare_logger(config, log_path = None):
    """Creates logging directory, and installs colorlogs 
    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace
    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    Code borrowed from RPMNet repository: https://github.com/yewzijian/RPMNet/
    """
    
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/console_output.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    print_info(config, log_path)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger, log_path
