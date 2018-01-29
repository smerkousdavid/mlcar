# -*- coding: utf-8 -*-
"""mlcar configuration files

Developed by: David Smerkous
"""

from json import loads, dumps
from os.path import dirname

from logger import Logger

log = Logger("config")

config_file = dirname(__file__) + "/configs/config.json"
configs = {}
"""Global module level definitions
logger: log - The module log object so that printed calls can be backtraced to this file
str: CONFIG_FILE - The relative path and filename of the configs json
"""


def linear_map(i, i_min, i_max, o_min, o_max):
    return (i - i_min) * (o_max - o_min) / (i_max - i_min) + o_min


def dump_configs(conf):
    """Json dumps wrapper to printy print dicts to the console

    Arguments:
        configs (dict): The dictionary to log with indendation

    Note:
        This will only print in the CONFIGS namespace
    """

    try:
        log.info("Configurations:\n%s" % dumps(conf, indent=4))
    except Exception as err:
        log.error("Failed to dump json! (err: %s)" % str(err))


def load_configs():
    """Load the configuration file

        Returns: (bool)
            True on success or False on failure to load configurations
    """
    global configs

    try:
        c_f = open(config_file, 'r')  # Open the config file for reading
        c_f_data = c_f.read()
        configs = loads(c_f_data)
        c_f.close()

        # Log the new configurations
        log.info("Dumping configs")
        dump_configs(configs)
        return True
    except Exception as err:
        log.error("Failed to load configuration file! (err: %s)" % str(err))
        if c_f is not None:
            c_f.close()
    return configs


def save_configs(new_configs):
    """Savethe configuration file

        Returns: (bool)
            True on success or False on failure to save the configurations
    """
    global configs

    try:
        c_f = open(config_file, 'w')  # Open the config file for writing
        c_f.write(dumps(new_configs, indent=2))
        c_f.close()
        return True
    except Exception as err:
        log.error("Failed to save the configuration file! (err: %s)" % str(err))
        if c_f is not None:
            c_f.close()

    return configs


def get_configs():
    global configs
    return configs