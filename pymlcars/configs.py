# -*- coding: utf-8 -*-
"""RemSphinx speech to text processing configuration files

This module is designed to load the configuration files
for the dynamic use of languages on the server. Please look at the config.json file
to see all of the application configurations.

Developed by: David Smerkous
"""

from json import loads, dumps
from sys import exit
from os.path import dirname, realpath, join, exists
from os import sep
from logger import logger
from threading import Thread

import re
import pyinotify

log = logger("CONFIGS")

CONFIG_FILE = "configs/config.json"
CONFIGS = {}
"""Global module level definitions
logger: log - The module log object so that printed calls can be backtraced to this file
str: CONFIG_FILE - The relative path and filename of the configs json
dict: CONFIGS - The global configurations for all other modules
"""

class Configs(object):
    """Configs handler and dynamic file change detection

    Attributes:
        _current_dir (str): The current global working directory 
        _json_config (str): The path to the json config file
    """

    class __ConfigFileEventHandler(pyinotify.ProcessEvent):
        """Config file change handler

        Attributes:
            _config_file (str): The full path of the configuration file to watch for
            _config_reload (obj: method): The method reference to reload the configuration file

        Note:
            This is a nested class for the sole purpose that the other
            Classes do not need access to this ConfigEvent listener
        """
        def __init__(self, config_file: str, config_reload: object):
            self._config_file = config_file
            self._config_reload = config_reload

        def process_IN_CLOSE_WRITE(self, event):
            """Pyinotify's method of handling file modification

            Note:
                This is called in the backend of pyinotify and can be called on ANY file
                within the selected folder. This is just the front end handler.
            """
            if self._config_file in event.pathname: # Make sure we are only checking for the loaded configuration file and not some other file
                log.info("The config file %s has been modified!" % event.pathname)
                log.info("Reloading configurations!")
                self._config_reload() # Reload the configuration files
                log.info("Reloading complete!")

    def __init__(self):
        self._current_dir = dirname(realpath(__file__)) 
        self._json_config = self.get_full_path(CONFIG_FILE)
        if not self.__load_configs(): # Attempt to read from the configuration file
            exit(0) # Exit the program on the first configuration error
        log.info("Succesfully loaded initial configs from %s" % self._json_config)

        # Create and attach the inotify file watch for the configuration files
        self._wm = pyinotify.WatchManager()
        self._handler = Configs.__ConfigFileEventHandler(self._json_config, self.__load_configs)
        self._notifier = pyinotify.Notifier(self._wm, self._handler)
        self._wdd = self._wm.add_watch(dirname(self._json_config), pyinotify.IN_CLOSE_WRITE)

        # Create and start the config file event loop
        self._event_thread = Thread(target=self._notifier.loop)
        self._event_thread.setName("ConfigFileEventLoop")
        self._event_thread.setDaemon(True)
        self._event_thread.start()

    @staticmethod
    def get_available_languages():
        global CONFIGS
        """Method to return all language codes from the configuration file

        Returns: (:obj: dict - string pairs)
            Pairs of language names and id's
        """
        try:
            return CONFIGS["language_codes"]
        except Exception as err:
            log.error("Failed loading available languages! (err: %s)" % str(err))
            return None

    @staticmethod
    def get_language_name_by_id(l_id):
        """Method to return the language model name based on the id

        Arguments:
            l_id (int): The language model id to get the language model name

        Returns (str):
            The language model name
        """

        try:
            a_l = Configs.get_available_languages()
            if a_l is None:
                return None

            for l in a_l:
                if l["id"] == l_id:
                    return l["name"]
        except Exception as err:
            log.error("Failed getting language name by id! (id: %d) (err: %s)" % (l_id, str(err)))
            return None

    @staticmethod
    def get_language_accents_by_id(l_id):
        """Method to return the language model name based on the id

        Arguments:
            l_id (int): The language model id to get the language model name

        Returns (str):
            The language model name
        """
        pass

    @staticmethod
    def get_server():
        global CONFIGS
        """Method to return all of the server configurations from the configuration file

        Returns: (:obj: dict - server configuration)
            The server configuration dictionary
        """

        try:
            return CONFIGS["server"]
        except Exception as err:
            log.error("Failed getting server configuration dictionary! (err: %s)" % str(err))
            return None

    def get_ssl(self):
        """Method to return all of the server ssl configurations from the configuration file

        Returns: (:obj: dict - server configuration)
            The server ssl configuration dictionary
        """

        try:
            server_configs = Configs.get_server()
            if server_configs is None:
                raise TypeError("Server configurations are not available!") 
            ssl_configs = server_configs["ssl"]
            ssl_configs["certfile"] = self.parse_config_path(ssl_configs["certfile"])
            ssl_configs["keyfile"] = self.parse_config_path(ssl_configs["keyfile"])
            return ssl_configs
        except Exception as err:
            log.error("Failed getting server configuration dictionary! (err: %s)" % str(err))
            return None

    @staticmethod
    def get_nltk():
        global CONFIGS
        """Public methdo to get the current nltk configurations

        Returns: (dict)
            The nltk configuration object
        """
        return CONFIGS["nltk"]

    def get_nltk_data(self, l_id):
        """Method to return all text processing configuration data

        Arguments:
            l_id (int): The language model id to get the text processing data from

        Returns (NLTKModel):
            The populated NLTKModel
        """
        try:
            n_id = str(l_id) # Turn the id into a str because the json only accepts str keys
            name = Configs.get_language_name_by_id(l_id)
            nltk = Configs.get_nltk() # Get the nltk sub object
            stop_words = nltk["stopwords"][n_id]
            return NLTKModel(name, stop_words) # Create the new nltk model object
        except Exception as err:
            log.error("Failed loading nltk model! (id: %s) (err: %s)" % (str(l_id), str(err)))
            return None

    @staticmethod
    def get_stt():
        global CONFIGS
        """Public method to get the current stt configurations

        Returns: (dict)
            The STT configuration object
        """
        return CONFIGS["stt"]

    def get_stt_data(self, l_id, accent):
        """Method to return all speech to text configuration data

        Arguments:
            l_id (int): The language model id to get speech to text data from
        
        Returns (LanguageModel):
            The populated LanguageModel
        """

        try:
            n_id = str(l_id) # Turn the id into a str because the json only accepts str keys
            name = Configs.get_language_name_by_id(l_id)
            stt = Configs.get_stt() # Get the speech to text sub object
            model_data = self.parse_config_path(stt["model_dir"]) # Parse the model directory from the configuration file
            # Get the current language's model data
            m_hmm = join(model_data, self.get_accent_path(stt["hmm"][n_id], accent))
            m_lm = join(model_data, self.get_accent_path(stt["lm"][n_id], accent))
            m_dict = join(model_data, self.get_accent_path(stt["dict"][n_id], accent))
            return LanguageModel(name, m_hmm, m_lm, m_dict) # Create the new language model object
        except Exception as err:
            log.error("Failed loading language model! (id: %s) (err: %s)" % (str(l_id), str(err)))
            return None

    @staticmethod
    def dump_configs(configs):
        """Json dumps wrapper to printy print dicts to the console

        Arguments:
            configs (dict): The dictionary to log with indendation

        Note:
            This will only print in the CONFIGS namespace
        """

        try:
            log.info("Configurations:\n%s" % dumps(configs, indent=4))
        except Exception as err:
            log.error("Failed to dump json! (err: %s)" % str(err))

    def __load_configs(self):
        global CONFIGS
        """Private method to reload the configuration file

            Note:
                This should really only be called on configs creation and on the file change listener

            Returns: (bool)
                True on success or False on failure to load configurations:we
        """

        try:
            c_f = open(self._json_config, 'r') # Open the config file for reading
            c_f_data = c_f.read()
            CONFIGS = loads(c_f_data)
            c_f.close()

            # Log the new configurations
            log.info("Dumping configs")
            Configs.dump_configs(CONFIGS)
            return True
        except Exception as err:
            log.error("Failed to load configuration file! (err: %s)" % str(err))
            if c_f is not None:
                c_f.close()
        return False

    def get_cwd(self):
        """Return the absolute path of the current working directory

        Returns: (str)
            The absolute path of the current working directory
        """

        return self._current_dir

    def get_full_path(self, relative_path):
        """Return the absolute path of a file that's located relative to the absolute path

        Returns: (str)
            The absolute path of the current file that's relative to the path
        """

        return join(self.get_cwd(), relative_path)

    def get_accent_path(self, path_parse, accent):
        """Method to replace common accent path symbols in the configuration files

        Note:
            If there's a path separator at the end of the path, then this method will remove it

        Arguments:
            path_parse (str): The configuration symbolic filled path to be parsed

        Returns: (str)
            The parsed and non-symbolic and non-variabled accent path data
        """

        try:
            if accent[-1] == sep:
                accent = accent[:len(accent) - 2]
            path_parse = re.sub(r'\(!accent!\)', accent, path_parse)
            return path_parse
        except Exception as err:
            log.error("Failed parsing accent config path! (path: %s) (err: %s)" % (path_parse, str(err)))
            return None

