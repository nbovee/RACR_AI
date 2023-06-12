import pathlib
import re
import json
import os
from singleton_decorator import singleton

from exceptions import InvalidConfigFileException

@singleton
class ConfigManager:

    """
    A class that handles configs and persistent data storage, including
    creating config files, saving, loading, editing, and parsing data.
    Implemented as singleton to prevent multiple instances from being created.

    Attributes
    ----------
    conf_fps : list of pathlib.Path objects representing necessary config files
    configs : dict of config data, with keys being config file names

    Methods
    -------
    create_config_file(name) : creates a new config file with the given name
    """

    def __init__(self):

        # Get abs path of the "api" directory
        self.api_path = pathlib.Path(__file__).parent.absolute()

        # Load networking configs or create them if they don't exist
        saved_networking_info_fp = self.api_path / "networking.json"
        if saved_networking_info_fp.exists():
            with open(saved_networking_info_fp, "r") as file:
                networking_info = json.load(file)
        else:
            networking_info = None
        self.networking = NetworkSection(data_file=networking_info)
        
        # Load preferences or create them if they don't exist
        saved_preferences_fp = self.api_path / "preferences.json"
        if saved_preferences_fp.exists():
            with open(saved_preferences_fp, "r") as file:
                preferences_info = json.load(file)
        else:
            preferences_info = None
        self.preferences = PreferencesSection(data_file=preferences_info)
        
    def save(self, section="all"):
        """Saves the given section of configs to disk."""

        configs_path = self.api_path / "Configs" 

        # Save networking config file
        if section in ("all", "net"):
            net_conf_path = configs_path / "networking.json"
            with open(net_conf_path, "w") as file:
                json.dump(self.networking, file, indent=8)

        # Save preferences config file
        if section in ("all", "pref"):
            pref_conf_path = configs_path / "preferences.json"
            with open(pref_conf_path, "w") as file:
                json.dump(self.preferences, file, indent=8)

    def 


class AbstractConfigSection:

    """
    An abstract class that represents a section of the config data. This class
    should be inherited by all config sections.

    Attributes
    ----------
    data : dict of config data

    Methods
    -------
    load_from(fp) : loads the config data from a given file path
    build_fresh() : builds a fresh config data dict

    Subclasses
    ----------
    NetworkSection : handles the networking section of the config data
    PreferencesSection : handles the preferences section of the config data

    Raises
    ------
    InvalidConfigFileException
    """

    def __init__(self, child_name, data_file=None):
        self.child_name = child_name
        self.api_path = pathlib.Path(__file__).parent.absolute()
        self.default_data_fp = pathlib.Path(self.setup_data
                                                .get(self.child_name)
                                                .get("default_path"))
        with open(self.api_path / "setup_data.json", "r") as file:
            self.setup_data = json.load(file)
        self.data_file = data_file
        self.save_to = self.data_file
        self.top_level_keys = set(
                self.setup_data[self.child_name]["top_level_keys"]
            )

        if data_file:
            try:
                self.load_from(self.data_file)
            # if we can't use the given file, move it to old and build fresh
            except InvalidConfigFileException:
                old_dir = self.api_path / "Configs" / "Old"
                orig_data_fn = self.data_file.stem
                similar_files = [f.stem
                    for f in old_dir.iterdir()
                    if f.stem.startswith(orig_data_fn)]

                def get_num(s):
                    # extracts int from end of filename
                    pattern = r'\D*(\d+)$'
                    match = re.search(pattern, s)
                    if match:
                        return int(match.group(1))
                    else:
                        return 0

                taken_nums = [get_num(s) for s in similar_files]
                max_num = max(taken_nums)
                new_fn = f"{orig_data_fn}_{max_num + 1}.json"
                
                shutil.move(self.data_file, self.data_file.parent
                    / "Old" / new_fn)

                self.data_file = self.default_data_fp
                self.build_fresh()
                
        else:
            self.data_file = self.default_data_fp
            self.build_fresh()

    def load_from(self, fp):
        """Loads the config data from a given file path."""
        
        with open(fp, "r") as file:
            data = json.load(file)

        # make sure the file is valid
        for key in self.top_level_keys:
            if key not in data:
                # TODO: log this
                raise InvalidConfigFileException(f"Invalid config file: {fp}")

        self.data = data
        
    def build_fresh(self, save_to=None):
        """Builds a fresh config data dict."""

        if not save_to and not self.data_file:


    def save(self, to=None):
        """Saves the config data to disk."""

        if to:
            self.save_to = to

class NetworkSection(AbstractConfigSection):

    """
    A small helper class with a composition relationship to ConfigManager. This
    class handles the networking section of the config data. It is a subclass
    of AbstractConfigSection.

    Attributes
    ----------
    controllers : list of Device objects representing controllers
    devices : list of Device objects representing devices
    saved_configurations : list of Configuration objects representing saved
        configs

    Methods
    -------
    add_controller(controller) : adds a controller to the controllers list
    add_device(device) : adds a device to the devices list
    add_saved_configuration(configuration) : adds a configuration to the saved
        configurations list
    """

    def __init__(self, data_file=None):
        super().__init__("NetworkSection", data_file=data_file)

    def load_from(self, fp):
        """Loads the networking section from a given file path."""

        super().load_from(fp)

        self.controllers = self.data["controllers"]
        self.devices = self.data["devices"]
        self.saved_configurations = self.data["saved_configurations"]

    def build_fresh(self, save_to=None):
        """Builds a fresh networking section."""

        super().build_fresh(save_to=save_to)

        self.controllers = []
        self.devices = []
        self.saved_configurations = []

        self.save()


class PreferencesSection(AbstractConfigSection):

    """
    A small helper class with a composition relationship to ConfigManager. This
    class handles the preferences section of the config data. It is a subclass
    of AbstractConfigSection.

    Attributes
    ----------
    verbosity : int representing the verbosity level

    Methods
    -------
    set_verbosity(verbosity) : sets the verbosity level
    """
    
    def __init__(self, data_file=None):
        super().__init__("PreferencesSection", data_file=data_file)

    def load_from(self, fp):
        """Loads the preferences section from a given file path."""

        super().load_from(fp)

        self.verbosity = self.data["verbosity"]

    def build_fresh(self, save_to=None):
        """Builds a fresh preferences section."""

        super().build_fresh(save_to=save_to)

        self.verbosity = 0

        self.save()










































