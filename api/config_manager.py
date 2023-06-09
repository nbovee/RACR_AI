import pathlib
import json


class ConfigManager:

    """
    A class that handles configs and persistent data storage, including
    creating config files, saving, loading, editing, and parsing data.

    Attributes
    ----------
    conf_fps : list of pathlib.Path objects representing necessary config files
    configs : dict of config data, with keys being config file names

    Methods
    -------
    create_config_file(name) : creates a new config file with the given name
    """

    def __init__(self):

        # Check if config files exist and create them if not
        cwd = pathlib.Path.cwd()
        with open(cwd / "setup_data.json", "r") as file:
            self.setup_data = json.load(file)

        self.conf_fps = [cwd / "Configs" / fn
                        for fn in self.setup_data["config_files"].keys()]
        missing_files = [fp for fp in self.conf_fps if not fp.exists()]
        if missing_files:
            for fp in missing_files:
                self.create_config_file(fp.stem)

        # Load config files
        self.configs = {}
        for fp in self.conf_fps:
            with open(fp, "r") as file:
                self.configs[fp.stem] = json.load(file)
        
    def create_config_file(self, name):
        """
        Creates a new config file with the given name, using the
        setup_data.json file to determine its structure and default values.

        Parameters
        ----------
        name : str
            The name of the config file to create (program will create new
            file in the Configs folder with name + ".json")
        """
        if not name.endswith(".json"):
            name += ".json"

        # Check if config file already exists
        cwd = pathlib.Path.cwd()
        fp = cwd / "Configs" / name
        if fp.exists():
            # TODO: log this
            return
        
        # Copy new file template from setup_data and write it to new file
        template = self.setup_data["config_files"][name]
        with open(fp, "w") as file:
            json.dump(template, file, indent=8)



