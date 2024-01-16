import yaml


class ModelConfigurationSetup:
    def __init__(self, path=None):
        self.config_details = self.__read_yaml_data(path)

    def __read_yaml_data(self, path):
        settings = {}
        try:
            with open(path, "r") as file:
                settings = yaml.safe_load(file)
        except Exception as error:
            print(
                "No valid configuration provided. Using default settings, behavior could be unexpected."
            )

        # add default entries here just in case
        # self.device = kwargs.get("device", "cpu")
        # self.mode = kwargs.get("mode", "eval")
        # self.hook_depth = kwargs.get("depth", np.inf)
        # self.base_input_size = kwargs.get("image_size", (3, 224, 224))
        # self.dataset_type = kwargs.get("dataset_type", "balanced")
        self.model_name = settings.get("model_name", "alexnet").lower().strip()

        return settings
