


class InvalidConfigFileException(Exception):
    """
    An exception raised when a config file is invalid.
    """

    def __init__(self, fp, message="Cannot load from invalid config file: "):
        self.message = message + fp
        super().__init__(self.message)

