# Exceptions used within the api module

class SSHAuthenticationException(Exception):
    def __init__(self, message):
        super().__init__(message)

class DeviceUnavailableException(Exception):
    def __init__(self, message):
        super().__init__(message)

