import socket
from pathlib import Path


def get_repo_root() -> Path:
    """
    Returns a pathlib.Path object representing the root directory of this repo.
    """
    return Path(__file__).parent.parent.parent.absolute()


def get_local_ip():
    """
    Returns the ip address currently used by the local machine. To automatically find the most
    appropriate network interface, we use the Google DNS server trick.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip

