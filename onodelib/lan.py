# The api module used for networking tasks like device discovery, setup, and communication during
# experiment runs.

import socket
import ipaddress

from concurrent.futures import ThreadPoolExecutor
from queue import Queue


class LAN:
    """
    Helps with general networking tasks that are not specific to one host.
    """
    LOCAL_CIDR_BLOCK: list[str] = [str(ip)
        for ip in ipaddress.ip_network("192.168.1.0/24").hosts()]

    @classmethod
    def host_is_reachable(cls, host: str, port: int, timeout: int | float) -> bool:
        """
        Checks if the host is available at all, but does not attempt to authenticate.
        """
        try:
            test_socket = socket.create_connection((host, port), timeout)
            test_socket.close()
            return True
        except Exception:
            return False

    @classmethod
    def get_available_hosts(cls,
        try_hosts: list[str] = LOCAL_CIDR_BLOCK,
        port: int =22,
        timeout: int | float = 0.5,
        max_threads: int = 50) -> list[str]:
        """
        Takes a list of strings (ip or hostname) and returns a new list containing only those that
        are available, without attempting to authenticate. Uses threading.
        """
        available_hosts = Queue()

        def check_host(host: str):
            """
            Adds host to queue if reachable.
            """
            if cls.host_is_reachable(host, port, timeout):
                available_hosts.put(host)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(check_host, try_hosts)

        return list(available_hosts.queue)


