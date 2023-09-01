# The api module used for networking tasks like device discovery, setup, and communication during
# experiment runs.

import paramiko
import json
import socket
import getpass
import pathlib
import uuid
import logging
import concurrent.futures
from contextlib import contextmanager
from typing import Self
import ipaddress
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from exceptions import SSHAuthenticationException

class LAN:
    """
    Helps with general networking tasks that are not specific to one host.
    """
    LOCAL_CIDR_BLOCK: list[str] = [str(ip)
        for ip in ipaddress.ip_network("192.168.1.0/24").hosts()]

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

        def check_host(host):
            """
            Checks if the host is available at all, but does not attempt to authenticate.
            """
            try:
                test_socket = socket.create_connection((host, port), timeout)
                available_hosts.put(host)
                test_socket.close()
            except Exception:
                return

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(check_host, try_hosts)

        return list(available_hosts.queue)


class SSHCredentials:
    """
    The networking module currently enforces a specific authentication strategy because it takes
    too much time to account for they myriad of ways one might establish a connection. Because the
    same strategy is always used, we can neatly bundle the required credentials (and remember
    them between sessions) using this class.
    """

    user: str
    pkey: paramiko.RSAKey
    pkey_fp: pathlib.Path

    def __init__(self, username: str, rsa_pkey_path: pathlib.Path | str) -> None:
        """
        Both parameters are validated and the given pkey path is converted to a paramiko.RSAKey
        instance before being stored as attributes.
        """
        self._set_user(username)
        self._set_pkey(rsa_pkey_path)

    @classmethod
    def from_dict(cls, source: dict) -> Self:
        """
        Construct an instance of SSHCredentials from its dictionary representation.
        """
        user, pkey_fp = source["user"], source["pkey_fp"]
        return cls(user, pkey_fp)

    def _set_user(self, username: str) -> None:
        """
        Validates the given username and stores it, raising an error if invalid.
        """
        u = username.strip()
        # the only enforced username constraint on Debian-based systems
        if 0 < len(u) < 32:
            self.user = u
        else:
            raise ValueError(f"Bad username '{username}' given.")

    def _set_pkey(self, rsa_pkey_path: pathlib.Path | str) -> None:
        """
        Validates the given path to the rsa key, converts it to a paramiko.RSAKey instance, and
        stores it, or raises an error if invalid.
        """
        if not isinstance(rsa_pkey_path, pathlib.Path):
            rsa_pkey_path = pathlib.Path(rsa_pkey_path)
        expanded_path = rsa_pkey_path.absolute().expanduser()

        if expanded_path.exists() and expanded_path.is_file():
            self.pkey = paramiko.RSAKey(filename=str(expanded_path))
            self.pkey_fp = expanded_path
        else:
            raise ValueError(f"Invalid path '{rsa_pkey_path}' specified for RSA key.")

    def as_dict(self) -> dict:
        """
        Returns the dictionary representation of the credentials. Used for persistent storage.
        """
        return {"user": self.user, "pkey_fp": self.pkey_fp}
    



        
class SSHSession(paramiko.SSHClient):
    """
    Because we're enforcing a specific authentication strategy, we can abstract away lots of the 
    setup involved with the paramiko.SSHClient class by automatically attempting authentication
    and updating an instance's state depending on the results. This class assumes the given host
    has already been validated as available (listening on port 22).
    """

    creds: SSHCredentials
    host: str  # IP address or hostname

    def __init__(self, credentials: SSHCredentials, hostname_or_ip: str) -> None:
        """
        Automatically attempts to connect, raising diffent exceptions for different points of
        failure. 
        """
        super().__init__()
        self.creds = credentials
        self._set_host(hostname_or_ip)

        try:
            self._establish()
        except Exception:
            raise SSHAuthenticationException(f"Problem while authenticating to host {self.host}.")

    def _set_host(self, hostname_or_ip: str):
        hostname_or_ip = hostname_or_ip.strip()
        self.host = hostname_or_ip

    def _establish(self) -> None:
        """
        Attempt to authenticate with the host and open the connection.
        """
        user = self.creds.user
        pkey = self.creds.pkey

        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.connect(
            self.host,
            username=user,
            pkey=pkey,
            auth_timeout=5,
            timeout=1
        )

