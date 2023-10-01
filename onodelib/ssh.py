import paramiko
from plumbum import SshMachine
import pathlib
import yaml

from typing import Self

from lan import LAN
from exceptions import SSHAuthenticationException, DeviceUnavailableException


class SSHConnectionParams:
    """
    The networking module currently enforces a specific authentication strategy because it takes
    too much time to account for they myriad of ways one might establish a connection. Because the
    same strategy is always used, we can neatly bundle the required credentials (and remember
    them between sessions) using this class.
    """

    host: str  # hostname or ip
    user: str
    pkey: paramiko.RSAKey
    pkey_fp: pathlib.Path

    _host_reachable: bool  # set during constructor

    def __init__(self,
                 host: str,
                 username: str,
                 rsa_pkey_path: pathlib.Path | str,
                 default: bool = True) -> None:
        """
        Both parameters are validated and the given pkey path is converted to a paramiko.RSAKey
        instance before being stored as attributes.
        """
        self._set_host(host)
        self._set_user(username)
        self._set_pkey(rsa_pkey_path)
        self._default = default

    @classmethod
    def from_dict(cls, source: dict) -> Self:
        """
        Construct an instance of SSHConnectionParams from its dictionary representation.
        """
        host, user, pkey_fp, default = source["host"], source["user"], source["pkey_fp"], source.get("default")
        if default is None:
            return cls(host, user, pkey_fp)
        else:
            return cls(host, user, pkey_fp, default=default)

    def _set_host(self, host: str) -> None:
        """
        Validates the given hostname or IP and stores it, updating the `_host_reachable` attribute 
        accordingly.
        """
        self.host = host

        SSH_PORT, TIMEOUT_SECONDS = 22, 0.5
        self._host_reachable = LAN.host_is_reachable(host, SSH_PORT, TIMEOUT_SECONDS)

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

    def host_reachable(self) -> bool:
        """
        Returns True if the host is listening on port 22, but does not guarantee authentication
        will succeed.
        """
        return bool(self._host_reachable)

    def as_dict(self) -> dict:
        """
        Returns the dictionary representation of the credentials. Used for persistent storage.
        """
        return {"host": self.host, "user": self.user, "pkey_fp": self.pkey_fp}

    def is_default(self) -> bool:
        """
        Returns "true" if this is the first connection method that should be tried for the host.
        """
        return self._default


class Device:
    """
    A basic interface for keeping track of devices.
    """

    _name: str
    _type: str
    _cparams: list[SSHConnectionParams]

    working_cparams: SSHConnectionParams | None

    def __init__(self, name: str, record: dict) -> None:
        self._name = name
        self._type = record["device_type"]
        self._cparams = [SSHConnectionParams.from_dict(d) for d in record["connection_params"]] 
        
        # check the default method first
        self._cparams.sort(key=lambda x: 1 if x.is_default() else 0, reverse=True)
        self.working_cparams = None
        for p in self._cparams:
            if p.host_reachable():
                self.working_cparams = p
                break

    def is_reachable(self) -> bool:
        """
        Returns true if a working connection method has been found.
        """
        return self.working_cparams is not None

    def serialized(self) -> tuple[str, dict[str, str | bool]]:
        """
        Used to serialize Device objects.
        """
        key = self._name
        value = {"device_type": self._type, "connection_params": [c.as_dict() for c in self._cparams]}
        return key, value

    def get_current(self, attr: str) -> str | None:
        """
        Gets the CURRENT host or user. Necessary because a single device may have multiple connection_params
        associated with it.
        """
        if self.working_cparams is not None:
            attr_clean = attr.lower().strip()
            if attr_clean in ("host", "hostname", "host name"):
                return self.working_cparams.host
            elif attr_clean in ("user", "username", "usr", "user name"):
                return self.working_cparams.user
        return None

    def as_pb_sshmachine(self) -> SshMachine:
        """
        Returns a plumbum.SshMachine instance to represent the device.
        """
        if self.working_cparams is not None:
            return SshMachine(self.working_cparams.host, user=self.working_cparams.user, keyfile=str(self.working_cparams.pkey_fp))
        else:
            raise DeviceUnavailableException(f"Cannot make plumbum object from device {self._name}: not available.")


class DeviceMgr:
    """
    Manages a collection of SSHConnectionParams objects. Responsible for reading and writing serialized
    instances to/from the persistent datafile.
    """

    devices: list[Device]

    def __init__(self, dfile_path: pathlib.Path) -> None:
        self.datafile_path = dfile_path
        self._load()

    def get_devices(self, available_only: bool = False) -> list[Device]:
        if available_only:
            return [d for d in self.devices if d.is_reachable()]
        return self.devices

    def _load(self) -> None:
        with open(self.datafile_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.SafeLoader)
        self.devices = [Device(dname, drecord) for dname, drecord in data.items()]

    def _save(self) -> None:
        serialized_devices = {name: details for name, details in [d.serialized() for d in self.devices]}
        with open(self.datafile_path, 'w') as file:
            yaml.dump(serialized_devices, file)


class SSHSession(paramiko.SSHClient):
    """
    Because we're enforcing a specific authentication strategy, we can abstract away lots of the 
    setup involved with the paramiko.SSHClient class by automatically attempting authentication
    and updating an instance's state depending on the results. This class assumes the given host
    has already been validated as available (listening on port 22).
    """

    login_params: SSHConnectionParams
    host: str  # IP address or hostname

    def __init__(self, device: Device) -> None:
        """
        Automatically attempts to connect, raising diffent exceptions for different points of
        failure. 
        """
        super().__init__()
        if device.working_cparams is None:
            raise DeviceUnavailableException(
                f"Cannot establish SSH connection to unavailable device {device._name}"
            )
        self.login_params = device.working_cparams
        self._set_host()

        try:
            self._establish()
        except Exception as e:
            raise SSHAuthenticationException(
                f"Problem while authenticating to host {self.host}: {e}"
            )

    def _set_host(self):
        self.host = self.login_params.host

    def _establish(self) -> None:
        """
        Attempt to authenticate with the host and open the connection.
        """
        user = self.login_params.user
        pkey = self.login_params.pkey

        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.connect(
            self.host,
            username=user,
            pkey=pkey,
            auth_timeout=5,
            timeout=1
        )

    def copy_over(self, from_path: pathlib.Path, to_path: pathlib.Path, exclude: list = []):
        """
        Copy a file or directory over to the remote device.
        """
        sftp = self.open_sftp()
        if not from_path.name in exclude:
            if from_path.is_dir():
                try:
                    sftp.stat(str(to_path))
                except FileNotFoundError:
                    sftp.mkdir(str(to_path))

                for item in from_path.iterdir():
                    # Recursive call to handle subdirectories and files
                    self.copy_over(item, to_path / item.name)
            else:
                # Upload the file
                sftp.put(str(from_path), str(to_path))
        sftp.close()

    def mkdir(self, to_path: pathlib.Path, perms: int = 511):
        sftp = self.open_sftp()
        try:
            sftp.mkdir(str(to_path), perms)
        except IOError:
            print(f"directory {to_path} already exists on remote device")
        sftp.close()

    def rpc_container_up(self):
        pass

