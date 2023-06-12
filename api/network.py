import socket
from getmac import get_mac_address
from scapy.all import ARP, Ether, srp
import os
import platform
import netifaces as ni
import uuid


class Device:

    """
    A class that represents a device on the LAN.

    Attributes
    ----------
    ip_address (string) : the local IP address of the device
    mac_address (string) : the MAC address of the device
    hostname (string) : the hostname of the device

    Methods
    -------
    get_mac_address : gets MAC
    get_hostname : gets hostname

    Subclasses
    ----------
    Controller : represent's the controller device (user's machine)
    """

    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.mac_address = self.get_mac_address()
        self.hostname = self.get_hostname()

    def get_mac_address(self):
        """Uses getmac to attempt to find MAC, memoizes if successful"""
        if self.mac_address:
            return self.mac_address    # if already memoized 
        try:
            return get_mac_address(ip=self.ip_address)
        except Exception as e:
            print(f"An error occurred while getting the MAC address: {e}")
            return None

    def get_hostname(self):
        """Uses socket to attempt to find hostname, memoizes if successful"""
        if self.hostname:
            return self.hostname    # if already memoized 
        try:
            return socket.gethostbyaddr(self.ip_address)[0]
        except Exception as e:
            print(f"An error occurred while getting the hostname: {e}")
            return None

    def set_up_passwordless_ssh(self):
        """Sets up passwordless ssh on the device"""
        # TODO: implement
        pass

    def is_setup(self):
        """Returns True if the device is ready to be deployed to"""
        # TODO:
        # Check for repo matching the controller device
        # Check for ability to ssh
        # Check for device info in config files
        # Check for working base image
        # Check for connectivity to other devices


class Controller(Device):

    """
    A child class of Device that represents the controller device, which is
    usually the machine the user is on.

    Attributes
    ----------
    ip_address (string) : the local IP address of the device
    mac_address (string) : the MAC address of the device
    hostname (string) : the hostname of the device
    
    Methods
    -------
    get_mac_address : gets MAC
    get_hostname : gets hostname
    """

    def __init__(self, net_interface=None):
        system_info = platform.uname()
        self.os_family = system_info.system
        self.release = system_info.release
        self.architecture = system_info.machine

        self.net_interface = net_interface
        if not self.net_interface:
            interface_list = ni.interfaces()
            self.net_interface = interface_list[0]

        self.hostname = socket.gethostname()
        self.ip_address = ni.ifaddresses(self.net_interface)[ni.AF_INET][0]['addr']
        if os.name == 'posix':
            self.mac_address = ni.ifaddresses(self.net_interface)[ni.AF_LINK][0]['addr']
        elif os.name == 'nt':
            self.mac_address = get_mac_address(interface=self.net_interface)


class LAN:

    """
    A class that represents the local network, specifically with respect to
    connectable IoT device setup and discovery.

    Attributes
    ----------
    devices : list
        a list of Device objects that have been discovered on the LAN

    Methods
    -------
    discover_devices(ip_range):
        Discovers devices on the LAN and adds them to the devices list.
    """

    def __init__(self):
        """Constructs all the necessary attributes for the LAN object."""
        self.devices = []

    def discover_devices(self, ip_range):
        """
        Discover devices on the LAN within the given IP range.

        Parameters:
            ip_range (str): The range of IPs to scan in CIDR format
            (e.g., "192.168.1.0/24").
        """
        for ip in ipaddress.IPv4Network(ip_range):
            try:
                device = Device(str(ip))
                self.devices.append(device)
            except Exception as e:
                print(f"An error occurred while adding a device: {e}")

    def display_devices(self):
        """
        Display the IP addresses and hostnames of all the devices in the LAN.
        """
        for device in self.devices:
            print(f"IP Address: {device.ip_address}, Hostname: {device.hostname}")


