#!/bin/bash

# Uses IP Forwarding and NAT to let the LAN devices access the internet
# through your PC.
#
# NOTE: LAN devices must have their /etc/dhcpcd.conf files configured to
# allow this, and your PC must either be running Linux or a subsystem like
# WSL

# like putting a "sudo" before the call to this script
[ "$UID" -eq 0 ] || exec sudo "$0" "$@"

INTERNET_INTERFACE="wlo1"
LAN_INTERFACE="enp3s0"

# Enable IP forwarding
echo "Enabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1

# Set up NAT
echo "Setting up NAT with iptables..."
sudo iptables -t nat -A POSTROUTING -o $INTERNET_INTERFACE -j MASQUERADE
sudo iptables -A FORWARD -i $LAN_INTERFACE -o $INTERNET_INTERFACE -j ACCEPT
sudo iptables -A FORWARD -i $INTERNET_INTERFACE -o $LAN_INTERFACE -m state --state RELATED,ESTABLISHED -j ACCEPT

# Wait for user input to disable IP forwarding and clean up iptables rules
read -p "Press ENTER to stop sharing the internet connection and clean up iptables rules..."

# Disable IP forwarding
echo "Disabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=0

# Clean up iptables rules
echo "Cleaning up iptables rules..."
sudo iptables -t nat -D POSTROUTING -o $INTERNET_INTERFACE -j MASQUERADE
sudo iptables -D FORWARD -i $LAN_INTERFACE -o $INTERNET_INTERFACE -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -D FORWARD -i $INTERNET_INTERFACE -o $LAN_INTERFACE -j ACCEPT

echo "Internet sharing stopped."

