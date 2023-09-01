#!/usr/bin/env python3

"""
This script runs the "tracr" CLI, which is powered by the API that lives in 
the "api" folder on this repo.

For a cleaner experience, add this directory to your PATH, which will allow
you to run the CLI from anywhere, and without preceding the command with
the word "python".
"""

import argparse
import json
import socket
import os
import shutil
import sys
import logging
from rich.logging import RichHandler
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Console
from rich.box import SQUARE
from rich.table import Table
from getmac import get_mac_address
from pathlib import Path
from collections import defaultdict

import api.experiment as exp
import api.device as dev
import api.utils as utils
import api.controller as control
import api.bash_script_wrappers as bashw


# path to root of tracr project
PROJECT_ROOT = utils.get_tracr_root()

# path to console text files
CONSOLE_TXT_DIR = PROJECT_ROOT / "Assets" / "console_text"

# path to the main logfile
MAIN_LOG_FP = PROJECT_ROOT / "PersistentData" / "Logs" / "app.log"


# logger setup
def setup_logging(verbosity):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    file_format = "%(asctime)s - %(module)s - %(levelname)s: %(message)s"

    logger = logging.getLogger("tracr_logger")
    logger.setLevel(logging.DEBUG)

    # all messages will be logged to this file
    file_handler = logging.FileHandler(MAIN_LOG_FP.expanduser())
    file_handler.setLevel(logging.DEBUG)

    # only messages of the given level or higher will be logged to console
    console_handler = RichHandler(
        show_time=False, show_path=False, rich_tracebacks=True
    )
    console_handler.setLevel(levels[min(verbosity, len(levels) - 1)])

    # different formats for file and console logs
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)

    # Adding the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Session:
    """
    Manages the current session, which is really just a single tracr command.
    Responsible for managing state, logging, and console output.
    """

    logger: logging.Logger
    console: Console
    controller: control.Controller
    device_mgr: dev.DeviceManager
    exp_mgr: exp.ExperimentManager

    def __init__(self, auto_init=True):
        """
        The constructor by default sets itself up with a logger, controller,
        and console, but this can be turned off.

        Parameters
        ----------
        auto_init : bool, optional
            Whether or not to automatically initialize the session with a
            logger, controller, and console, by default True
        """
        if auto_init:
            self.console = Console()
            self.controller = control.Controller()

            # check whether the controller has been setup
            if not self.controller.is_setup(showprogress=True):
                self.print(
                    f"Your machine is not yet set up as a tracr controller. "
                    + f"Run the bootstrap script to get started.",
                    style="bold red",
                )
                sys.exit(1)

            # set up the logger according to the user's verbosity setting
            verbosity = self.controller.get_settings("Preferences").get("verbosity", 2)
            self.logger = setup_logging(verbosity)
            self.device_mgr = dev.DeviceManager()
            self.exp_mgr = exp.ExperimentManager(PROJECT_ROOT / "TestCases")
        else:
            self.logger = None
            self.console = None
            self.controller = None
            self.device_mgr = None
            self.exp_mgr = None

    def log(self, level, message):
        """
        Logs a message to the session's logger, which may also print the log
        to the console, depending on the user's verbosity setting.
        """
        if not self.logger:
            raise RuntimeError(
                "Can't log without first setting a logger for the Session instance."
            )

        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            raise ValueError(f"Invalid log level: {level}")

    def print(self, message, **kwargs):
        """
        Prints a message to the console, using the session's console.
        """
        if not self.console:
            raise RuntimeError(
                "Can't print without first setting a console for the Session instance."
            )

        self.console.print(message, **kwargs)


def launch_experiment(name, session: Session, preset=None, **kwargs):
    """
    Launches an experiment with the given name, and ideally a preset
    from the experiment's config file. Otherwise, all options must
    be specified in the kwargs.
    """
    if preset:
        param_dicts = []
        for nodetype, specs in preset.items():
            nodename = nodetype
            num_devices = specs["num_devices"]
            if specs["run_on"] == "any":
                devices = session.device_mgr.get_devices_by(
                    is_available=True, is_setup=True
                )[:num_devices]
            elif specs["run_on"] == "controller":
                devices = [session.controller]
            else:
                session.log("warning", "Only 'any' run_on is supported right now.")
                return
            logto = Path(specs.get("output").get("log_directory"))
            monitor = bool(specs.get("output").get("monitor_realtime"))
            load_data_from = specs.get("input").get("dataset_path")
            data_type = specs.get("input").get("type")
            data_loader = specs.get("input").get("loader")
            port = specs.get("input").get("port")
            metrics = specs.get("metrics")
            nodespecs = session.exp_mgr.get_experiment(name).main_config.getval(
                "nodes", "types", nodename
            )
            wait_for = nodespecs.get("depends_on")
            python_version = nodespecs.get("environment").get("python_version")
            pip_version = nodespecs.get("environment").get("pip_version")
            main_file = nodespecs.get("main_file")

            params = {
                "node_type": nodename,
                "devices": devices,
                "logto": logto,
                "monitor": monitor,
                "load_data_from": load_data_from,
                "data_type": data_type,
                "data_loader": data_loader,
                "port": port,
                "metrics": metrics,
                "wait_for": wait_for,
                "python_version": python_version,
                "pip_version": pip_version,
                "main_file": main_file,
            }

            for device in devices:
                if device == session.controller:
                    controller_ready = all(
                        bashw.validate_node_setup(
                            name, nodename, python_version, pip_version, main_file
                        ).values()
                    )
                    if not controller_ready:
                        session.log("error", f"Controller is not ready to run {name}.")
                        return
                else:
                    device_ready = all(
                        bashw.validate_node_setup(
                            name, nodename, python_version, pip_version, main_file
                        ).values()
                    )
                    if not device_ready:
                        bashw.node_setup(
                            device.name,
                            name,
                            nodename,
                            python_version,
                            pip_version,
                            overwrite=True,
                        )
                        device_ready = all(
                            bashw.validate_node_setup(
                                name, nodename, python_version, pip_version, main_file
                            ).values()
                        )
                        if not device_ready:
                            session.log(
                                "error", f"{device.name} is not ready to run {name}."
                            )
                            return
        ports = [p.get("port") for p in param_dicts if p.get("port")]
        if ports:
            port = ports[0]
            session.controller.open_fileserver("load_data_from", port)
        else:
            session.controller.open_fileserver("load_data_from")

        for p in sorted(param_dicts, key=lambda x: 1 if x["wait_for"] else 0):
            pass

            param_dicts.append(params)


# CLI is split up into submodules responsible for different operations, so
# there are lots of arguments that can be used, each with their own option
# flags and arguments. This is all organized using argparse, which reads
# the arguments and options and passes them to the appropriate function.


##############################################################################
######################### SETUP SUBMODULE FUNCTIONS ##########################
##############################################################################


def setup_controller(args):
    pass


def setup_device(args):
    # TODO: define setup device function
    pass


def setup_experiment(args):
    # TODO: define setup experiment function
    pass


##############################################################################
##################### DEVICE SUBMODULE FUNCTIONS #############################
##############################################################################


def device_ls(args):
    """
    Shows the user a list of devices.
    Called by the "tracr device ls [-options]" command.

    Parameters:
    -----------
    args: argparse.Namespace
        The arguments and options passed to the CLI.
    """
    devices = session.device_mgr.tracr_devices
    console = session.console

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Available")
    table.add_column("Ready")
    table.add_column("UUID")
    table.add_column("Host")

    for d in devices:
        name = d.name
        available = (
            "[bold green]Yes[/bold green]"
            if d.is_available()
            else "[bold red]No[/bold red]"
        )
        ready = (
            "[bold green]Yes[/bold green]"
            if d.is_setup()
            else "[bold red]No[/bold red]"
        )
        uuid = str(d.id)
        host = d.host

        table.add_row(name, available, ready, uuid, host)

    console.print(table)


def device_add(args):
    if args.wizard:
        pass
    if args.host:
        pass
    if args.user:
        pass
    if args.pw:
        pass
    if args.keys:
        pass
    if args.nickname:
        pass
    if args.description:
        pass


##############################################################################
#################### EXPERIMENT SUBMODULE FUNCTIONS ##########################
##############################################################################


def experiment_add(args):
    """
    Adds a new experiment to the controller's local system by setting up
    a 'blank slate' directory inside of TestCases.
    """
    pass


def experiment_ls(args):
    """
    Displays the list of experiments that are currently available.
    """
    experiments = session.exp_mgr.experiments
    console = session.console

    main_table = Table(show_header=True, header_style="bold magenta")
    main_table = Table(show_header=True, header_style="bold magenta", box=SQUARE)
    main_table.add_column("Experiment Name")
    main_table.add_column("Description")
    main_table.add_column("Node Types")

    for e in experiments:
        name = e.name
        description = e.main_config.getval("meta", "description")

        # Create a new table for the node types
        node_table = Table(show_header=True, header_style="bold cyan", box=SQUARE)
        node_table.add_column("Name")
        node_table.add_column("Environment")
        node_table.add_column("Supported CPUs")
        node_table.add_column("Dependencies")
        node_table.add_column("Main File")

        node_types = e.main_config.getval("nodes").get("types")
        for key in node_types.keys():
            nt_name = key
            nt_env = (
                f"Python: {node_types[key].get('environment').get('python_version')},"
                + f"\nPip: {node_types[key].get('environment').get('pip_version')}"
            )
            nt_arch = ", ".join(node_types[key].get("supported_cpu_architectures"))
            nt_deps = node_types[key].get("depends_on")
            nt_main = node_types[key].get("main_file")

            node_table.add_row(nt_name, nt_env, nt_arch, nt_deps, nt_main)

        # Add the node table to the main table
        main_table.add_row(name, description, node_table)

    console.print(main_table)


def experiment_run(args):
    """
    Runs an experiment.
    """
    exp_mgr = session.exp_mgr
    console = session.console

    name = args.name

    if args.output:
        pass

    if args.preset:
        # gather preset information
        try:
            exp = exp_mgr.get_experiment(name)
        except ValueError:
            session.log("warning", f"Experiment {name} not found.")
            return
        if not args.preset in exp.main_config.getval("runtime", "presets"):
            session.log(
                "warning",
                f"Preset {args.preset} not found in experiment {name}.",
            )
            return
        preset = exp.main_config.getval("runtime", "presets", args.preset)

        session.controller.launch_experiment(name, preset=preset)


def network(args):
    if args.d:
        print(f"Status for host: {args.d}")
    elif args.e:
        print(f"Status for name: {args.e}")
    else:
        print("Running network")


def run(args):
    if args.e:
        print(f"Running with name: {args.e}")
    else:
        print("Running")


def setup(args):
    if args.d:
        print(f"Setup for host: {args.d}")
    elif args.e:
        print(f"Setup for name: {args.e}")
    else:
        print("Running setup")


def main(session: Session):
    parser = argparse.ArgumentParser(
        description=utils.get_text(CONSOLE_TXT_DIR / "tracr_description.txt"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=utils.get_text(CONSOLE_TXT_DIR / "about_tracr.txt"),
    ),
    subparsers = parser.add_subparsers(title="SUBMODULES")

    # parser for "device"
    parser_device = subparsers.add_parser(
        "device", help="Local device setup and configuration"
    )
    # add sub-sub parsers for the device module
    device_subparsers = parser_device.add_subparsers(title="DEVICE MODULE COMMANDS")
    parser_device_ls = device_subparsers.add_parser("ls", help="List devices")
    parser_device_ls.set_defaults(func=device_ls)

    parser_device_add = device_subparsers.add_parser(
        "add", help="add new devices and configure them for experiments"
    )
    parser_device_add.add_argument(
        "-w",
        "--wizard",
        action="store_true",
        help="use a wizard to help with adding the new device",
        dest="wizard",
    )
    parser_device_add.add_argument(
        "-a",
        "--host",
        help="specify the hostname or IP address of the device to add",
        nargs=1,
        dest="host",
    )
    parser_device_add.add_argument(
        "-u",
        "--user",
        help="specify the username to connect with via SSH",
        nargs=1,
        dest="user",
    )
    parser_device_add.add_argument(
        "-p",
        "--pass",
        help="specify the password for SSH connections to the device",
        nargs=1,
        dest="pw",
    )
    parser_device_add.add_argument(
        "-k",
        "--keys",
        help="specify the public and private keys, separated by a space",
        nargs=2,
        dest="keys",
    )
    parser_device_add.add_argument(
        "-n",
        "--nickname",
        help="assign a nickname to the device",
        nargs=1,
        dest="nickname",
    )
    parser_device_add.add_argument(
        "-d",
        "--description",
        help="give a description to the device",
        nargs=1,
        dest="description",
    )
    parser_device_add.set_defaults(func=device_add)

    # Parser for "experiment"
    parser_experiment = subparsers.add_parser(
        "experiment", help="Manage and run experiments"
    )
    # Add sub-sub parsers for the experiment module
    exp_subparsers = parser_experiment.add_subparsers(help="experiment submodule help")
    parser_experiment_ls = exp_subparsers.add_parser(
        "ls", help="list experiments and experiment data"
    )
    parser_experiment_ls.add_argument(
        "-n",
        "--name",
        help="list the experiment names",
        action="store_true",
        dest="name",
    )
    parser_experiment_ls.add_argument(
        "-l",
        "--last-run",
        help="list the last time each experiment was run",
        action="store_true",
        dest="last_run",
    )
    parser_experiment_ls.add_argument(
        "-s",
        "--settings",
        help="list the settings for each experiment",
        action="store_true",
        dest="settings",
    )
    parser_experiment_ls.set_defaults(func=experiment_ls)

    parser_experiment_run = exp_subparsers.add_parser("run", help="run an experiment")
    parser_experiment_run.add_argument(
        "name", nargs=1, help="the name of the experiment to be run"
    )
    parser_experiment_run.add_argument(
        "-l",
        "--local",
        help="run the experiment locally using simulated devices",
        action="store_true",
        dest="local",
    )
    parser_experiment_run.add_argument(
        "-o",
        "--output",
        help="specify a location for performance logging output",
        nargs="?",
        dest="output",
    )
    parser_experiment_run.add_argument(
        "-p",
        "--preset",
        help="use a preset runtime config for this launch",
        nargs="?",
        dest="preset",
    )
    parser_experiment_run.set_defaults(func=experiment_run)

    parser_experiment_add = exp_subparsers.add_parser(
        "add", help="add a new experiment"
    )
    parser_experiment_add.add_argument(
        "name", nargs=1, help="the name of the experiment to create"
    )
    parser_experiment_add.set_defaults(func=experiment_add)

    # Parser for 'setup'
    parser_setup = subparsers.add_parser("setup", help="Guided initial setup")

    # Add sub-sub parsers for the setup module
    setup_subparsers = parser_setup.add_subparsers(title="SETUP MODULE COMMANDS")
    parser_setup_controller = setup_subparsers.add_parser(
        "controller", help="Set up the controller"
    )
    parser_setup_controller.add_argument(
        "-r",
        "--reset",
        action="store_true",
        dest="reset",
        help="reset the controller configuration",
    )
    parser_setup_controller.add_argument(
        "-o",
        "--overwrite",
        nargs="+",
        dest="overwrite",
        help="controllerParam=overwriteValue",
    )
    parser_setup_controller.set_defaults(func=setup_controller)

    parser_setup_device = setup_subparsers.add_parser("device", help="Set up a device")
    # TODO: add arguments for device setup
    parser_setup_device.set_defaults(func=setup_device)

    parser_setup_experiment = setup_subparsers.add_parser(
        "experiment", help="Set up an experiment"
    )
    # TODO: add arguments for experiment setup
    parser_setup_experiment.set_defaults(func=setup_experiment)

    args = parser.parse_args()
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Start the session
    session = Session()
    session.log("debug", "tracr CLI run as main. Logging setup complete.")

    # Run the main function
    main(session)
