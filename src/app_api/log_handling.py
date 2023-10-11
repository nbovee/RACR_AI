import atexit
import random
import logging
import logging.handlers
import socketserver
import struct
import pickle
from rich.console import Console

from app_api import utils


MAIN_LOG_FP = utils.get_repo_root() / "src" / "app_api" / "AppData" / "app.log"


class ColorByDeviceFormatter(logging.Formatter):
    """
    The formatter used by the central logger's socket handler (which handles all the logs from)
    remote nodes as they initialize and run). Each device is assigned a random color from the list
    COLORS the first time a log is received from that device. All of their logs will be rendered
    with that color to help with readability.
    """
    COLORS: list[str] = [
        "light_coral", "orchid2", "dark_sea_green2", "steel_blue1", "gold1"
    ]

    device_color_map: dict[str, str] = {"MAIN": "white"}
    console: Console = Console()

    def format(self, record):
        log_message = super().format(record)
        device = record.name.split('_')[0].upper()
        color = self.get_color(device)
        message = self.console.render(f"[{color}]{log_message}[/]")
        return message

    def get_color(self, device: str) -> str:
        if device not in self.device_color_map.keys():
            available_color = random.choice(
                [c for c in self.COLORS if c not in self.device_color_map.values()]
            )
            assert available_color is not None
            self.device_color_map[device] = available_color

        return self.device_color_map[device]


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        logger = logging.getLogger("main_logger")
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            length = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(length)
            record = logging.makeLogRecord(pickle.loads(chunk))
            logger.handle(record)


class ConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        log_message = self.format(record)
        device = record.name.split("_")[0].upper()

        if device == "MAIN":
            print(f"OBSERVER@localhost: {log_message}")
        else:
            print(log_message)


def setup_logging(verbosity: int = 3):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    file_format = "%(asctime)s - %(module)s - %(levelname)s: %(message)s"

    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG)

    # all messages will be logged to this file
    file_handler = logging.FileHandler(MAIN_LOG_FP.expanduser())
    file_handler.setLevel(logging.DEBUG)

    # only messages of the given level or higher will be logged to console
    console_handler = ConsoleHandler()
    console_handler.setFormatter(ColorByDeviceFormatter())
    console_handler.setLevel(levels[min(verbosity, len(levels) - 1)])

    # different formats for file and console logs
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)

    # Adding the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def start_remote_log_server():
    server = socketserver.TCPServer(("localhost", 9000), LogRecordStreamHandler)
    server.serve_forever()
    atexit.register(server.shutdown)
