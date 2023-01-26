import os
import colorlog
import logging
from logzio.handler import LogzioHandler

class ExtraFieldsFormatter(colorlog.ColoredFormatter):
    def __init__(self, *args, **kwargs):
        self.__orig_fmt = args[0]

        super().__init__(*args, **kwargs)

    def format(self, record):
        extra_keys = ExtraKeysResolver.get_extra_keys(record)

        if not extra_keys:
            return super().format(record)

        def map_placeholder(field_name):
            return "{}: %({})s".format(field_name, field_name)

        extra_keys_placeholders = list(map(map_placeholder, extra_keys))

        self.__set_format(self.__orig_fmt + "\n" + "{" + ", ".join(extra_keys_placeholders) + "}")
        formated = super().format(record)
        self.__set_format(self.__orig_fmt)

        return formated

    def __set_format(self, fmt: str):
        self._fmt = fmt
        self._style = logging.PercentStyle(self._fmt)

class ExtraKeysResolver:

    ignored_record_keys = [
        "name",
        "msg",
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    ]

    @staticmethod
    def get_extra_keys(record):
        return record.__dict__.keys() - ExtraKeysResolver.ignored_record_keys

def create_logger(name):
    handler = colorlog.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter('%(log_color)s%(message)s'))

    logger = colorlog.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    app_env = os.environ["APP_ENV"]

    logz_formatter = logging.Formatter('{"app_env": "' + app_env + '"}', validate=False)

    logz_handler = LogzioHandler(os.environ["LOGZIO_TOKEN"], url="https://listener-eu.logz.io:8071")
    logz_handler.setLevel(logging.INFO)
    logz_handler.setFormatter(logz_formatter)
    logger.addHandler(logz_handler)

    return logger
