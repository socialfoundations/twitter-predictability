import logging, os, sys
from ssl_smtp_handler import SSLSMTPHandler

log_formatter = logging.Formatter(
    "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
)


def set_logger_levels(logger_names, level=logging.DEBUG):
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def log_to_stdout(logger_name, level=None):
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)


def logs_to_stdout(logger_names, level=None):
    for name in logger_names:
        log_to_stdout(name, level)


def log_to_file(logger_name, logfile, level=None):
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(filename=logfile, mode="w")
    handler.setFormatter(log_formatter)
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)


def logs_to_file(logger_names, logdir, level=None):
    for name in logger_names:
        log_to_file(name, logfile=os.path.join(logdir, f"{name}.log"), level=level)


def get_email_logger(subject):
    email_logger = logging.getLogger("email")
    email_logger.setLevel("INFO")

    handler = SSLSMTPHandler(
        mailhost="smtp.gmail.com",
        fromaddr=os.environ["EMAIL_FROM"],
        toaddrs=os.environ["EMAIL_TO"],
        credentials=(os.environ["EMAIL_FROM"], os.environ["EMAIL_PASSWORD"]),
        subject=subject,
    )
    handler.setLevel("INFO")
    handler.setFormatter(log_formatter)

    email_logger.addHandler(handler)
    return email_logger
