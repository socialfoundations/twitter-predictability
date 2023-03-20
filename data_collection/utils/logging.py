import logging, os, sys
from ssl_smtp_handler import SSLSMTPHandler


def log_to_stdout(logger_name, level=None):
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_to_file(logger_name, logfile, level=None):
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    handler = logging.FileHandler(filename=logfile, mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    email_logger.addHandler(handler)
    return email_logger
