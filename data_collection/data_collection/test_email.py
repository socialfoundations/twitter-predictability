from dotenv import load_dotenv
import os, logging
from logging import basicConfig, info
from ssl_smtp_handler import SSLSMTPHandler

load_dotenv()

handler = SSLSMTPHandler(
    mailhost="smtp.gmail.com",
    fromaddr=os.environ["EMAIL_FROM"],
    toaddrs=os.environ["EMAIL_TO"],
    credentials=(os.environ["EMAIL_FROM"], os.environ["EMAIL_PASSWORD"]),
    subject="Hello from python",
)
handler.setLevel("INFO")

# basicConfig(handlers=[handler], level='INFO')
# info('This is an example message')

logger = logging.getLogger("test")
logger.addHandler(handler)
logger.setLevel("INFO")
logger.info("This is an example message.")

print("Done.")
