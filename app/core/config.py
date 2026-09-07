import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv("APP_NAME", "Dataset Research")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")