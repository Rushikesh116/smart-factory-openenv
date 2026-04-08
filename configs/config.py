import os

class Config:
    API_BASE_URL = os.getenv("API_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENV_URL = os.getenv("ENV_URL")

config = Config()
