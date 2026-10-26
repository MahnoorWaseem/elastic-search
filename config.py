"""
config.py

All settings are read from the .env file next to this one.
The second argument of each getenv() is the default, used if .env lacks it.
After changing .env, restart the server (--reload only watches .py files).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Elasticsearch (leave ES_PASSWORD empty when security is OFF)
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
INDEX_NAME = os.getenv("INDEX_NAME", "nexa-alerts")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_TIME_MODEL = os.getenv("OLLAMA_TIME_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_THINK = _bool("OLLAMA_THINK")

# Time search
MAX_YEARS_BACK = int(os.getenv("MAX_YEARS_BACK", "1"))

# MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.100.127")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "person_attributes/results")

# SeaweedFS
SEAWEED_S3_ENDPOINT = os.getenv("SEAWEED_S3_ENDPOINT", "http://localhost:8333")
SEAWEED_ACCESS_KEY = os.environ.get("SEAWEED_ACCESS_KEY")
SEAWEED_SECRET_KEY = os.environ.get("SEAWEED_SECRET_KEY")
SEAWEED_BUCKET = os.getenv("SEAWEED_BUCKET", "person-images")

# Langfuse keys are not read here: the langfuse SDK reads LANGFUSE_* from
# the environment itself, which load_dotenv() above has already filled.
