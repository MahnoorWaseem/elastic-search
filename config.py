"""
config.py

All connection settings in one place, so no other file has a URL or a
password written inside it. Change the values here to match your Docker setup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file in this folder and make its lines available
# through os.getenv(). It doesn't overwrite variables already set in Windows.
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------

# Where Elasticsearch is running.
#   Security OFF -> "http://localhost:9200"
#   Security ON  -> "https://localhost:9200"   (note the "s")
ES_URL = "http://localhost:9200"

# Only needed when security is ON. Leave ES_PASSWORD empty ("") when it's OFF.
ES_USERNAME = "elastic"
ES_PASSWORD = ""

# The index (similar to a table in a database) that holds the flattened alerts.
# One index for all pipelines; the "pipeline_id" field tells them apart.
INDEX_NAME = "nexa-alerts"

# ---------------------------------------------------------------------------
# Ollama (runs the Qwen model on your machine)
# ---------------------------------------------------------------------------

# Ollama always listens on port 11434 unless you changed it.
OLLAMA_URL = "http://localhost:11434"

# The exact model name you downloaded with "ollama pull".
OLLAMA_MODEL = "qwen3.5:2b"

# ---------------------------------------------------------------------------
# MQTT (where the CV pipeline publishes new alerts)
# Values come from .env. The second value is used if .env doesn't have it.
# ---------------------------------------------------------------------------

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.100.127")
# Everything in .env is text, so we turn the port into a number.
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "person_attributes/results")
