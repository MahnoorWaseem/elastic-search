"""
config.py

All connection settings in one place, so no other file has a URL or a
password written inside it. Change the values here to match your setup.

The MQTT settings are read from the .env file next to this one, so you can
change the broker without touching any code.
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
INDEX_NAME = "nexa-alerts"

# ---------------------------------------------------------------------------
# Ollama (runs the Qwen model on your machine)
# ---------------------------------------------------------------------------

# Ollama always listens on port 11434 unless you changed it.
# OLLAMA_URL = "http://192.168.100.4:11434"
OLLAMA_URL = "http://localhost:11434"


# Every search asks the LLM two separate questions, each with its own
# short prompt: one for the attributes (gender, colours, age, camera) and
# one for the time. Each can use a different model.
# The exact model names you downloaded with "ollama pull":
#     ollama pull qwen3:4b-instruct-2507-q4_K_M
#     ollama pull qwen3.5:2b

# OLLAMA_MODEL = "hf.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:IQ2_M"  # attributes (llm_parser.py)
# OLLAMA_TIME_MODEL = "hf.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:IQ2_M"  # time (time_parser.py)

OLLAMA_MODEL = "qwen3.5:2b"  # attributes (llm_parser.py)
# OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # attributes (llm_parser.py)
OLLAMA_TIME_MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # time (time_parser.py)

# Let the model think (write its reasoning) before it writes the JSON.
# Off: on a CPU, the 2B model thought for ~100 s and never reached the JSON.
# The Instruct models can't think at all, so this MUST be False for them.

OLLAMA_URL = "http://192.168.100.63:11434"
OLLAMA_THINK = False

# ---------------------------------------------------------------------------
# Time search
# ---------------------------------------------------------------------------

# The time zone the users and cameras are in. "today", "at 2 pm", ... all
# mean Pakistan time. Elasticsearch stores times in UTC and converts.
TIMEZONE = "Asia/Karachi"

# How many years back a search may go. With 1, in 2026 you can search
# 2025 and 2026. A date outside this range is ignored (logged as a warning).
MAX_YEARS_BACK = 1

# ---------------------------------------------------------------------------
# MQTT (where the CV pipeline publishes new alerts)
# Values come from .env. The second value is used if .env doesn't have it.
# ---------------------------------------------------------------------------

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.100.127")
# Everything in .env is text, so we turn the port into a number.
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "person_attributes/results")
