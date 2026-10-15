"""
config.py

All connection settings in one place, so no other file has a URL or a
password written inside it. Change the values here to match your Docker setup.
"""

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
