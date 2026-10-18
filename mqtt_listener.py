"""
mqtt_listener.py

What it does:

Connects to the broker, subscribes to the topic, and for every message runs flatten → save to file → save to Elasticsearch in a background thread, so searches never wait.
If the broker is off when the server starts, it retries every 5 s. If the connection drops later, gmqtt reconnects on its own, and the listener subscribes again.
One bad alert is logged and skipped. It never stops the listener.

Listens to the MQTT topic where the CV pipeline publishes alerts.
For every alert that arrives:
    flatten it  ->  add it to output/flattened_alerts.jsonl  ->  put it in Elasticsearch
so it can be searched about 1 second later.

app.py starts the listener when the server starts and stops it when the
server stops. You never run this file yourself.
"""

import asyncio
import json
import logging
import uuid

from gmqtt import Client as MQTTClient

from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
from es_index import save_to_index
from flattener import flatten_and_save

log = logging.getLogger("mqtt")

# Which pipeline each topic belongs to, so the right flattener is used.
# A new pipeline later = one more line here (and its flattener).
PIPELINE_FOR_TOPIC = {
    MQTT_TOPIC: "person_attributes",
}

# Filled in by start_listening().
_client = None
_es = None
_connect_task = None


def handle_message(topic, payload):
    """
    The real work for one MQTT message. It runs in a background thread
    (see on_message), so saving alerts never slows down a search.
    """
    # Show the message exactly as it arrived, before we change anything.
    # payload is bytes, so we turn it into text first.
    log.info("raw alert on %s: %s", topic, payload.decode("utf-8", errors="replace"))

    pipeline = PIPELINE_FOR_TOPIC.get(topic)
    if pipeline is None:
        log.warning("message on unknown topic %s, skipped", topic)
        return

    try:
        data = json.loads(payload)
    except ValueError:
        log.warning(
            "message on %s is not valid JSON, skipped: %r", topic, payload[:200]
        )
        return

    # Usually one alert per message. If the pipeline ever sends a list,
    # we handle each alert in it.
    alerts = data if isinstance(data, list) else [data]

    for alert in alerts:
        try:
            doc = flatten_and_save(alert, pipeline)
            save_to_index(_es, doc)
            # One short line per alert, so you can watch them arrive.
            # "None" means the value was unknown and wasn't stored.
            log.info(
                "alert %s saved (%s, gender %s, age_group %s, top %s, bottom %s)",
                doc["alert_id"],
                doc.get("camera_id"),
                doc.get("gender"),
                doc.get("age_group"),
                doc.get("upper_color"),
                doc.get("lower_color"),
            )
        except Exception as error:
            # One broken alert must not stop the listener: log it, keep going.
            log.warning("could not save alert %r: %s", alert, error)


def on_connect(client, flags, rc, properties):
    """Called every time we connect, including after a reconnect."""
    log.info("connected to MQTT broker %s:%s", MQTT_BROKER, MQTT_PORT)
    # Subscribe here (not once at start), so we subscribe again after a
    # reconnect. qos=1: the broker re-sends a message if it wasn't received.
    client.subscribe(MQTT_TOPIC, qos=1)
    log.info("listening on topic '%s'", MQTT_TOPIC)


def on_disconnect(client, packet, exc=None):
    # gmqtt reconnects by itself every few seconds; we just say what happened.
    log.warning("lost connection to the MQTT broker, reconnecting...")


async def on_message(client, topic, payload, qos, properties):
    """
    Called by gmqtt for every message. Saving to the file and Elasticsearch
    waits on the disk and the network, so we do it in a background thread.
    That keeps the server free to answer searches at the same time.
    """
    await asyncio.to_thread(handle_message, topic, payload)
    return 0  # 0 = "message received OK" for the broker


async def _connect_with_retry():
    """
    gmqtt reconnects on its own after a lost connection, but NOT if the very
    first connection fails (for example, the broker machine is off when the
    server starts). So we keep trying every 5 seconds until it works.
    """
    while True:
        try:
            await _client.connect(MQTT_BROKER, MQTT_PORT)
            return
        except Exception as error:
            log.warning(
                "can't reach MQTT broker %s:%s (%s). Trying again in 5 seconds...",
                MQTT_BROKER,
                MQTT_PORT,
                error,
            )
            await asyncio.sleep(5)


async def start_listening(es):
    """Called by app.py when the server starts."""
    global _client, _es, _connect_task
    _es = es

    # Every client needs its own name on the broker. If two used the same
    # name, the broker would kick one off. --reload starts a new copy of the
    # server on every save, so we add a random part to the name.
    _client = MQTTClient(f"nexa-search-{uuid.uuid4().hex[:8]}")
    _client.on_connect = on_connect
    _client.on_disconnect = on_disconnect
    _client.on_message = on_message

    # Connect in the background, so the server (and search) starts right away
    # even if the broker can't be reached yet.
    _connect_task = asyncio.create_task(_connect_with_retry())


async def stop_listening():
    """Called by app.py when the server stops."""
    if _connect_task:
        _connect_task.cancel()
    if _client and _client.is_connected:
        await _client.disconnect()
