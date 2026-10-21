"""
run_time_tests.py

Sends ready-made search phrases to the TIME call of the LLM and prints what
it writes, and how long it took. Nothing is searched in Elasticsearch, and
Elasticsearch doesn't need to be running (the first/last alert are fixed below).

Run it from the project folder (Ollama must be running):
    python run_time_tests.py

For each phrase it prints:
    time       : the window exactly as the model wrote it
    after code : the window after clean_time() (only shown if it changed)

The model is told it is NOW (below), not the real time, so the expected
answers stay the same whatever day you run this. The expected window is
written as a comment next to each phrase, so you can compare by eye.
"at" answers have the same time in from and to (the query builder widens it).
"""

import contextlib
import io
import json
import time
from datetime import datetime

# The parsers print their prompts when they load. We hide that here,
# so the screen only shows the results.
with contextlib.redirect_stdout(io.StringIO()):
    from llm_parser import ask_attributes
    from time_parser import ask_time, clean_time

from config import OLLAMA_MODEL, OLLAMA_THINK, OLLAMA_TIME_MODEL

# Pretend it is Tuesday 20 October 2026, 12:42 ...
NOW = datetime(2026, 10, 20, 12, 42)
# ... and that the stored alerts go from 1 Nov 2025 09:00 to 20 Oct 2026 12:30.
DATA_RANGE = (datetime(2025, 11, 1, 9, 0), datetime(2026, 10, 20, 12, 30))

# True = also ask the attribute call for every phrase (twice as slow).
ALSO_ATTRIBUTES = False
# True = also print what the model thought before answering (can be long).
SHOW_THINKING = False

PHRASES = [
    "a person wearing red at 2 o'clock on 15 oct",  # 2026-10-15 02:00 (no am/pm: as it is)
    "woman in brown today",  # 2026-10-20 00:00 - 23:59
    "a kid on camera 3 on friday",  # 2026-10-16 00:00 - 23:59
    "a man in red last sunday",  # 2026-10-18 00:00 - 23:59
    "someone in a blue jacket on 16 october 2026",  # 2026-10-16 00:00 - 23:59
    "a girl at 12 on 16 oct",  # 2026-10-16 12:00
    "a boy in the afternoon",  # 2026-10-20 12:00 - 17:00
    "man in black yesterday evening",  # 2026-10-19 17:00 - 21:00
    "people on camera 1 last night",  # 2026-10-19 21:00 - 2026-10-20 06:00
    "a guy at night on 13 oct",  # 2026-10-13 21:00 - 2026-10-14 06:00
    "a man at 4:45 pm",  # 2026-10-19 16:45 (today's is later than now)
    "person in gray at 21:30",  # 2026-10-19 21:30 (today's is later than now)
    "a woman around 5 in the morning",  # 2026-10-20 05:00
    "a woman at 8 in the evening on saturday",  # 2026-10-17 20:00
    "anyone in white after 10 pm yesterday",  # 2026-10-19 22:00 - 23:59
    "a man before 11 am today",  # 2026-10-20 00:00 - 11:00
    "a lady in pink from 1 to 3 pm on 14/10",  # 2026-10-14 13:00 - 15:00
    "someone on camera 2 between 9 and 11",  # 2026-10-20 09:00 - 11:00 (no am/pm: as it is)
    "woman in black in the last 3 hours",  # 2026-10-20 09:42 - 12:42
    "a child in the past 15 minutes",  # 2026-10-20 12:27 - 12:42
    "anyone in the last 2 days",  # 2026-10-18 12:42 - 2026-10-20 12:42
    "a man in the last hour",  # 2026-10-20 11:42 - 12:42
    "a girl in green earlier",  # no time
    "man in a white shirt on camera 4",  # no time
    "a man on 25 december",  # 2025-12-25 00:00 - 23:59 (2026's is after today)
    "a woman in black on 15 oct 2025",  # 2025-10-15 00:00 - 23:59 (said exactly: stays)
    "someone in red the day before yesterday",  # 2026-10-18 00:00 - 23:59
    "a guy two days ago at noon",  # 2026-10-18 12:00
    # counting back (the lead's point 3)
    "a girl in green or white shirt since last 15 days",  # 2026-10-05 12:42 - 2026-10-20 12:42
    "a man in black 15 days back",  # 2026-10-05 00:00 - 23:59
    "a woman in the last 30 days",  # 2026-09-20 12:42 - 2026-10-20 12:42
    "someone at 2 on 10 oct",  # 2026-10-10 02:00 (no am/pm: as it is)
]

print(f"Time model: {OLLAMA_TIME_MODEL}, thinking: {OLLAMA_THINK}")
if ALSO_ATTRIBUTES:
    print(f"Attributes model: {OLLAMA_MODEL}")
print("Pretending now is", NOW.strftime("%A %Y-%m-%d %H:%M"))
print("Pretending the alerts go from", DATA_RANGE[0], "to", DATA_RANGE[1])
print()

total = 0.0
for number, phrase in enumerate(PHRASES, start=1):
    started = time.perf_counter()
    answer, thinking = ask_time(phrase, NOW, DATA_RANGE)
    seconds = time.perf_counter() - started
    total += seconds

    print(f"{number}. {phrase}")
    print(f"   time ({seconds:5.1f} s): {json.dumps(answer)}")
    # What clean_time() turns it into. Only shown if it changed something.
    fixed = clean_time(answer, NOW)
    written = [answer["time"]] if isinstance(answer, dict) and "time" in answer else []
    if fixed != written:
        print(f"   after code     : {json.dumps(fixed)}")
    if ALSO_ATTRIBUTES:
        attributes, _ = ask_attributes(phrase)
        print(f"   attributes     : {json.dumps(attributes)}")
    if SHOW_THINKING and thinking:
        print(f"   thinking: {thinking.replace(chr(10), ' ')}")
    print()

print(f"Average: {total / len(PHRASES):.1f} s per phrase (time call only)")
