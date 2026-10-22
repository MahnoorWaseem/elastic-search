"""
time_parser.py

The TIME part of a search: "woman in black last night" -> when?

Time has its own LLM call with its own short prompt, which only talks about
time. (The attributes, like "woman in black", come from a separate call in
llm_parser.py.)

For every search, llm_parser.parse_query() calls two functions from here:
    ask_time(text, now)      -> asks the model, returns its answer
    clean_time(answer, now)  -> checks the answer, returns [window] or []

The system prompt is RULES, with the facts for this search put in at {FACTS}
(by facts_text(), on every search): now, today, yesterday, the first and the
last alert stored in Elasticsearch, and the most recent date of each weekday.
Everything in RULES before {FACTS} stays the same between searches, so Ollama
can reuse that part.

There is NO time-zone conversion anywhere: the model writes times exactly
as the user says them, and they are compared as they are with the stored
timestamps. The model answers with ONE window:
    {"time": {"from_date": "2026-10-20", "from_time": "00:00",
              "to_date":   "2026-10-20", "to_time":   "23:59"}}
"""

import logging
from datetime import datetime, timedelta, timezone

from config import INDEX_NAME, OLLAMA_TIME_MODEL
from es_index import get_client
from ollama_client import ask_ollama

log = logging.getLogger("time_parser")

# Our own Elasticsearch connection, only used to find the first and last alert.
# (It doesn't connect until the first question, so making it here is cheap.)
_es = get_client()


# ---------------------------------------------------------------------------
# The facts: now, and the first and last alert
# ---------------------------------------------------------------------------


def current_time():
    """
    Why:     "today", "yesterday" and "at 9 pm hasn't happened yet" need "now".
             It must be on the SAME clock as the stored alerts, and the CV
             pipeline sends their times in UTC (+00:00). So "now" is read on
             the UTC clock too. Nothing is converted: it's just the clock the
             data uses. (Your computer's clock is 5 hours ahead of it, so
             "today" would start 5 hours off with it.)
    Used by: ask_time() and llm_parser.parse_query(), on every real search.

    Returns now on the UTC clock, without seconds, e.g. 2026-10-22 06:41.
    """
    now = datetime.now(timezone.utc)
    return now.replace(tzinfo=None, second=0, microsecond=0)


def stored_time(milliseconds):
    """
    Why:     Elasticsearch gives dates as milliseconds since 1970. This turns
             them back into the date and time exactly as they are stored.
    Used by: first_and_last_alert().

    1792091381008 -> 2026-10-15 19:09   (the same as the stored "2026-10-15T19:09:41Z")
    """
    moment = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
    return moment.replace(tzinfo=None, second=0, microsecond=0)


def first_and_last_alert():
    """
    Why:     the model must know which dates have alerts at all, so it picks
             dates inside that range when the user doesn't say them exactly.
    Used by: ask_time(), on every search (new alerts keep arriving).

    Asks Elasticsearch for the oldest (min) and newest (max) alert timestamp.
    Returns them as stored, e.g. (2026-10-15 19:09, 2026-10-21 13:14),
    or (None, None) if there are no alerts or Elasticsearch can't be reached.
    When the user doesn’t say a date exactly, the model picks dates inside this range. When the user does say a date exactly, it keeps it, even if it’s outside the range.
    """
    try:
        response = _es.search(
            index=INDEX_NAME,
            size=0,  # we don't want any alerts back, only the two numbers
            aggs={
                "first": {"min": {"field": "timestamp"}},
                "last": {"max": {"field": "timestamp"}},
            },
        )
    except Exception as error:
        log.warning("couldn't get the first and last alert: %s", error)
        return None, None

    first = response["aggregations"]["first"]["value"]  # None when there are no alerts
    last = response["aggregations"]["last"]["value"]
    if first is None or last is None:
        return None, None
    return stored_time(first), stored_time(last)
    # "aggregations": {
    #   "first": {"value": 1792091381008, "value_as_string": "2026-10-15T19:09:41.008Z"},
    #   "last":  {"value": 1792091381xxx, ...}
    # }
    # stored_time() turns 1792091381008 back into 2026-10-15 19:09.


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

# Part 1: the rules. Always the same, so Ollama can reuse them between searches.
RULES = """You are a temporal time-filter extractor.

TASK
Extract ONLY dates and times from user text.
This is text parsing, NOT a camera search.
Ignore people, clothing, colors, cameras, locations,
and leading list numbers.

OUTPUT
Return ONLY valid JSON:
{"time":{"from_date":"YYYY-MM-DD","to_date":"YYYY-MM-DD","from_time":"HH:MM","to_time":"HH:MM"}}

Return {} only if no temporal expression exists or it
cannot be represented by this schema.
No explanations, reasoning, or extra fields.
Use 24-hour HH:MM. Write every time exactly as the user says it.
Never convert or adjust times for a time zone.

SEMANTICS
- Dates define an inclusive date range.
- Times define a DAILY filter applied to every date.
- Never treat the output as one continuous interval.
- No clock time or daypart -> 00:00 to 23:59.
- Overnight filters may have from_time > to_time.
  Each selected date is the start of the overnight interval.
- Explicit dates, times, and operators take precedence
  over inferred defaults.

STEP 1: EXTRACT CLOCK TIMES

OPERATOR PRIORITY
Apply the most specific matching operator:

- "at X" / standalone exact clock time -> X to X.
- "after X" -> X to 23:59, SAME DATE.
- "before X" -> 00:00 to X, SAME DATE.
- "between X and Y" -> X to Y.
- "from X to Y" -> X to Y.
- "around/about/roughly X" -> X ± 30 minutes.

CRITICAL EXACT-TIME RULE
- A single exact clock time MUST have identical
  from_time and to_time.
- A date, weekday, or duration NEVER expands an exact
  time into a longer clock interval.
- "at 5pm" -> 17:00 to 17:00.
- "last 5 days at 5pm" -> daily 17:00 to 17:00.
- "at 8 in the evening" -> 20:00 to 20:00.
- Do not replace an exact clock time with a daypart range.

CLOCK INTERPRETATION
- 5 pm = 17:00; 5 am = 05:00.
- No AM/PM -> literal hour: 5 = 05:00.
- "7 in the evening" = 19:00.
- "8 in the morning" = 08:00.
- "18:30" = 18:30.
- Explicit clock operators override daypart rules.
- "after 10 pm" -> 22:00 to 23:59,
  NOT 22:00 to 06:00.
- "around 5" -> 04:30 to 05:30,
  NOT 05:00 to 23:59.

DAYPARTS
Use only when no more specific clock interval is given.
A daypart alone IS valid temporal information.

- morning -> 06:00 to 12:00
- afternoon -> 12:00 to 17:00
- evening -> 17:00 to 21:00
- night -> 21:00 to 06:00
- last night -> Yesterday, 21:00 to 06:00
- noon -> 12:00 to 12:00
- midnight -> 00:00 to 00:00

OVERNIGHT
- "10 pm to 3 am" -> 22:00 to 03:00.
- Only explicit crossing-midnight ranges, approximate
  windows crossing midnight, or night dayparts may
  produce from_time > to_time.
- Each date is the START date of the overnight window.

STEP 2: RESOLVE DATES

PRIORITY
1. Explicit calendar date or date range
2. Relative date or weekday
3. Relative duration
4. Default date inference

DATES
- today -> Today; yesterday -> Yesterday.
- day before yesterday -> Today minus 2 days.
- Dates use DD/MM/YYYY or written month names.
- Missing year -> Today's year, or previous year if future.
- Preserve explicitly specified years.
- Explicit date ranges override inferred dates.

WEEKDAYS
- "Monday" -> most recent Monday, including Today.
- "last Monday" -> most recent Monday before Today.
- "this Monday" -> Monday in the current calendar week.
- Use WEEKDAY LOOKUP when available.
- Never substitute another weekday.

DURATIONS
Historical durations are lookbacks ending Today:

- last/past/for N days -> Today minus N days to Today.
- N days ago/back -> ONLY the day Today minus N days.
- last/past/for N weeks -> Today minus (N * 7) days
  through Today.
- a week -> Today minus 7 days through Today.
- last/past/for N months -> Today minus N calendar months
  through Today.
- a month -> Today minus 1 calendar month through Today.
- last/past/for N years -> Today minus N calendar years
  through Today.
- a year -> Today minus 1 calendar year through Today.

CRITICAL DURATION RULES
- "last 5 weeks" = subtract exactly 35 calendar days.
- "last 15 days" = subtract exactly 15 calendar days.
- "15 days ago" means one day, not a 15-day range.
- Do NOT count Today as day 1 when subtracting N days.
- Calendar months/years are not fixed day counts.
- Clamp invalid month-days to the last valid day.
- Preserve explicit daily clock intervals with durations.
- Rolling hours/minutes not representable accurately
  by the daily-filter schema -> {}.

DEFAULT DATE
Only when no date or duration is specified:

- Exact time <= Now -> Today; otherwise Yesterday.
- For a daytime range or daypart, compare its END
  against Now.
- If the end is later than Now -> Yesterday.
- Otherwise -> Today.
- For overnight ranges, choose the most recent
  completed interval.
- Never return {} merely because the date is missing.
- Vague words alone ("recently", "earlier") -> {}.

ALERT RULES
- Alert availability is NOT the requested time.
- Use alert dates only to resolve genuine ambiguity.
- Never override explicit dates, weekdays, or durations.
- Never clip a requested interval to stored alert dates.

{FACTS}

STEP 3: GENERATE JSON

EXAMPLES

Input: since last 5 days at 5pm
Output: {"time":{"from_date":"2026-10-17","to_date":"2026-10-22","from_time":"17:00","to_time":"17:00"}}

Input: at 8 in the evening on Saturday
Output: {"time":{"from_date":"2026-10-17","to_date":"2026-10-17","from_time":"20:00","to_time":"20:00"}}

Input: since last 5 weeks
Output: {"time":{"from_date":"2026-09-17","to_date":"2026-10-22","from_time":"00:00","to_time":"23:59"}}

Input: since 5 weeks
Output: {"time":{"from_date":"2026-09-17","to_date":"2026-10-22","from_time":"00:00","to_time":"23:59"}}

Input: anyone in white after 10 pm yesterday
Output: {"time":{"from_date":"2026-10-21","to_date":"2026-10-21","from_time":"22:00","to_time":"23:59"}}

Input: a boy in the afternoon
Output: {"time":{"from_date":"2026-10-21","to_date":"2026-10-21","from_time":"12:00","to_time":"17:00"}}

Input: a woman around 5 today
Output: {"time":{"from_date":"2026-10-22","to_date":"2026-10-22","from_time":"04:30","to_time":"05:30"}}

Input: lady from 5 pm to 10 pm for a month
Output: {"time":{"from_date":"2026-09-22","to_date":"2026-10-22","from_time":"17:00","to_time":"22:00"}}

Input: person from 10 pm to 3 am last night
Output: {"time":{"from_date":"2026-10-21","to_date":"2026-10-21","from_time":"22:00","to_time":"03:00"}}

Input: woman wearing red
Output: {}

FINAL VALIDATION
- Return JSON only with exactly four fields, or {}.
- Dates: YYYY-MM-DD; times: HH:MM.
- from_date must not exceed to_date.
- Exact clock times must have equal start and end.
- "after X" ends at 23:59, never at 06:00.
- N weeks means exactly N * 7 days.
- Explicit times and dates override defaults.
- Never add explanations or extra fields.
"""


def facts_text(now, first, last):
    """
    Why:     the model must know today's date and time to understand "today",
             "monday" or "last 15 days", and the first/last alert (ALERT RULES).
             These change with every search, so they are made here, never typed in.
    Used by: build_prompt(), on every search.

        FACTS
        Now: Thursday 2026-10-22 06:41
        Today: 2026-10-22
        Yesterday: 2026-10-21
        First alert stored: 2026-10-15 19:09
        Last alert stored: 2026-10-21 13:14

        WEEKDAY LOOKUP (most recent)
        Monday: 2026-10-19
        ...
    """
    today = now.date()
    lines = [
        "FACTS",
        f"Now: {now:%A %Y-%m-%d %H:%M}",
        f"Today: {today:%Y-%m-%d}",
        f"Yesterday: {today - timedelta(days=1):%Y-%m-%d}",
    ]
    if first and last:
        lines.append(f"First alert stored: {first:%Y-%m-%d %H:%M}")
        lines.append(f"Last alert stored: {last:%Y-%m-%d %H:%M}")
    else:
        lines.append("First and last alert stored: unknown")

    # The most recent date of each weekday, today included.
    lines += ["", "WEEKDAY LOOKUP (most recent)"]
    for weekday in range(7):  # 0 = Monday ... 6 = Sunday
        days_back = (today.weekday() - weekday) % 7
        date = today - timedelta(days=days_back)
        lines.append(f"{date:%A}: {date:%Y-%m-%d}")
    return "\n".join(lines)


def build_prompt(now, first, last):
    """
    Why:     the facts change with every search, so the prompt is built fresh
             each time: the rules with this search's facts put in at {FACTS}.
    Used by: ask_time().
    """
    return RULES.replace("{FACTS}", facts_text(now, first, last))


# The shape the answer must have: {"time": {one window}}, or {} for no time.
# Ollama only lets the model write JSON that fits this. Every window has all
# four parts. They are plain text here; clean_time() checks they are real.
_text = {"type": "string"}
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "time": {
            "type": "object",
            "properties": {
                "from_date": _text,
                "to_date": _text,
                "from_time": _text,
                "to_time": _text,
            },
            "required": ["from_date", "from_time", "to_date", "to_time"],
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Asking the model
# ---------------------------------------------------------------------------


def ask_time(text, now=None, data_range=None):
    """
    Why:     the one LLM call that finds the time in what the user typed.
    Used by: llm_parser.parse_query() on every search, and run_time_tests.py.

    text       : what the user typed, sent as it is (the date is in the prompt)
    now        : the current time; leave it out to use the real time (UTC clock)
    data_range : (first alert, last alert); leave it out to ask Elasticsearch.
                 The test script passes a fixed one, so it needs no Elasticsearch.

    Returns (answer, thinking): the answer exactly as the model wrote it.
    """
    now = now or current_time()
    first, last = data_range or first_and_last_alert()
    prompt = build_prompt(now, first, last)
    return ask_ollama(OLLAMA_TIME_MODEL, prompt, text, JSON_SCHEMA)


# ---------------------------------------------------------------------------
# Checking the answer
# ---------------------------------------------------------------------------


def read_date(text):
    """
    Why:     the model writes dates as text; to check and compare them they
             must become real dates.
    Used by: clean_time().

    "2026-10-20" -> 2026-10-20.  Returns None if it isn't a real date ("2026-02-30", "today").
    """
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def read_clock(text):
    """
    Why:     the same for clock times.
    Used by: clean_time().

    "14:00" -> 14:00.  Returns None if it isn't a real time ("25:00", "5pm").
    """
    if text == "24:00":  # some people write midnight as 24:00
        text = "23:59"
    try:
        return datetime.strptime(text, "%H:%M").time()
    except (ValueError, TypeError):
        return None


def clean_time(answer, now):
    """
    Why:     never search with a broken window, and fix two mistakes small
             models often make with dates.
    Used by: llm_parser.parse_query(), right after ask_time().

    The window is a DAILY filter: every day from from_date to to_date, and on
    each day from from_time to to_time. So dates and times are checked
    separately, and from_time may be later than to_time (22:00 to 03:00 is a
    real overnight window, it is NOT swapped).

      - no time in the answer             -> []
      - not a real date or time           -> [] (a warning says why)
      - missing to_date                   -> the same day as from_date
      - missing times                     -> the whole day, 00:00 to 23:59
      - to_date before from_date          -> the dates are swapped
      - a from_date after today           -> one year earlier ("25 december" in october)
      - only today, and its time hasn't started yet -> yesterday ("at 9 pm" asked at 12:53)

    Returns a list with the one checked window (the query builder takes a list).
    It doesn't check the first/last alert: a window with no alerts in it simply
    finds 0 alerts, which is the right answer.
    """
    window = answer.get("time") if isinstance(answer, dict) else None
    if not isinstance(window, dict):
        return []

    from_date = read_date(window.get("from_date"))
    to_date = read_date(window.get("to_date")) or from_date
    from_time = read_clock(window.get("from_time") or "00:00")
    to_time = read_clock(window.get("to_time") or "23:59")
    if None in (from_date, to_date, from_time, to_time):
        log.warning("time is not a real date/time, not used: %s", window)
        return []

    if to_date < from_date:
        from_date, to_date = to_date, from_date

    today = now.date()
    if from_date > today:
        # No alerts can exist after today, so the user meant last year.
        try:
            from_date = from_date.replace(year=from_date.year - 1)
            to_date = to_date.replace(year=to_date.year - 1)
        except ValueError:  # 29 february in a year without one
            return []
    elif from_date == to_date == today and from_time > now.time():
        # Only today, but that time hasn't come yet, so the user meant yesterday.
        from_date = to_date = today - timedelta(days=1)

    return [
        {
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "from_time": from_time.strftime("%H:%M"),
            "to_time": to_time.strftime("%H:%M"),
        }
    ]
