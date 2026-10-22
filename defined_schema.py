"""
defined_schema.py

The ONE list of everything a user can search for, and the allowed values.

Why one list?
Many parts of the project need to know the searchable fields:
the flattener, (flattener.py)
the Elasticsearch index,
the LLM prompt,
the validator
and the query builder.
If each part had its own copy, they would slowly stop matching. So they all
read from this file. To add a field (or a new pipeline later), change it here.

Rule: allowed values are EXACTLY what the CV pipeline writes, letter for letter.
"""

# All colours the colour model can output.
# Copied from the alerts: details.models.crops.palette
COLORS = [
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "maroon",
    "orange",
    "brown",
    "beige",
    "yellow",
    "olive",
    "green",
    "dark green",
    "teal",
    "light blue",
    "blue",
    "navy",
    "purple",
    "pink",
]

# Colour families: when the LLM writes a broad colour, the query builder
# also searches for its shades. Works for color, upper_color and lower_color,
# in both include and exclude:
#   include "red"  -> red OR maroon
#   exclude "red"  -> not red AND not maroon
#   include "navy" -> navy only (a specific shade stays specific)
# Colours not listed here (black, white, orange, ...) only match themselves.
# To change a family, edit only this dictionary.
# Without the check, the server starts normally. When someone searches “red”, the shades are added (["red", "marron"]). But pick_values() only keeps values that are in COLORS, so it throws "marron" away. “Red” then searches only red, and maroon shirts are missed. The only sign is a warning line in the terminal on every search, which is easy to overlook
COLOR_FAMILIES = {
    "red": ["maroon"],
    "blue": ["light blue", "navy"],
    "green": ["dark green", "olive"],
    "gray": ["silver"],
    "brown": ["beige"],
}

# Safety check when the file loads: a typo above (e.g. "marron") would
# silently match nothing, so the server refuses to start and says which one.
for broad, shades in COLOR_FAMILIES.items():
    for colour in [broad] + shades:
        assert colour in COLORS, f"COLOR_FAMILIES: '{colour}' is not in COLORS"

# Values the CV pipeline uses when it wasn't sure.
# Nobody searches for an "uncertain" shirt, so we never store these.
# "null" and "none" are here because the new pipeline sends them as TEXT,
# e.g. "age": "null", which is not the same as an empty value in Python.
NOT_KNOWN = {None, "", "uncertain", "unknown", "null", "none"}

DEFINED_SCHEMA = {
    "gender": {
        "type": "enum",  # enum = one value from a fixed list
        "values": ["male", "female"],
        "description": "apparent gender of the person",
    },
    # The pipeline only tells adult or child (no age number), so this is the
    # only age field. The LLM turns ages and words like "kid" into one of these.
    "age_group": {
        "type": "enum",
        "values": ["adult", "child"],
        "description": (
            "ONLY when the request says an age or a stage of life "
            "(old, elderly, kid, teenager, 30 years old). "
            "child = 15 years old or younger, adult = 16 or older"
        ),
    },
    # "color" is only used when SEARCHING, it is never stored.
    # It covers queries like "person wearing red" where the user doesn't say
    # top or bottom. The query builder will turn it into
    # "upper_color is red OR lower_color is red".
    # It comes BEFORE upper_color and lower_color on purpose: a colour with no
    # garment is the more common search, so the LLM should see it first.
    "color": {
        "type": "any_of",
        "fields": ["upper_color", "lower_color"],
        "description": "a color when the user does NOT say top or bottom (a woman in brown)",
    },
    "upper_color": {
        "type": "enum",
        "values": COLORS,
        "description": "color of the top: shirt, t-shirt, jacket, hoodie, coat, sweater",
    },
    "lower_color": {
        "type": "enum",
        "values": COLORS,
        "description": "color of the bottom: pants, trousers, jeans, shorts, skirt",
    },
    "camera_id": {
        # Cameras are not a fixed list like colours: new ones get added.
        # So the LLM only writes the camera NUMBER the user said ("camera 2" -> 2),
        # and the code adds the prefix to make the real ID ("cam_" + 2 = "cam_2").
        # Any number is allowed, so nothing gets forced. “camera 2” now searches cam_2 and correctly returns 0 results, since your data only has cam_1.
        "type": "number_id",
        "prefix": "camera_",
        "description": "the camera number the person was seen on, only when the text says camera or cam (cam 2 -> 2, camera 5 -> 5)",
    },
    "time": {
        "type": "time",
        "field": "timestamp",  # the stored field that time searches look at
        "description": (
            "when the person was seen: a day, a date, a clock time or a part of the day"
        ),
    },
}
