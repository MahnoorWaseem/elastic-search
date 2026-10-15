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

# Values the CV pipeline uses when it wasn't sure.
# Nobody searches for an "uncertain" shirt, so we never store these.
NOT_KNOWN = {None, "", "uncertain", "unknown"}

DEFINED_SCHEMA = {
    "gender": {
        "type": "enum",  # enum = one value from a fixed list
        "values": ["male", "female"],
        "description": "apparent gender of the person",
    },
    "age_group": {
        "type": "enum",
        "values": ["adult", "child"],
        "description": "adult or child",
    },
    "age": {
        "type": "range",  # range = a number between min and max
        "min": 1,
        "max": 85,
        # Life stages copied from the alerts: details.models.crops.age_head.age_bins
        # The LLM will read this to turn "teenager" or "elderly" into numbers.
        "description": (
            "estimated age in years. Life stages: baby or toddler 1-3, "
            "young child 4-9, pre-teen 10-12, young teenager 13-15, "
            "older teenager 16-19, young adult 20-34, middle-aged 35-54, "
            "elderly 55-85"
        ),
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
        "type": "enum",
        # Not a CV value: it comes from your list of cameras.
        # Only cam_1 exists in the data for now, so it's written here by hand.
        "values": ["cam_1"],
        "description": "the camera that saw the person",
    },
    # "color" is only used when SEARCHING, it is never stored.
    # It covers queries like "person wearing red" where the user doesn't say
    # top or bottom. The query builder will turn it into
    # "upper_color is red OR lower_color is red".
    "color": {
        "type": "any_of",
        "fields": ["upper_color", "lower_color"],
        "description": "a color when the user doesn't say top or bottom",
    },
}
