SYSTEM_PROMPT = (
    "You are a smart home assistant. Call tools to control the home and confirm actions "
    "in a short sentence. Ask for clarification if a request is ambiguous. No emojis. Think as quickly/succinctly as possible."
    "Use the talk tool to respond to your user."
)

def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

_room = {"type": "string", "description": "Room/area to control, e.g. 'living room', 'all'."}

TOOLS = [
    _fn("set_light_power", "Turn lights on or off in a room.", {
        "room": _room,
        "state": {"type": "string", "enum": ["on", "off"], "description": "on or off."},
    }, ["room", "state"]),

    _fn("set_light_color", "Change smart light colour (and optionally brightness) in a room.", {
        "room": _room,
        "color": {"type": "string", "description": "Colour name or hex code, e.g. 'red', '#FF5733'."},
        "brightness": {"type": "integer", "description": "1–100. Omit to leave unchanged."},
    }, ["room", "color"]),

    _fn("play_music", "Control music playback.", {
        "action": {"type": "string", "enum": ["play", "pause", "resume", "stop", "next", "previous"],
                   "description": "Playback action."},
        "query":  {"type": "string", "description": "Song/artist/album/playlist. Required for 'play'."},
        "room":   {"type": "string", "description": "Speaker/room. Defaults to main speaker."},
        "volume": {"type": "integer", "description": "0–100. Omit to leave unchanged."},
    }, ["action"]),

    _fn("water_garden", "Start or stop watering a garden zone.", {
        "zone":             {"type": "string", "description": "Zone to water, e.g. 'front lawn', 'all'."},
        "action":           {"type": "string", "enum": ["start", "stop"], "description": "start or stop."},
        "duration_minutes": {"type": "integer", "description": "Minutes to water. Omit for default duration."},
    }, ["zone", "action"]),

    _fn("set_ac_temperature", "Set AC/heating temperature or mode for a room.", {
        "room":        _room,
        "temperature": {"type": "number",  "description": "Target °C. Omit to change mode only."},
        "mode":        {"type": "string",  "enum": ["cool", "heat", "fan", "auto", "off"],
                        "description": "Operating mode. Omit to keep current mode."},
    }, ["room"]),
]
