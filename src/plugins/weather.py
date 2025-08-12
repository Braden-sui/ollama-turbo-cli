"""Weather tool plugin providing real weather via wttr.in"""
from __future__ import annotations

import json
from typing import Any

try:  # Optional dependency pattern consistent with legacy tool
    import requests
except Exception:  # pragma: no cover - exercised in runtime environments without requests
    requests = None  # type: ignore

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get real-time current weather for a city via wttr.in. Use when the user asks about current weather or near-term conditions for a specific location. Always pass a concrete city name (not 'here'). Prefer the user's unit preference; default is celsius. The tool returns a concise plain-text summary and should not be used for historical data or multi-day forecasts.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name to query (e.g., 'London', 'Paris', 'Tokyo'). Do not pass vague values like 'here' or 'near me'."
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit to return. Choose based on user preference if provided; otherwise use the default.",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
}

def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather using real data from wttr.in with no static fallback.

    This function requires network access and the 'requests' library. It will return
    a clear error message if the lookup fails for any reason.
    """
    try:
        city = (city or "").strip()
        if not city:
            return "Error: city must be provided"

        # Validate unit
        unit_l = (unit or "celsius").lower()
        if unit_l not in ("celsius", "fahrenheit"):
            return "Error: unit must be 'celsius' or 'fahrenheit'"

        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use the weather tool."

        # Use wttr.in JSON API (no API key required)
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        headers = {"User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)"}
        try:
            response = requests.get(url, timeout=8, headers=headers)
        except Exception as e:  # requests.RequestException
            return f"Error fetching weather for '{city}': network error: {e}"

        if response.status_code != 200:
            return f"Error fetching weather for '{city}': HTTP {response.status_code}"

        try:
            data = response.json()
        except json.JSONDecodeError:
            return f"Error fetching weather for '{city}': invalid JSON response"

        try:
            current = data["current_condition"][0]
            condition = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            wind_mph = current["windspeedMiles"]
            wind_dir = current.get("winddir16Point") or current.get("winddirDegree") or "N/A"
            if unit_l == "fahrenheit":
                temp = current["temp_F"]
                feels_like = current["FeelsLikeF"]
                unit_symbol = "°F"
            else:
                temp = current["temp_C"]
                feels_like = current["FeelsLikeC"]
                unit_symbol = "°C"
        except (KeyError, IndexError, TypeError):
            return f"Error: unexpected API response while fetching weather for '{city}'"

        return (
            f"Weather in {city.title()}: {condition}, {temp}{unit_symbol} "
            f"(feels like {feels_like}{unit_symbol}), "
            f"Humidity: {humidity}%, Wind: {wind_mph} mph {wind_dir}"
        )
    except Exception as e:
        return f"Error getting weather for '{city}': {str(e)}"


TOOL_IMPLEMENTATION = get_current_weather
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
