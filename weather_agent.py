"""
Tool-Using Agent: Sum Temperatures of New York and Delhi

Demonstrates the tool-use (function calling) agent pattern:
  1. Define tools (get_weather, calculator)
  2. An LLM (Claude) decides which tool to call and with what args
  3. The runtime executes the tool and feeds the result back
  4. The LLM repeats until it has enough information to answer

Requirements:
  pip install anthropic httpx

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python python/agent/weather_agent.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 1. Tool implementations (the "runtime")
# ---------------------------------------------------------------------------

WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"

# City → latitude/longitude lookup (static for this example)
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.7143, -74.0060),
    "delhi":    (28.6519, 77.2315),
    "london":   (51.5074, -0.1278),
    "tokyo":    (35.6895, 139.6917),
    "paris":    (48.8566, 2.3522),
    "sydney":   (-33.8688, 151.2093),
}


def get_weather(city: str) -> dict[str, Any]:
    """
    Fetch the current temperature (°C) for a given city.

    Uses the free Open-Meteo API (no key required).
    """
    city_lower = city.strip().lower()
    coords = CITY_COORDS.get(city_lower)
    if coords is None:
        raise ValueError(
            f"Unknown city '{city}'. Known: {list(CITY_COORDS.keys())}"
        )

    lat, lon = coords
    params: dict[str, str | int] = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "temperature_unit": "celsius",
    }

    resp = httpx.get(WEATHER_API_BASE, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    current = data["current_weather"]
    return {
        "city": city.strip(),
        "temperature_celsius": current["temperature"],
        "wind_speed_kmh": current["windspeed"],
        "weather_code": current["weathercode"],
    }


def calculator(expression: str) -> dict[str, Any]:
    """
    Evaluate a simple arithmetic expression and return the result.

    WARNING: Uses eval() — only safe in a controlled environment.
    For production, use a proper expression parser (e.g. `asteval`).
    """
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        raise ValueError("Expression contains disallowed characters")

    result = eval(expression, {"__builtins__": {}}, {})
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# 2. Tool schemas (sent to the LLM so it knows what tools exist)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current temperature (in Celsius) for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'New York'",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a numeric expression (e.g. '15 + 23.5').",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
]

# Map tool names → Python functions
TOOL_IMPLS: dict[str, Any] = {
    "get_weather": get_weather,
    "calculator": calculator,
}


# ---------------------------------------------------------------------------
# 3. Agent loop
# ---------------------------------------------------------------------------


def run_agent(user_query: str, max_turns: int = 10) -> str:
    """
    Run the tool-use agent loop with Claude.

    Steps:
      1. Send the conversation so far + tool definitions to the LLM.
      2. If the LLM returns a text response → we are done (return it).
      3. If the LLM returns a tool-use block → execute the tool,
         append the result, and go back to step 1.
    """
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_query}
    ]

    for _ in range(max_turns):
        print (f"messages : {messages}")
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            system="You are a helpful assistant with access to weather data "
                   "and a calculator. Use the tools step by step to answer "
                   f"the user's question accurately. Available cities for weather app are : {CITY_COORDS.keys()}",
            messages=messages,
            tools=TOOLS,
        )

        print(f"response : {response}")
        # Collect all blocks in the response
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text" and b.text.strip()]

        # If there are NO tool calls, the text is the final answer
        if not tool_use_blocks and text_blocks:
            return text_blocks[-1].text.strip()

        # Otherwise process tool-use blocks
        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input
                print(
                    f"  └─ Calling {tool_name}({json.dumps(tool_args)}) ... ",
                    end="",
                    flush=True,
                )

                try:
                    fn = TOOL_IMPLS[tool_name]
                    result = fn(**tool_args)
                    output = json.dumps(result, indent=2)
                    print("done")
                except Exception as exc:
                    output = json.dumps({"error": str(exc)})
                    print(f"error: {exc}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # Append the assistant's (tool-use) message + tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Agent reached max turns without producing a final answer."


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Get a key at https://console.anthropic.com/ and export it:")
        print("  export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your query: ")

    print(f"\nQuery: {query}")
    print("Agent thinking ...")

    try:
        answer = run_agent(query)
        print(f"\nAnswer:\n{answer}")
    except Exception as exc:
        print(f"\nAgent failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
