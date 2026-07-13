"""
Tool-Using Agent (LangGraph version)

Same agent as weather_agent.py but rewritten using the LangGraph framework.

The agent flow is modelled as a state graph:

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     tool calls     ┌──────────────┐
    │ call_llm  │ ─────────────────► │ execute_tools │
    │ (Claude)  │                   │ (get_weather  │
    │           │ ◄──────────────── │  /calculator) │
    └─────┬─────┘     results       └──────────────┘
          │
          │ no tool calls
          ▼
    ┌──────────┐
    │   END    │
    └──────────┘

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python python/agent/weather_agent_langgraph.py
  python python/agent/weather_agent_langgraph.py "Your custom query"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Literal

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# ---------------------------------------------------------------------------
# 1. Tool runtime (unchanged from the original)
# ---------------------------------------------------------------------------

WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"

CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.7143, -74.0060),
    "delhi":    (28.6519, 77.2315),
    "london":   (51.5074, -0.1278),
    "tokyo":    (35.6895, 139.6917),
    "paris":    (48.8566, 2.3522),
    "sydney":   (-33.8688, 151.2093),
}


def get_weather(city: str) -> dict[str, Any]:
    """Fetch the current temperature (Celsius) for a city via Open-Meteo."""
    city_lower = city.strip().lower()
    coords = CITY_COORDS.get(city_lower)
    if coords is None:
        raise ValueError(
            f"Unknown city '{city}'. Known: {list(CITY_COORDS.keys())}"
        )

    lat, lon = coords
    resp = httpx.get(
        WEATHER_API_BASE,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "temperature_unit": "celsius",
        },
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json()["current_weather"]
    return {
        "city": city.strip(),
        "temperature_celsius": current["temperature"],
        "wind_speed_kmh": current["windspeed"],
        "weather_code": current["weathercode"],
    }


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a simple arithmetic expression.

    WARNING: Uses eval() — only safe in a controlled environment.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        raise ValueError("Expression contains disallowed characters")

    result = eval(expression, {"__builtins__": {}}, {})
    return {"expression": expression, "result": result}


# Tool name → implementation mapping
TOOL_IMPLS: dict[str, Any] = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# ---------------------------------------------------------------------------
# 2. LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State that flows through the graph nodes."""
    messages: Annotated[list, add_messages]  # properly merged by add_messages
    user_query: str


# ---------------------------------------------------------------------------
# 3. Graph nodes
# ---------------------------------------------------------------------------

# Initialise the LLM with tool bindings
llm = ChatAnthropic(model="claude-sonnet-5")
llm_with_tools = llm.bind_tools(
    [
        {
            "name": "get_weather",
            "description": "Get the current temperature (in Celsius) for a city.",
            "parameters": {
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
            "parameters": {
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
    ],
)


def call_llm(state: AgentState) -> AgentState:
    """Node: send the message history (with tool definitions) to Claude."""
    CITY_LIST = ", ".join(CITY_COORDS.keys())
    result = llm_with_tools.invoke(
        state["messages"],
        system=(
            "You are a helpful assistant with access to weather data "
            "and a calculator. Use the tools step by step to answer "
            f"the user's question accurately. Available cities: {CITY_LIST}."
        ),
    )
    return {"messages": [result]}


def execute_tools(state: AgentState) -> AgentState:
    """
    Node: iterate over any tool calls in the latest AI message and
    execute them, producing ToolMessage results.
    """
    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(
            f"  └─ Calling {tool_name}({json.dumps(tool_args)}) ... ",
            end="",
            flush=True,
        )

        try:
            fn = TOOL_IMPLS[tool_name]
            result = fn(**tool_args)
            content = json.dumps(result, indent=2)
            print("done")
        except Exception as exc:
            content = json.dumps({"error": str(exc)})
            print(f"error: {exc}")

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id)
        )

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# 4. Conditional edge: route after call_llm
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["execute_tools", "__end__"]:
    """
    If the LLM's last message has tool calls → route to execute_tools.
    Otherwise → we're done (END).
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "__end__"


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------

def build_agent() -> StateGraph:
    """Construct the LangGraph state graph for the weather agent."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("execute_tools", execute_tools)

    # Entry point
    workflow.set_entry_point("call_llm")

    # Edges
    workflow.add_conditional_edges(
        "call_llm",
        should_continue,
        {"execute_tools": "execute_tools", "__end__": END},
    )
    workflow.add_edge("execute_tools", "call_llm")  # always loop back

    return workflow.compile()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your query: ")
    print(f"\nQuery: {query}")
    print("Agent thinking ...\n")

    try:
        agent = build_agent()

        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
        }

        # Run the graph
        final_state = agent.invoke(initial_state)

        # Extract the final answer from the last AIMessage
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                # content can be a string or a list of content blocks
                if isinstance(msg.content, str):
                    text = msg.content.strip()
                elif isinstance(msg.content, list):
                    text = "".join(
                        b["text"] for b in msg.content if isinstance(b, dict) and b.get("type") == "text"
                    ).strip()
                else:
                    text = str(msg.content).strip()

                if text:
                    print(f"\nAnswer:\n{text}")
                    return

        print("\nAnswer:\n(no text response produced)")
    except Exception as exc:
        print(f"\nAgent failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
