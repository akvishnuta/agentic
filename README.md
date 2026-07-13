# Agents: Weather + Calculator

Tool-using agents that answer questions like
*"By how much degree is Delhi warmer than New York currently?"*

Two implementations showing different frameworks:

| File | Framework | Pattern |
|------|-----------|---------|
| `weather_agent.py` | **Anthropic SDK** (vanilla) | Manual agent loop: send → tool_use → execute → repeat |
| `weather_agent_langgraph.py` | **LangGraph** | State-graph: nodes (call_llm ↔ execute_tools) + conditional routing |

## Architecture

| Layer | Responsibility |
|-------|----------------|
| **LLM** (Claude) | Understands the query, decides which tool to call & with what args |
| **Agent Loop** | Sends messages ↔ LLM, intercepts tool-use blocks, runs tools, feeds results back |
| **Tool Runtime** | `get_weather()` → Open-Meteo free API | `calculator()` → expression evaluator |

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  run_agent() loop                                   │
│                                                     │
│  ┌──────────┐   tools + history    ┌────────────┐  │
│  │  LLM      │ ◄───────────────── │  messages[]  │  │
│  │ (Claude)  │ ─── tool_use ────► │              │  │
│  │ (planner) │                    │  append      │  │
│  └─────┬─────┘                    │  result as   │  │
│        │                          │  "user" role │  │
│        ▼ tool_use                 └──────────────┘  │
│  ┌──────────────┐                                   │
│  │  Tool Runtime│  get_weather() / calculator()     │
│  └──────┬───────┘                                   │
│         │ result                                    │
└─────────┼───────────────────────────────────────────┘
          │ final text (no tool_use blocks)
          ▼
       Answer
```

## Supported Cities

`get_weather` currently supports: New York, Delhi, London, Tokyo, Paris, Sydney.

## Tools

### `get_weather(city: str)`
- **Source**: [Open-Meteo](https://open-meteo.com/) — free, no API key required
- **Returns**: current temperature (°C), wind speed, weather code

### `calculator(expression: str)`
- Evaluates simple arithmetic (no external dependency)
- Only allows `0-9 + - * / ( ) .`

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...

# Run with a CLI argument
python weather_agent.py "By how much degree is Delhi warmer than New York currently?"
python weather_agent_langgraph.py "By how much degree is Delhi warmer than New York currently?"

# Or run interactively — you'll be prompted for a query
python weather_agent.py
```

## Message Flow

The Anthropic Messages API uses only `"user"` and `"assistant"` roles (no `"tool"` role). Tool results are wrapped in a `"user"` message with `type: "tool_result"`:

```
user:      "By how much degree is Delhi warmer than New York currently?"
assistant: tool_use(id="abc", name="get_weather", input={city: "Delhi"})
user:      tool_result(tool_use_id="abc", content={temp: 32.1})
assistant: tool_use(id="def", name="get_weather", input={city: "New York"})
user:      tool_result(tool_use_id="def", content={temp: 28.3})
assistant: tool_use(id="ghi", name="calculator", input={expr: "32.1 - 28.3"})
user:      tool_result(tool_use_id="ghi", content={result: 3.8})
assistant: "Delhi is currently 3.8°C warmer than New York."
```

Each `tool_result` links back to its `tool_use` via `tool_use_id`. The `"user"` role is just the container — the `type: "tool_result"` field tells the API it's a tool response, not free-form user text.

## Extending

Add a new tool in three places:

1. **Implement** the function (e.g. `def search_web(...)`)
2. **Register it** in the `TOOLS` list with its JSON Schema
3. **Map it** in the `TOOL_IMPLS` dict

The agent loop handles the rest automatically.
