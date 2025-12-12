# Anthropic Agents Cheat Sheet

**Date**: December 11, 2025
**Prep completed**: 3 days (weather/calculator agents + research agent + workflow patterns)

---

## What I Built

### 1. Tool Use Agents (Days 1-2)
**Files**: `weather_agent.py`, `calculator_agent.py`

**Weather Agent** (2 tools):
- `get_current_weather`: Current conditions
- `get_weather_forecast`: 7-day forecast
- Demonstrated: Tool selection, parallel execution, error recovery

**Calculator Agent** (3 tools):
- `add`, `multiply`, `power`
- Demonstrated: Tool chaining (multi-step: add then multiply)

### 2. Production-Ready Research Agent (Day 3)
**File**: `notebooks/research_agent.ipynb`

**Tools** (3):
- `search_papers`: Find papers by keyword
- `get_paper_details`: Get full paper info by ID
- `summarize_text`: Condense long text

**Error handling implemented**:
- ✅ Retry with exponential backoff (TimeoutError, RateLimitError)
- ✅ Input validation (ValueError for bad inputs)
- ✅ Max iterations (prevent infinite loops)
- ✅ Graceful degradation (when tools fail, explain limitation)

### 3. Workflow Patterns (Day 3)
**File**: `notebooks/w7d4-agent_workflow_playground.ipynb`

**Implemented**:
- ✅ **Prompt chaining**: Sequential transformations (flowery → translate → uppercase)
- ✅ **Routing**: Task-based agent selection (math vs language)
- ✅ **Parallel execution**: Concurrent LLM calls (translate to 3 languages)

---

## Core Agent Patterns

### Agent Loop (Basic)
```python
messages = [{"role": "user", "content": query}]

while True:
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "end_turn":
        # Extract final answer
        break

    if response.stop_reason == "tool_use":
        # Append assistant response
        messages.append({"role": "assistant", "content": response.content})

        # Execute tools

        # Append tool results
        messages.append({"role": "user", "content": tool_results})
```

**Key concepts**:
- Claude is stateless → Must pass full message history
- `stop_reason == "tool_use"` → Execute tools and continue loop
- `stop_reason == "end_turn"` → Claude is done, extract final answer

---

## Production Error Handling Patterns

### 1. Retry with Exponential Backoff
```python
def process_tool_call(tool_name: str, tool_input: dict, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            result = execute_tool(tool_name, tool_input)
            return result

        except TimeoutError as e:
            print(f"Attempt {attempt + 1}/{max_retries} Timeout: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            return f"Error: {e} (failed after {max_retries} retries)"

        except RateLimitError as e:
            print(f"Attempt {attempt + 1}/{max_retries} Rate limited: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Longer wait for rate limits
                continue
            return f"Error: {e}"

        except ValueError as e:
            # Permanent error - don't retry
            return f"Error: {e}"

    return "Error: Max retries exceeded"
```

**Key distinctions**:
- **Transient errors** (TimeoutError, RateLimitError) → Retry
- **Permanent errors** (ValueError, NotFoundError) → Don't retry

### 2. Max Iterations Safety
```python
MAX_ITERATIONS = 10
iteration = 0

while iteration < MAX_ITERATIONS:
    iteration += 1
    # ... agent loop ...

if iteration >= MAX_ITERATIONS:
    print("Warning: Reached max iterations - possible infinite loop")
```

**Why**: Prevents runaway API costs if Claude gets stuck in tool-calling loop

### 3. Input Validation
```python
# Option A: In tool function
def search_papers(query: str) -> str:
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters")
    # ...

# Option B: In dispatcher (before calling tool)
if tool_name == "search_papers" and "query" not in tool_input:
    return "Error: search_papers requires 'query' parameter"
```

---

## Tool Schema Best Practices

### Schema Structure
```python
tool = {
    "name": "search_papers",  # Function name
    "description": "Search for academic papers by keyword. Use this to find papers on a topic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword, e.g. 'transformers', 'agents'"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

**Key points**:
- **Clear description**: Claude reads this to decide when to use tool
- **Specific examples**: "e.g. 'transformers', 'agents'" helps Claude format inputs
- **Required vs optional**: Mark required fields explicitly
- **Defaults**: Specify default values for optional parameters

### Writing Good Tool Descriptions

**Good ✅**:
```python
"description": "Get the 7-day weather forecast for a location. Use this when the user asks about future weather, upcoming conditions, or whether it will rain/snow in the coming days."
```
- Clear trigger words: "future weather", "upcoming", "will rain"
- Distinguishes from current weather tool

**Bad ❌**:
```python
"description": "Gets weather"
```
- Too vague - Claude won't know when to use it

---

## Advanced Workflow Patterns

### 1. Prompt Chaining
**Use case**: Break complex task into sequential steps

```python
def chain_prompt(input: str, prompts: list[str]):
    result = input
    for prompt in prompts:
        query = f"{prompt}\nInput: {result}"
        result = llm_call(query)
    return result

# Example: input → flowery language → translate → uppercase
prompts = [
    "Rewrite the input with flowery language",
    "Translate the input to Chinese",
    "Return the input in uppercase"
]
chain_prompt("Hello world", prompts)
```

**When to use**: Sequential transformations where each step depends on previous output

### 2. Routing
**Use case**: Select specialized agent based on input type

```python
def route(input: str, routes: list[Route]):
    # Step 1: Route selection
    routing_prompt = f"""
    Analyze the input and select the best route:
    <routes>
        math: Handles math questions
        language: Handles linguistic questions
    </routes>

    Return: <route>route_name</route>
    """
    route_name = extract_xml(llm_call(routing_prompt), "route")

    # Step 2: Execute selected route
    route = routes[route_name]
    return llm_call(f"{route.prompt}\nInput: {input}")
```

**When to use**: Input can be categorized into distinct types (math, language, code, etc.)

### 3. Parallel Execution
**Use case**: Multiple independent operations

```python
from concurrent.futures import ThreadPoolExecutor

def parallel(prompt: str, inputs: list[dict], n_workers: int = 3):
    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures = [
        executor.submit(llm_call, prompt=prompt.format(**input))
        for input in inputs
    ]
    return [f.result() for f in futures]

# Example: Translate to 3 languages in parallel
inputs = [
    {"target": "French", "text": "Hello"},
    {"target": "Japanese", "text": "Hello"},
    {"target": "Chinese", "text": "Hello"}
]
parallel("Translate to {target}: {text}", inputs)
```

**When to use**: Independent operations that don't depend on each other

---

## Interview Questions I Can Answer

### Q: "Build an agent with 2-3 tools"
**A**:
1. Define tool schemas (name, description, input_schema)
2. Implement tool functions
3. Build agent loop (while True with stop_reason checks)
4. Test with queries that use different tools

**Live coding**: Can build in Colab in 15-20 minutes

---

### Q: "How does Claude choose which tool to call?"
**A**: Claude matches the user query to tool descriptions. I write clear, specific descriptions with trigger words. For example:
- "Get **current** weather" vs "Get **7-day forecast**"
- Claude maps "now" → current weather, "next week" → forecast

**Example**: Weather agent correctly routes "now" vs "next week" queries

---

### Q: "How do you handle API timeouts?"
**A**: Retry with exponential backoff:
- Catch TimeoutError
- Retry up to 3 times with increasing delays (1s, 2s, 4s)
- After max retries, return error string to Claude
- Claude can then try alternative approach or inform user

**Code**: `process_tool_call` in research_agent.ipynb

---

### Q: "How do you prevent infinite loops?"
**A**: Max iterations counter:
- Set MAX_ITERATIONS = 10 (or 20 for complex tasks)
- Increment counter each loop
- Break and warn if limit reached
- This prevents runaway API costs if Claude gets stuck

**Code**: `handle_query` in research_agent.ipynb, line ~20

---

### Q: "What if a tool returns invalid data?"
**A**: Two-layer validation:
1. **Input validation**: Check parameters before calling tool (ValueError for bad input)
2. **Output validation**: Check tool result format, return error if malformed
3. **Return error strings** (not exceptions) so Claude can see and adapt

**Example**: Research agent Test 3 - when tools fail, Claude gracefully degrades

---

### Q: "How would you scale this for production?"
**A**:
1. **Async tool execution**: Use asyncio for I/O-bound tools (API calls)
2. **Caching**: Cache idempotent tool results (e.g., paper details by ID)
3. **Rate limiting**: Implement token bucket for external APIs
4. **Orchestrator-workers**: Break into coordinator agent + specialized worker agents
5. **Context management**: Summarize old messages when approaching context limit
6. **Metrics**: Track latency, success rate, tool usage patterns

---

### Q: "When should you NOT use agents?"
**A**:
1. **Simple tasks**: Single LLM call sufficient (no tools needed)
2. **Latency-sensitive**: Agents add overhead (multiple API calls)
3. **Deterministic workflows**: Use code instead (more reliable)
4. **High-stakes decisions**: Where errors are unacceptable
5. **Cost constraints**: Tool loops can be expensive

**Alternative**: Use structured prompts, RAG, or traditional code

---

### Q: "Difference between tool use and RAG?"
**A**:
- **RAG**: Augments Claude's context with retrieved information (read-only)
- **Tool use**: Enables Claude to take actions (API calls, calculations, writes)

**Example**:
- RAG: "Here's relevant documentation" → Claude synthesizes answer
- Tool: Claude decides "I need weather data" → Calls get_weather → Uses result

---

### Q: "How do you debug agent failures?"
**A**:
1. **Log all tool calls**: Print tool name, inputs, outputs
2. **Inspect message history**: See full conversation flow
3. **Check stop_reason**: Understand why Claude stopped
4. **Test tools in isolation**: Verify tool functions work independently
5. **Add max iterations early**: Prevent runaway loops during development

**Example**: My notebooks have extensive logging (***LOG: messages)

---

## System Prompt Best Practices

**Good system prompt template**:
```python
system_prompt = f"""
You are a [role] with access to: {", ".join(tool_names)}.

Rules:
- ONLY use the provided tools - do not invent or use other tools
- If a question doesn't require a tool, answer directly using your knowledge
- When multiple independent tool calls are needed, make them in parallel for efficiency
- When a tool returns an error, try an alternative approach or inform the user

Available tools: {", ".join(tool_names)}
"""
```

**Key elements**:
1. ✅ Explicit tool list (prevents hallucination)
2. ✅ When NOT to use tools (avoid unnecessary calls)
3. ✅ Parallel execution guidance
4. ✅ Error handling instruction
5. ✅ Repeat tool list at end (recency bias)

---

## Notebook Development Tips

**Cell organization**:
1. Setup (imports, API key)
2. Tool definitions
3. Tool functions (test individually)
4. Tool dispatcher
5. Agent loop
6. Test cases (separate cells)

**Benefits**:
- ✅ Test components independently
- ✅ Iterative development (run cell, fix, rerun)
- ✅ Clear structure to follow
- ✅ Easy debugging (add print statements in any cell)

**Keyboard shortcuts** (Colab):
- Shift+Enter: Run cell
- Ctrl+M B: Add cell below
- Tab: Autocomplete

---

## Production Patterns Summary

| Pattern | When to Use | Implementation |
|---------|------------|----------------|
| **Tool Use** | Need external data or actions | Define tools, agent loop |
| **Prompt Chaining** | Sequential transformations | Pass output as next input |
| **Routing** | Categorizable inputs | Route selector + specialized prompts |
| **Parallel** | Independent operations | ThreadPoolExecutor or async |
| **Retry** | Transient errors | Exponential backoff |
| **Max Iterations** | Prevent runaway | Counter in loop |
| **Validation** | Bad inputs | Check before tool execution |
