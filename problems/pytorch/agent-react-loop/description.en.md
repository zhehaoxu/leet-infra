# Agent Infrastructure: ReAct Loop State Machine

## Problem

An LLM only emits text; the work happens in the agent runtime around it. The runtime parses `Thought` / `Action` / `Action Input` / `Final Answer` out of each model turn, calls the tool, appends the `Observation` to the context, and decides whether to continue or stop. ReAct (Yao et al., 2210.03629) fixed that loop as a protocol, and LangChain's `AgentExecutor`, AutoGPT and every function-calling loop are variants of it.

This problem isolates the state machine: **what the model emitted on each turn is given to you explicitly**, and you only parse, execute and decide when to stop — no generation involved.

## Task

Implement `solve(transcripts, tools, max_steps=None)`:

- `transcripts`: list of raw model turns; `tools`: `{name: callable(str) -> str}`; `max_steps`: how many turns to consume at most, `None` for unlimited.
- Parsing (line by line, top down, first occurrence of each marker): `Final Answer:` ends the run immediately, with the stripped remainder as the answer; `Action:` names the tool and is followed by `Action Input:`, and an Action without an input raises `ValueError`; neither present means an empty turn, which terminates with no answer.
- Execution: a known tool yields `str(tools[name](action_input))`; an unknown tool yields `Error: unknown tool '<name>'` and does **not** abort; an `(action, input)` pair identical to the previous executed one is treated as a loop and terminates before executing.
- Return `(observations, steps, final_answer)`: the observation list, the number of turns consumed, and the answer (`None` if absent).

## Example

```
transcripts = [
    "Thought: check the weather first\nAction: weather\nAction Input: Beijing",
    "Thought: now the clock\nAction: clock\nAction Input: now",
    "Thought: enough\nFinal Answer: Beijing 22C at 10:00",
]
tools = {"weather": lambda city: "22C", "clock": lambda when: "10:00"}

solve(transcripts, tools) -> (["22C", "10:00"], 3, "Beijing 22C at 10:00")
```

## Constraints

- **Four termination sources**: a `Final Answer`, a turn with neither Action nor answer, a repeated action, and running out of `max_steps` or script.
- **`steps` counts consumed turns**, including the turn that produced the answer and the turn that tripped the loop guard; `max_steps=0` consumes none.
- **Unknown tools are a normal path**: real models hallucinate tool names, so the runtime feeds back an error observation and lets the model recover instead of crashing.
- **An empty answer is not a missing answer**: `Final Answer:` with nothing after it returns `""`, not `None`.
- **The answer wins**: if a turn contains both an Action and a Final Answer, the answer is taken and the tool is not executed.
