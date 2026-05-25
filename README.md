# 🦜🔗 LangGraph Basics — Understanding Stateful Graphs for AI Agents

A beginner-friendly notebook that walks you through the **core concepts of LangGraph** — from defining state and nodes to building conditional branching workflows.

---

## 📖 Table of Contents

- [What is LangGraph?](#what-is-langgraph)
- [Why Do We Need LangGraph?](#why-do-we-need-langgraph)
- [Core Concepts](#core-concepts)
- [Code Walkthrough](#code-walkthrough)
  - [Part 1 — Minimal Graph](#part-1--minimal-graph)
  - [Part 2 — Conditional Branching Graph](#part-2--conditional-branching-graph)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## What is LangGraph?

**LangGraph** is an open-source library built on top of [LangChain](https://www.langchain.com/) that lets you build **stateful, multi-step AI workflows** as graphs. It treats your AI pipeline as a directed graph where:

- **Nodes** are functions (or LLM calls) that perform a specific task
- **Edges** define the flow — which node runs after which
- **State** is a shared data object that gets passed through and updated at each step

LangGraph is developed by the LangChain team and is the backbone of many modern **AI agent** and **multi-agent** architectures.

---

## Why Do We Need LangGraph?

Standard LLM calls are stateless and linear — you send a prompt, you get a response, and that's it. But real-world AI applications often need:

| Challenge | What LangGraph Solves |
|---|---|
| **Multi-step reasoning** | Chain multiple LLM calls and tools in a controlled sequence |
| **Shared state across steps** | Maintain and update context as data flows through nodes |
| **Conditional logic** | Route execution dynamically based on intermediate outputs |
| **Human-in-the-loop** | Pause, review, and resume workflows at any node |
| **Cyclic workflows** | Build agents that can loop, retry, and self-correct |
| **Visibility & control** | Visualize the exact flow of your agent as a graph |

Before LangGraph, building agents with complex control flow required a lot of boilerplate and custom orchestration. LangGraph gives you a clean, reusable abstraction for all of this.

---

## Core Concepts

### 🗂 State
A `TypedDict` that holds all the data shared across nodes. Every node reads from and writes back to this state object.

```python
class State(TypedDict):
    graph_state: str
```

### 🔵 Node
A plain Python function that receives the current state and returns an updated version of it.

```python
def my_node(state: State):
    return {"graph_state": state["graph_state"] + " — processed"}
```

### ➡️ Edge
Connects two nodes to define execution order. Can be:
- **Static edge** — always goes from Node A to Node B
- **Conditional edge** — decides the next node based on state at runtime

### 🏁 START & END
Special built-in nodes that mark the entry and exit points of the graph.

### 🔨 Compile
Once nodes and edges are defined, calling `.compile()` locks the graph and prepares it for execution via `.invoke()`.

---

## Code Walkthrough

### Part 1 — Minimal Graph

This section demonstrates the absolute minimum required to build and run a LangGraph graph.

#### 1. Define the State

```python
from typing import TypedDict

class SomeState(TypedDict):
    att1: str
    att2: str
```

A `TypedDict` acts as the schema for your graph's shared state. Here `att1` and `att2` are two string fields that any node can read or modify.

#### 2. Create a Node Function

```python
def some_function(state: SomeState):
    state['att1'] = "Value changed by node some_function()"
    return state
```

The node receives the full state, modifies `att1`, and returns the updated state. Every node follows this pattern.

#### 3. Build and Connect the Graph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(SomeState)          # Initialize with your state schema
graph.add_node("node1", some_function) # Register the node
graph.add_edge("node1", END)           # Connect node1 → END
graph.set_entry_point("node1")         # Declare where execution begins
```

#### 4. Compile and Run

```python
compiled_graph = graph.compile()
```

`.compile()` validates the graph structure and returns an executable object. You can call `.invoke({...})` on it to run the graph with an initial state.

---

### Part 2 — Conditional Branching Graph

This is the more complete example — it demonstrates **conditional edges**, where the graph dynamically decides which node to visit next based on runtime logic.

#### 1. Define the State

```python
from typing import TypedDict

class State(TypedDict):
    graph_state: str
```

A single string field `graph_state` accumulates messages as it travels through the graph.

#### 2. Define Three Nodes

```python
def first_node(state):
    print("This is my first node")
    return {"graph_state": state['graph_state'] + "Hi, I am travelling "}

def second_node(state):
    print("This is my second node")
    return {"graph_state": state['graph_state'] + "Meghalaya - India"}

def third_node(state):
    print("This is my third node")
    return {"graph_state": state['graph_state'] + "Kerala - India"}
```

- `first_node` — the starting node; appends a travel message to the state
- `second_node` — one possible destination (Meghalaya)
- `third_node` — another possible destination (Kerala)

#### 3. Write the Conditional Logic

```python
import random
from typing import Literal

def decide_location(state) -> Literal['second_node', 'third_node']:
    if random.random() < 0.5:
        return 'second_node'
    else:
        return 'third_node'
```

This function is a **router** — it inspects the state and returns the *name* of the next node to execute. Here it randomly picks between Meghalaya and Kerala, simulating a dynamic decision. In real applications this could be an LLM classification, a rule check, or a confidence threshold.

#### 4. Assemble the Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# Register all nodes
builder.add_node("first_node", first_node)
builder.add_node("second_node", second_node)
builder.add_node("third_node", third_node)

# Define edges
builder.add_edge(START, "first_node")                               # Entry point
builder.add_conditional_edges("first_node", decide_location)        # Branch after first_node
builder.add_edge("second_node", END)                                # second_node → done
builder.add_edge("third_node", END)                                 # third_node → done
```

**`add_conditional_edges`** is the key API here — it tells LangGraph: *"after `first_node` runs, call `decide_location` with the current state and go to whichever node it returns."*

#### 5. Compile, Visualize, and Invoke

```python
from IPython.display import Image, display

graph = builder.compile()

# Visualize the graph structure as a diagram
display(Image(graph.get_graph().draw_mermaid_png()))

# Run the graph with an initial state
result = graph.invoke({"graph_state": "Hi, My name is Abhijeet. "})
print(result)
```

**Graph flow diagram:**

```
START
  │
  ▼
first_node
  │
  ├─── (random < 0.5) ──▶ second_node ──▶ END
  │
  └─── (random >= 0.5) ─▶ third_node  ──▶ END
```

**Example outputs** (non-deterministic due to `random`):

```
# Run 1
This is my first node
This is my second node
{'graph_state': 'Hi, My name is Abhijeet. Hi, I am travelling Meghalaya - India'}

# Run 2
This is my first node
This is my third node
{'graph_state': 'Hi, My name is Abhijeet. Hi, I am travelling Kerala - India'}
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
pip install langgraph
```

For graph visualization:

```bash
pip install grandalf  # or install playwright for PNG rendering
```

### Run the Notebook

```bash
jupyter notebook lang_graph_basics_understanding.ipynb
```

---

## Project Structure

```
.
├── lang_graph_basics_understanding.ipynb   # Main notebook with all examples
└── README.md                               # This file
```

---

## Key Takeaways

| Concept | What You Learned |
|---|---|
| `TypedDict` State | Defines the schema of data shared across all nodes |
| `add_node` | Registers a Python function as an executable step in the graph |
| `add_edge` | Creates a fixed, unconditional connection between two nodes |
| `add_conditional_edges` | Routes execution dynamically using a router function |
| `set_entry_point` / `START` | Marks where graph execution begins |
| `END` | Marks the terminal node — execution stops here |
| `.compile()` | Validates and prepares the graph for invocation |
| `.invoke({...})` | Runs the graph from START with a given initial state |

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangChain Documentation](https://docs.langchain.com/)


## 📄 License

This project is for educational and demonstration purposes. Feel free to fork and extend.
This project is for educational purposes. Feel free to use and adapt it for your own learning.
