# Copilot Instructions

## Project Overview

This project implements a RAG system for exploring the ACL anthology.

The system receives natural language input requesting papers with a given topic.

The system produces a list of relevant papers based on the input topic.

The vector store holds embeddings of the papers for retrieval as well as metadata such as publication dates, authors or venues. The database needs to be constructed using the acl-anthology Python API.

The implementation must be based on LangGraph and a open-weight LLM such as Qwen that is hosted on a custom server via llama.cpp or vLLM (IP address: 127.0.0.1, port: 8345).

Use uv for project management.

## Core Design Principles

- Keep the implementation as simple as possible.
- Prioritize concise, readable, and transparent code over abstractions, extensibility, or clever optimizations.
- Use LangGraph for the core execution flow; do not introduce additional orchestration frameworks.
- Make data flow and LLM invocation explicit and easy to trace.
- Prefer a small number of functions, modules, and graph nodes.
- Avoid unnecessary classes, interfaces, factories, wrappers, and design patterns.
- Do not introduce abstractions until a concrete requirement justifies them.
- Reuse existing dependencies and project conventions.
- Do not add features that have not been requested.
- Make the smallest change that correctly implements the requested behavior.

## Code Style and Implementation

- Follow the repository's existing formatting, linting, and typing conventions.
- Use type hints for public functions, graph state, and important data structures.
- Prefer descriptive names and short functions with clear responsibilities.
- Keep control flow explicit; avoid deeply nested logic and unnecessary indirection.
- Avoid Any and broad exception handling unless justified.
- Raise meaningful errors rather than silently ignoring failures.
- Add comments only when they clarify non-obvious decisions.
- Do not add docstrings that merely restate a function's name or implementation.
- Avoid premature generalization and speculative future-proofing.
- Do not modify unrelated code or refactor working components without a clear reason.

## Testing Requirements

Tests must verify the core properties of the system, not merely that individual functions execute without errors.

Use a deterministic fake or mock model for unit tests. Tests must not require API credentials or incur model API costs.

## Implementation Workflow

For each task:

- Inspect the existing repository, dependencies, tests, and conventions.
- Identify the smallest change that satisfies the requirement.
- Trace the relevant data flow from input through LangChain and the LLM to output.
- Implement the change using the existing architecture wherever possible.
- Add or update tests for the affected behavior.
- Run the relevant tests and available quality checks.
- Review the diff for unnecessary abstractions, unrelated changes, and untested behavior.
- Summarize what changed and report which checks actually passed.

For architectural changes, explain the proposed approach before introducing additional complexity.