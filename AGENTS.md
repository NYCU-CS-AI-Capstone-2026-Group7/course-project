# AGENTS.md

This file acts as a persistent memory and configuration register for AI Coding Agents working in this repository.

> [!IMPORTANT]
> All AI agents entering this workspace MUST read, respect, and strictly adhere to the credentials, boundaries, and environmental constraints listed in this document.

---

## 🔑 Remote VM Credentials (Glows AI - Session 2)

These are the credentials for accessing the remote development machine used in the second session of this project:

- **User**: `glows`
- **Host**: `tw-06.access.glows.ai`
- **SSH Port**: `25298`
- **Password**: `4edUC]mYPVWuzqH%`
- **Remote Project Path**: `/root/course-project`

---

## 📌 Rules & Environmental Constraints (Things to "Remember")

### 1. Local Codebase First (Git Scope)
- **Constraint**: If any part of a task can be solved or controlled via the local Git repository, **do NOT modify or read from the remote VM**. Keep all edits and verification strictly in the local workspace.
- **Constraint**: As long as a file or part of the codebase is not gitignored or sparsed out (meaning it can be managed within the local Git repository scope), there is no need to actively push or write it to the remote VM.
- **Constraint**: NEVER EVER GIT COMMIT OR PUSH FOR THE USER except they CLEARLY require it!

### 2. Repository Type
- **Constraint**: This local repository is a **sparse checkout**. Do not attempt to pull, checkout, or track the entire codebase recursively unless explicitly instructed, as many files are omitted.

### 3. Space Management (`uv sync` Ban)
- **Constraint**: **NEVER** run `uv sync` or any disk-heavy package-sync commands in this repository or its directories. These commands generate extremely large dependency graphs and caches that consume all available disk space.

### 4. Local Laptop Virtual Environment
- **Constraint**: The local machine's virtual environment `/tmp/.venv/bin/activate` is pre-configured and already contains essential validation libraries such as `pybullet` and `numpy`. Use this environment for local verification tasks.

### 5. LaTeX Formatting Ban (No LaTeX in responses)
- **Constraint**: **NEVER** use LaTeX rendering or math blocks (such as $...$ or $$...$$ or \pm) in direct text responses. It causes rendering errors in the CLI. Use plain text representation or applicable Unicode Characters instead (e.g., '+- 15 degrees', '180°', etc.). This doesn't include Markdown files or artifacts.

### 6. Language Constraint (Traditional Chinese Replies, English Code)
- **Constraint**: **ALWAYS** reply to the user in Traditional Chinese (including direct responses and artifacts). Only code implementations, code comments, and modifications inside `AGENTS.md` should be written in English.

### 7. Global Rules Only (No Notes)
- **Constraint**: This file is for global rules only, not a note! Never modify it w/o the user's clear instruction.

---

## Workspace design
Workspaces organize large codebases by splitting them into multiple packages with common dependencies. Think: a FastAPI-based web application, alongside a series of libraries that are versioned and maintained as separate Python packages, all in the same Git repository.

In a workspace, each package defines its own pyproject.toml, but the workspace shares a single lockfile, ensuring that the workspace operates with a consistent set of dependencies.

As such, uv lock operates on the entire workspace at once, while uv run and uv sync operate on the workspace root by default, though both accept a --package argument, allowing you to run a command in a particular workspace member from any workspace directory.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
