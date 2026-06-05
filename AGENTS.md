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

### 2. Repository Type
- **Constraint**: This local repository is a **sparse checkout**. Do not attempt to pull, checkout, or track the entire codebase recursively unless explicitly instructed, as many files are omitted.

### 3. Space Management (`uv sync` Ban)
- **Constraint**: **NEVER** run `uv sync` or any disk-heavy package-sync commands in this repository or its directories. These commands generate extremely large dependency graphs and caches that consume all available disk space.

### 4. Local Laptop Virtual Environment
- **Constraint**: The local machine's virtual environment `/tmp/.venv/bin/activate` is pre-configured and already contains essential validation libraries such as `pybullet` and `numpy`. Use this environment for local verification tasks.

### 5. LaTeX Formatting Ban (No LaTeX in responses)
- **Constraint**: **NEVER** use LaTeX rendering or math blocks (such as $...$ or $$...$$ or \pm) in text responses. It causes rendering errors in the CLI and web chat interface. Use plain text representation instead (e.g., '+- 15 degrees', '180 degrees', etc.).

### 6. Language Constraint (Traditional Chinese Replies, English Code)
- **Constraint**: **ALWAYS** reply to the user in Traditional Chinese. Only code implementations, code comments, and modifications inside `AGENTS.md` should be written in English.
