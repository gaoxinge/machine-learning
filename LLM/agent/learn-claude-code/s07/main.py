import os
import re
import json
import time
import subprocess
from pathlib import Path
from anthropic import Anthropic

MODEL = "GLM-5"
WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES = ["user", "feedback", "project", "reference"]
CONSOLIDATE_THRESHOLD = 10


client = Anthropic(
    base_url="https://antchat.alipay.com/api/anthropic",
    api_key="{API-KEY}"
)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def _rebuild_index():
    """Rebuild MEMORY.md index from all memory files."""
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}({f.name}) — {desc}]")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")


def read_memory_index() -> str:
    """Read MEMORY.md index (injected into SYSTEM every turn)."""
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""


def list_memory_files():
    """List all memory files with metadata."""
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result


def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """Write a single memory file with YAML frontmatter."""
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n")
    _rebuild_index()
    return filepath


def read_memory_file(filename: str) -> str | None:
    """Read a single memory file's full content."""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """Select relevant memory filenames by matching recent conversation against
    memory names/descriptions. Uses a simple LLM call (or falls back to keyword
    matching on name+description)."""
    files = list_memory_files()
    if not files:
        return []

    # Collect recent user text for context
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]
    if not recent.strip():
        return []

    # Build catalog of name + description for LLM to choose from
    catalog_lines = []
    for i, f in enumerate(files):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)

    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = extract_text(response.content).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass

    # Fallback: keyword matching on name + description
    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected


def load_memories(messages: list) -> str:
    """Load relevant memory content for injection into context."""
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""
    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def extract_memories(messages: list):
    """Extract new memories from recent dialogue. Runs after each turn."""
    # Collect recent conversation text
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return

    # Check existing memories to avoid duplicates
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        text = extract_text(response.content).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass


def consolidate_memories():
    """Merge duplicate/stale memories. Triggered when file count ≥ threshold."""
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000
        )
        text = extract_text(response.content).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # Remove old memory files (keep MEMORY.md)
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)

        print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
    except Exception:
        pass


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit}) more lines"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def extract_text(content) -> str:
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


def build_system() -> str:
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"You are a coding agent at {WORKDIR}."
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    }
]


def agent_loop(messages: list):
    memories_content = load_memories(messages)
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None
    system = build_system()

    while True:
        pre_compress = [
            m if isinstance(m, dict) else {
                "role": m.get("role",""),
                "content": str(m.get("content",""))
            } for m in messages
        ]

        request_messages = messages
        if memories_content and memory_turn is not None and memory_turn < len(messages):
            request_messages = messages.copy()
            request_messages[memory_turn] = {
                **messages[memory_turn],
                "content": memories_content + "\n\n" + messages[memory_turn]["content"],
            }
        response = client.messages.create(
            model=MODEL,
            system=system,
            messages=request_messages,
            tools=TOOLS,
            max_tokens=8000
        )

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            extract_memories(pre_compress)
            consolidate_memories()
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s09: Memory — persistent cross-session knowledge")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
