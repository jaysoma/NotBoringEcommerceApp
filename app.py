import fastapi
import uvicorn
import ollama
import json
from pathlib import Path


_PAIRS = {'{': '}', '[': ']'}

def _find_close(s: str, open_pos: int) -> int:
    """Return index of the matching closing bracket, or -1 if unmatched."""
    open_ch = s[open_pos]
    close_ch = _PAIRS[open_ch]
    depth = 0
    i = open_pos
    while i < len(s):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        elif s[i] == '"':
            i += 1
            while i < len(s):
                if s[i] == '\\':
                    i += 1
                elif s[i] == '"':
                    break
                i += 1
        i += 1
    return -1

def _read_string(s: str, i: int):
    """Read a quoted string starting at i. Returns (string_content, next_i)."""
    j = i + 1
    while j < len(s):
        if s[j] == '\\':
            j += 2
        elif s[j] == '"':
            return s[i+1:j], j + 1
        else:
            j += 1
    return s[i+1:j], j

def _parse_object(s: str) -> dict:
    """Parse the interior of a {} block into a flat dict of string key-value pairs."""
    result = {}
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i] in ' \t\n\r,':
            i += 1
        if i >= n or s[i] != '"':
            i += 1
            continue
        key, i = _read_string(s, i)
        while i < n and s[i] in ' \t\n\r':
            i += 1
        if i >= n or s[i] != ':':
            continue
        i += 1
        while i < n and s[i] in ' \t\n\r':
            i += 1
        if i >= n:
            break
        if s[i] == '"':
            val, i = _read_string(s, i)
            if val.strip():
                result[key] = val.strip()
        elif s[i] in _PAIRS:
            close = _find_close(s, i)
            if close == -1:
                break
            i = close + 1
        else:
            # number, bool, null — skip to next comma
            while i < n and s[i] not in ',}]':
                i += 1
    return result

def _parse_nodes(s: str) -> list[dict]:
    """
    Recursively walk bracket-structured text.
    Every matched {} becomes a flat dict node.
    Unmatched brackets are discarded.
    Returns a list of all nodes found at any depth.
    """
    nodes = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{':
            close = _find_close(s, i)
            if close == -1:
                break  # unmatched — discard rest
            inner = s[i+1:close]
            node = _parse_object(inner)
            if node:
                nodes.append(node)
            nodes.extend(_parse_nodes(inner))
            i = close + 1
        elif s[i] == '[':
            close = _find_close(s, i)
            if close == -1:
                break  # unmatched — discard rest
            nodes.extend(_parse_nodes(s[i+1:close]))
            i = close + 1
        else:
            i += 1
    return nodes

def extract_list(raw) -> list[dict]:
    """Extract a list of flat dicts from any source — one dict per {} node."""
    if isinstance(raw, (dict, list)):
        raw = json.dumps(raw)

    text = str(raw).strip()

    if text and text[0] in ('{', '['):
        nodes = _parse_nodes(text)
        # Deduplicate by frozenset of items
        seen = set()
        result = []
        for n in nodes:
            key = frozenset(n.items())
            if key not in seen:
                seen.add(key)
                result.append(n)
        return result

    # Plain text fallback
    for delim in ('\n', ',', ';', '|'):
        parts = [p.strip() for p in text.split(delim) if p.strip()]
        if len(parts) > 1:
            return [{"name": p} for p in parts]
    return [{"name": text}] if text else []


app = fastapi.FastAPI()

def _pick_name(node: dict) -> str:
    """Pick the most label-like value from a node for display/comparison."""
    for k, v in node.items():
        if k.lower() in ("name", "title", "label"):
            return v
    return next(iter(node.values()), "")

@app.get("/categories")
def get_categories():
    prompt = "Invent 5 fun and creative department names for a fictional ecommerce store."
    raw = ollama.generate("qwen2.5:3b", prompt, format="json")["response"]
    nodes = extract_list(raw)
    return {"categories": [_pick_name(n) for n in nodes if _pick_name(n)]}

@app.get("/")
def read_root():
    return fastapi.responses.FileResponse(Path(__file__).parent / "index.html")

@app.get("/products/{category}")
def get_products(category: str):
    prompt = f'Invent 5-7 fun fictional products with name and price for the "{category}" department of a whimsical ecommerce store.'
    raw = ollama.generate("qwen2.5:3b", prompt, format="json")["response"]
    nodes = extract_list(raw)
    return {"products": [n for n in nodes if _pick_name(n).lower() != category.lower()]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
