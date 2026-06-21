import fastapi
import uvicorn
import ollama
import json
from pathlib import Path


_Q = '\x00'  # sentinel replacing all " characters before parsing
_OPEN_CLOSE = {'{': '}', '[': ']', _Q: _Q}

def _preprocess(s: str) -> str:
    s = s.replace('**', '"')
    s = s.replace('\\"', '').replace('"', _Q)
    return s

def _find_match(s: str, open_pos: int) -> int:
    open_ch = s[open_pos]
    close_ch = _OPEN_CLOSE.get(open_ch)
    if not close_ch:
        return -1
    if open_ch == _Q:
        return s.find(_Q, open_pos + 1)
    depth = 0
    i = open_pos
    while i < len(s):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        elif s[i] == _Q:
            end = s.find(_Q, i + 1)
            if end == -1:
                break
            i = end
        i += 1
    return -1

def _parse(s: str) -> list:
    s = s.strip()
    if not s:
        return []
    open_ch = s[0]
    if open_ch not in _OPEN_CLOSE:
        return [s]
    close_pos = _find_match(s, 0)
    if close_pos == -1:
        return []
    if open_ch == _Q:
        content = s[1:close_pos]
        return [content] if content else ['']
    inner = s[1:close_pos]
    if not inner.strip():
        return []
    inner_delim = inner.lstrip()[0]
    if inner_delim not in _OPEN_CLOSE:
        return [inner.strip()]
    children = []
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c == inner_delim:
            match = _find_match(inner, i)
            if match == -1:
                break
            subtree = _parse(inner[i:match+1])
            if subtree:
                children.append(subtree if len(subtree) > 1 else subtree[0])
            i = match + 1
        elif c in _OPEN_CLOSE and c != inner_delim:
            match = _find_match(inner, i)
            if match == -1:
                break
            subtree = _parse(inner[i:match+1])
            if subtree:
                children.append(subtree)
            i = match + 1
        else:
            i += 1
    # Tag the result so _flatten_nodes knows the container type
    return ('arr', children) if open_ch == '[' else ('obj', children)

def _flatten_nodes(tree) -> list[dict]:
    if isinstance(tree, str):
        return []
    if not isinstance(tree, tuple):
        # bare list — recurse
        nodes = []
        for child in tree:
            nodes.extend(_flatten_nodes(child))
        return nodes

    kind, children = tree

    if kind == 'arr':
        # Each child is its own item
        nodes = []
        for child in children:
            if isinstance(child, str):
                if child:
                    nodes.append({child: ''})
            else:
                nodes.extend(_flatten_nodes(child))
        return nodes

    # kind == 'obj': pair consecutive strings as key:value
    nodes = []
    obj = {}
    pending_key = None
    for child in children:
        if isinstance(child, str):
            if pending_key is None:
                pending_key = child
            elif child == '':
                obj[pending_key] = ''
                pending_key = None
            else:
                obj[pending_key] = child
                pending_key = None
        else:
            if obj:
                nodes.append(obj)
                obj = {}
                pending_key = None
            nodes.extend(_flatten_nodes(child))
    if obj:
        nodes.append(obj)
    return nodes

def extract_list(raw) -> list[dict]:
    """
    Parse any source into a list of flat dicts — one per {} node.
    No assumptions about what the data means. That's the caller's job.
    """
    if isinstance(raw, (dict, list)):
        raw = json.dumps(raw)
    text = _preprocess(str(raw).strip())
    tree = _parse(text)
    nodes = _flatten_nodes(tree)
    if not nodes:
        # Plain text fallback: extract sentinel-delimited tokens
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == _Q:
                match = text.find(_Q, i + 1)
                if match == -1:
                    break
                token = text[i+1:match].strip()
                if token:
                    tokens.append({token: ''})
                i = match + 1
            else:
                i += 1
        nodes = tokens
    seen = set()
    result = []
    for node in nodes:
        # Discard nodes where any value is empty (empty branch)
        if any(v == '' for v in node.values()) and len(node) > 1:
            continue
        key = frozenset(node.items())
        if key not in seen:
            seen.add(key)
            result.append(node)
    return result


app = fastapi.FastAPI()

@app.get("/categories")
def get_categories():
    prompt = "Invent 5 fun and creative department names for a fictional ecommerce store."
    raw = ollama.generate("qwen2.5:7b", prompt, format="json")["response"]
    print("RAW:", repr(raw))
    nodes = extract_list(raw)
    print("NODES:", nodes)
    names = []
    seen = set()
    for node in nodes:
        # Value-as-label: pick shortest non-empty value
        name = min(
            (v for v in node.values() if isinstance(v, str) and len(v) > 2),
            key=len, default=None
        )
        # Key-as-label: if node is from a string array (single key, empty value), key is the label
        if not name and len(node) == 1 and list(node.values())[0] == '':
            name = list(node.keys())[0] if len(list(node.keys())[0]) > 2 else None
        print("NODE:", node, "-> NAME:", name)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    print("NAMES:", names)
    return {"categories": names}

@app.get("/")
def read_root():
    return fastapi.responses.FileResponse(Path(__file__).parent / "index.html")

@app.get("/products/{category}")
def get_products(category: str):
    prompt = f'Invent 5-7 fun fictional products with name and price for the "{category}" department of a whimsical ecommerce store.'
    raw = ollama.generate("qwen2.5:7b", prompt, format="json")["response"]
    nodes = extract_list(raw)
    # Each node is a product dict — pass through as-is, filter out the category echo
    return {"products": [
        n for n in nodes
        if not any(v.lower() == category.lower() for v in n.values() if isinstance(v, str))
    ]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
