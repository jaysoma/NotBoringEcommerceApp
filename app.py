import fastapi
import uvicorn
import ollama
import json
import urllib.request
import urllib.parse
import re
import unicodedata
from pathlib import Path


def _load_config():
    cfg = {}
    model_txt = Path(__file__).parent / "model.txt"
    current_section = None
    for line in model_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            cfg[current_section] = ''
        elif '=' in line and current_section is None:
            k, _, v = line.partition('=')
            cfg[k.strip()] = v.strip()
        elif current_section:
            cfg[current_section] = (cfg[current_section] + '\n' + line).lstrip('\n')
    return cfg

_CFG = _load_config()
_MODEL = _CFG['model']
PEXELS_KEY = (Path(__file__).parent / "pexels.key").read_text().strip()



_Q = '\x1f'  # ASCII Unit Separator — sentinel replacing all " before parsing

_SYMMETRIC = frozenset(f"\"'`|~{_Q}")  # chars that are their own closing delimiter

def _close(open_ch: str) -> str | None:
    """Derive closing token from opening token — no hardcoded map.
    Unicode Ps (open punctuation) finds its Pe pair by ASCII offset.
    Known quote-like chars are symmetric (close == open).
    Everything else (: , . - $ % etc.) is NOT a delimiter.
    """
    if open_ch in _SYMMETRIC:
        return open_ch
    if unicodedata.category(open_ch) == 'Ps':
        for delta in (1, 2):
            candidate = chr(ord(open_ch) + delta)
            if unicodedata.category(candidate) == 'Pe':
                return candidate
    return None

def _preprocess(s: str) -> str:
    s = s.replace('**', '"')
    s = s.replace('\\"', '').replace('"', _Q)
    return s

def _find_match(s: str, open_pos: int) -> int:
    open_ch = s[open_pos]
    close_ch = _close(open_ch)
    if close_ch is None:
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
    if _close(open_ch) is None:
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
    if _close(inner_delim) is None:
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
        elif _close(c) is not None and c != inner_delim:
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
    def stream():
        prompt = _CFG['categories']
        iterator = ollama.generate(_MODEL, prompt, stream=True, format="json")
        full_response = ""
        for chunk in iterator:
            token = chunk["response"]
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        names = []
        try:
            parsed = json.loads(full_response)
            # find the first list value — model may wrap under any key
            candidates = parsed if isinstance(parsed, list) else (
                parsed.get("departments") or parsed.get("categories") or
                next((v for v in parsed.values() if isinstance(v, list)), []))
            for item in candidates:
                if isinstance(item, str) and len(item) > 2:
                    names.append(item)
                elif isinstance(item, dict):
                    v = next((v for v in item.values() if isinstance(v, str) and len(v) > 2), None)
                    if v:
                        names.append(v)
        except Exception:
            pass
        if not names:
            # fallback to extract_list for unexpected shapes
            for node in extract_list(full_response):
                if len(node) == 1:
                    k, v = next(iter(node.items()))
                    name = v if (isinstance(v, str) and len(v) > 2) else (k if len(k) > 2 else None)
                    if name and name not in names:
                        names.append(name)
        yield f"data: {json.dumps({'categories': names})}\n\n"
    return fastapi.responses.StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/image/{keywords}")
def get_image(keywords: str):
    encoded = urllib.parse.quote(keywords)
    req = urllib.request.Request(
        f"https://api.pexels.com/v1/search?query={encoded}&per_page=1&orientation=portrait",
        headers={"Authorization": PEXELS_KEY, "User-Agent": "NotAmazon/1.0"}
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    photos = data.get("photos", [])
    if not photos:
        return {"ascii": None, "error": "no photos found"}
    url = photos[0]["src"]["medium"]
    return {"url": url, "photographer": photos[0].get("photographer", "")}

@app.get("/")
def read_root():
    return fastapi.responses.FileResponse(Path(__file__).parent / "index.html")

@app.get("/products/{category}")
def get_products(category: str):
    def stream():
        prompt = _CFG['products'].replace('{category}', category)
        iterator = ollama.generate(_MODEL, prompt, stream=True, format="json")
        full_response = ""
        for chunk in iterator:
            token = chunk["response"]
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        products = []
        try:
            parsed = json.loads(full_response)
            # model may wrap in any key; find the first list value
            if isinstance(parsed, list):
                products = parsed
            else:
                for v in parsed.values():
                    if isinstance(v, list):
                        products = v
                        break
        except Exception:
            products = extract_list(full_response)
        yield f"data: {json.dumps({'products': products})}\n\n"
    return fastapi.responses.StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
