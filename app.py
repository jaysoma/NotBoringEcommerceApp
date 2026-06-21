import fastapi
import uvicorn
import ollama
import json
from pathlib import Path

sample_output = '{ "data": { "store_categories": { "title": "Welcome to the Intergalactic E-commerce Emporium", "categories": [ { "id": "Galactic Gadgets", "subcategories": [ { "id": "Space-Age Smartphones", "description": "The latest in interstellar communication technology" }, { "id": "Quantum Laptops", "description": "Powerful processors that can warp time and space" }, { "id": "Black Hole Televisions", "description": "Watch your favorite shows while bending reality" } ] }, { "id": "Stellar Fashion", "subcategories": [ { "id": "Andromedan Apparel for Men", "description": "Look dapper as you journey through the cosmos" }, { "id": "Alien Attire for Women", "description": "Fashion-forward clothing for the intergalactic explorer in you" }, { "id": "Outer Space Outfits for Kids", "description": "Bright, fun, and designed to withstand zero gravity" } ] }, { "id": "Cosmic Home & Kitchen", "subcategories": [ { "id": "Universal Appliances", "description": "The best equipment for cooking up a storm in space" }, { "id": "Time-Traveling Tech", "description": "Innovative gadgets to help you navigate the timestream" }, { "id": "Starship Supplies", "description": "Everything you need for a successful voyage" } ] } ] } } }'


app = fastapi.FastAPI()

def parse_ollama_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())

@app.get("/categories")
def get_categories():
    prompt = (
        'Respond with ONLY valid JSON in exactly this shape: {"categories": ["Name1", "Name2", ...]}. '
        "No explanation, no markdown, no extra keys. "
        "Invent 5-7 fun and creative category names for a fictional ecommerce app."
    )
    raw = ollama.generate("qwen2.5:3b", prompt, format="json")["response"]
    print("RAW CATEGORIES:", raw)
    parsed = parse_ollama_json(raw)
    print("PARSED:", parsed)
    return parsed

@app.get("/")
def read_root():
    return fastapi.responses.FileResponse(Path(__file__).parent / "index.html")

@app.get("/products/{category}")
def get_products(category: str):
    prompt = (
        f"Respond with ONLY valid JSON, no explanation, no markdown. "
        f"Create a fun and creative products list for the '{category}' category of a fictional ecommerce app."
    )
    ai_response = ollama.generate("qwen2.5:3b", prompt, format="json")
    ai_response = parse_ollama_json(ai_response["response"])
    return ai_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
