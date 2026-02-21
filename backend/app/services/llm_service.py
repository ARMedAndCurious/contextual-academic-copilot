from google import genai
from app.config import settings
from pathlib import Path
import json


# Initialize Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)


# Prompt directory
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = BASE_DIR / "prompts"


def load_prompt(file_name: str) -> str:
    with open(PROMPT_DIR / file_name, "r") as f:
        return f.read()


def call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.0
        }
    )
    return response.text.strip()


def clean_json_response(text: str) -> str:
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading "json" if present
    if text.lower().startswith("json"):
        text = text[4:].strip()

    return text


def generate_answer(context: str, question: str) -> str:
    template = load_prompt("qa_prompt.txt")

    prompt = template.format(
        context=context,
        question=question
    )

    return call_gemini(prompt)


def generate_comparison(context: str, question: str) -> dict:
    template = load_prompt("compare_prompt.txt")

    prompt = template.format(
        context=context,
        question=question
    )

    raw_response = call_gemini(prompt)
    cleaned = clean_json_response(raw_response)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse comparison output",
            "raw_response": cleaned
        }
