from os import getenv
from openai import OpenAI

client = OpenAI(
    api_key=getenv("OLLAMA_API_KEY", "ollama"),
    base_url=getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
)