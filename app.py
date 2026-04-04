import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

st.title("Chat With Docs")