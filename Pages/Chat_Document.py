import streamlit as st
from typing import Literal, Optional, TypedDict, Union
import numpy as np
import re
import math
from functools import lru_cache
from constants import RESPOND_TO_MESSAGE_SYSTEM_PROMPT
from db import DocumentInformationChunks, set_diskann_query_rescore, db, get_or_create_user, Conversations, Users
from peewee import SQL
from ollama_client import client
from reranker import rerank, get_reranker
import os
import time
import random

st.set_page_config(page_title="Chat With Documents")
st.title("Chat With Documents")

# ========== Configuration ==========
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
MAX_CONTEXT_CHARS = 8000          # Increased to allow more story context
CANDIDATE_LIMIT = 20              # Increased from 10
TOP_K_RESULTS = 5                 # Increased from 3
SIMILARITY_THRESHOLD = 0.65       # Lowered from 0.65 to catch more relevant chunks
# ===================================

# ========== Greeting responses ==========
GREETINGS = [
    "Hello! How can I help you today?",
    "Hi there! Ask me anything about your documents.",
    "Greetings! What would you like to know?",
    "Hey! Ready to answer your questions.",
    "Good to see you! What can I assist with?",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|sup|yo|good morning|good afternoon|good evening)(\s|$)",
    r"^(hiya|heya|hola|namaste|bonjour|ciao)(\s|$)",
    r"^what'?s up(\s|$)",
    r"^how's it going(\s|$)",
]

def is_greeting(message: str) -> bool:
    msg_lower = message.lower().strip()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, msg_lower):
            return True
    short_greetings = {"hi", "hello", "hey", "hiya", "heya", "yo", "sup"}
    return msg_lower in short_greetings

def is_followup_question(message: str) -> bool:
    """Check if the user is asking to elaborate, summarize, or explain something mentioned earlier."""
    followup_indicators = [
        "summarize", "tell me more", "elaborate", "explain", "details", 
        "full story", "what else", "go deeper", "can you", "describe",
        "list", "outline", "break down"
    ]
    msg_lower = message.lower()
    return any(indicator in msg_lower for indicator in followup_indicators)
# =====================================

@st.cache_resource
def get_cached_reranker():
    return get_reranker()

@lru_cache(maxsize=1)
def get_model_name() -> str:
    model = os.getenv("FINE_TUNED_MODEL", "")
    return model if model.strip() else "llama3.2"

# Preload reranker model once
if "reranker_loaded" not in st.session_state:
    with st.spinner("Loading models..."):
        get_cached_reranker()
        st.session_state["reranker_loaded"] = True
    set_diskann_query_rescore(50)

# Check BM25 availability
bm25_available = False
if USE_HYBRID_SEARCH:
    try:
        from bm25_index import get_bm25
        bm25, _, _ = get_bm25()
        bm25_available = bm25 is not None
    except:
        bm25_available = False

class Message(TypedDict):
    role: Union[Literal["user"], Literal["assistant"]]
    content: str
    references: Optional[list[str]]

def cosine_similarity_from_euclidean(euclidean_dist):
    return 1 - (euclidean_dist * euclidean_dist) / 2

def keyword_search(query: str, top_k: int = 3) -> list[str]:
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        return []
    conditions = " OR ".join(["chunk ILIKE %s"] * len(keywords))
    sql = f"SELECT chunk FROM document_information_chunks WHERE {conditions} LIMIT %s"
    params = [f'%{kw}%' for kw in keywords] + [top_k * 2]
    cursor = db.execute_sql(sql, params)
    rows = cursor.fetchall()
    chunk_scores = {}
    for row in rows:
        chunk = row[0]
        chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:top_k]]

def extract_section_number(query: str) -> Optional[str]:
    patterns = [
        r'\b[Ss]ection\s+(\d+)\b',
        r'\b[Ss]ec\.?\s+(\d+)\b',
        r'\b[Ss]ec\s+(\d+)\b'
    ]
    for pat in patterns:
        match = re.search(pat, query)
        if match:
            return f"Section {match.group(1)}"
    return None

def get_section_chunks(section: str) -> list[str]:
    patterns = [
        f'%{section}%',
        f'%{section.lower()}%',
        f'%{section.upper()}%',
        f'%{section} %',
        f'%{section}:%',
        f'%{section}.%',
        f'%{section} of%'
    ]
    conditions = " OR ".join(["chunk ILIKE %s"] * len(patterns))
    sql = f"SELECT chunk FROM document_information_chunks WHERE {conditions} LIMIT 10"
    cursor = db.execute_sql(sql, patterns)
    rows = cursor.fetchall()
    seen = set()
    unique = []
    for row in rows:
        ch = row[0]
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique[:5]

@st.cache_data(ttl=3600)
def get_embedding_cached(text: str):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text
    )
    return response.data[0].embedding

def retrieve_chunks(query: str, limit: int = CANDIDATE_LIMIT) -> list[str]:
    """Perform vector search with similarity threshold."""
    query_embedding = get_embedding_cached(query)
    with db.atomic():
        cursor = db.execute_sql("""
            SELECT chunk, embedding <-> %s::vector as distance
            FROM document_information_chunks
            ORDER BY distance
            LIMIT %s
        """, (query_embedding, limit * 2))
        raw_results = cursor.fetchall()
    chunks = []
    for chunk, dist in raw_results:
        sim = cosine_similarity_from_euclidean(dist)
        if sim >= SIMILARITY_THRESHOLD:
            chunks.append(chunk)
    return chunks

# Sidebar (unchanged)
with st.sidebar:
    st.header("User")
    all_users = [u.username for u in Users.select()]
    if not all_users:
        all_users = ["default_user"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        username = st.selectbox("Select user", all_users, index=0)
    with col2:
        if st.button("Delete User", type="primary", help="Delete selected user"):
            if username != "default_user":
                user_to_delete = Users.get_or_none(Users.username == username)
                if user_to_delete:
                    Conversations.delete().where(Conversations.user_id == user_to_delete.id).execute()
                    user_to_delete.delete_instance()
                    if f"messages_{user_to_delete.id}" in st.session_state:
                        del st.session_state[f"messages_{user_to_delete.id}"]
                    st.success(f"User '{username}' deleted!")
                    st.rerun()
            else:
                st.error("Cannot delete default user")
    
    new_user = st.text_input("Create new user")
    if new_user and new_user not in all_users:
        get_or_create_user(new_user)
        st.rerun()
    
    current_user = get_or_create_user(username)
    st.session_state["user_id"] = current_user.id

    st.header("History")
    if st.button("Clear All Chats"):
        Conversations.delete().where(Conversations.user_id == current_user.id).execute()
        st.session_state[f"messages_{current_user.id}"] = []
        st.rerun()
    
    doc_count = DocumentInformationChunks.select().count()
    if doc_count == 0:
        st.warning("⚠️ No documents uploaded. Upload a PDF to get answers.")

# Load messages for this user
if f"messages_{current_user.id}" not in st.session_state:
    db_messages = Conversations.select().where(Conversations.user_id == current_user.id).order_by(Conversations.timestamp.asc())
    loaded = [{"role": m.role, "content": m.message, "references": None} for m in db_messages]
    st.session_state[f"messages_{current_user.id}"] = loaded

def get_messages():
    return st.session_state[f"messages_{current_user.id}"]

def push_message(message: Message):
    messages = get_messages()
    messages.append(message)
    st.session_state[f"messages_{current_user.id}"] = messages
    Conversations.create(
        user_id=current_user.id,
        message=message["content"],
        role=message["role"]
    )

# ========== IMPROVED RESPONSE FUNCTION ==========
def send_message_fast(input_message: str):
    start_time = time.time()
    
    if is_greeting(input_message):
        greeting = random.choice(GREETINGS)
        push_message({"role": "assistant", "content": greeting, "references": None})
        return
    
    try:
        messages = get_messages()
        # Check if this is a follow-up to a previous answer
        is_followup = is_followup_question(input_message) and len(messages) >= 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "assistant"
        
        # --- 1. Retrieve chunks for the current query ---
        section = extract_section_number(input_message)
        if section:
            candidate_chunks = get_section_chunks(section)
        else:
            candidate_chunks = retrieve_chunks(input_message, CANDIDATE_LIMIT)
            if not candidate_chunks:
                candidate_chunks = keyword_search(input_message, TOP_K_RESULTS)
        
        # --- 2. If follow-up, also retrieve using the previous assistant's answer ---
        if is_followup and len(messages) >= 2:
            prev_answer = messages[-2]["content"]
            # Extract key entities from prev_answer (simple: first 200 chars)
            # Use the entire previous answer as a query for additional chunks
            additional_chunks = retrieve_chunks(prev_answer, CANDIDATE_LIMIT)
            # Merge and deduplicate
            combined = list(dict.fromkeys(candidate_chunks + additional_chunks))
            candidate_chunks = combined[:CANDIDATE_LIMIT * 2]  # Allow more chunks for follow-ups
        
        # --- 3. Hybrid search if available ---
        if USE_HYBRID_SEARCH and bm25_available and candidate_chunks:
            try:
                from bm25_index import get_bm25
                bm25, all_chunks, _ = get_bm25()
                if bm25 and all_chunks:
                    tokenized_query = input_message.split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    bm25_indices = np.argsort(bm25_scores)[-10:][::-1]  # more candidates
                    bm25_chunks = [all_chunks[i] for i in bm25_indices if i < len(all_chunks)]
                    combined = list(dict.fromkeys(candidate_chunks + bm25_chunks))
                    candidate_chunks = combined[:CANDIDATE_LIMIT * 2]
            except:
                pass
        
        # --- 4. Rerank ---
        if len(candidate_chunks) > TOP_K_RESULTS:
            top_chunks = rerank(input_message, candidate_chunks, top_k=TOP_K_RESULTS)
        else:
            top_chunks = candidate_chunks[:TOP_K_RESULTS]
        
        # --- 5. Build context (include previous assistant answer) ---
        context = "\n".join(f"{i+1}. {chunk[:MAX_CONTEXT_CHARS]}" for i, chunk in enumerate(top_chunks))
        
        # Inject previous assistant answer (if exists)
        if len(messages) >= 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "assistant":
            prev_answer = messages[-2]["content"]
            context += f"\n\nPrevious answer (for reference):\n{prev_answer[:MAX_CONTEXT_CHARS]}"
        
        # If no chunks but we have a previous answer, still try to answer using history
        if not top_chunks and len(messages) >= 2 and messages[-2]["role"] == "assistant":
            # Allow answering from history only
            context = f"Previous answer (use this to respond):\n{messages[-2]['content'][:MAX_CONTEXT_CHARS]}"
        elif not top_chunks:
            answer = "I don't have enough information in the documents to answer that."
            push_message({"role": "assistant", "content": answer, "references": None})
            return
        
        # --- 6. Generate response ---
        system_content = RESPOND_TO_MESSAGE_SYSTEM_PROMPT.replace("{{knowledge}}", context)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        messages_for_api = [
            {"role": "system", "content": system_content},
            *[{"role": m["role"], "content": m["content"]} for m in recent_messages]
        ]
        
        model_name = get_model_name()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
            max_tokens=500,  # Increased for summaries
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        if not answer:
            answer = "I couldn't generate a response. Please try again."
        
        push_message({
            "role": "assistant",
            "content": answer,
            "references": top_chunks[:5] if top_chunks else None
        })
        
        elapsed = time.time() - start_time
        # st.caption(f"Response time: {elapsed:.2f}s")  # optional
        
    except Exception as e:
        push_message({
            "role": "assistant",
            "content": f"Error: {str(e)[:200]}",
            "references": None
        })
# ===============================================

# Display conversation
for message in get_messages():
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("references"):
            with st.expander("Sources"):
                for ref in message["references"][:5]:
                    st.write(ref[:200] + "...")

# Handle user input
input_message = st.chat_input("Ask a question...")
if input_message:
    push_message({"role": "user", "content": input_message, "references": None})
    with st.spinner("Thinking..."):
        send_message_fast(input_message)
    st.rerun()