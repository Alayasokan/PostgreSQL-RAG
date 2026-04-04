import asyncio
import json
from io import BytesIO
import os

import streamlit as st
import pdftotext
from functools import lru_cache

from constants import GET_MATCHING_TAGS_SYSTEM_PROMPT
from db import DocumentInformationChunks, DocumentTags, Tags, db, Documents
from peewee import JOIN, NodeList, SQL
from ollama_client import client
from bm25_index import rebuild_bm25_index

st.set_page_config(page_title="Manage Documents")
st.title("Manage Documents")

# ========== Configuration ==========
IDEAL_CHUNK_LENGTH = 1000
CHUNK_OVERLAP = 200
# ====================================

@lru_cache(maxsize=1)
def get_model_name() -> str:
    model = os.getenv("FINE_TUNED_MODEL", "")
    return model if model.strip() else "llama3.2"

def delete_document(document_id: int):
    Documents.delete().where(Documents.id == document_id).execute()
    rebuild_bm25_index()

def chunk_text(text: str, chunk_size: int = IDEAL_CHUNK_LENGTH, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks

async def get_matching_tags(pdf_text: str):
    tags_result = list(Tags.select())
    if not tags_result:
        return []
    # Create a dict for O(1) lookup
    tag_name_to_id = {tag.name.lower(): tag.id for tag in tags_result}
    tag_names = list(tag_name_to_id.keys())
    
    model_name = get_model_name()
    retries = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": GET_MATCHING_TAGS_SYSTEM_PROMPT.replace(
                            "{{tags_to_match_with}}",
                            str(tag_names)
                        )
                    },
                    {"role": "user", "content": pdf_text}
                ]
            )
            content = response.choices[0].message.content
            # Expecting JSON: {"tags": ["tag1", "tag2"]}
            import json
            data = json.loads(content)
            matching_tag_names = data.get("tags", [])
            matching_tag_ids = []
            for tname in matching_tag_names:
                tname_lower = tname.lower()
                if tname_lower in tag_name_to_id:
                    matching_tag_ids.append(tag_name_to_id[tname_lower])
            return matching_tag_ids
        except Exception as e:
            retries += 1
            if retries > 5:
                raise e
            await asyncio.sleep(1)

def create_embedding(text: str):
    embedding = client.embeddings.create(
        model="nomic-embed-text",
        input=text
    )
    return embedding.data[0].embedding

def upload_document(name: str, pdf_file: bytes, progress_bar):
    """Upload document – stores original text chunks (no LLM fact extraction)."""
    progress_bar.progress(5, text="Reading PDF...")
    parsed_pdf = pdftotext.PDF(BytesIO(pdf_file))
    pdf_text = "\n\n".join(parsed_pdf)
    progress_bar.progress(10, text="Text extracted")
    
    text_chunks = chunk_text(pdf_text)
    total_chunks = len(text_chunks)
    progress_bar.progress(15, text=f"Split into {total_chunks} chunks")
    
    # Generate tags (async)
    progress_bar.progress(60, text="Generating tags...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        matching_tag_ids = loop.run_until_complete(get_matching_tags(pdf_text[:5000]))
    finally:
        loop.close()
    progress_bar.progress(65, text=f"Tags generated: {len(matching_tag_ids)} tags")
    
    # Create embeddings for each original chunk
    rows = []
    for idx, chunk in enumerate(text_chunks):
        embedding = create_embedding(chunk)
        rows.append({
            "document_id": None,
            "chunk": chunk,
            "embedding": embedding,
            "chunk_index": idx,
            "chunk_type": "original"
        })
        # Update progress from 65% to 95%
        embed_progress = 65 + (idx + 1) / total_chunks * 30
        progress_bar.progress(int(embed_progress), 
                              text=f"Creating embeddings: {idx+1}/{total_chunks}")
    
    # Insert into database
    progress_bar.progress(95, text="Saving to database...")
    with db.atomic():
        document_id = Documents.insert(name=name).execute()
        for row in rows:
            row["document_id"] = document_id
        DocumentInformationChunks.insert_many(rows).execute()
        if matching_tag_ids:
            DocumentTags.insert_many([
                {"document_id": document_id, "tag_id": tag_id} for tag_id in matching_tag_ids
            ]).execute()
    
    rebuild_bm25_index()
    progress_bar.progress(100, text="Upload complete!")

@st.dialog("Upload document")
def upload_document_dialog_open():
    pdf_file = st.file_uploader("Upload PDF file", type="pdf")
    if pdf_file:
        if st.button("Upload"):
            progress_bar = st.progress(0, text="Starting upload...")
            try:
                upload_document(pdf_file.name, pdf_file.getvalue(), progress_bar)
                st.success(f"Document '{pdf_file.name}' uploaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
                progress_bar.empty()

st.button("Upload Document", on_click=upload_document_dialog_open)

# List documents
documents = Documents.select(
    Documents.id,
    Documents.name,
    NodeList([
        SQL("array_remove(array_agg("),
        Tags.name,
        SQL("), NULL)")
    ]).alias("tags")
).join(
    DocumentTags,
    JOIN.LEFT_OUTER
).join(
    Tags,
    JOIN.LEFT_OUTER
).group_by(
    Documents.id
).execute()

if documents:
    for document in documents:
        with st.container(border=True):
            st.write(document.name)
            if document.tags:
                st.write(f"Tags: {', '.join(document.tags)}")
            st.button("Delete", key=f"{document.id}-delete", on_click=delete_document, args=(document.id,))
else:
    st.info("No documents created yet. Upload one.")