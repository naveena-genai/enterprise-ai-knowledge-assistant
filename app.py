
import streamlit as st
import os
import hashlib
import numpy as np

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise AI Knowledge Assistant")
st.write("Ask questions from your PDFs or use General AI.")


# -------------------- API SETUP --------------------


# -------------------- API SETUP --------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# If local .env key is unavailable, read Streamlit Cloud Secrets
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GEMINI_API_KEY not found. Please configure it in .env or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"



# -------------------- SESSION STATE --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "PDF Only"


# -------------------- LOAD EMBEDDING MODEL --------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------- PDF PROCESSING --------------------

def create_chunks(text, filename, page_number,
                  chunk_size=800, overlap=150):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "filename": filename,
                "page": page_number
            })

        start += chunk_size - overlap

    return chunks


def process_pdfs(uploaded_files):

    all_chunks = []

    for uploaded_file in uploaded_files:

        try:
            reader = PdfReader(uploaded_file)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""

                if text.strip():
                    page_chunks = create_chunks(
                        text,
                        uploaded_file.name,
                        page_number
                    )

                    all_chunks.extend(page_chunks)

        except Exception as error:
            st.warning(
                f"Could not read {uploaded_file.name}: {error}"
            )

    return all_chunks


def get_file_hash(uploaded_files):

    file_hash = hashlib.sha256()

    for uploaded_file in uploaded_files:
        file_hash.update(uploaded_file.name.encode())
        file_hash.update(uploaded_file.getvalue())

    return file_hash.hexdigest()


def build_pdf_index(uploaded_files):

    current_hash = get_file_hash(uploaded_files)

    if current_hash == st.session_state.file_hash:
        return

    st.session_state.chat_history = []
    st.session_state.pdf_chunks = []
    st.session_state.embeddings = None

    with st.spinner("Reading PDFs and creating search index..."):

        chunks = process_pdfs(uploaded_files)

        if not chunks:
            st.session_state.file_hash = current_hash
            st.warning(
                "No readable text found in these PDFs. "
                "Scanned PDFs may require OCR."
            )
            return

        model = load_embedding_model()

        texts = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(texts)

        st.session_state.pdf_chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.file_hash = current_hash

    st.success(
        f"Successfully processed {len(uploaded_files)} PDF(s) "
        f"and created {len(chunks)} text chunks."
    )


# -------------------- PDF SEARCH --------------------

def search_pdfs(question):

    if (
        not st.session_state.pdf_chunks
        or st.session_state.embeddings is None
    ):
        return []

    model = load_embedding_model()

    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        st.session_state.embeddings
    )[0]

    top_indices = np.argsort(similarities)[-5:][::-1]

    results = []

    for index in top_indices:

        chunk = st.session_state.pdf_chunks[index].copy()

        chunk["similarity"] = float(similarities[index])

        results.append(chunk)

    return results


# -------------------- FOLLOW-UP QUESTION --------------------

def get_search_question(question):

    follow_up_phrases = [
        "explain it",
        "explain this",
        "tell me more",
        "give an example",
        "one example",
        "in simple words",
        "simplify it",
        "what about it",
        "how about it",
        "why is that",
        "how does it work",
        "elaborate",
        "difference between them"
    ]

    question_lower = question.lower()

    if any(phrase in question_lower for phrase in follow_up_phrases):

        previous_questions = [
            message["content"]
            for message in st.session_state.chat_history
            if message["role"] == "user"
        ]

        if previous_questions:
            return previous_questions[-1] + "\n" + question

    return question


# -------------------- CHAT HISTORY --------------------

def get_conversation_history():

    history = []

    for message in st.session_state.chat_history[-8:]:

        role = "user" if message["role"] == "user" else "model"

        history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])]
            )
        )

    return history


# -------------------- GEMINI RESPONSE --------------------

def generate_answer(question, mode, retrieved_chunks):

    history = get_conversation_history()

    if mode == "General AI":

        system_prompt = """
You are a helpful general AI assistant.

Answer questions using your general knowledge.
You do not need to use uploaded PDF documents.

Explain concepts in simple, beginner-friendly language.
Give examples whenever useful.
Understand follow-up questions using conversation history.
Be friendly, clear and accurate.
"""

        user_prompt = question

    elif mode == "PDF Only":

        context = ""

        for index, chunk in enumerate(retrieved_chunks, start=1):

            context += (
                f"\nSOURCE {index}\n"
                f"Filename: {chunk['filename']}\n"
                f"Page: {chunk['page']}\n"
                f"Content: {chunk['text']}\n"
            )

        system_prompt = """
You are a PDF-based knowledge assistant.

IMPORTANT RULES:

1. Answer factual questions ONLY using the provided PDF context.
2. Do not use your general knowledge to add unsupported facts.
3. If the answer is not available in the retrieved PDF content,
   clearly say that the information was not found in the provided
   PDF sections.
4. Never invent filenames or page numbers.
5. Cite sources using the exact filename and page number provided.
6. Explain the available information in simple words.
7. You may give a clearly labelled hypothetical example to explain
   a concept, but do not present it as a fact from the PDF.
"""

        user_prompt = f"""
Retrieved PDF information:

{context}

User question:
{question}

Answer using only the retrieved PDF information.
Include the exact source filename and page number when relevant.
"""

    else:  # HYBRID MODE

        context = ""

        for index, chunk in enumerate(retrieved_chunks, start=1):

            context += (
                f"\nSOURCE {index}\n"
                f"Filename: {chunk['filename']}\n"
                f"Page: {chunk['page']}\n"
                f"Content: {chunk['text']}\n"
            )

        system_prompt = """
You are a helpful Hybrid AI assistant.

Combine information from uploaded PDFs with your general knowledge.

Keep these two sections separate:

1. From Uploaded PDFs:
   Use only the provided PDF context.
   Mention exact filenames and page numbers.
   Never invent PDF citations.

2. Additional General Explanation:
   You may use your general knowledge to explain the topic,
   provide examples and add useful background information.
   Clearly label this section as general knowledge, not PDF content.

If the PDF does not contain the answer, say so clearly.
Then provide a separate general explanation if possible.

Use simple, beginner-friendly language.
Understand follow-up questions using conversation history.
"""

        if retrieved_chunks:

            user_prompt = f"""
Retrieved PDF information:

{context}

User question:
{question}

Give the answer in separate PDF and general knowledge sections.
"""

        else:

            user_prompt = f"""
No readable PDF information is available.

User question:
{question}

Explain using your general knowledge.
Clearly mention that no PDF information was used.
"""

    contents = history + [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)]
        )
    ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3
        )
    )

    return response.text


# -------------------- MODE SELECTION --------------------

mode = st.radio(
    "Choose Answer Mode",
    ["PDF Only", "General AI", "Hybrid"],
    horizontal=True
)

if mode != st.session_state.current_mode:

    st.session_state.chat_history = []
    st.session_state.current_mode = mode


if mode == "PDF Only":
    st.info("📄 Answers will be based only on uploaded PDF content.")

elif mode == "General AI":
    st.info(
        "🤖 Ask any normal question. "
        "Uploaded PDFs will not be used in this mode."
    )

else:
    st.info(
        "🔄 Answers can combine uploaded PDF information "
        "and general AI knowledge."
    )


# -------------------- PDF UPLOAD --------------------

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# -------------------- BUILD INDEX WHEN NEEDED --------------------

if mode in ["PDF Only", "Hybrid"] and uploaded_files:

    build_pdf_index(uploaded_files)


# -------------------- CHAT DISPLAY --------------------

st.subheader("💬 Chat")

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📄 Retrieved PDF Sources"):

                for source in message["sources"]:

                    st.write(
                        f"**File:** {source['filename']} | "
                        f"**Page:** {source['page']} | "
                        f"**Similarity:** {source['similarity']:.3f}"
                    )


# -------------------- CHAT INPUT --------------------

if mode == "PDF Only" and not uploaded_files:

    st.warning("Please upload at least one PDF to use PDF Only mode.")

else:

    question = st.chat_input("Ask your question here...")

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                retrieved_chunks = []

                if mode in ["PDF Only", "Hybrid"]:

                    if st.session_state.pdf_chunks:

                        search_question = get_search_question(question)

                        retrieved_chunks = search_pdfs(search_question)

                with st.spinner("Thinking..."):

                    answer = generate_answer(
                        question,
                        mode,
                        retrieved_chunks
                    )

                st.markdown(answer)

                sources = (
                    retrieved_chunks
                    if mode in ["PDF Only", "Hybrid"]
                    else []
                )

                if sources:

                    with st.expander("📄 Retrieved PDF Sources"):

                        for source in sources:

                            st.write(
                                f"**File:** {source['filename']} | "
                                f"**Page:** {source['page']} | "
                                f"**Similarity:** {source['similarity']:.3f}"
                            )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(error_message)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })


# -------------------- CLEAR CHAT --------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()
