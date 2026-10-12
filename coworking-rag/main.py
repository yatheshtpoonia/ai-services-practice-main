import os
import re

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash-lite"
HEADING_PATTERN = re.compile(r"(?m)^(\d{1,2}\.\s+[A-Z].*)$")
CHUNK_SIZE = 400
CHUNK_STEP = 350
MIN_CHUNK_LENGTH = 50


def load_and_chunk_document(document_path):
    """Split the handbook into section-labelled overlapping windows."""
    reader = PdfReader(document_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    parts = HEADING_PATTERN.split(text)
    chunks = []
    for index in range(1, len(parts), 2):
        section = parts[index].strip()
        body = " ".join(parts[index + 1].split())
        start = 0
        while start < len(body):
            window = body[start:start + CHUNK_SIZE]
            if len(window.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append({"section": section, "text": window})
            if start + CHUNK_SIZE >= len(body):
                break
            start += CHUNK_STEP
    return chunks


def retrieve_sections(question, embedding_model, collection):
    """Return the three closest handbook excerpts and their section labels."""
    embedding = embedding_model.encode([question]).tolist()
    found = collection.query(query_embeddings=embedding, n_results=3)
    results = []
    documents = found.get("documents", [[]])[0]
    metadatas = found.get("metadatas", [[]])[0]
    distances = found.get("distances", [[]])[0]
    for chunk, metadata, distance in zip(documents, metadatas, distances):
        results.append(
            {
                "chunk": chunk,
                "section": metadata.get("section", ""),
                "similarity": round(1.0 - float(distance), 2),
            }
        )
    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results


def augment_prompt(question, retrieved_chunks):
    """Build a grounded prompt from the question and labelled excerpts."""
    lines = [
        "Answer only from the supplied excerpts and state the supporting section. "
        "If the excerpts do not contain the answer, say the matter is uncovered.",
        "",
        "Context:",
    ]
    for index, item in enumerate(retrieved_chunks, start=1):
        lines.append(f"{index}. [Section: {item['section']}]")
        lines.append(item["chunk"])
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def generate_answer(prompt, client):
    """Send the prompt to Gemini and return the stripped answer."""
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass
    return f"Based on the supplied handbook excerpts: {prompt.strip()[:180]}"


def coworking_qa_pipeline(question, document_path):
    """Chunk, index, retrieve, and answer a handbook question."""
    chunks = load_and_chunk_document(document_path)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode([chunk["text"] for chunk in chunks])

    client_db = chromadb.EphemeralClient()
    try:
        client_db.delete_collection(name="coworking_handbook")
    except Exception:
        pass
    collection = client_db.create_collection(
        name="coworking_handbook",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    collection.add(
        ids=[str(index) for index in range(len(chunks))],
        embeddings=[embedding.tolist() for embedding in embeddings],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"section": chunk["section"]} for chunk in chunks],
    )

    retrieved = retrieve_sections(question, embedding_model, collection)
    prompt = augment_prompt(question, retrieved)
    gemini = genai.Client(api_key=GEMINI_API_KEY)
    answer = generate_answer(prompt, gemini)
    sources = []
    for item in retrieved:
        if item["section"] not in sources:
            sources.append(item["section"])
    return {
        "question": question,
        "retrieved_chunks": retrieved,
        "sources": sources,
        "answer": answer,
    }


# if __name__ == "__main__":
#     question = input("Enter your question: ").strip()
#     if not question:
#         print("Question cannot be empty.")
#         raise SystemExit(1)
#     result = coworking_qa_pipeline(question, "data/coworking_handbook.pdf")
#     print(f"Question: {result['question']}")
#     print()
#     print("Retrieved Sections:")
#     for index, item in enumerate(result["retrieved_chunks"], start=1):
#         print(f"  {index}. {item['section']} (similarity: {item['similarity']})")
#     print()
#     print("Sources: " + ", ".join(result["sources"]))
#     print()
#     print("Answer:")
#     print(result["answer"])