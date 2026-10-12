import os
import re
import chromadb 
from dotenv import load_dotenv 
from sentence_transformers import SentenceTransformer 
from google import genai 
from pypdf import PdfReader

load_dotenv()

DOCUMENT_PATH = "data/employee_policy_handbook.pdf"
HEADING_PATTERN = re.compile(r"(?m)^(\d{1,2}\.\s+[A-Z].*)$")

CHUNK_SIZE = 500
CHUNK_STEP = 420
MIN_CHUNK_LENGTH = 60
EXPECTED_OVERLAP = CHUNK_SIZE - CHUNK_STEP
MIN_COVERAGE = 0.75
N_RESULTS = 4

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash"

def load_and_chunk_document(document_path) :
    reader = PdfReader(document_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    parts = HEADING_PATTERN.split(text)
    chunks = []
    for index in range(1, len(parts), 2) :
        section = parts[index].strip()
        body = " ".join(parts[index + 1].split())
        start = 0
        while start < len(body) :
            window = body[start : start + CHUNK_SIZE]
            if len(window.strip()) >= MIN_CHUNK_LENGTH :
                chunks.append(
                    {
                        "section" : section,
                        "text" : window
                    }
                )
            if start + CHUNK_SIZE > len(body) : break 
            start += CHUNK_STEP 
    return chunks 

def retrieve_sections(question, embedding_model : SentenceTransformer, collection : chromadb.Collection) :
    embedding = embedding_model.encode([question]).tolist() 
    found = collection.query(query_embeddings= embedding, n_results= N_RESULTS)
    documents = found.get("documents", [[]])[0]
    metadatas = found.get("metadatas", [[]])[0]
    distances = found.get("distances", [[]])[0]
    retrieved = []
    for chunk, metadata, distance in zip(documents, metadatas, distances) :
        retrieved.append(
            {
                "chunk" : chunk,
                "section" : metadata.get("section", ""),
                "similarity" : round(1 - float(distance), 2)
            }
        )
    retrieved.sort(key= lambda item : item["similarity"], reverse= True)
    return retrieved 

def augment_prompt(question, retrieved_chunks) :
    lines = [
        "Answer only from the supplied excerpts and name the supporting section.",
        "State the matter is not covered when the excerpts do not contain the answer.",
        "",
        "Context:"
    ]
    for index, item in enumerate(retrieved_chunks, start= 1) :
        lines.append(f"{index}. [Section: {item['section']}]")
        lines.append(item["chunk"])
    lines.append("")
    lines.append(f"Question : {question}")
    return "\n".join(lines)

def generate_answer(prompt, client : genai.Client) : 
    response = client.models.generate_content(model= MODEL_NAME, contents= prompt)
    return (response.text or "").strip()

def policy_qa_pipeline(question, document_path) :
    chunks = load_and_chunk_document(document_path)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode([chunk["text"] for chunk in chunks])
    store = chromadb.EphemeralClient()
    try :
        store.delete_collection(name = "employee_policy")
    except Exception :
        pass 
    collection = store.create_collection(
        name= "employee_policy",
        configuration= {"hnsw" : {"space" : "cosine"}},
        embedding_function= None
    )
    collection.add(
        ids= [str(index) for index in range(len(chunks))],
        embeddings= [embedding.tolist() for embedding in embeddings],
        documents= [chunk["text"] for chunk in chunks],
        metadatas= [{"section" : chunk["section"]} for chunk in chunks]
    )
    retrieved = retrieve_sections(question, embedding_model, collection)
    prompt = augment_prompt(question, retrieved)
    client = genai.Client(api_key= GEMINI_API_KEY)
    answer = generate_answer(prompt, client)
    sources = []
    for r in retrieved :
        if r["section"] not in sources :
            sources.append(r["section"])
    return {
        "question" : question,
        "retrieved_chunks" : retrieved,
        "sources" : sources,
        "answer" : answer
    }


