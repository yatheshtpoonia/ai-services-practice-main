# Weekly assessment practice: Coworking Handbook RAG

Implement the five functions and terminal flow in `main.py` using `PROJECT_INSTRUCTIONS.md`. The data PDF contains fictional HarborWorks Coworking rules. The solution file is a skeleton only.

```bash
cd coworking-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` and `MODEL_NAME` in `.env` and load it in your implementation.

```bash
python3 main.py
python3 -m pytest tests.py -vv
```

The suite has exactly 15 tests: 5 chunking, 3 retrieval, 2 prompt, 2 generation and 3 pipeline checks. It uses the same test logic as the reference assessment, with only domain-specific inputs and the pipeline name changed. Settings remain 400-character chunks, 350-character steps, a 50-character minimum and 3 retrieval results.

Tests fail until the skeleton is implemented. Retrieval requires the SentenceTransformer model, downloaded on first use. Live generation and pipeline checks require valid Gemini credentials and network access. Do not commit your `.env`.
