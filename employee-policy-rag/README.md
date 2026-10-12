# Weekly assessment practice: Employee Policy RAG

Read `Question_Policy_RAG.pdf` for the assessment requirements. Implement the five functions and terminal block in the `main.py` skeleton. The completed employee-policy solution is available on the `solutions` branch.

## Setup

From the repository root, enter `employee-policy-rag`, then run these commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. Load the environment in your implementation before using Gemini.

## Practice and validation

```bash
python3 main.py
python3 -m pytest tests.py -vv
```

The suite contains 15 test cases. Retrieval tests require the SentenceTransformer model (downloaded on first use). Live-answer and pipeline tests require valid Gemini credentials and network access. Tests are expected to fail until you implement the skeleton. The full suite requires the installed dependencies and working Gemini configuration.

The three-page handbook contains fictional policies created for this assessment. It is the knowledge base at `data/employee_policy_handbook.pdf`.
