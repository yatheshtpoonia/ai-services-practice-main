# AI Services Practice

Self-contained weekly assessment exercises for section-attributed retrieval-augmented generation (RAG). Each exercise includes a problem statement, fictional handbook PDF, tests, dependencies and environment-variable template.

## Exercises

| Folder | Domain | Tests | Chunk / overlap | Retrieval results |
| --- | --- | --- | --- | --- |
| `employee-policy-rag` | Employee policies | 15 | 500 / 80 characters | 4 |
| `coworking-rag` | Coworking membership rules | 15 | 400 / 50 characters | 3 |

## Branches

- `main`: starter skeletons for both exercises.
- `solutions`: completed employee-policy implementation. The coworking exercise remains a skeleton; its solution has not been written yet.

Choose an exercise and follow its README. Run commands from that exercise's directory so the PDF paths resolve correctly. Install its dependencies, copy `.env.example` to `.env`, and supply your own Gemini API key and model name. Do not commit credentials.

The embedding model runs locally and may download on first use. Live generation and pipeline tests require Gemini access. Read the problem statement before implementing the TODOs.

To view the available solution locally:

```bash
git switch solutions
```

Return to starter exercises with `git switch main`. Commit or save your changes before switching branches.
