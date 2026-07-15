# Architecture Rules

## Immutable Constraints
1. **Monolithic Pipeline**: The architecture is a deterministic sequential pipeline.
2. **Single Conditional Loop**: Re-retrieval is the only non-linear control flow in the system.
3. **No Multi-Agent Orchestration**: Multi-agent frameworks are strictly forbidden to preserve the clarity of the triage head's contribution.
4. **No External Vector DBs**: Retrieval must use local FAISS and BM25 indices loaded into RAM.
5. **No Networked Databases**: Database state is restricted to SQLite via SQLAlchemy.
6. **No General-Purpose LLMs**: ChatGPT, Claude, and similar LLMs are explicitly excluded from the verification and triage pipeline.
7. **No Live Web Search**: Retrieval is strictly scoped to the static FEVER Wikipedia snapshot.

*Violation of these rules requires explicit architectural board approval and an update to the DECISION_LOG.md.*
