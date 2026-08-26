# Streamlit app — DataPilot AI

**Priority 8.** Chat UI over the existing RAG chatbot (Systems B and C).

## Run

From the project root:

```bash
python scripts/run_app.py
# or
streamlit run app/streamlit_app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Modes

| Sidebar option | Behaviour |
|----------------|-----------|
| Fine-tuned + RAG (System C) | LoRA adapter + FAISS (default when adapter exists) |
| RAG only (System B) | Base Qwen + FAISS |
| Retrieve only | FAISS snippets, **no LLM** (works on CPU without downloading the 1.5B model) |

First full-chat question loads Qwen 1.5B (fp16 if 4-bit bitsandbytes is unavailable) and can take several minutes on CPU. Prefer Colab T4 or a local NVIDIA GPU for a live demo. Local CPU demos can use **Retrieve only**.

## UI features

- Example BI/DE questions
- Session chat history
- Source links from retrieved chunks
- Clear conversation
- Out-of-domain refusal via the existing chatbot rules

V1 does not execute SQL or connect to a warehouse.
