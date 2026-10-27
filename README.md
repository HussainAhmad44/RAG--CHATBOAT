# Simple Document RAG

A local Streamlit app to upload PDF, DOCX and TXT files and ask grounded questions. Uses TF-IDF retrieval and an OpenAI-compatible chat API; no external vector database is required.

## Windows quickstart

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with a **new** API key, compatible endpoint (if needed), and supported model.

```powershell
streamlit run app.py
```

## Notes
- Text-based PDFs only; scanned PDFs need OCR (not included).
- Documents are held in the app session, not saved to a database.
- Text from relevant chunks is sent to the configured LLM provider.
- Retrieval uses local TF-IDF, a lightweight lexical RAG baseline, rather than semantic embeddings.
- Do not commit `.env` to Git.
