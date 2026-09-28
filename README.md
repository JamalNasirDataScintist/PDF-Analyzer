# PDF Analyzer (RAG with Ollama and Chroma)

Upload a PDF and ask questions about it. The document is split into chunks, embedded with Ollama, stored in Chroma and answered by LLaMA-3 running locally through Ollama, so it works offline.

## Setup

Install [Ollama](https://ollama.com) and run `ollama pull llama3`, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```n
## Tech stack

Python, Streamlit, LangChain, Chroma, Ollama
