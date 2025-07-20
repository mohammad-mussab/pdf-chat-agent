# PDF RAG Chatbot (Streamlit + Pinecone + BAAI/bge)

A modern, interactive chatbot UI for chatting with your own PDF documents using Retrieval-Augmented Generation (RAG), Pinecone vector store, and BAAI/bge-small-en-v1.5 embeddings.

---

## ✨ Features

- **PDF Upload in Sidebar:** Upload your PDF document from the sidebar. Only one PDF is loaded at a time.
- **Modern Chat UI:**
  - User and bot messages appear in chat bubbles with avatars (👤 for user, 🤖 for bot).
  - Dark mode, clean layout, and responsive design.
  - Chat history is preserved for the current PDF.
- **RAG with Pinecone:**
  - PDF is split into chunks and embedded using BAAI/bge-small-en-v1.5.
  - Chunks are stored in Pinecone for fast retrieval.
  - When you ask a question, the app retrieves relevant chunks and answers using an LLM.
- **Clear Feedback:**
  - Shows spinners for "Thinking..." and "Calling tool to search PDF..." when appropriate.
  - Error handling and user-friendly messages.
- **Clear Chat Button:**
  - Easily clear the chat history for the current PDF.

---

## 🚀 Getting Started

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 2. **Set Environment Variables**

Create a `.streamlit` folder in your project root and create the file `secrets.toml` in that folder with your API keys:

```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
PINECONE_INDEX_NAME=rag-agent
OPENAI_API_KEY=your-openai-api-key  
```

### 3. **Run the App**

```bash
streamlit run streamlit_app.py
```

Open the provided local URL in your browser.

---

## 📝 Usage

1. **Upload a PDF** in the sidebar. Wait for processing to complete.
2. **Chat with your PDF** in the main area. Type your question and press Send.
3. **See chat history** with avatars and bubbles. User messages are blue (👤), bot messages are gray (🤖).
4. **Clear chat** with the button in the sidebar if you want to start over.
5. **Upload a new PDF** to replace the current one and start a new chat.

---

## 💡 Tips
- Ask specific questions about the content of your PDF for best results.
- Only one PDF is active at a time. Uploading a new PDF will clear the previous chat.
- The app uses Pinecone for fast semantic search and BAAI/bge-small-en-v1.5 for high-quality embeddings.

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) for the UI
- [Pinecone](https://www.pinecone.io/) for vector storage
- [LangChain](https://python.langchain.com/) for RAG logic
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for embeddings
- [OpenAI](https://openai.com/) for LLM responses

---

## 📄 License
MIT 
