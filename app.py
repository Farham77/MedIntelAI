import gradio as gr
import os
import requests
import json
import pickle
from datetime import datetime
from pypdf import PdfReader
from dotenv import load_dotenv
from collections import Counter
import re
import math

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment variables!")
    print("Please create a .env file with your API key or set the environment variable.")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

# Assistant persona / system prompt (default)
BASE_PROMPT = """
You are MedIntelAI, an assistant that answers questions based on uploaded medical PDF documents.
Use only the provided document context to answer questions. If the answer is not
in the provided context, say so clearly and provide guidance on where to look in the documents.
Be concise, professional, and show citations (document name + chunk index) when possible.
Intended use: educational and informational only for medical students and healthcare learners.
Do NOT provide medical diagnoses or personalized medical advice.
"""

# Simple in-memory document store
document_chunks = []

# -----------------------------
# Simple Text Processing
# -----------------------------
def simple_chunk_text(text, chunk_size=1000, overlap=100):
    """Simple text chunking by character count."""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence end
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start:
                end = sentence_end + 1
            else:
                # Look for word boundary
                word_end = text.rfind(' ', start, end)
                if word_end > start:
                    end = word_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def extract_text_from_pdfs(files):
    """Extract and chunk text from PDFs."""
    if not files:
        return []
    
    all_chunks = []
    
    for file_idx, file in enumerate(files):
        try:
            reader = PdfReader(file.name)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if text.strip():
                chunks = simple_chunk_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        'text': chunk,
                        'file_name': os.path.basename(file.name),
                        'file_idx': file_idx,
                        'chunk_idx': chunk_idx
                    })
                
        except Exception as e:
            all_chunks.append({
                'text': f"[Error reading {os.path.basename(file.name)}: {e}]",
                'file_name': os.path.basename(file.name),
                'file_idx': file_idx,
                'chunk_idx': 0
            })
    
    return all_chunks

# -----------------------------
# Simple TF-IDF Based Retrieval
# -----------------------------
def preprocess_text(text):
    """Simple text preprocessing."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    return words

def calculate_tf_idf(query_words, chunks):
    """Simple TF-IDF calculation for ranking chunks."""
    if not chunks:
        return []
    
    # Calculate TF for query
    query_tf = Counter(query_words)
    
    # Calculate document frequencies
    doc_freq = Counter()
    all_docs = []
    
    for chunk in chunks:
        words = preprocess_text(chunk['text'])
        all_docs.append(set(words))
        for word in set(words):
            doc_freq[word] += 1
    
    # Score each chunk
    scores = []
    num_docs = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        chunk_words = preprocess_text(chunk['text'])
        if not chunk_words:
            continue
        chunk_tf = Counter(chunk_words)
        
        score = 0
        for word in query_words:
            if word in chunk_tf:
                tf = chunk_tf[word] / len(chunk_words)
                idf = math.log((num_docs + 1) / (doc_freq[word] + 1))  # smoothing
                score += tf * idf
        
        scores.append((score, idx, chunk))
    
    # Sort by score and return top chunks
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, idx, chunk in scores[:5] if score > 0]

def retrieve_relevant_chunks(query, max_chunks=3):
    """Retrieve most relevant chunks using simple TF-IDF."""
    if not document_chunks:
        return []
    
    query_words = preprocess_text(query)
    if not query_words:
        return []
    
    relevant_chunks = calculate_tf_idf(query_words, document_chunks)
    return relevant_chunks[:max_chunks]

# -----------------------------
# Document Management
# -----------------------------
def save_documents():
    """Save documents to file for persistence."""
    try:
        with open('documents.pkl', 'wb') as f:
            pickle.dump(document_chunks, f)
    except:
        pass

def load_documents():
    """Load documents from file."""
    global document_chunks
    try:
        with open('documents.pkl', 'rb') as f:
            document_chunks = pickle.load(f)
        return len(document_chunks)
    except:
        document_chunks = []
        return 0

# -----------------------------
# Query with Simple RAG
# -----------------------------
def query_groq_with_rag(message, temperature=0.3, max_tokens=1000, system_prompt=BASE_PROMPT):
    """Query Groq API with retrieved context. Returns (reply_text, context_html)."""
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set. Please add it to environment variables.", "<i>No context</i>"

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(message)
    
    # Build context with better formatting
    if relevant_chunks:
        context_parts = []
        context_html_parts = []
        for i, chunk in enumerate(relevant_chunks):
            preview = chunk['text'][:400] + ("..." if len(chunk['text']) > 400 else "")
            context_parts.append(f"Document {i+1} (from {chunk['file_name']}, chunk {chunk['chunk_idx']}):\n{chunk['text']}")
            # HTML card for UI (dark / neon friendly)
            context_html_parts.append(
                f"<div style='padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.04);margin-bottom:8px;background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));'>"
                f"<strong style='color:#dfefff'>{chunk['file_name']}</strong> — <span style='color:#9fb7ff'>chunk {chunk['chunk_idx']}</span><br>"
                f"<div style='color:#c9d8ff;margin-top:6px;font-size:13px'>{gradio_safe_html(preview)}</div>"
                f"</div>"
            )
        context = "\n\n".join(context_parts)
        context_html = "".join(context_html_parts)
        context_info = f"\n\n[Found {len(relevant_chunks)} relevant document sections]"
    else:
        context = "No relevant documents found."
        context_html = "<i style='color:#9aa6b2'>No relevant content found in uploaded documents.</i>"
        context_info = "\n\n[No matching content found in uploaded documents]"
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    if relevant_chunks:
        messages.append({
            "role": "system", 
            "content": f"Relevant document context:\n{context}"
        })
    
    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL_NAME, 
                "messages": messages, 
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"]
            return ai_response + context_info, context_html
        return f"API error {response.status_code}: {response.text}", context_html
    
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not reach Groq API. {e}", context_html

# Helper to escape HTML for safe preview rendering
def gradio_safe_html(text):
    if not text:
        return ""
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Preserve line breaks
    text = text.replace("\n", "<br>")
    # Shorten long whitespace sequences
    return text

# -----------------------------
# Gradio Interface Functions
# -----------------------------
def _normalize_history_to_messages(history):
    """
    Ensure history is in 'messages' format: a list of dicts like {'role': 'user'|'assistant', 'content': str}
    Accepts legacy formats too (list of tuples/lists [(user, assistant), ...]) and converts them.
    """
    if not history:
        return []
    # If already dict-based messages, do nothing
    if isinstance(history, list) and all(isinstance(h, dict) and 'role' in h and 'content' in h for h in history):
        return history
    # If legacy tuples, convert
    new = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            new.append({'role': 'user', 'content': user})
            new.append({'role': 'assistant', 'content': assistant})
        elif isinstance(item, dict) and 'role' in item and 'content' in item:
            new.append(item)
    return new

def process_pdfs(files):
    """Process uploaded PDFs."""
    global document_chunks
    
    if not files:
        return "No files uploaded.", "<i>No files</i>", "0 chunks"
    
    try:
        # Extract and chunk documents
        chunks = extract_text_from_pdfs(files)
        
        if not chunks:
            return "No text could be extracted from the uploaded PDFs.", "<i>No text</i>", "0 chunks"
        
        # Update global document store
        document_chunks = chunks
        save_documents()
        
        file_names = [os.path.basename(f.name) for f in files]
        status_msg = f"✅ Processed {len(files)} PDF(s): {', '.join(file_names)}"
        files_html = "<ul style='margin:6px 0;padding-left:18px'>" + "".join([f"<li style='color:#cfe8ff'>{n}</li>" for n in file_names]) + "</ul>"
        chunks_info = f"{len(chunks)} text chunks created."
        return status_msg, files_html, chunks_info
    
    except Exception as e:
        return f"❌ Error processing PDFs: {e}", "<i>Error</i>", "0 chunks"

def chat_fn(message, history, temperature, max_tokens, system_prompt):
    """Chat function for Gradio with retrieval display. Uses 'messages' format for chatbot history."""
    # Normalize history to messages format
    history = _normalize_history_to_messages(history)

    if not message or not message.strip():
        return history, "", "<i>No query entered</i>"
    
    if not document_chunks:
        reply = "Please upload and process some PDF documents first!"
        # Append messages style entries
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': reply})
        return history, "", "<i>No context</i>"
    else:
        reply, context_html = query_groq_with_rag(message, temperature, max_tokens, system_prompt)
    
    # Append message and assistant response as messages dicts
    history.append({'role': 'user', 'content': message})
    history.append({'role': 'assistant', 'content': reply})
    
    # Log interaction (store plain text)
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] | user={message} | bot={reply}\n"
        with open("chat_logs.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
    except:
        pass
    
    return history, "", context_html

def clear_all():
    """Clear everything."""
    global document_chunks
    document_chunks = []
    try:
        os.remove('documents.pkl')
    except:
        pass
    return [], "Upload PDFs to get started!", "<i>No files</i>", "0 chunks", "<i>No context</i>"

# Load existing documents on startup
loaded_count = load_documents()

# -----------------------------
# Neon / Dark CSS
# -----------------------------
CSS = """
:root{
  --bg-0: #07080c;        /* page background */
  --bg-1: linear-gradient(180deg,#071022 0%, #0a1224 100%);
  --card: rgba(10,16,30,0.7); /* card backgrounds - non-white */
  --muted: #8ea3c7;
  --text: #dfe9ff;
  --border: rgba(255,255,255,0.04);
  --neon-green: #39ff14;
  --neon-cyan: #00f5ff;
  --neon-pink: #ff3bd6;
  --glass: rgba(255,255,255,0.02);
}
body {
  background: var(--bg-1);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
/* Header */
.header {
  display:flex; align-items:center; gap:12px; padding:18px 20px;
  background: linear-gradient(90deg, rgba(3,8,23,0.65), rgba(6,12,34,0.5));
  border-radius: 10px; margin-bottom: 12px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.6), 0 0 40px rgba(0,245,255,0.03) inset;
}
.logo {
  background: linear-gradient(90deg,var(--neon-cyan),var(--neon-pink));
  color: #01020a; font-weight:800; padding:10px 14px; border-radius:10px; font-size:18px;
  box-shadow: 0 4px 30px rgba(0,245,255,0.08);
}
.subtle { color:var(--muted); font-size:14px; }
/* Cards & panels */
.card { background: var(--card); padding:14px; border-radius:10px; border:1px solid var(--border); color:var(--text); }
.small { font-size:13px; color:var(--muted); }
/* Chatbox / components */
.chatbox { background: transparent; border-radius:12px; padding:6px; }
.gradio-container .gr-chatbot {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
  border: 1px solid var(--border);
  border-radius: 12px;
  color: var(--text);
}
/* Inputs */
input, textarea, .gradio-container .form-control, .gradio-container .gr-textbox {
  background: rgba(255,255,255,0.02) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  box-shadow: none !important;
}
/* Retrieval / snippet cards */
.retrieval { max-height:260px; overflow:auto; padding-right:6px; color:var(--text); }
/* Buttons - neon styles */
.gr-button, button, .gradio-container .gr-button {
  border: none !important;
  color: #030313 !important;
  padding: 10px 14px;
  font-weight:700;
  border-radius:10px;
  cursor: pointer;
  transition: transform .08s ease, box-shadow .12s ease;
  box-shadow: 0 6px 30px rgba(0,0,0,0.6);
}
/* primary (neon cyan) */
.gr-button.primary, button.primary, .gr-button[variant="primary"] {
  background: linear-gradient(90deg, rgba(0,245,255,1), rgba(57,255,20,0.9));
  box-shadow: 0 6px 30px rgba(0,245,255,0.12), 0 0 18px rgba(0,245,255,0.12);
  color: #041017 !important;
}
/* secondary (neon pink) */
.gr-button.secondary, button.secondary, .gr-button[variant="secondary"] {
  background: linear-gradient(90deg, rgba(255,59,214,0.95), rgba(0,245,255,0.65));
  box-shadow: 0 6px 30px rgba(255,59,214,0.08), 0 0 18px rgba(255,59,214,0.06);
  color: #031018 !important;
}
/* generic neon alternative for other buttons (neon green) */
.gr-button:not(.primary):not(.secondary) {
  background: linear-gradient(90deg, rgba(57,255,20,0.95), rgba(0,245,255,0.2));
  box-shadow: 0 6px 30px rgba(57,255,20,0.06);
}
/* hover / active */
.gr-button:hover, button:hover {
  transform: translateY(-2px);
  filter: brightness(1.05);
}
.gr-button:active, button:active { transform: translateY(0); }
/* small footer text */
.footer-small { color:var(--muted); font-size:13px; margin-top:8px; }
/* make list items visible on dark bg */
.retrieval li, .card li { color: #cfe8ff; }
/* adjust gradio file component */
.gradio-container .gr-file {
  background: transparent;
  border: 1px dashed rgba(255,255,255,0.03);
  color: var(--text);
}
/* subtle scrollbar for retrieval */
.retrieval::-webkit-scrollbar { height:8px; width:8px; }
.retrieval::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.04); border-radius:8px; }
"""

with gr.Blocks(css=CSS, title="MedIntelAI — Medical Document Q&A") as demo:
    # Header
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(
                """
                <div class="header">
                  <div class="logo">MedIntelAI</div>
                  <div>
                    <div style="font-weight:700;font-size:18px;color:#e9f6ff">MedIntelAI</div>
                    <div class="subtle">Medical document intelligence — upload clinical/medical PDFs and ask context-aware questions</div>
                  </div>
                </div>
                """
            )
        with gr.Column(scale=1, min_width=220):
            gr.HTML(
                f"""
                <div class="card small">
                  <div style="font-weight:700;color:#e9f6ff">Status</div>
                  <div style="margin-top:8px;color:#cfe8ff">{loaded_count} chunks loaded</div>
                  <div style="margin-top:10px" class="footer-small">Tip: Upload medical PDFs and click Process to get started</div>
                </div>
                """
            )

    with gr.Row():
        # Left: Chat area
        with gr.Column(scale=3):
            with gr.Row():
                # Use messages format
                chatbot = gr.Chatbot([], label="Conversation with MedIntelAI", elem_classes="chatbox", type="messages")
            with gr.Row():
                with gr.Column(scale=5):
                    msg = gr.Textbox(placeholder="Ask something about the uploaded medical PDFs (e.g. 'Summarize the treatment recommendations on page 12')", label="Your question")
                with gr.Column(scale=1, min_width=120):
                    send = gr.Button("Send", variant="primary")
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
                download_btn = gr.Button("Download Transcript")
            with gr.Accordion("System prompt (assistant persona) — advanced", open=False):
                system_prompt_box = gr.Textbox(value=BASE_PROMPT, lines=4, label="System prompt")
            with gr.Row():
                with gr.Column():
                    temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="Temperature")
                with gr.Column():
                    max_tok = gr.Slider(minimum=200, maximum=2000, value=1000, step=50, label="Max tokens")

        # Right: File upload + processing + retrieved context
        with gr.Column(scale=1, min_width=320):
            with gr.Column():
                gr.HTML("<div style='font-weight:700;margin-bottom:6px;color:#e9f6ff'>Upload & process medical PDFs</div>")
                pdf_upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs", show_label=False)
                process_btn = gr.Button("Process PDFs", variant="primary")
                status = gr.Textbox(label="Status", value="Upload PDFs to get started!" if loaded_count==0 else f"Loaded {loaded_count} chunks", interactive=False)
                files_html = gr.HTML("<i style='color:#9aa6b2'>No files</i>")
                chunks_info = gr.Textbox(label="Chunks", value=f"{loaded_count} chunks" if loaded_count else "0 chunks", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div style='font-weight:700;margin-top:12px;color:#e9f6ff'>Retrieved context</div>")
                    retrieval_box = gr.HTML("<i style='color:#9aa6b2'>No context yet</i>", elem_id="retrieval")
                    gr.HTML("<div class='footer-small'>Top relevant medical document snippets are shown when you ask a question.</div>")

    # Events / Wiring
    process_btn.click(process_pdfs, inputs=[pdf_upload], outputs=[status, files_html, chunks_info])
    
    # Chat wiring: returns (chat history(as messages), clear input text, context_html)
    msg.submit(chat_fn, inputs=[msg, chatbot, temp, max_tok, system_prompt_box], outputs=[chatbot, msg, retrieval_box])
    send.click(chat_fn, inputs=[msg, chatbot, temp, max_tok, system_prompt_box], outputs=[chatbot, msg, retrieval_box])
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    clear_btn.click(lambda: ([], "Upload PDFs to get started!", "<i>No files</i>", "0 chunks", "<i>No context</i>"), outputs=[chatbot, status, files_html, chunks_info, retrieval_box])
    
    # Download transcript
    def download_transcript(history):
        """
        Accepts history in messages format (list of dicts with role/content).
        Builds a plain-text transcript pairing users and assistants where possible.
        """
        history = _normalize_history_to_messages(history)
        if not history:
            return "No conversation to download."
        lines = []
        last_user = None
        for msg in history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                last_user = content
            elif role == 'assistant':
                if last_user:
                    lines.append(f"User: {last_user}\n\nAssistant: {content}\n\n---\n")
                    last_user = None
                else:
                    lines.append(f"Assistant: {content}\n\n---\n")
            else:
                pass
        if last_user:
            lines.append(f"User: {last_user}\n\nAssistant: \n\n---\n")
        content = "\n".join(lines)
        fname = "transcript.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)
        return fname
    
    download_btn.click(download_transcript, inputs=[chatbot], outputs=[gr.File(label="Download transcript")])

    # Small footer with medical-use disclaimer
    gr.HTML("<div class='footer-small' style='margin-top:16px;color:#9aa6b2'>Built with MedIntelAI • For educational and informational use only — not for diagnosis. Intended for medical students and healthcare learners.</div>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)