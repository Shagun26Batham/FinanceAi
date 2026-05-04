import json
import chromadb
import ollama
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_PATH     = "db/chroma_db"
COLLECTION_NAME = "rag_chatbot"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "phi3:latest"
TOP_K           = 3
MAX_HISTORY     = 5

SYSTEM_PROMPT = (
    "You are a smart financial advisor who explains things clearly and naturally. "
    "Avoid robotic responses. Do not give generic greetings. "
    "Be practical and helpful. Keep answers concise."
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# In-memory conversation history  [{user: ..., assistant: ...}, ...]
conversation_history = []

# ── RAG helpers ───────────────────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_query(query: str):
    res = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return res["embedding"]


def retrieve(query: str, collection):
    if collection.count() == 0:
        return []
    emb     = embed_query(query)
    results = collection.query(query_embeddings=[emb], n_results=TOP_K)
    return results["documents"][0] if results["documents"] else []


def format_history(history):
    lines = []
    for h in history:
        lines.append(f"User: {h['user']}")
        lines.append(f"Assistant: {h['assistant']}")
    return "\n".join(lines)


def build_prompt(history, context, question: str) -> str:
    ctx  = "\n---\n".join(context) if context else ""
    hist = format_history(history)

    prompt = ""
    if hist:
        prompt += f"Conversation:\n{hist}\n\n"

    prompt += (
        "You are a financial literacy assistant who explains concepts in a simple "
        "and practical way.\n\n"
        "Stay within the domain of finance, but respond naturally and conversationally.\n"
        "Use the provided context if it is helpful, otherwise rely on your general knowledge.\n"
        "Keep responses clear, useful, and easy to understand.\n\n"
    )

    if ctx:
        prompt += f"Context:\n{ctx}\n\n"

    prompt += f"User: {question}\nAssistant:"
    return prompt.strip()


# ── SSE helper ────────────────────────────────────────────────────────────────
def sse(event: str, data: str) -> str:
    """Format a single Server-Sent Event frame."""
    payload = json.dumps({"data": data})
    return f"event: {event}\ndata: {payload}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Streaming endpoint.
    Emits SSE events:
      event: token  →  one token of the assistant reply
      event: done   →  signals end-of-stream (no data needed)
      event: error  →  error message string
    """
    global conversation_history

    data         = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build prompt synchronously (fast) before opening the stream
    try:
        collection = get_collection()
        context    = retrieve(user_message, collection)
        prompt     = build_prompt(conversation_history[-MAX_HISTORY:], context, user_message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def generate():
        full_reply = []
        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p":       0.9,
                    "top_k":       20,    # limits vocab search → faster sampling
                    "num_predict": 150,   # shorter max reply → faster overall
                    "num_ctx":     1024,  # smaller context window → faster prefill
                },
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    full_reply.append(token)
                    yield sse("token", token)

            # Persist to history once stream is complete
            reply_text = "".join(full_reply).strip()
            conversation_history.append({"user": user_message, "assistant": reply_text})
            if len(conversation_history) > MAX_HISTORY:
                conversation_history[:] = conversation_history[-MAX_HISTORY:]

            yield sse("done", "")

        except Exception as e:
            yield sse("error", str(e))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True is required for SSE to work correctly in dev mode
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)