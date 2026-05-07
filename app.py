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

SYSTEM_PROMPT = """You are FinanceAI, a highly specialized, privacy-first financial advisory assistant.

You operate within a Retrieval-Augmented Generation (RAG) system. You must use the provided context (retrieved from the user's private financial data and knowledge base) to answer queries accurately and safely.

━━━━━━━━━━━━━━━━━━━━━━━
🎯 DOMAIN SCOPE
━━━━━━━━━━━━━━━━━━━━━━━
You ONLY handle topics related to finance, including:
- Personal finance (budgeting, saving, expenses, emergency funds)
- Investments (stocks, mutual funds, SIPs, ETFs, bonds)
- Banking (accounts, interest rates, credit cards, loans, EMI)
- Tax basics (especially India)
- Financial planning and wealth management

If a query is NOT related to finance:
→ Respond: "I am a finance-specific assistant and cannot help with that request."

━━━━━━━━━━━━━━━━━━━━━━━
📚 CONTEXT USAGE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━
- Use ONLY the retrieved context to generate answers.
- Do NOT invent or hallucinate information.
- If the context is insufficient or missing:
  → Respond: "I don't have enough financial data to answer this accurately."
- If multiple context chunks exist:
  → Combine them logically into a single coherent answer.
- If context conflicts:
  → Mention the uncertainty clearly.

━━━━━━━━━━━━━━━━━━━━━━━
🇮🇳 LOCALIZATION (INDIA)
━━━━━━━━━━━━━━━━━━━━━━━
- Prefer INR (₹) for currency.
- Suggest India-relevant options:
  - SIPs, mutual funds, PPF, FD, NPS, ELSS
- Use Indian tax concepts where applicable.
- Keep examples relatable to Indian users.

━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SAFETY & RISK RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Do NOT give guaranteed returns or unrealistic promises.
- Do NOT suggest illegal or unethical financial actions.
- Avoid high-risk advice unless explicitly asked, and always include caution.
- Prefer safe, conservative, and educational guidance.
- If the query involves risk (e.g., trading, crypto):
  → Include a brief risk disclaimer.

━━━━━━━━━━━━━━━━━━━━━━━
👤 PERSONALIZATION
━━━━━━━━━━━━━━━━━━━━━━━
- If user data is available (income, expenses, etc.), tailor your response.
- If key information is missing:
  → Ask clarifying questions before giving advice.

━━━━━━━━━━━━━━━━━━━━━━━
🧑‍🎓 USER LEVEL HANDLING
━━━━━━━━━━━━━━━━━━━━━━━
- Assume the user may be a beginner.
- Explain concepts in simple language.
- Avoid jargon unless requested.
- Use examples where helpful.

━━━━━━━━━━━━━━━━━━━━━━━
🗣️ RESPONSE STYLE
━━━━━━━━━━━━━━━━━━━━━━━
- Be clear, structured, and professional.
- Keep answers concise but useful.
- Use bullet points or steps when helpful.

━━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT FORMAT (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━
Always structure responses as:

1. **Answer:**
   Direct response to the question

2. **Explanation:**
   Brief reasoning using context

3. **Actionable Steps (if applicable):**
   - Step 1
   - Step 2
   - Step 3

4. **(Optional) Clarifying Question:**
   Ask if more info is needed

━━━━━━━━━━━━━━━━━━━━━━━
🚫 REFUSAL CASES
━━━━━━━━━━━━━━━━━━━━━━━
- Non-finance queries
- Missing/insufficient data
- Unsafe or illegal requests

Use polite refusal: "I am a finance-specific assistant and cannot help with that request."

━━━━━━━━━━━━━━━━━━━━━━━
🧩 FINAL INSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━
Always prioritize:
Accuracy > Creativity
Safety > Risk
Context > Assumptions

Never generate an answer without grounding it in the provided financial context.
"""

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# In-memory conversation history  [{user: ..., assistant: ...}, ...]
conversation_history = []

# ── Finance domain guard ──────────────────────────────────────────────────────
# A lightweight keyword check that rejects clearly off-topic queries before
# they reach the LLM, saving time and enforcing domain scope.
FINANCE_KEYWORDS = {
    # core finance
    "money", "finance", "financial", "invest", "investment", "investing",
    "stock", "stocks", "share", "shares", "equity", "market", "trading", "trade",
    "mutual fund", "sip", "elss", "etf", "bond", "nifty", "sensex", "bse", "nse",
    # banking
    "bank", "savings", "account", "deposit", "fd", "rd", "interest", "loan",
    "emi", "credit", "debit", "mortgage", "ppf", "nps", "epf",
    # personal finance
    "budget", "budgeting", "expense", "income", "salary", "spend", "spending",
    "saving", "emergency fund", "wealth", "net worth", "debt", "insurance",
    # tax
    "tax", "itr", "80c", "gst", "capital gain", "tds", "deduction", "exemption",
    # crypto (allowed but with disclaimer)
    "crypto", "bitcoin", "ethereum", "blockchain",
    # general financial planning
    "retire", "retirement", "pension", "portfolio", "diversif", "asset",
    "inflation", "return", "profit", "loss", "dividend", "fund",
}

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "what's up","thankyou","thanks"
}

def is_greeting(text: str) -> bool:
    """Return True if the message is a casual greeting."""
    lowered = text.strip().lower()
    return any(kw in lowered for kw in GREETING_KEYWORDS)


def is_finance_query(text: str) -> bool:
    lowered = text.lower()

    # Allow greetings through — they are handled separately
    if is_greeting(lowered):
        return True

    # Fast keyword check
    if any(kw in lowered for kw in FINANCE_KEYWORDS):
        return True

    # LLM fallback (better understanding)
    prompt = f"""
Classify if this query is related to finance.
Answer only YES or NO.

Query: {text}
"""
    try:
        res = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return "YES" in res["message"]["content"].upper()
    except:
        return False


# ── High-risk topic detector ──────────────────────────────────────────────────────
HIGH_RISK_KEYWORDS = {
    "crypto", "bitcoin", "ethereum", "blockchain", "nft", "defi",
    "trading", "day trading", "options", "futures", "derivatives",
    "leverage", "margin", "short sell", "penny stock", "forex",
    "speculation", "speculative", "high risk",
}

def is_high_risk_query(text: str) -> bool:
    """Return True if the query touches high-risk financial topics."""
    lowered = text.lower()
    return any(kw in lowered for kw in HIGH_RISK_KEYWORDS)


# ── Personalization signal detector ─────────────────────────────────────────────────
PERSONAL_DATA_KEYWORDS = {
    "my salary", "my income", "i earn", "i make", "my budget",
    "my expense", "my savings", "i save", "i spend", "my loan",
    "my emi", "my debt", "my portfolio", "my investment", "i have",
    "i invested", "i want to invest", "i can invest",
}

def has_personal_data(text: str) -> bool:
    """Return True if the user has shared personal financial data."""
    lowered = text.lower()
    return any(kw in lowered for kw in PERSONAL_DATA_KEYWORDS)

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

    # ── Domain guard: reject non-finance queries immediately ──────────────────
    if not is_finance_query(user_message):
        def _refuse_domain():
            msg = "I am a finance-specific assistant and cannot help with that request."
            yield sse("token", msg)
            yield sse("done", "")
        return Response(
            stream_with_context(_refuse_domain()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Greeting handler: fast SSE reply, no LLM / RAG needed ──────────────
    if is_greeting(user_message):
        def _greet():
            msg = "Hello! 👋 I'm your FinanceAI assistant. How can I help you with your finances today?"
            yield sse("token", msg)
            yield sse("done", "")
        return Response(
            stream_with_context(_greet()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Build prompt synchronously (fast) before opening the stream
    try:
        collection = get_collection()
        context    = retrieve(user_message, collection)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ── Context guard: fallback if RAG found nothing ─────────────────────────
    if not context:
        context = ["No relevant documents found. Answer using general financial knowledge safely."]

    def generate():
        full_reply = []
        try:
            # Build messages list
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            # Conversation history
            for h in conversation_history[-MAX_HISTORY:]:
                messages.append({"role": "user",      "content": h["user"]})
                messages.append({"role": "assistant",  "content": h["assistant"]})

            # RAG context as a system message
            messages.append({
                "role":    "system",
                "content": f"Financial Context:\n{chr(10).join(context)}",
            })

            # Risk disclaimer flag
            if is_high_risk_query(user_message):
                messages.append({
                    "role":    "system",
                    "content": "⚠️ RISK NOTICE: This query involves a high-risk topic. Include a brief risk disclaimer.",
                })

            # Personalization signal
            if has_personal_data(user_message):
                messages.append({
                    "role":    "system",
                    "content": "👤 PERSONALIZATION: The user shared personal financial data. Tailor your advice.",
                })

            # Output format reminder
            messages.append({
                "role":    "system",
                "content": (
                    "Respond using this exact structure:\n"
                    "1. **Answer:** (direct response)\n"
                    "2. **Explanation:** (brief reasoning from context)\n"
                    "3. **Actionable Steps:** (bullet points, if applicable)\n"
                    "4. **Clarifying Question:** (only if key info is missing)"
                ),
            })

            # Current user query
            messages.append({"role": "user", "content": user_message})

            # Stream from LLM
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.3,
                    "top_p":       0.9,
                    "top_k":       40,
                    "num_predict": 512,
                    "num_ctx":     2048,
                },
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    full_reply.append(token)
                    yield sse("token", token)

            # Persist to history
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
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    """Quick health-check endpoint."""
    return jsonify({"status": "ok", "model": LLM_MODEL, "embed": EMBED_MODEL})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True is required for SSE to work correctly in dev mode
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)