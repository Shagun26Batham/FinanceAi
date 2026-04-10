import chromadb
import ollama

CHROMA_PATH = "db/chroma_db"
COLLECTION_NAME = "rag_chatbot"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:latest"

TOP_K = 3
MAX_HISTORY = 5


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_query(query):
    res = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return res["embedding"]


def retrieve(query, collection):
    if collection.count() == 0:
        return []
        
    emb = embed_query(query)
    results = collection.query(query_embeddings=[emb], n_results=TOP_K)
    return results["documents"][0] if results["documents"] else []


def format_history(history):
    lines = []
    for h in history:
        lines.append(f"User: {h['user']}")
        lines.append(f"Assistant: {h['assistant']}")
    return "\n".join(lines)


def build_prompt(history, context, question):
    ctx = "\n---\n".join(context)
    hist = format_history(history)

    prompt = ""

    if hist:
        prompt += f"Conversation:\n{hist}\n\n"

    prompt += f"Context:\n{ctx}\n\n"
    prompt += f"User: {question}\nAssistant:"

    return prompt


def chat(prompt):
    res = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer concisely using provided context. Do not hallucinate."},
            {"role": "user", "content": prompt}
        ],
        options={"num_predict": 512, "temperature": 0.3}
    )
    return res["message"]["content"].strip()


def main():
    collection = get_collection()
    history = []

    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        try:
            context = retrieve(user_input, collection)
            prompt = build_prompt(history[-MAX_HISTORY:], context, user_input)
            response = chat(prompt)

            print(f"\nAssistant: {response}\n")

            history.append({"user": user_input, "assistant": response})

            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

        except Exception as e:
            print(f"\n[Error: {e}]\nPlease make sure Ollama is running and both 'phi3:latest' and 'nomic-embed-text' are installed via 'ollama pull <model>'\n")


if __name__ == "__main__":
    main()