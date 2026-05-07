import chromadb
import ollama

CHROMA_PATH = "db/chroma_db"
COLLECTION_NAME = "rag_chatbot"
EMBED_MODEL = "nomic-embed-text"

def check_db():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Collection '{COLLECTION_NAME}' has {count} documents.")
        
        if count > 0:
            query = "What is an index fund?"
            print(f"Testing embedding for: {query}")
            res = ollama.embeddings(model=EMBED_MODEL, prompt=query)
            emb = res["embedding"]
            print("Embedding successful.")
            
            results = collection.query(query_embeddings=[emb], n_results=1)
            print("Query successful.")
            print(f"Top result: {results['documents'][0]}")
        else:
            print("Collection is empty. RAG will not have local context.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
