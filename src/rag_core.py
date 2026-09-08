import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import re

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledgeBase"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

class RAG_CORE:

    def __init__ (self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client= chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.coll_fixed = self.client.get_or_create_collection(name="kb_fixed_size", metadata={"hnsw:space": "cosine"})
        self.coll_sentence = self.client.get_or_create_collection(name="kb_sentence_based", metadata={"hnsw:space": "cosine"})
        self.similarity_threshold = 0.45

    def load_documents(self):
        """Loads all .txt documents from the knowledgeBase directory."""
        documents = {}
        if not KB_DIR.exists():
            print(f"Warning: {KB_DIR} does not exist.")
            return documents
            
        for file_path in KB_DIR.glob("*.txt"):
            doc_id = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                documents[doc_id] = f.read().strip()
        return documents

    def chunk_fixed_size(self, text, chunk_size=200, overlap=50):
        """Strategy A: Fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def chunk_sentence_based(self, text):
        """Strategy B: Sentence-based chunks."""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i+2])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def index_knowledge_base(self):
        """Indexes documents using both chunking strategies into separate ChromaDB collections."""
        docs = self.load_documents()
        if not docs:
            print("No documents found to index.")
            return

        try:
            self.client.delete_collection("kb_fixed_size")
            self.client.delete_collection("kb_sentence_based")
        except Exception:
            pass
            
        self.coll_fixed = self.client.get_or_create_collection(name="kb_fixed_size", metadata={"hnsw:space": "cosine"})
        self.coll_sentence = self.client.get_or_create_collection(name="kb_sentence_based", metadata={"hnsw:space": "cosine"})

        fixed_ids, fixed_texts, fixed_metas = [], [], []
        sent_ids, sent_texts, sent_metas = [], [], []

        for doc_id, text in docs.items():
            f_chunks = self.chunk_fixed_size(text)
            for idx, chunk in enumerate(f_chunks):
                chunk_id = f"{doc_id}_f_{idx}"
                fixed_ids.append(chunk_id)
                fixed_texts.append(chunk)
                fixed_metas.append({"parent_doc": doc_id, "strategy": "fixed"})

            s_chunks = self.chunk_sentence_based(text)
            for idx, chunk in enumerate(s_chunks):
                chunk_id = f"{doc_id}_s_{idx}"
                sent_ids.append(chunk_id)
                sent_texts.append(chunk)
                sent_metas.append({"parent_doc": doc_id, "strategy": "sentence"})

        if fixed_texts:
            fixed_embeddings = self.model.encode(fixed_texts).tolist()
            self.coll_fixed.add(
                ids=fixed_ids,
                documents=fixed_texts,
                embeddings=fixed_embeddings,
                metadatas=fixed_metas
            )

        if sent_texts:
            sent_embeddings = self.model.encode(sent_texts).tolist()
            self.coll_sentence.add(
                ids=sent_ids,
                documents=sent_texts,
                embeddings=sent_embeddings,
                metadatas=sent_metas
            )

        print(f"Indexed {len(fixed_ids)} fixed-size chunks and {len(sent_ids)} sentence-based chunks successfully.")

    def query_rag(self, query: str, strategy: str = "sentence", top_k: int = 3):
        """Retrieves top-k chunks and applies the calibration threshold ("I don't know" fallback)."""
        collection = self.coll_sentence if strategy == "sentence" else self.coll_fixed
        
        query_embedding = self.model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        similarities = [1.0 - d for d in distances] if distances else [0.0]
        
        top_similarity = max(similarities) if similarities else 0.0

        if top_similarity < self.similarity_threshold:
            return {
                "answer": "I don't know the answer to that based on the available knowledge base.",
                "retrieved_chunks": [],
                "top_similarity": top_similarity,
                "fallback_triggered": True
            }
            
        context_text = " ".join(documents)
        answer = f"Based on Naukri.com policies: {context_text[:300]}..."
        
        return {
            "answer": answer,
            "retrieved_chunks": documents,
            "metadatas": metadatas,
            "top_similarity": top_similarity,
            "fallback_triggered": False
        }

if __name__ == "__main__":
    rag = RAG_CORE()
    rag.index_knowledge_base()
