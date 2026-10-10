"""RAG core with two chunking strategies and empirically measured threshold."""
from __future__ import annotations
import json, math, re
from pathlib import Path
from collections import Counter
from functools import lru_cache

KB_DIR = Path(__file__).parent / "knowledge_base"
OUT_DIR = Path(__file__).parent / "artifacts"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def load_documents() -> list[dict]:
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = next((x[2:].strip() for x in lines if x.startswith("# ")), path.stem)
        body = "\n".join(x for x in lines if not x.startswith("#")).strip()
        docs.append({"doc_id": path.stem, "title": title, "text": body})
    return docs

def fixed_chunks(documents, size=320, overlap=60):
    out=[]
    for d in documents:
        text=d["text"]
        start=0
        while start < len(text):
            chunk=text[start:start+size].strip()
            if chunk: out.append({**d,"chunk_id":f'{d["doc_id"]}-f-{start}',"chunk":chunk})
            if start+size >= len(text): break
            start += size-overlap
    return out

def sentence_chunks(documents, max_sentences=2):
    out=[]
    for d in documents:
        sentences=[s.strip() for s in SENTENCE_RE.split(d["text"]) if s.strip()]
        for i in range(0,len(sentences),max_sentences):
            chunk=" ".join(sentences[i:i+max_sentences])
            out.append({**d,"chunk_id":f'{d["doc_id"]}-s-{i}',"chunk":chunk})
    return out

def _tokens(s):
    stop={"the","is","a","an","what","how","can","i","for","to","and","or","of","in","on","it","my","me","who","won","does","do","with","this","that","are","be","after"}
    tokens=[t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in stop]
    def stem(t):
        if len(t)>5 and t.endswith("ies"): return t[:-3]+"y"
        if len(t)>5 and t.endswith("ing"): return t[:-3]
        if len(t)>4 and t.endswith("ed"): return t[:-2]
        if len(t)>4 and t.endswith("s") and not t.endswith("ss"): return t[:-1]
        return t
    return [stem(t) for t in tokens]

class LocalRetriever:
    """SentenceTransformer + Chroma when installed; lexical fallback otherwise."""
    def __init__(self, strategy="sentence", documents=None):
        self.strategy=strategy
        self.documents=documents or load_documents()
        self.chunks=sentence_chunks(self.documents) if strategy=="sentence" else fixed_chunks(self.documents)
        self.mode="lexical-fallback"
        self.collection=None
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.client=chromadb.Client()
            cname=f"nykaa_{strategy}_chunks"
            try: self.client.delete_collection(cname)
            except Exception: pass
            self.collection=self.client.create_collection(name=cname, metadata={"hnsw:space":"cosine"})
            embeddings=self.model.encode([c["chunk"] for c in self.chunks], normalize_embeddings=True).tolist()
            self.collection.upsert(
                ids=[c["chunk_id"] for c in self.chunks],
                documents=[c["chunk"] for c in self.chunks],
                metadatas=[{"doc_id":c["doc_id"],"title":c["title"]} for c in self.chunks],
                embeddings=embeddings
            )
            self.mode="sentence-transformers+chromadb"
        except Exception as exc:
            self.fallback_reason=f"{type(exc).__name__}: {exc}"
            self._vectors=[Counter(_tokens(c["chunk"])) for c in self.chunks]

    @staticmethod
    def _cosine(a,b):
        dot=sum(v*b.get(k,0) for k,v in a.items())
        na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.0

    def search(self, query, top_k=3):
        if self.collection is not None:
            qv=self.model.encode([query], normalize_embeddings=True).tolist()[0]
            result=self.collection.query(query_embeddings=[qv],n_results=min(top_k,len(self.chunks)),include=["documents","metadatas","distances"])
            rows=[]
            for i,txt in enumerate(result["documents"][0]):
                meta=result["metadatas"][0][i]
                rows.append({"doc_id":meta["doc_id"],"title":meta["title"],"chunk":txt,
                             "similarity":max(0.0,1-float(result["distances"][0][i]))})
            return rows
        q=Counter(_tokens(query))
        rows=[]
        for c,v in zip(self.chunks,self._vectors):
            sim=self._cosine(q,v)
            if sim:
                rows.append({"doc_id":c["doc_id"],"title":c["title"],"chunk":c["chunk"],"similarity":sim})
        return sorted(rows,key=lambda r:r["similarity"],reverse=True)[:top_k]

    def answer(self, query, threshold=None, top_k=3):
        hits=self.search(query,top_k)
        threshold = threshold if threshold is not None else calibrated_threshold(self)["threshold"]
        if not hits or hits[0]["similarity"] < threshold:
            return {"answer":"I don't know based on the available policy documents. Please contact customer support for confirmation.",
                    "grounded":False,"similarity":hits[0]["similarity"] if hits else 0.0,"sources":[],"cache_hit":False}
        # Extractive mock generation: only retrieved text is used.
        answer=" ".join(h["chunk"] for h in hits[:2])
        return {"answer":answer,"grounded":True,"similarity":hits[0]["similarity"],
                "sources":list(dict.fromkeys(h["doc_id"] for h in hits)),"cache_hit":False}

def calibrated_threshold(retriever):
    in_scope=[
      "What is the return window for Beauty products?",
      "How long can a COD refund take?",
      "What is the delivery SLA after dispatch?",
    ]
    out_scope=[
      "Who won the 2030 World Cup?",
      "Explain quantum entanglement and black holes.",
    ]
    good=[retriever.search(q,1)[0]["similarity"] if retriever.search(q,1) else 0 for q in in_scope]
    bad=[retriever.search(q,1)[0]["similarity"] if retriever.search(q,1) else 0 for q in out_scope]
    max_bad=max(bad); min_good=min(good)
    threshold=(max_bad+min_good)/2 if max_bad < min_good else min_good
    return {"threshold":threshold,"in_scope":dict(zip(in_scope,good)),"out_of_scope":dict(zip(out_scope,bad)),
            "clusters_separated":max_bad < min_good,"strategy":retriever.strategy,"mode":retriever.mode}

def relevance_for_query(query, doc_id):
    """Small explicit gold map used for reproducible document-level metrics."""
    q=query.lower()
    mapping=[
      (("return window","beauty return","return policy"),"return-window"),
      (("cod refund","cash-on-delivery refund"),"cod-refunds"),
      (("delivery sla","delivery time","after dispatch"),"delivery-sla"),
      (("reverse pickup","reverse-pickup"),"reverse-pickup"),
      (("warranty","manufacturer warranty"),"warranty"),
      (("cancel","cancellation"),"cancellation"),
      (("loyalty points","redeem points"),"loyalty-points"),
      (("payment failed","payment retry","debit"),"payment-retry"),
      (("size exchange","exchange size","exchange apparel size"),"size-exchange"),
      (("damaged item","damage claim"),"damaged-claim"),
      (("international shipping","ship internationally"),"international-shipping"),
      (("escalation matrix","escalate"),"escalation-matrix"),
    ]
    expected={d for terms,d in mapping if any(t in q for t in terms)}
    return doc_id in expected if expected else False

def precision_recall(retriever, queries):
    rows=[]
    for query in queries:
        hits=retriever.search(query,top_k=3)
        retrieved=list(dict.fromkeys(h["doc_id"] for h in hits))
        relevant={d["doc_id"] for d in retriever.documents if relevance_for_query(query,d["doc_id"])}
        tp=len(set(retrieved)&relevant)
        precision=tp/len(retrieved) if retrieved else 0.0
        recall=tp/len(relevant) if relevant else 0.0
        rows.append({"query":query,"retrieved_docs":retrieved,"relevant_docs":sorted(relevant),
                     "arithmetic":f"precision={tp}/{len(retrieved)}; recall={tp}/{len(relevant)}",
                     "precision":round(precision,3),"recall":round(recall,3)})
    return rows
