from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

#data 
CSV_PATH = os.environ.get("DATA_CSV", "/root/oman_chatbot/oman_investment_chatbot_unique.csv")


#FastAPI
app = FastAPI(title="Oman Investment RAG Chatbot", version="0.2")

# Model Embedding
emb_model_name = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

#CSV
_df = pd.read_csv(CSV_PATH)
_df.fillna("", inplace=True)

docs: List[Document] = []

CHUNK_SIZE = 300  
for _, r in _df.iterrows():
    content = f"Q: {r['question']}\nA: {r['answer']}"
  
    chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    for chunk in chunks:
        docs.append(Document(
            page_content=chunk,
            metadata={"topic": r.get("topic", ""), "question": r["question"]}
        ))

vectorstore = FAISS.from_documents(docs, _embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Load flan-t5-small model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

#Prompt Template
prompt_tmpl = PromptTemplate(
template=(
"You are an assistant that answers questions about investment projects in Oman.\n"
"Use ONLY the following context. If the answer isn't in the context, say you don't know and suggest asking a domain expert.\n\n"
"Context:\n{context}\n\n"
"Question: {question}\n"
"Answer concisely and factually."
),
input_variables=["context", "question"],
)

# API Models
class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = 4
    topic: Optional[str] = None

# Creating RetrievalQA
qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=retriever,
chain_type_kwargs={"prompt": prompt_tmpl},
return_source_documents=True,
)
#FastAPI Routes
@app.get("/")
def root():
    return {"status": "ok", "entries": len(_df), "embeddings": emb_model_name}

@app.post("/chat")
def chat(req: ChatRequest):
    if req.topic:
        filtered_docs = [d for d in docs if d.metadata.get("topic") == req.topic]
        vectorstore_filtered = FAISS.from_documents(filtered_docs, _embeddings)
        retriever_filtered = vectorstore_filtered.as_retriever(search_kwargs={"k": req.k})
        qa_filtered = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_filtered,
            chain_type_kwargs={"prompt": prompt_tmpl},
            return_source_documents=True
        )
        res = qa_filtered.invoke({"query": req.question})
    else:
        res = qa.invoke({"query": req.question})

    answer = res["result"]

    query_emb = _embeddings.embed_query(req.question)
    doc_embs = [_embeddings.embed_query(d.page_content) for d in res.get("source_documents", [])]

    if doc_embs:
        sims = cosine_similarity([query_emb], doc_embs)[0]
        max_sim = max(sims)
        if max_sim < 0.5:
            answer = "I don't know. Please consult a domain expert."
    else:
        answer = "I don't know. Please consult a domain expert."

    sources = []
    for d in res.get("source_documents", [])[:req.k]:
        sources.append({
            "snippet": d.page_content[:180],
            "metadata": d.metadata
        })

    return {"answer": answer, "sources": sources}
