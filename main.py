
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import logging
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import os

app = FastAPI()

prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant virtuel nommé **WixAI**.

    Rôle :
    Tu es un assistant virtuel expert du lvire manuel de l'influenceur.

    Règles strictes :
    - Tu réponds **uniquement** aux questions liées au livre **manuel de l'influenceur**
    - Tu t’appuies **exclusivement** sur le contexte fourni
    - Si l’information n’existe pas dans le contexte, dis clairement :
    « je n'ai pas d'information à ce sujet»
    Style de réponse :
    - Réponses **précises, claires et concises**
    - Ton **professionnel et formel**
    - Pas de disgressions inutiles
    contexte: {context}
    Question: {question}
    """)
llm = OllamaLLM(model="gemma3:1b", temperature=2.5)

chain = prompt | llm

# Request schema 
class QuestionRequest(BaseModel):
    question: str


# API Endpoint 
@app.post("/ask")
def ask_question(payload: QuestionRequest):
    # Retrieve relevant documents
    docs = retriever.invoke(payload.question)

    # Run LLM
    answer = chain.invoke({
        "context": docs,
        "question": payload.question
    })

    # Return JSON
    return {
        "success": True,
        "question": payload.question,
        "answer": answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

