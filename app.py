from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import json
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY is required to run the application.")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is required to use the Gemini client.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))

gemini_safety_settings_raw = os.environ.get("GEMINI_SAFETY_SETTINGS")
gemini_safety_settings = None
if gemini_safety_settings_raw:
    try:
        gemini_safety_settings = json.loads(gemini_safety_settings_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GEMINI_SAFETY_SETTINGS must be valid JSON if provided.") from exc


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chat_model_kwargs = {
    "model": GEMINI_MODEL,
    "temperature": GEMINI_TEMPERATURE,
}

if gemini_safety_settings is not None:
    chat_model_kwargs["safety_settings"] = gemini_safety_settings

chatModel = ChatGoogleGenerativeAI(**chat_model_kwargs)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Given a chat history and the latest user"
            " question which might reference context in the history, rewrite the"
            " question to be a standalone query. If no rewriting is needed, return"
            " the question as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt + "\n\nRelevant context from medical literature:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

session_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    data = request.get_json(silent=True) or request.form
    msg = data.get("msg") if data else None
    if not msg:
        return jsonify({"error": "Message is required."}), 400

    session_id = (
        data.get("session_id")
        if isinstance(data, dict)
        else request.form.get("session_id")
    ) or request.remote_addr

    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},
    )
    answer = response.get("answer", "")
    return jsonify({"answer": answer})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
