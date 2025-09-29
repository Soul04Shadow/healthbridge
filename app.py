from flask import Flask, jsonify, render_template, request, session
from uuid import uuid4
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.history import load_message_history, message_history_factory
from src.prompt import *
import json
import os


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")


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
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    message_history_factory,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)


@app.route("/")
def index():
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid4())
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid4())
        session["conversation_id"] = conversation_id
    msg = request.form["msg"]
    print(msg)
    response = conversational_rag.invoke(
        {"input": msg},
        config={"configurable": {"session_id": conversation_id}},
    )
    print("Response : ", response["answer"])
    return str(response["answer"])


@app.route("/history", methods=["GET"])
def history():
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Unauthorized"}), 401
    history_store = load_message_history(conversation_id)
    messages = [
        {"role": message.type, "content": message.content}
        for message in history_store.messages
    ]
    return jsonify(messages)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
