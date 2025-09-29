from flask import Flask, jsonify, render_template, request, session
from uuid import uuid4

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.history import load_message_history, message_history_factory
from src.prompt import *
import os


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o")
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
