from models import db
from flask_sqlalchemy import SQLAlchemy
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import random
from flask_migrate import Migrate


# create a Flask app
app = Flask(__name__)

# Initialize database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345@localhost:5432/chat_bot_main'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a_random_and_secret_key_here'

db.init_app(app)

migrate = Migrate(app, db)
with app.app_context():
    from models.user import User
    from models.chathistroy import ChatHistory

# Load the documents
loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Creating the Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load the model
# llm = Ollama(model="phi3:medium")

llm = OllamaLLM(model="gemma2")
# stable

# llm = Ollama(model="llama3.2")
# llm = Ollama(model="llama3")
# llm = Ollama(model="neural-chat", temperature=0.7)


# load the memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create the chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)
greeting_responses = [
    "Hi! How may I assist you today?",
    "Hello there! What can I help you with?",
    "Greetings! Feel free to ask me anything.",
    "Hey! How's it going? How can I assist you?",
    "Hi! Ready to help with your questions."
]


# render the template
@app.route("/")
def index():
    return render_template("index.html")


# Posting the user query
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    greetings = ["hi", "hello", "hey", "greetings"]
    if user_input in greetings:
        return random.choice(greeting_responses)
    result = chain.invoke({"question": user_input, "chat_history": []})
    return result["answer"]

# Posting the user query


# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     print(data)
#     user_input = data['user_input']
#     unique_id = data['uuid']
#     username = data['username']
#     # Ensure necessary data is provided
#     if user_input and unique_id and username:
#         # Check if the user exists, otherwise create a new user
#         user = User.query.filter_by(username=username).first()
#         if not user:
#             user = User(username=username, unique_id=unique_id)
#             db.session.add(user)
#             db.session.commit()
#         # Check if the input is a greeting
#         greetings = ["hi", "hello", "hey", "greetings"]
#         if user_input.lower() in greetings:
#             return random.choice(greeting_responses)
#         # Check for matching question in the database
#         existing_chat = ChatHistory.query.filter_by(
#             question=user_input).first()
#         if existing_chat:
#             return existing_chat.answer
#         # Generate a new answer from the model
#         try:
#             result = chain.invoke({"question": user_input, "chat_history": []})
#             answer = result["answer"]
#         except Exception as e:
#             return {"error": f"Failed to generate answer: {str(e)}"}, 500
#         # Save the question and answer in the chat history
#         new_chat = ChatHistory(
#             user_id=user.id, question=user_input, answer=answer)
#         db.session.add(new_chat)
#         db.session.commit()
#         # Return the generated answer
#         return answer
#     else:
#         return {"error": "Invalid input. Please provide 'user_input', 'uuid', and 'username'."}, 400


@app.route("/user-feedback", methods=["POST"])
def chat():
    data = request.get_json()
    if data:
        username = data['user']
        status = data['status']
        feeback = data['status']

        user = User.query.filter_by(username=username).first()
        if user:
            pass
        else:
            return {'error': 'user not found'}, 404
    else:
        return {'error': 'data not provided'}, 400
    return result["answer"]


if __name__ == "__main__":
    app.run(debug=True)
