from fastapi import FastAPI
from pydantic import BaseModel
from typing import Tuple
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer("all-MinLM")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to load and preprocess blog data
def load_and_preprocess_blog(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        blog_data = file.read()
    blog_posts = re.split(r"\n----\n", blog_data)
    cleaned_data = [
        re.sub(r"\s+", " ", text.strip()) for text in blog_posts if text.strip()
    ]
    return cleaned_data


# Preprocess the query
def preprocess_query(query, model):
    query_clean = re.sub(r"\s+", " ", query.strip())
    query_embedding = model.encode([query_clean])
    return query_embedding


# Create FAISS index
def create_faiss_index(blog_data, model):
    blog_embeddings = model.encode(blog_data)
    dim = blog_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(blog_embeddings))
    return index, blog_embeddings


# Retrieve the answer
def retrieve_answer(query, index, blog_data, model):
    query_embedding = preprocess_query(query, model)
    D, I = index.search(query_embedding, k=1)
    most_similar_idx = I[0][0]
    answer = blog_data[most_similar_idx]
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


# Function to generate follow-up question
def generate_follow_up(query, answer, model):
    context = f"Question: {query}\nAnswer: {answer}"
    context_embedding = model.encode([context])
    predefined_questions = [
        "Would you like more details on this topic?",
        "Would you like to know how this applies to your specific case?",
        "Is there a particular aspect you'd like to learn more about?",
        "Do you have any further questions on this subject?",
    ]
    follow_up_embeddings = model.encode(predefined_questions)
    D, I = faiss.IndexFlatL2(follow_up_embeddings.shape[1]).search(
        np.array(follow_up_embeddings), k=1
    )
    follow_up_question = predefined_questions[I[0][0]]
    return follow_up_question


# Run chatbot logic
def run_chatbot(file_path, query):
    blog_data = load_and_preprocess_blog(file_path)
    index, blog_embeddings = create_faiss_index(blog_data, model)
    answer = retrieve_answer(query, index, blog_data, model)
    follow_up_question = generate_follow_up(query, answer, model)
    return answer, follow_up_question


# Define a Pydantic model to accept requests
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    follow_up_question: str


# FastAPI endpoint to handle chatbot requests
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"Received query: {request.query}")  # This will log the received query
    file_path = "blog_data.txt"  # Specify your blog data file path
    query = request.query
    answer, follow_up_question = run_chatbot(file_path, query)
    return ChatResponse(answer=answer, follow_up_question=follow_up_question)


# Run the FastAPI server (use Uvicorn to run this file) send this file to me
