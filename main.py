from fastapi import FastAPI
from Answer import findCompleteAnswer
from pydantic import BaseModel
app= FastAPI()

class Question(BaseModel):
    question:str

@app.get("/")
async def root():
    print("root activated")
    return "Welcome to the Question answer api"

@app.post("/answer")
async def get_answer(Question:Question):
    print("get_answer activated")
    return findCompleteAnswer(Question.question)
