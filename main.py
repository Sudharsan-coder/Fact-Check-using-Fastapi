from fastapi import FastAPI
from Answer import findCompleteAnswer

app= FastAPI()

@app.get("/")
async def root():
    print("root activated")
    return "Welcome to the Question answer api"

@app.get("/answer")
async def get_answer():
    print("get_answer activated")
    question="did ratan tata announce reward for rashid khan?"
    return findCompleteAnswer(question)
