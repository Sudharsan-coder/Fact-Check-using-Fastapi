# try:
  # import venv
  # import os
  # venv.create("test")
  # os.system("source test/bin/activate")
  # import pip
  # pip.main(["install", "pandas"])
  # pip.main(["install", "sentence-transformers"])
  # pip.main(["install", "torch"])
  # pip.main(["install", "transformers"])
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline
from pandas import *
# import sys
from sentence_transformers import SentenceTransformer, util
import json
from transformers import logging
logging.set_verbosity_error()


data = read_csv("datas.csv")
links = data['Links'].tolist()
more_details=data['Passages'].tolist()
summarized_data=data['Summary'].tolist()

def findCompleteAnswer(question):

  model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

  query_emb = model.encode(question)
  doc_emb = model.encode(summarized_data)

  scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

  doc_score_pairs = list(zip(summarized_data, scores))

  max_index=0
  for index,value in enumerate(doc_score_pairs):
    cur=value[1]
    if(cur>doc_score_pairs[max_index][1]):
      max_index=index
  # print(doc_score_pairs)
  selected_heading=max_index
  # print(more_details[selected_heading])
  boolean_ans=findBooleanAns(more_details[selected_heading],question)
  # boolean_ans_str="It is a true news," if boolean_ans else "It is a fake news,"
  answer=findAnswer(question,more_details[selected_heading])
  if(answer==None):
    ans= {"validation":"","answer":"Sorry I can't answer the question due to insufficient data, try asking different question","link":""}
  else:
    ans= {"validation":boolean_ans,"answer":answer,"link":links[selected_heading]}
  print(ans)
  return ans

def findBooleanAns(doc,question):
  model = AutoModelForSequenceClassification.from_pretrained("nfliu/roberta-large_boolq")
  tokenizer = AutoTokenizer.from_pretrained("nfliu/roberta-large_boolq")
  sample_item=[(question,doc)]
  encoded_input = tokenizer(sample_item, padding=True, truncation=True, return_tensors="pt")

  with torch.no_grad():
    model_output = model(**encoded_input)
    probabilities = torch.softmax(model_output.logits, dim=-1).cpu().tolist()

  probability_no = [round(prob[0], 2) for prob in probabilities]
  probability_yes = [round(prob[1], 2) for prob in probabilities]
  return probability_no<probability_yes

def findAnswer(question,context):
  model= pipeline("question-answering", model='bert-large-uncased-whole-word-masking-finetuned-squad')
  result = model(question=question,context=context)
  if(result['score']<0.1):
    return None
  # to get the selected sentence from the passage
  #   start=result["start"]
  #   end=result['end']
  #   while start >=-1:
  #     if context[start] ==".":
  #         break
  #     start -= 1
  #   while end <= len(context):
  #     if context[end] == ".":
  #         break
  #     end += 1
  # print("sentence :")
  # result["answer"]=(context[start+1:end]).strip()

  return (result['answer'])

# question="did ratan tata announce reward for rashid khan?"
# ans=findCompleteAnswer(question)
# ans=findAssociation(sys.argv[1])
# print(json.dumps(ans))

# except Exception as e:
#   print(e)