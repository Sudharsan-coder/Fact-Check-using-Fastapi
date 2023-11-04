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
# import json
from transformers import logging
logging.set_verbosity_error()


# data = read_csv("datas.csv")
links = ['https://newsmobile.in/articles/2023/10/30/fact-check-ratan-tata-did-not-announce-reward-for-rashid-khan-after-afg-defeated-pak/','https://newsmobile.in/articles/2023/10/28/fact-check-viral-post-claiming-n-koreas-kim-jong-un-criticising-us-for-isreal-palestine-conflict-is-false/']
more_details=['''In the midsts of ongoing ICC Men’s Cricket World Cup, a claim surfaced on the internet according to which Industralist Ratan Tata announced a reward of whopping Rs 10 crore for Afghanistan cricketer Rashid Khan. The claim also stated that Pakistan had complain to ICC aganist Rasid Khan during his victory celebration with Indian flag and ICC fined him for Rs 55 lakh aganist Rasid Khan but Ratan Tata declare 10 crore to Rasid Khan.

One of the posts read: “Ratan Tata has once again shown his greatness.

Rashid Khan, Afghanistan spinner, after defeating NaPak, took the Indian flag and ran around the ground on victory lap and shouted Bharat Mata ki Jai.

NaPak complained to ICC about his action.

ICC and all other world sports bodies, as always, show staunch opposition any ‘Bharatiya’ eulogization.

ICC imposed a fine of Rs 55 lakhs on Rashid Khan.

Ratan Tata, by saying that the person who showed respect to our National Flag should be lauded. Tata announced that he would not only pay the fine amount of Rs 55 lakhs but also reward Rashid Khan with a whopping amount of Rs 10 crore.

Hail Ratan Tata 🙏🙏🙏

*Jai Hind*🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳

This is *TATA*”

The above post can be seen here. More such posts can be seen here and here.

The claim started doing rounds after Afghanistan defeated Pakistan in the World Cup clash on October 23, 2023, following which the Afghanistan team took a victory lap of the MA Chidambaram Stadium in Chennai to celebrate their historic win. During the celebration, Rashid Khan could be seen holding the Indian flag.

NewsMobile fact-checked the above claim and found it to be false.

Ratan Tata on Monday took to X (formerly known as Twitter) to refute the claim.

Tata clarified that he did not make any such announcement and he has no connection with cricket.

“I have made no suggestions to the ICC or any cricket faculty about any cricket member regarding a fine or reward to any players. I have no connection to cricket whatsoever Please do not believe WhatsApp forwards and videos of such nature unless they come from my official platforms,” he wrote on X.

I have made no suggestions to the ICC or any cricket faculty about any cricket member regarding a fine or reward to any players.

I have no connection to cricket whatsoever

Please do not believe WhatsApp forwards and videos of such nature unless they come from my official…

Hence, it can be concluded that the viral claim is false.''','''The conflict between Israel and Hamas has so far resulted in over 1,400 reported casualties in Israel and nearly 5,000 in Gaza. Meanwhile, a video of North Korea’s leader Kim Jong-Un speaking about the Israel-Palestine conflict has gone viral on social media claiming that he was talking about the recent Isreal-Palestine war and criticising US President Joe Biden for the conflict.

A Facebook user shared this video and wrote: “Guess Who Made A Statement About Israel-Hamas. First time Kim Jong-Un has come out to speak on global issues, this is serious.”

This Facebook post can be seen here.

It is being widely shared on Facebook and X with a similar claim.

NewMobile fact-checked the above claim, and found it to be false.

Running a Reverse Image Search of the video keyframes, the NM team found the same visual in The Guardian report, dated October 2020, saying, the viral video is of the 75th anniversary of North Korea’s ruling Party, the Workers’ Party of Korea where Kim Jong-Un got emotional while speaking at a military parade in Pyongyang. He also addressed his impoverished people who have been left battered by typhoons, the coronavirus pandemic, and sanctions. Nowhere in the report, Jong-Un has addressed the Israel-Palestine conflict.

A video report by BBC, dated October 13, 2020, shows similar dress as seen in the viral video. According to the report, North Korean leader Kim Jong-Un got emotional during a speech at a military parade and issued a rare apology for his failure to guide the country through tumultuous times exacerbated by the coronavirus outbreak. He also thanked his troops for their efforts against the pandemic and recent natural disasters.

Searching further, our team also found the transcript of the speech made by Kim Jong Un on October 10, 2020, which was translated into English by the Korean Central News Agency. However, there was no mention of the Israel-Palestine conflict in the entire speech. Click here to read the translated speech.

Thus, it is confirmed that the viral video is from 2020 when Kim Jong-Un gave a speech at a military parade. It was NOT related to the ongoing Israel-Hamas conflict.''']
summarized_data=['''The claim started doing rounds after Afghanistan defeated Pakistan in the World Cup clash on October 23, 2023. The Afghanistan team took a victory lap of the MA Chidambaram Stadium in Chennai to celebrate their historic win. During the celebration, Rashid Khan could be seen holding the Indian flag.''','''A video of North Korea’s leader Kim Jong-Un speaking about the Israel-Palestine conflict has gone viral on social media. A Facebook user shared this video and wrote: “Guess Who Made A Statement About Israel-Hamas.” NM fact-checked the above claim, and found it to be false.''']
def findCompleteAnswer(question):
  try:
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
  except Exception as e:
    return {"error":e}

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
