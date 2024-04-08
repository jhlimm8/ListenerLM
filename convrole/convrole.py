apikey = '' # ur openai api key
data_path = '' # path to ur data

topics = ['Partner', 'School', 'Work', 'Family', 'Friends', 'Financials', 'Health']
moods = ['Angry', 'Anxious', 'Happy', 'Depressed']

from openai import OpenAI
client = OpenAI(
    api_key= apikey
)

import random
# random.seed(42) # uncomment for reproducible combi
lengths = [len(topics), len(moods)]
combi = [random.randint(0, length-1) for length in lengths]
print(topics[combi[0]], moods[combi[1]])
messages=[
  {"role": "system", "content": f"Act as someone who is currently feeling {moods[combi[1]]} because of an experience related to their {topics[combi[0]]}, and is looking for someone to talk about the experience."},
]
completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=messages,
  temperature=1.5,
  max_tokens=1024, # don't set too high in the infrequent case of a long and incoherent generation
  #top_p = 0.5
)
print(completion.usage)
while True:
  messages.append({"role":"assistant", "content":completion.choices[0].message.content})
  print(messages[-1]["content"])
  # for the creation of the dataset, we will use our own preferred listening style.
  response = input()
  #print(response) # uncomment if input isnt printed
  if response == "quit":
      break
  messages.append({"role":"user", "content":response})
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    temperature=1.5,
    max_tokens=1024,
  )
  print(completion.usage)

import json 
import os 
  
files = os.listdir(data_path)
n_files = len(files)
with open(f'{data_path}/data'+str(n_files+1)+'.json', 'w') as outfile:
    json.dump(messages, outfile)