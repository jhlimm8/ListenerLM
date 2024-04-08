apikey = '' # ur openai api key
continued_path = '' # path to the convo u wanna continue

from openai import OpenAI
client = OpenAI(
    api_key= apikey
)

import json

with open(continued_path) as json_file:
    messages = json.load(json_file)
    print('orig messages', messages)
num_prune = 1
for i in range(num_prune):
    messages.pop()
print('pruned messages', messages)
completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=messages,
  temperature=1.5
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
    temperature=1.5
    #top_p = 0.5
  )
  print(completion.usage)

import json 
import os 

with open(continued_path, 'w') as outfile:
    json.dump(messages, outfile)