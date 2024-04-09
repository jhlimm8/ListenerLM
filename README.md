# Introduction
A repo for fine-tuning and inferencing LLMs with personalised listening styles.

ROUGE F1 and BLEU precision scores comparing the base Llama 2-Chat 7B model against our fine-tuned version
![image](https://github.com/jhlimm8/ListenerLM/assets/103594440/786458cf-0392-489a-b4f1-0fcb093e1c43)

An example convo using infer_unmerged.py (black is an irrelevant code warning)
![image](https://github.com/jhlimm8/ListenerLM/assets/103594440/f24ba5ec-bd91-4717-a2b9-ea71ab41e916)

# Usage Instructions

## Inference

will require ~17GB VRAM minimum, preferably a 3090/A5000

for a ipynb run all the cells in 

```
infer_unmerged.ipynb
```

while changing the huggingface_token variable in the second cell to your huggingface_token that has llama 2 access.

for a .py CLI first run

```
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

then run

```
python infer_unmerged.py --hf_key 'huggingface token'
```

where 'huggingface token' is your huggingface_token that has llama 2 access.

Quantized and merged versions made from 

```
mrege_and_quantize_beta
```

are technically available, but they perform significantly worse due to a bug making the merged model perform worse than the unmerged model
(even though it should be fixed as per https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930)
so we don't recommend using them for now.

## Fine-tuning 

tested on a 3090 so 24GB VRAM minimum

Create your data using 

```
python convrole/convrole.py
```

, then combine your created data using

```
python convrole/convcombiner.py
```

, then replace the data_files parameter and huggingface_token variable in the 2nd and 4th cells of 

```
train/ft.ipynb
```

. Also add the huggingface repo you want to upload ur fine-tuned model to in the second last cell if you wish to do so. After that run all the cells to do the fine-tuning.

Examples of data obtained from convrole.py are shown in 

```
train/individual_convos
```

and examples of the combined data is shown in

```
train/combined_convos.json
```
