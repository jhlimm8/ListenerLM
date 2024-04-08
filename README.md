# Introduction
A repo for fine-tuning LLMs with personalised listening styles and inferencing them.
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
infer_unmerged.py --hf_key 'huggingface token'
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

Create your data using 

```
convrole/convrole.py
```

, then combine your created data using

```
convrole/convcombiner.py
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
