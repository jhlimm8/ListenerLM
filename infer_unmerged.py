import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

path_parser = argparse.ArgumentParser(description='parser for args')

path_parser.add_argument('--base_model_path', '-m', help='Path to the base model, defaults to Llama 2 7B-Instruct', required=False, default="meta-llama/Llama-2-7b-chat-hf") 
path_parser.add_argument('--adapter_path', '-p', help='Path to the trained adapters, defaults to the original trained adapters', required=False, default="jhlim8/ListenerLM")
path_parser.add_argument('--device', '-d', help='Which device to do inference on, defaults to cuda:0', required=False, default="cuda:0")
path_parser.add_argument('--hf_key', '-hf', help='Huggingface key to access the models', required=True)

args = path_parser.parse_args()
base_model_path = args.base_model_path
adapter_path = args.adapter_path
device = args.device
hf_key = args.hf_key

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant = True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map={'':device}, # have to specifically set each layer to device
    torch_dtype=torch.bfloat16, 
    token=hf_key
)
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, token=hf_key)
model.eval()
tok = AutoTokenizer.from_pretrained(adapter_path, token=hf_key)

input_text = input("Enter your message: ")
msg = [{"role":"user", "content":input_text}]

while True:
  encmessages = tok.apply_chat_template(msg, return_tensors="pt", tokenize=True).to(device)
  #print(msg)
  generated_ids = model.generate(input_ids=encmessages, max_new_tokens=500, do_sample=False, use_cache=True)
  decoded = tok.batch_decode(generated_ids)
  output = decoded[0][:-4].split('[/INST] ')[-1]
  msg.append({"role":"assistant", "content": output})
  print(output)
  next_input = input()
  msg.append({"role":"user", "content": next_input})
  if next_input == 'exit':
    break