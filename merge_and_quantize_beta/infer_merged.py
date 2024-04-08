import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

path_parser = argparse.ArgumentParser(description='parser for args')

path_parser.add_argument('--base_model_path', '-m', help="Path to the base model, defaults to the original FT'd model", required=False, default="jhlim8/merged_15_20mar")
path_parser.add_argument('--device', '-d', help='Which device to do inference on, defaults to cuda:0', required=False, default="cuda:0")
path_parser.add_argument('--flash_attn', '-f', help='Whether to use flash attention 2, defaults to True', required=False, default=True)

args = path_parser.parse_args()
base_model_path = args.base_model_path
device = args.device
flash_attn = args.flash_attn

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant = True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2" if flash_attn else "sdpa",
    device_map={'':device}, # have to specifically set each layer to device
    torch_dtype=torch.bfloat16, 
)
model.eval()
tok = AutoTokenizer.from_pretrained(base_model_path)

input_text = input("Enter your message: ")
msg = [{"role":"user", "content":input_text}]

while True:
  encmessages = tok.apply_chat_template(msg, return_tensors="pt", tokenize=True).to(device)
  print(msg)
  generated_ids = model.generate(input_ids=encmessages, max_new_tokens=500, do_sample=False, use_cache=True)
  decoded = tok.batch_decode(generated_ids)
  output = decoded[0][:-4].split('[/INST] ')[-1]
  msg.append({"role":"assistant", "content": output})
  print(output)
  next_input = input()
  msg.append({"role":"user", "content": next_input})
  if next_input == 'exit':
    break