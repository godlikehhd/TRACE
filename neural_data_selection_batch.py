from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
import argparse
import json
import sys
import os
from TRACE_data import TraceData, DataCollatorForChat
from torch.utils.data import DataLoader
import time
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import os
def ensure_dir_for_file(file_path):
    """
    检查给定的文件路径中的目录是否存在，不存在则创建目录。
    
    参数:
        file_path (str): 要保存的文件的完整路径。
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录已创建：{directory}")
    else:
        print(f"目录已存在：{directory}")


class TrainDataNeuronSelection:
    def __init__(self, args):
        self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path_source)
        self.tok.pad_token = self.tok.eos_token
        self.source_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_source, device_map="auto", output_hidden_states=False)
        self.max_length = args.max_length
        self.activations = []
        self.current_attention_mask = None
        self.args = args
        self.length = []
        self.init_hook()
        self.source_model.eval()
        self.target_neuron_activation = []

        
    def init_hook(self):
        def hook(module, input, output):
            self.activations.append(output.detach())
        for name, module in self.source_model.named_modules():
            if "mlp.act_fn" in name:
                # 提取层号（假设 name 格式为 model.layers.{num}.mlp.act_fn）
                parts = name.split(".")
                # print("name", parts)
                if "layers" in parts:
                    layer_idx = parts[2]
                    try:
                        layer_num = int(layer_idx)
                        # print("layer_num", layer_num)
                        if layer_num == self.args.num_layers:
                            module.register_forward_hook(hook)
                    except ValueError:
                        print("Invalid layer number:", parts[layer_idx])
                        pass  # 避免非数字 index 报错

 


    def get_score(self):
        output = self.activations[0]
        masked = output * self.current_attention_mask.unsqueeze(-1)
        neuron_total = masked.mean(dim=1)
        self.activations.clear()
        return neuron_total



    def get_rpe(self, inputs, i):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            self.current_attention_mask = attention_mask
            outputs_ori = self.source_model(
                input_ids=input_ids,
                labels=input_ids
            )

        torch.cuda.synchronize()
        end_time = time.time()

        target_ac = self.get_score()
        time_use = end_time - start_time
        print(f"Forward time: {time_use:.4f}s")

        return time_use
     
def load_data(data_path):
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path_source", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)



 
    args = parser.parse_args()
    return args


from tqdm import tqdm
import math
def main():

    args = parse_args()
    tdns = TrainDataNeuronSelection(args)
    data_collator = DataCollatorForChat(tokenizer_name_or_path=args.model_name_or_path_source)
    dataset = TraceData(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=data_collator,
        drop_last=False,
        shuffle=False # Set to True for training
    )
    time_use_total = 0
    for i, batch in tqdm(enumerate(dataloader), desc='Calculating activations'):
        time_use = tdns.get_rpe(batch, i)
        time_use_total += time_use
    print(time_use_total)
    activations = tdns.target_neuron_activation
    ensure_dir_for_file(args.save_path)
    torch.save(activations, args.save_path)
if __name__ == '__main__':
    main()


