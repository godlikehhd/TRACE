from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
import argparse
import json
import sys
import os
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

start_ids = {
    "llama": [128006, 78191, 128007, 271],
    "qwen": [198, 151644, 77091, 198],
    "olmo":[515, 5567, 49651, 187]
}
end_ids = {
    "llama": 128009,
    "qwen": 151645,
    "olmo": 50279,
}
def create_sft_labels(input_ids, model_name):
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    labels[:] = -100  # 默认所有位置为 -100
    start_ids_target = start_ids[model_name]
    end_id = end_ids[model_name]
    assistant_spans = []
    in_assistant_block = False
    start_index = None

    for i in range(len(input_ids[0])):
        token = input_ids[0, i].item()

        if token in start_ids_target:
            if i + 3 < len(input_ids[0]) and \
               input_ids[0, i] == start_ids_target[0] and \
               input_ids[0, i+1] == start_ids_target[1] and \
               input_ids[0, i+2] == start_ids_target[2] and \
               input_ids[0, i+3] == start_ids_target[3]:
                in_assistant_block = True
                start_index = i + 4  # assistant内容开始位置
        elif token == end_id and in_assistant_block:
            end_index = i   # 包括 <|eot_id|> 本身
            assistant_spans.append((start_index, end_index))
            in_assistant_block = False

    for start, end in assistant_spans:
        labels[0, start:end] = input_ids[0, start:end]

    return labels


def generate_input_output_mask(labels: torch.Tensor):
    """
    根据 SFT 中的 labels 生成 input_mask 和 output_mask，
    并将 output_mask 向左 shift 一位，使其对齐到 input token 的表示。
    """
    # labels: shape [batch_size, seq_len]
    # mask: 1 表示输出部分（需要监督），0 表示输入部分
    raw_output_mask = (labels != -100).long()  # [B, T]

    # 向左 shift 一位，前面补 0
    shifted_output_mask = torch.zeros_like(raw_output_mask)
    shifted_output_mask[:, 1:] = raw_output_mask[:, :-1]

    # input mask 是反过来的
    input_mask = 1 - shifted_output_mask
    shifted_output_mask_2 = torch.zeros_like(shifted_output_mask)
    shifted_output_mask_2[:, :-1] = shifted_output_mask[:, 1:]

    return input_mask, shifted_output_mask_2

def extract_by_mask(representations, mask):
    """
    从每一层的表示中提取掩码位置的 token 表示。

    Args:
        representations: Tensor[num_layers, seq_len, hidden_size]
        mask: Tensor[seq_len]，布尔型或整型，True/1 表示要保留的位置

    Returns:
        Tensor[num_layers, num_selected_tokens, hidden_size]
    """
    # 确保 mask 是 bool 类型
    mask = mask.squeeze(0)
    mask = mask.bool()  # shape: [seq_len]

    # 通过广播提取每层中对应位置的表示
    # 等价于对每一层做： layer[mask]
    # print("mask.shape", mask.shape)
    # print("representations.shape", representations.shape)
    selected = representations[:, mask, :]  # shape: [num_layers, num_selected_tokens, hidden_size]
    return selected

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


def average_valid_embeddings(embedding, mask):
    # embedding: [num_layers, bs, seq_len, hidden_size]
    # mask: [bs, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(-1).float().to(device)  # [1, bs, seq_len, 1]
    
    masked_emb = embedding * mask  # [num_layers, bs, seq_len, hidden]
    token_counts = mask.sum(dim=2)  # [1, bs, 1]
    
    avg_emb = masked_emb.sum(dim=2) / token_counts.clamp(min=1.0)  # [num_layers, bs, hidden]
    return avg_emb

class TrainDataNeuronSelection:
    def __init__(self, args):
        self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path_source)
        self.tok.pad_token = self.tok.eos_token
        self.source_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_source, device_map="auto", output_hidden_states=True)
      
        self.max_length = args.max_length
        self.activations = []
        self.current_attention_mask = None
        self.args = args
        self.length = []
        self.init_hook()
        self.start_idx = args.start_idx
        self.source_model.eval()
        self.target_neuron_activation = {}
        self.last_hidden_state = {}
        self.loss = {}
        self.target_sign = {}
        # self.init_target_neurons(args.neuron_file)
    

    # def init_target_neurons(self, neuron_file):
    #     loaded_data = torch.load(neuron_file)
    #     neuron_idx_total = loaded_data['idx_total']
    #     total_neurons = sorted(list(neuron_idx_total.keys()))
    #     # total_neurons_idx = {k: v for v, k in enumerate(total_neurons)}
    #     # with open(self.args.target_neurons_path, "w", encoding="utf-8") as f:
    #     #     json.dump(total_neurons_idx, f, indent=4, ensure_ascii=False)
    #     self.target_neurons = torch.tensor(total_neurons, dtype=torch.int64)
        
    def init_hook(self):
        def hook(module, input, output):
            if self.current_attention_mask is None:
                print("Warning: current_attention_mask is not set in hook!")
                return
            mask = self.current_attention_mask.to(output[0].device, non_blocking=True)
            num_tokens = mask.sum(dim=1, keepdim=True)
            masked_output = output * mask.unsqueeze(-1)
            
            sum_activations = masked_output.sum(dim=1)
            avg_activations = sum_activations / num_tokens.clamp(min=1.0)
            self.activations.append(avg_activations.detach())
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
                        if layer_num == 15:
                            module.register_forward_hook(hook)
                    except ValueError:
                        print("Invalid layer number:", parts[layer_idx])
                        pass  # 避免非数字 index 报错

 

    
    def format_input(self, inputs):
        # print("inputs", inputs)
        if "messages" in inputs:
            messages = inputs['messages']
        else:
            
            instruction, input, output = inputs['instruction'], inputs['input'], inputs['output']
            messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
            ]
        tok_only_query = self.tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", max_length=self.max_length, truncation=True)
        labels = create_sft_labels(tok_only_query, 'qwen')
        length = tok_only_query.shape[1]
        input_mask, output_mask = generate_input_output_mask(labels)

        return tok_only_query, input_mask, output_mask, labels, length

    def format_input_batch(self, inputs):
        # print("inputs", inputs)
        messages_total = []
        for data in inputs:
            messages = data['messages']
            messages_total.append(messages)
        
        tok_only_query = self.tok.apply_chat_template(messages_total, tokenize=True, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        

        return tok_only_query


    def tokenize(self, input):
        # print("input", input)
        lengths = []
        # for data in input:
        #     tok_only_query, input_mask, output_mask, labels, length = self.format_input(data)
        #     lengths.append(length)
        # print(lengths)
        tok_only_query = self.format_input_batch(input)
        bs, length = tok_only_query.shape
        positions = torch.arange(length).unsqueeze(0).expand(bs, -1)
        mask = positions < torch.tensor(lengths).unsqueeze(1)
        e = {
            "input_ids": tok_only_query,
            "mask": mask
        }
        return e



    def get_score(self, input):
        # activation_source = activation_source[:, choice_start-1:choice_end-1]
        # activation_target = activation_target[:, choice_start-1:choice_end-1]
        neuron_total = torch.stack(self.activations, dim=0)
        # print("neuron_total", neuron_total.shape)
        # neuron_input = extract_by_mask(neuron_total, input_mask)
        # # print("neuron_input", neuron_input.shape)
        # neuron_input = torch.mean(neuron_total, dim=1)
        # # print("neuron_input", neuron_input.shape)
        # neuron_input = neuron_input.view(1, -1).squeeze(0)
        # target_ac = neuron_input[self.target_neurons]
        del self.activations
        self.activations = []
        return neuron_total.cpu()
            



    def get_rpe(self, inputs, i):
        # Tokenize the text

        inputs_ori = self.tokenize(inputs)

        self.current_attention_mask = inputs_ori['mask']
        # sys.exit(0)
        # Get the logits
        with torch.no_grad():
            input_ids=inputs_ori['input_ids'].to(device)
            outputs_ori = self.source_model(
                input_ids=input_ids,
                labels=input_ids
                )
        target_ac = self.get_score(inputs_ori)
        # print("target_ac", target_ac.shape)
        for j in range(target_ac.shape[1]):
            ac = target_ac[:, j, :].reshape(1, -1).squeeze(0)
            # print("ac", ac.shape)
            self.target_neuron_activation[self.start_idx + i*self.args.batch_size + j] = ac
        
        torch.cuda.empty_cache()
        return target_ac
     
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

    data = load_data(args.data_path)
    if args.end_idx == 0:
        args.end_idx = len(data)
    # data = data[:33838]
    data_sample = data[args.start_idx:args.end_idx]
    num_batch = math.ceil(len(data_sample) / args.batch_size)
    for i in tqdm(range(num_batch), desc='Calculating activations'):
        input = data_sample[i*args.batch_size:(i+1)*args.batch_size]
        # if input.get('examples') is not None:
            # if args.icl == False:
            #     input.pop('examples')
        target_ac = tdns.get_rpe(input, i)

    activations = tdns.target_neuron_activation
    ensure_dir_for_file(args.save_path)
    torch.save(activations, args.save_path)

    
    

    

if __name__ == '__main__':
    main()


