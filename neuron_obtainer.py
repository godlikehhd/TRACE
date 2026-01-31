from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
import argparse
import json
import sys
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def create_sft_labels(input_ids, start_ids={128006, 78191, 128007}, end_id=128009):
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    labels[:] = -100  # 默认所有位置为 -100

    assistant_spans = []
    in_assistant_block = False
    start_index = None

    for i in range(len(input_ids[0])):
        token = input_ids[0, i].item()

        if token in start_ids:
            if i + 2 < len(input_ids[0]) and \
               input_ids[0, i] == 128006 and \
               input_ids[0, i+1] == 78191 and \
               input_ids[0, i+2] == 128007:
                in_assistant_block = True
                start_index = i + 3  # assistant内容开始位置
        elif token == end_id and in_assistant_block:
            end_index = i + 1  # 包括 <|eot_id|> 本身
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

    return input_mask, shifted_output_mask

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

class NeuronObtainer:
    def __init__(self, args):
        self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", output_hidden_states=True)
        self.max_length = args.max_length
        self.activations = []
        self.init_hook()
        self.model.eval()
        self.args = args
        self.length = {}
        self.neurons= {}
    
    def init_hook(self):
        def hook(module, input, output):
            self.activations.append(output.detach().squeeze(0))

        for name, module in self.model.named_modules():
            if "mlp.act_fn" in name:
                # 提取层号（假设 name 格式为 model.layers.{num}.mlp.down_proj）
                parts = name.split(".")
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
        tok_only_query = self.tok.apply_chat_template(messages, tokenize=True, return_tensors="pt")
        labels = create_sft_labels(tok_only_query)
        input_mask, output_mask = generate_input_output_mask(labels)

        return tok_only_query, input_mask, output_mask, labels

    def tokenize(self, input):
        # print("input", input)
        tok_only_query, input_mask, output_mask, labels = self.format_input(input)
        e = {
            "input_ids": tok_only_query,
            "input_mask": input_mask,
            "output_mask": output_mask,
            "label": labels
        }
        return e

    def get_logprob(self, input, logprobs):
        input_ids = input['input_ids']
        choice_start = input['choice_start']
        choice_end = input['choice_end']
        choice_tokens = input_ids[0][choice_start:choice_end].unsqueeze(1)
        choice_logprobs = logprobs[0][choice_start - 1:choice_end - 1]
        extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)
        choice_length = choice_end - choice_start
        assert choice_length > 0, "choice length should be greater than 0"
        lm_log_p = torch.sum(extracted).item()


        # sys.exit(0)
        norm_lm_log_p = (lm_log_p / choice_length)

        return lm_log_p, norm_lm_log_p
    
    def get_neuron(self, inputs_ori):
        neuron_total = None
        input_mask = inputs_ori['input_mask']
        output_mask = inputs_ori['output_mask']
        neuron_total = torch.stack(self.activations, dim=0)
        # neuron_input = extract_by_mask(neuron_total, input_mask)
        # neuron_output = extract_by_mask(neuron_total, output_mask)
        # neuron_input = torch.mean(neuron_total, dim=1)
        # neuron_output = torch.mean(neuron_output, dim=1)
        # neuron_total = torch.mean(neuron_total, dim=1)
        del self.activations
        torch.cuda.empty_cache()
        self.activations = []
        return neuron_total
                

    
    def get_neural_activation(self, inputs, i):
        # Tokenize the text

        inputs_ori = self.tokenize(inputs)

        with torch.no_grad():
            outputs_ori = self.model(
                input_ids=inputs_ori['input_ids'].to(device),  # [B, L']
                ).logits
        neurons = self.get_neuron(inputs_ori)
        if "task" in inputs:
            if self.neurons.get(inputs['task']) is None:
                self.neurons[inputs['task']] = []
            self.neurons[inputs['task']].append(neurons.cpu())
        else:
            self.neurons[i] = neurons.cpu()
        return 
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--icl", type=int, default=0)
    args = parser.parse_args()
    return args


from tqdm import tqdm
def main():

    args = parse_args()
    nb = NeuronObtainer(args)

    data = load_data(args.data_path)
    # data = data[:3]
    # data_sample = data[args.start_idx:args.end_idx]

    for i in tqdm(range(len(data)), desc='Calculating activations'):
        input = data[i]
        if args.icl == 0:
            if input.get('examples') is not None:
                input.pop('examples')
        nb.get_neural_activation(input, i)
    activations = nb.neurons
    ensure_dir_for_file(args.save_path)
    torch.save(activations, args.save_path)
    # with open(args.length_save_path, 'w', encoding='utf-8') as f:
    #     json.dump(rpe.length, f, ensure_ascii=False, indent=4)

    

if __name__ == '__main__':
    main()
    