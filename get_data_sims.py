import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
def get_target_layer_val(data, layers=list(range(24, 32))):
    
    if layers == "all":
        return data
    else:
        return data[layers, :, :]
    
def compute_diff(neuron_base, neuron_ckpt):
    diff_total = []
    naf_total = []
    sign_total = []
    for base_data, ckpt_data in tqdm(zip(neuron_base, neuron_ckpt)):
        # print("Base data shape:", base_data.shape)
        # print("Ckpt data shape:", ckpt_data.shape)
        
        naf = (ckpt_data - base_data) ** 2
        naf = torch.sqrt(torch.sum(naf, dim=1) / naf.shape[1])
        naf = naf.view(1, -1).squeeze(0)
        ckpt_data = torch.mean(ckpt_data, dim=1)
        base_data = torch.mean(base_data, dim=1)
        diff = ckpt_data - base_data
        
        diff = diff.view(1, -1).squeeze(0)
        sign = torch.sign(diff)
        diff_total.append(diff)
        naf_total.append(naf)
        sign_total.append(sign)
    diff_total = torch.stack(diff_total, dim=0)
    naf_total = torch.stack(naf_total, dim=0)
    sign_total = 
    print("Diff shape:", diff_total.shape)
    print("Naf shape:", naf_total.shape)
    return diff_total, naf_total, sign_total
    
def get_layers(data, start, end):
    layer_idx_start = start * 14336
    layer_idx_end = end * 14336
    return data[:, layer_idx_start:layer_idx_end]

def load_neuron_data(path, layers=list(range(24, 32))):
    data = torch.load(path, map_location=torch.device('cpu'))
    retrun_data = []
    for k, v in data.items():

        for tensor in v:
            tensor = get_target_layer_val(tensor, layers)
            retrun_data.append(tensor)
    
    return retrun_data


import torch

def layered_topk_indices(flat_activations: torch.Tensor, num_layers: int, topk_percent: float):
    """
    计算每一层 top a% 神经元在展平向量中的位置索引
    
    参数:
    - flat_activations: torch.Tensor, 形状为 [num_layers * hidden_size]
    - num_layers: int, 层数 n
    - topk_percent: float, 取值 (0, 1)，表示每层 top a%

    返回:
    - topk_indices_flat: torch.Tensor, 所有 topk 神经元在展平向量中的索引
    """
    hidden_size = flat_activations.numel() // num_layers
    assert flat_activations.numel() == num_layers * hidden_size, "尺寸不匹配"

    # reshape 为 [n, 14336]
    activations = flat_activations.view(num_layers, hidden_size)

    # 每层 topk 个数
    k = int(hidden_size * topk_percent)
    if k == 0:
        raise ValueError("topk_percent 太小，导致 k=0")

    # 存储所有展平索引
    topk_indices_flat = []

    for layer_idx in range(num_layers):
        layer_acts = activations[layer_idx]  # shape: [14336]
        topk_vals, topk_pos = torch.topk(layer_acts, k, largest=True)

        # 将层内索引转换为展平索引
        flat_indices = layer_idx * hidden_size + topk_pos
        topk_indices_flat.append(flat_indices)

    # 拼接所有层的结果
    return torch.cat(topk_indices_flat)

def cosine_similarity_large_A(
    A: torch.Tensor,  # [m, d] - very large, stays on CPU
    B: torch.Tensor,  # [n, d] - small enough to move to GPU
    batch_size: int = 500,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Efficient pairwise cosine similarity when A is too large for GPU.
    A stays on CPU and is processed in batches.
    B is moved to GPU once.
    Returns: [m, n] similarity matrix on CPU
    """
    B = B.to(device)
    B_norm = F.normalize(B, dim=1)  # [n, d]
    results = []

    m = A.size(0)

    for i in tqdm(range(0, m, batch_size)):
        A_chunk = A[i:i + batch_size]  # [b, d], still on CPU
        A_chunk = A_chunk.to(device, non_blocking=True)
        A_chunk_norm = F.normalize(A_chunk, dim=1)  # [b, d]

        sim_chunk = A_chunk_norm @ B_norm.T  # [b, n]
        results.append(sim_chunk.cpu())  # Move result back to CPU

        del A_chunk, A_chunk_norm, sim_chunk  # Free GPU mem
        torch.cuda.empty_cache()

    return torch.cat(results, dim=0)  # [m, n]
def get_ckpt_specific_data(base_dir, train_dir, val_neuron_path_pre, val_neuron_path_post, train_file_name_pre, train_file_name_post, layers_total, do_neuron_selection=False, neuron_selection_partition=0.05):
    base_files = os.listdir(base_dir)
    neuron_train_base = {}
    for file in tqdm(base_files):
        if file.endswith('base'):
            data = torch.load(os.path.join(base_dir, file), map_location=torch.device('cpu'))
            neuron_train_base.update(data)
    neuron_val_pre = load_neuron_data(val_neuron_path_pre)
    neuron_val_post = load_neuron_data(val_neuron_path_post)
    files = os.listdir(train_dir)
    neuron_train_pre = neuron_train_base
    neuron_train_post = {}
    for file in tqdm(files):
        if file.endswith(train_file_name_post):
            data = torch.load(os.path.join(train_dir, file), map_location=torch.device('cpu'))
            neuron_train_post.update(data)
    neuron_train_pre = torch.stack([neuron_train_pre[k] for k in sorted(neuron_train_pre.keys())], dim=0)
    neuron_train_post = torch.stack([neuron_train_post[k] for k in sorted(neuron_train_post.keys())], dim=0)
    diff_val, naf_val, sign_val = compute_diff(neuron_val_pre, neuron_val_post)
    diff_train = neuron_train_post - neuron_train_pre
    for layers in layers_total:
        start, end = layers[0], layers[1]
        diff_val_layer = get_layers(diff_val, start, end)
        diff_train_layer = get_layers(diff_train, start, end)
        print("Diff val layer shape:", diff_val_layer.shape)
        print("Diff train layer shape:", diff_train_layer.shape)
        
        neuron_selection_partition = "all"
        sim = cosine_similarity_large_A(diff_train_layer, diff_val_layer)
        

        score_mean = torch.mean(sim, dim=1)
        print("Score mean shape:", score_mean.shape)
        saved_path = f"sims/bbh/bbh_{train_file_name_pre}_{train_file_name_post}_{layers[0]}_{layers[1]}_neuron_partition_{neuron_selection_partition}_2.pt"
        torch.save(score_mean, saved_path)

if __name__ == "__main__":
    base_dir = 'results/bbh'
    train_dir = 'results/bbh'
    val_neuron_path_pre = 'neuron_activations/bbh/act_bbh_val_base'
    val_neuron_path_post = 'neuron_activations/bbh/act_bbh_val_ckpt1'
    train_file_name_pre = 'base'
    train_file_name_post = 'ckpt1'
    layers = [0, 8]
    layers_total = [layers]

    get_ckpt_specific_data(val_neuron_path_pre, val_neuron_path_post, train_file_name_pre, train_file_name_post, layers_total)
        