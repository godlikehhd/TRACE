import json 
import random
import os

          



def random_sample_data(sample_proportion, random_seed, use_revision=True):
    data_path = 'TRACE/datasets/data_repo/data_repo_deduplicated.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    random.seed(random_seed)
    sample_size = int(len(data) * sample_proportion) if not use_revision else int(270680 * sample_proportion)
    random.shuffle(data)
    
    random_sample = random.sample(data, sample_size)
    print(len(random_sample))
    save_path = f'LLaMA-Factory/data/random_sample_{sample_proportion}_seed{random_seed}.json'
    with open(save_path, 'w') as f:
        json.dump(random_sample, f, ensure_ascii=False, indent=4)
    

seeds = [3, 6, 9]
for seed in seeds:
    random_sample_data(0.05, seed, use_revision=True)

