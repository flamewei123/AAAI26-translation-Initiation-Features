from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pdb
import json
import os
import csv
import torch._dynamo
torch._dynamo.disable()
import argparse
import re
import numpy as np
from sparsify import Sae


def get_delta_saelen(model, sae, tgt_layer, inputs, p, v):
    sae_inter_dict = dict()
    # 修改后的模型生成
    def layer_hook(module, inp, out, pos=p, value=v):
        out0 = out[0]
        out0_inter = sae.encode(out0)
        out0_recon = sae.decode(out0_inter)
        error = out0 - out0_recon  # 重建前后的差值
        sae_features = out0_inter.clone()
        out0_val = sae_features[0, -1, pos]
        sae_features[0, -1, pos] = torch.where(out0_val != 0, out0_val * value, out0_val)             
        modified_out0 = sae.decode(sae_features)
        new_out0 = modified_out0 + error
        sae_inter_dict['vector'] = new_out0 - out0

        return (new_out0,) + out[1:]


    change_hook = model.model.layers[tgt_layer].register_forward_hook(layer_hook)
    outputs = model(inputs)
    change_hook.remove()
    delta = sae_inter_dict['vector'][0, -1, :].detach().cpu()
    return delta

def get_delta_sparsify(model, sae, tgt_layer, inputs, p, v):
    sae_inter_dict = dict()
    # 修改后的模型生成
    def layer_hook(module, inp, out, pos=p, value=v):
        out0 = out[0]
        flat_out0 = out0.view(-1, out0.shape[-1])
        top_acts, top_indices, pre_acts = sae.encode(out0)
        out0_recon = sae.decode(top_acts, top_indices)
        error = out0 - out0_recon
        sae_features = top_acts.clone()

        last_token_indices = top_indices[-1]
        mask = (last_token_indices == pos)
        if mask.any():
            sae_features[-1, mask] = sae_features[-1, mask] * value
        modified_out0 = sae.decode(sae_features, top_indices)
        new_out0 = modified_out0 + error
        sae_inter_dict['vector'] = new_out0 - out0
        return (new_out0,) + out[1:]


    change_hook = model.model.layers[tgt_layer].register_forward_hook(layer_hook)
    outputs = model(inputs)
    change_hook.remove()
    delta = sae_inter_dict['vector'][0, -1, :].detach().cpu()
    return delta

def consistency_metrics(vectors):
    # vectors: (n, d)
    u = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # unit
    
    # 1. mean pairwise cosine
    cos_mat = u @ u.T
    n = len(u)
    mean_pair_cos = (cos_mat.sum() - n) * 2 / (n * (n - 1))
    
    # 2. resultant length
    r = np.linalg.norm(u.sum(axis=0)) / n
    
    # 3. PCA first eigenvalue
    S = u.T @ u / n
    eigvals = np.linalg.eigvalsh(S)[::-1]
    rho = eigvals[0]        # since trace=1
    
    return {"mean_cos": mean_pair_cos, "resultant_r": r, "pca_ratio": rho}


def read_position(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(int(line.strip().split('\t')[1]))
    return datas

def read_test_data(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) != 3: continue
            langpari, source_text, target_text = items
            if '-' in langpari:
                source_lang, target_lang = langpari.split("-")
            elif '2' in langpari:
                source_lang, target_lang = langpari.split("2")
            data = {
                "src": source_text, 
                "tgt": target_text, 
                "src_lang": source_lang, 
                "tgt_lang": target_lang.rstrip()
                }
            datas.append(data)
    return datas

def is_trivial(delta, 
               eps_abs=1e-3, eps_rel=1e-2):
    abs_norm = torch.norm(delta).item()
    if abs_norm < eps_abs:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. ")
    parser.add_argument("--feature_pos_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The information of sparse feature positions. ")
    parser.add_argument("--model_name", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_id",
                        type=str,
                        default='0',
                        help="available gpu id")
    # parameters about integrated grad
    parser.add_argument("--layer",
                        default=24,
                        type=int,
                        help="layer")
    args = parser.parse_args()
    # 设置 GPU 设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    tgt_layer = args.layer
    # 加载 SAE 模型
    if args.model_name == 'qwen3-4b-it':
        model_path = '/models/Qwen3-4B-Instruct-2507'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()        
        sae = SAE.load_from_disk('/sae_len/checkpoints/Qwen3-4B-Instruct-2507-resid-l20')
        sae.to(device)
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
        
    if args.model_name == "gemma-2-2b-it":
        model_path = "/models/gemma-2-2b-it"
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_{}/width_16k/canonical".format(tgt_layer),
            device=device,
        )
    if args.model_name == "gemma-2-9b-it":
        model_path = '/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819'
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id="layer_{}/width_16k/canonical".format(tgt_layer),
            device=device,
        )
    if args.model_name == "gemma-2-9b":
        model_path = '/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6'
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id="layer_{}/width_16k/canonical".format(tgt_layer),
            device=device,
        )
    if args.model_name == 'llama-3-1b':
        model_path = '/models/Llama-3.2-1B'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "chanind/sae-llama-3.2-1b-res",
            sae_id = "blocks.{}.hook_resid_post".format(tgt_layer), 
        )
        sae.to(device)
    if args.model_name == 'llama-3-8b':
        model_path = '/models/Llama-3.1-8B'
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # cache_path = "/huggingface/hub/models--EleutherAI--sae-llama-3-8b-32x/snapshots/32926540825db694b6228df703f4528df4793d67/layers.12"
        # sae = Sae.load_from_disk(cache_path, device=device)
        sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", 
            hookpoint="layers.{}".format(tgt_layer), 
            device=device
        )
    positions = read_position(args.feature_pos_path)
    data_path = args.data_path
    output_filepath = os.path.join(
        os.path.dirname(args.feature_pos_path),
        f'l{tgt_layer}' + '.csv'
    )
    # output_prefix = os.path.join(args.output_dir, f"{args.model_name}/layer{tgt_layer}.v{v}.editing.results")
    
    value_list = [0.0]
    prompt_templates = {
            "prompt1": "Please translate the following query into {tgt_lang}, Provide only the translation: {src}:",
            "prompt2": 'Please translate the following query {src} into {tgt_lang}: {tgt}',
            "prompt3": 'Translate {src} into {tgt_lang}:'
            }
    
    
    # 打开输出文件
    print('Start processing...')
    datas = read_test_data(args.data_path)
    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['layer','index','value','cosine','resultant_r','pca_ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in positions:
            # 读取测试数据
            for v in value_list:
                vector_list = []
                for example in tqdm(datas, desc=f"Processing p={p}, v={v}"):
                    src = example['src']
                    tgt_lang = example['tgt_lang']
                    
                    for prompt_id, prompt_template in prompt_templates.items():
                        prompt = prompt_template.format(src=src, tgt_lang=tgt_lang)
                        inputs = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

                        if 'gemma-2' or 'llama-3-1b' in args.model_name:
                            delta = get_delta_saelen(model, sae, tgt_layer, inputs, p, v)
                        if args.model_name == 'llama-3-8b':
                            delta = get_delta_sparsify(model, sae, tgt_layer, inputs, p, v)
    
                        # if is_trivial(delta): 
                        #     vector_list.append(delta)
                        # else: pass
                        vector_list.append(delta)

                print(f"vector count: {len(vector_list)}")

                vectors_np = torch.stack(vector_list, dim=0).numpy()   # (n, d) ndarray
                metrics = consistency_metrics(vectors_np)
                # metrics = consistency_metrics(vectors_np, p=0) 

                writer.writerow({
                'layer'       : tgt_layer,
                'index'       : p,
                'value'       : v,
                'cosine'      : f'{metrics["mean_cos"]:.4f}',
                'resultant_r' : f'{metrics["resultant_r"]:.4f}',
                'pca_ratio'   : f'{metrics["pca_ratio"]:.4f}',
                })
                csvfile.flush()  
    print(f"已写入结果到: {output_filepath}")
        
            

if __name__ == "__main__":
    main()

