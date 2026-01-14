import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import csv, json, jsonlines
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import defaultdict
from sparsify import Sae
import warnings
warnings.filterwarnings("ignore")
from sae_lens import SAE
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path):
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

def get_key_index(encoded_inputs,prompt,src,tgt,tgt_lang):
    offset_mapping = encoded_inputs['offset_mapping'][0] # 对于单条输入，取出其offset_mapping
    encoding = encoded_inputs.encodings[0] # 获取底层的 Encoding 对象，它有 char_to_token 方法

    # get src index
    src_char_start = prompt.find(src)
    src_char_end = src_char_start + len(src)
    src_end_token_idx = encoding.char_to_token(src_char_end - 1)

    # get prompt index(src的第一个字符位置)
    src_first_char_idx = prompt.find(src)
    src_first_token_idx = encoding.char_to_token(src_first_char_idx)

    # 如果 src_first_token_idx 不是 None 且大于 0 (不是序列的第一个 token)
    src_before_token_idx = None
    if src_first_token_idx is not None and src_first_token_idx > 0:
        src_before_token_idx = src_first_token_idx - 1

    # get tgt_lang index
    tgt_lang_char_start = prompt.find(tgt_lang)
    tgt_lang_char_end = tgt_lang_char_start + len(tgt_lang)
    tgt_lang_end_token_idx = encoding.char_to_token(tgt_lang_char_end - 1)

    # get tgt index
    tgt_char_start = prompt.find(tgt)
    tgt_char_end = tgt_char_start + len(tgt)
    tgt_end_token_idx = encoding.char_to_token(tgt_char_end - 1)

    return src_end_token_idx, src_before_token_idx, tgt_lang_end_token_idx, tgt_end_token_idx
    # return src_idx, prompt_idx, tgt_lang_idx, tgt_idx

def filter_attrs(res_dict,top_k,min_act):
    total_lines = len(res_dict)
    all_combined_attrs = []
    for single_res in res_dict:
        for prompt_attrs_list in single_res['all_attrs']:
            all_combined_attrs.extend(prompt_attrs_list)

    acts_by_key = defaultdict(list)  # (layer,pos) -> [act1, act2, ...]
    for layer_id, index, value in all_combined_attrs:
        acts_by_key[(layer_id, index)].append(value)

    n_arrays = len(res_dict[0]['all_attrs'])           # = len(js['all_attrs'])
    threshold = 0.6 * n_arrays * total_lines

    # 过滤
    qualified = {k: v for k, v in acts_by_key.items() if len(v) > threshold}

    stats = []   # (avg_act, (layer,pos), occurrences)
    for key, val_list in qualified.items():
        avg_act = sum(val_list) / len(val_list)
        if avg_act > min_act:
            stats.append((avg_act, key, len(val_list)))

    stats.sort(reverse=True)                # 按平均激活值降序
    top_stats = stats[:top_k]

    unique_attrs = [(layer, pos) for avg_act, (layer, pos), _ in top_stats]
    return unique_attrs


def get_context_attr_saelen(idx, src, tgt, tgt_lang, args, model, tokenizer, sae, device):
    layer_id = args.layer
    
    all_attrs = []
    res_dict = {
                'idx': idx,
                'src': src,
                'tgt_lang': tgt_lang,
                'tgt': tgt,
                'prompt': [],
                'all_attrs': None
            }
    # prompts = [f'Translate {src} into {tgt_lang}: {tgt}']
    prompts = [
        f"Please translate the following query into {tgt_lang}, Provide only the translation: {src}: {tgt}",
        f'Please translate the following query {src} into {tgt_lang}: {tgt}',
        f'Translate {src} into {tgt_lang}: {tgt}'
    ]
    
    for prompt in prompts:
        encoded_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True 
        )
        inputs = encoded_inputs['input_ids'].to(device)
        src_idx, prompt_idx, tgt_lang_idx, tgt_idx = get_key_index(encoded_inputs,prompt,src,tgt,tgt_lang)
        if args.interp_tgt == 'prompt':
            obs_idx = prompt_idx
        if args.interp_tgt == 'tgt_lang':
            obs_idx = tgt_lang_idx
        if args.interp_tgt == 'tgt_idx':
            obs_idx = tgt_idx

        for tgt_layer in range(layer_id, layer_id + 1): # range(model.model.config.num_hidden_layers - 1):
            # print("第{}层".format(tgt_layer))
            sae_inter_dict = dict()
            def w_forward_hook_fn(module, inp, outp):
                out0 = outp[0]
                inter = sae.encode(out0)
                sae_inter_dict["inter"] = inter
                out0_rec = sae.decode(inter)
                out0[:, -1:, :] = out0_rec[:, -1:, :]
                out0 = out0_rec
                # sae_inter_dict["inter"].retain_grad()
                out = (out0, )
                return out

            hook = model.model.layers[tgt_layer].register_forward_hook(w_forward_hook_fn)
            outputs = model(inputs)
            hook.remove()
            
            sae_inter = sae_inter_dict["inter"]
            # print(sae_inter.size())
            # hidden_states = hidden_states[:, prompt_idx, :]
            
            acts = sae_inter[0, obs_idx, :]
            non_zero_indices = acts.nonzero(as_tuple=True)[0] #[nonzero]
            attr = []
            for idx in non_zero_indices:
                attr.append([tgt_layer, idx.item(), acts[idx].item()]) #[nonzero, 4]
            
            all_attrs.append(attr)
        res_dict['prompt'].append(prompt)
        res_dict['all_attrs'] = all_attrs

    return res_dict

def get_context_attr_sparify(idx, src, tgt, tgt_lang, args, model, tokenizer, sae, device):
    layer_id = args.layer
    
    all_attrs = []
    res_dict = {
                'idx': idx,
                'src': src,
                'tgt_lang': tgt_lang,
                'tgt': tgt,
                'prompt': [],
                'all_attrs': None
            }
    prompts = [
        f'Please translate {src} into {tgt_lang}: {tgt}',
        f'Please translate the following query {src} into {tgt_lang}: {tgt}',
        f'Translate {src} into {tgt_lang}: {tgt}'
    ]
    
    for prompt in prompts:
        encoded_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True 
        )
        inputs = encoded_inputs.to(device)
        src_idx, prompt_idx, tgt_lang_idx, tgt_idx = get_key_index(encoded_inputs,prompt,src,tgt,tgt_lang)
        if args.interp_tgt == 'prompt':
            obs_idx = prompt_idx
        if args.interp_tgt == 'tgt_lang':
            obs_idx = tgt_lang_idx
        if args.interp_tgt == 'tgt_idx':
            obs_idx = tgt_idx

        for tgt_layer in range(layer_id, layer_id + 1):
            sae_inter_dict = dict()
            
            def w_forward_hook_fn(module, inp, outp):
                original_hidden_state = outp[0]
                flat_hidden_state = original_hidden_state.view(-1, original_hidden_state.shape[-1])#.to("cpu")
                top_acts, top_indices, pre_acts = sae.encode(flat_hidden_state)

                sae_inter_dict["inter"] = (top_acts, top_indices)
                reconstructed_flat = sae.decode(top_acts, top_indices)
                reconstructed_hidden_state = reconstructed_flat.view_as(original_hidden_state)
                reconstructed_hidden_state = reconstructed_hidden_state.to(original_hidden_state.dtype).to(original_hidden_state.device)

                return (reconstructed_hidden_state,) + outp[1:]

            # Llama-3的模型结构是 model.model.layers
            hook_target = model.model.layers[tgt_layer]
            hook = hook_target.register_forward_hook(w_forward_hook_fn)
            
            with torch.no_grad(): # 推理时最好加上no_grad以节省显存和计算
                outputs = model(**inputs) # 确保只传递input_ids
            
            hook.remove()
            
            # 从字典中获取存储的稀疏激活值
            sae_inter, sae_indic = sae_inter_dict["inter"]
            # print("Shape of sae_inter:", sae_inter.shape) 

            acts = sae_inter[obs_idx, :]
            indices = sae_indic[obs_idx, :]
            attr = []
            for act_idx, indice_idx in zip(acts, indices): # 变量名用 act_idx 更清晰
                attr.append([tgt_layer, indice_idx.item(), act_idx.item()])
            
            all_attrs.append(attr)
        res_dict['prompt'].append(prompt)
        res_dict['all_attrs'] = all_attrs
    return res_dict


def main():
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    # parameters about integrated grad
    parser.add_argument("--layer",
                        default=24,
                        type=int,
                        help="layer")
    parser.add_argument("--interp_tgt",
                        default='prompt',
                        type=str)
    parser.add_argument("--topk",
                        default=50,
                        type=int,
                        help="filter threshold")
    args = parser.parse_args()


    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # TODO: To implement multi gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # set output dir
    model_path = ''
    tgt_layer = args.layer
    if args.model_name == 'gemma-2-2b-it':
        model_path = '/models/gemma-2-2b-it'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()        
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gemma-scope-2b-pt-res-canonical",
            sae_id = "layer_{}/width_16k/canonical".format(tgt_layer), 
            device=device,
        )
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
    if args.model_name == 'gemma-2-9b-it':
        model_path = '/models/gemma-2-9b-it'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gemma-scope-9b-pt-res-canonical",
            sae_id = "layer_{}/width_16k/canonical".format(tgt_layer), 
            device=device,
        )
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
    if args.model_name == 'gemma-2-9b':
        model_path = '/models/gemma-2-9b'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gemma-scope-9b-pt-res-canonical",
            sae_id = "layer_{}/width_16k/canonical".format(tgt_layer), 
            device=device,
        )
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
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
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
    if args.model_name == 'llama-3-8b':
        model_path = '/models/Llama-3.1-8B'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae = Sae.load_from_hub(
            "EleutherAI/sae-llama-3-8b-32x", 
            hookpoint="layers.{}".format(tgt_layer),
            device=device
        )
        print("SAE sparse_matrix的维度:", sae.W_dec.size())

    output_path = os.path.join(args.output_dir, args.model_name, args.data_path.split('/')[-1].split('.')[1])
    output_prefix = f"{args.data_path.split('/')[-1]}.layer{args.layer}"
    os.makedirs(output_path, exist_ok=True)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    # load data
    data = read_data(args.data_path) 
    
    tic = time.perf_counter()
    print(f"Start process, dataset size: {len(data)}")
    stragegys = ['prompt', 'tgt_lang', 'tgt_idx']
    filtered_results = []
    if 'gemma-2' in args.model_name:
        for strategy in stragegys:
            args.interp_tgt = strategy  
            results = []      
            for idx, example in enumerate(tqdm(data)):            
                src = example['src']
                tgt_lang = example['tgt_lang']
                tgt = example['tgt']
                res_dict = get_context_attr_saelen(idx, src, tgt, tgt_lang, args, model, tokenizer, sae, device)
                results.append(res_dict)
            filtered_results.extend(filter_attrs(results, args.topk, 10))
    if 'llama-3-1b' in args.model_name:
        for strategy in stragegys:
            args.interp_tgt = strategy  
            results = []      
            for idx, example in enumerate(tqdm(data)):            
                src = example['src']
                tgt_lang = example['tgt_lang']
                tgt = example['tgt']
                res_dict = get_context_attr_saelen(idx, src, tgt, tgt_lang, args, model, tokenizer, sae, device)
                results.append(res_dict)
            filtered_results.extend(filter_attrs(results, args.topk, 1))
    if args.model_name == 'llama-3-8b':
        for strategy in stragegys:
            args.interp_tgt = strategy  
            results = []      
            for idx, example in enumerate(tqdm(data)):            
                src = example['src']
                tgt_lang = example['tgt_lang']
                tgt = example['tgt']
                res_dict = get_context_attr_sparify(idx, src, tgt, tgt_lang, args, model, tokenizer, sae, device)
                results.append(res_dict)
            filtered_results.extend(filter_attrs(results, args.topk, 0.4))
    final_results = list(set(filtered_results))     

    with open(os.path.join(output_path, output_prefix), 'w', encoding='utf-8') as fw:        
        for result in final_results:
            fw.write('\t'.join(map(str, list(result))) + '\n')

        
    print(f"Saved in {os.path.join(output_path, output_prefix)}")

    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")

if __name__ == "__main__":
    main()
