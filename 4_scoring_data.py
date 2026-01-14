import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from tqdm import tqdm
import argparse
import os
import json
from collections import defaultdict
from utils.load_model import load_model
import warnings


def read_target_features(path: str) -> defaultdict:
    """
    读取目标特征文件，格式为 "层索引,特征索引"。
    例如:
    12,2291
    13,3517
    
    返回: 一个字典，键是层索引(int)，值是该层需要追踪的特征索引列表(list[int])。
    """
    features_by_layer = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                layer_idx, feature_idx = map(int, line.split(','))
                features_by_layer[layer_idx].append(feature_idx)
            except ValueError:
                print(f"警告: 跳过格式不正确的行: {line}")
    print(features_by_layer)
    return features_by_layer


def read_translation_dataset(path: str) -> list[dict]:
    """
    读取翻译任务数据集，格式为 "英文源句\t中文目标句"。

    返回: 一个字典列表，每个字典包含 'src' 和 'tgt' 键。
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' not in line:
                continue
            src, tgt, comet_score = line.split('\t', 2)
            samples.append({'src': src.strip(), 'tgt': tgt.strip(), 'comet': comet_score.strip()})
    return samples


# --- 主逻辑函数 ---

def main():
    parser = argparse.ArgumentParser(description="记录数据集中样本的PPL和SAE特征激活值")
    # 基本参数
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace上的模型标识符，例如 'google/gemma-2-2b-it'")
    parser.add_argument("--dataset_path", type=str, required=True, help="待筛选的数据集文件路径 (例如 en2zh.txt)")
    parser.add_argument("--target_features_path", type=str, required=True, help="目标特征位置文件路径 (target_features.txt)")
    parser.add_argument("--output_dir", type=str, default="./data/selection", help="输出JSON文件的目录")
    
    # ADDED: 新增一个可选参数，用于直接指定目标语言，如果未提供则从文件名推断
    parser.add_argument("--target_lang", type=str, default=None, help="目标语言代码 (例如 'zh', 'th')。如果未提供，将尝试从文件名中推断。")

    # 其他参数
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--gpu_id", type=str, default='0', help="使用的GPU ID")
    parser.add_argument("--no_cuda", action='store_true', help="不使用CUDA")

    args = parser.parse_args()
    
    # ADDED: 从文件名或参数中获取目标语言
    if args.target_lang:
        tgt_lang = args.target_lang
        print(f"使用指定的目标语言: {tgt_lang}")
    else:
        try:
            # 例如从文件名 "en2th.txt" 中提取 "th"
            filename = os.path.basename(args.dataset_path)
            lang_pair = filename.split('.')[0]  # 'en2th'
            tgt_lang = lang_pair.split('2')[1]   # 'th'
            print(f"从文件名 {filename} 中推断出目标语言为: {tgt_lang}")
        except Exception as e:
            print(f"错误: 无法从文件名 {args.dataset_path} 推断目标语言。文件名应为'src2tgt.txt'格式。请使用 --target_lang 参数指定。")
            print(f"具体错误: {e}")
            return
            
    # --- 设备设置 ---
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
    print(f"使用设备: {device}")

    # --- 加载数据和模型 ---
    target_features = read_target_features(args.target_features_path)
    if not target_features:
        print("错误: 未能从文件中读取任何目标特征。请检查文件格式。")
        return
        
    dataset = read_translation_dataset(args.dataset_path)
    target_layers = sorted(target_features.keys())
    
    model, tokenizer, saes = load_model(args.model_name, target_layers, device)
    
    # MODIFIED: 定义 prompt 模板
    prompt_template = "Please translate the following text into {tgt_lang}: \n{src} \n"
    # prompt_template = "Human: Please translate the following text into {tgt_lang}: \n{src}\nAssistant: "

    # --- 开始处理 ---
    results_data = []
    print("开始处理数据集...")
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="分析样本"):
            src_text = sample['src']
            tgt_text = sample['tgt']
            comet_score = sample['comet']
            
            # MODIFIED: 根据模板构建正确的prompt和用于PPL计算的全文本
            prompt_text = prompt_template.format(tgt_lang=tgt_lang, src=src_text)
            full_text = prompt_text + tgt_text
            
            inputs = tokenizer(full_text, return_tensors='pt', max_length=args.max_seq_length, truncation=True).to(device)
            input_ids = inputs.input_ids
            if input_ids.shape[1] == 0: continue

            # --- 1. 计算困惑度 (Perplexity) ---
            outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss
            ppl = torch.exp(loss).item()

            # --- 2. 获取SAE激活值 ---
            prompt_only_inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
            src_end_token_idx = prompt_only_inputs.input_ids.shape[1] - 1
            
            # 检查索引是否越界 (在truncation发生时可能出现)
            if src_end_token_idx >= input_ids.shape[1]:
                src_end_token_idx = input_ids.shape[1] - 1

            hidden_states = outputs.hidden_states
            sample_sae_activations = {}
            
            for layer_idx, feature_indices in target_features.items():
                sae = saes[layer_idx]
                layer_hidden_state = hidden_states[layer_idx].to(torch.bfloat16)

                feature_activations = sae.encode(layer_hidden_state)
                
                # 在prompt结束位置提取激活值
                activations_at_pos = feature_activations[0, src_end_token_idx, :]

                for feat_idx in feature_indices:
                    key = f"{layer_idx}_{feat_idx}"
                    if feat_idx < activations_at_pos.shape[0]:
                        value = activations_at_pos[feat_idx].item()
                        sample_sae_activations[key] = value
                    else:
                        print(f"警告: 特征索引 {feat_idx} 超出SAE维度 {activations_at_pos.shape[0]}，跳过。")

            # --- 汇总结果 ---
            results_data.append({
                'src': src_text,
                'tgt': tgt_text,
                'comet': comet_score,
                'ppl': ppl,
                'sae_activations': sample_sae_activations
            })

    # --- 保存输出文件 (无需修改) ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name_safe = args.model_name.split('/')[-1]
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    output_path = os.path.join(args.output_dir, f"activations_{model_name_safe}_{dataset_name}.json")
    
    print(f"处理完成，结果保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
