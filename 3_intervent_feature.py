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
from math import exp

# try:
#     import torch._dynamo
#     torch._dynamo.config.suppress_errors = True
# except ImportError:
#     pass

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

def read_position(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datas.append(int(row['index']))
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
    parser.add_argument("--value",
                        default=0,
                        type=float,
                        help="editing value")
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

    tgt_layer = args.layer
    # 加载 SAE 模型
    if args.model_name == "gemma-2-2b-it":
        model_path = "/models/gemma-2-2b-it"
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_{}/width_16k/canonical".format(tgt_layer),
            device=device,
        )
    if args.model_name == "gemma-2-2b":
        model_path = "google/gemma-2-2b"
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_{}/width_16k/canonical".format(tgt_layer),
            device=device,
        )
    if args.model_name == 'gemma-2-9b-it':
        model_path = '/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819'
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
        model_path = '/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gemma-scope-9b-pt-res-canonical",
            sae_id = "layer_{}/width_16k/canonical".format(tgt_layer), 
            device=device,
        )
        print("SAE sparse_matrix的维度:", sae.W_dec.size())
    if args.model_name == 'llama-3-8b':
        model_path = '/models/Llama-3.1-8B'
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, low_cpu_mem_usage=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", 
            hookpoint="layers.{}".format(tgt_layer), 
            device=device
        )
    positions = read_position(args.feature_pos_path)
    v = args.value
    data_path = args.data_path
    # output_prefix = os.path.join(args.output_dir, f"{args.model_name}/layer{tgt_layer}.v{v}.editing.results")
    
    prompt_templates = {
            "prompt1": "Please translate the following query into {tgt_lang}, Provide only the translation: {src}:",
            "prompt2": 'Please translate the following query {src} into {tgt_lang}: {tgt}',
            "prompt3": 'Please translate {src} into {tgt_lang}: \n'
            }
    
    
    # 打开输出文件
    print('Start processing...')
    for p in positions:
        original_files = {}
        modified_files = {}
        for prompt_id, _ in prompt_templates.items():
            output_dir_for_prompt = os.path.join(
                args.output_dir, 
                args.model_name,
                f"{args.data_path.split('/')[-1].split('.')[1]}_edit",
                f"l{tgt_layer}_p{p}", 
                f"v{v}_{prompt_id}"
                )
            os.makedirs(output_dir_for_prompt, exist_ok=True)
            
            original_output_path = os.path.join(output_dir_for_prompt, f"{args.model_name}.original")
            modified_output_path = os.path.join(output_dir_for_prompt, f"{args.model_name}.modified")

            original_files[prompt_id] = open(original_output_path, "w", encoding='utf-8')
            modified_files[prompt_id] = open(modified_output_path, "w", encoding='utf-8')
        # 读取测试数据
        datas = read_test_data(args.data_path)
        for example in tqdm(datas, desc=f"Processing p={p}, v={v}"):
            src = example['src']
            src_lang = example['src_lang']
            tgt_lang = example['tgt_lang']
            tgt = example['tgt']
            lang_pair = f"{src_lang}-{tgt_lang}"
            
            for prompt_id, prompt_template in prompt_templates.items():
                prompt = prompt_template.format(src=src, tgt_lang=tgt_lang)
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                # --- 原始模型生成 ---
                pred_ori = model.generate(**inputs, max_new_tokens=args.max_seq_length, repetition_penalty=1.1, do_sample=False)
                answer_ori = tokenizer.batch_decode(pred_ori, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # 移除prompt部分
                answer_ori_clean = re.sub(re.escape(prompt), "", answer_ori, count=1).strip().replace('\t', '\n').replace('\n', ' ')
                # if "Here's" in answer_ori_clean:
                #     answer_ori_clean = answer_ori_clean.split("Here's")[0]

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

                    return (new_out0,) + out[1:]
            

                change_hook = model.model.layers[tgt_layer].register_forward_hook(layer_hook)
                pred_changed = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_seq_length, 
                    repetition_penalty=1.1, 
                    do_sample=False
                    )

                answer_changed = tokenizer.batch_decode(pred_changed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                change_hook.remove()
                answer_changed_clean = re.sub(re.escape(prompt), "", answer_changed, count=1).replace('\t', ' ').replace('\n', ' ')

                # 写入指定文档
                clean_src = src.strip()
                
                f_orig = original_files[prompt_id]
                f_mod = modified_files[prompt_id]

                f_orig.write(f"{lang_pair}\t{clean_src}\t{answer_ori_clean}\n")
                f_mod.write(f"{lang_pair}\t{clean_src}\t{answer_changed_clean}\n")
        for f in original_files.values():
            f.close()
        for f in modified_files.values():
            f.close()

if __name__ == "__main__":
    main()

