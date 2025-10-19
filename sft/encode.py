import os
import json
import argparse
import torch
import pathlib
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
# ==================================
from dataset import MultipleKnowledgeDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default="./data",type=str)
    parser.add_argument("--config_path",default="./config",type=str)
    parser.add_argument("--styled",action="store_true")
    parser.add_argument("--model_path",type=str,default="./models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file",type=str,default="./Data/processed/train.json")
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    # 文件保存目录
    model_name = pathlib.Path(args.model_path).name
    style_name = "styled" if args.styled else "no_styled"
    save_file_path = os.path.join(args.data_path,model_name)
    os.makedirs(save_file_path,exist_ok=True)
    save_file_name = os.path.join(save_file_path,f"{style_name}.pth")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if not os.path.exists(save_file_name):
        # 训练数据
        with open(args.train_file,mode="r",encoding="utf-8") as rfp:
            all_raw_data_list = json.load(rfp)
        # 系统提示词
        sys_prompt_file=os.path.join(args.config_path,"sys_prompt.json")
        with open(sys_prompt_file,mode="r",encoding="utf-8") as rfp:
            sys_prompt_dict = json.load(rfp)
            sys_prompt = sys_prompt_dict["prefix"] + sys_prompt_dict["suffix"]
        # 风格提示词
        style_prompt_file=os.path.join(args.config_path,"style_prompt.json")
        with open(style_prompt_file,mode="r",encoding="utf-8") as rfp:
            style_prompt_dict = json.load(rfp)
        saved_data_list = []
        max_length = 0
        min_length = float("inf")
        for data_dict in tqdm(all_raw_data_list,desc="Processing"):
            # 格式化数据
            # 用户提示词
            ques_prompt = data_dict["problem"]["question"]
            ans_prompt = f"<think>{data_dict['think_cot']}</think><answer>{data_dict['problem']['answer']}</answer>"
            if args.styled:
                language = data_dict["language"]
                knowledge_category = data_dict["knowledge_category"]
                granularity_subclass = data_dict["granularity_subclass"]
                # print(language,style_prompt_dict[language].keys(),knowledge_category)
                style_prompt = style_prompt_dict[language][knowledge_category]["Granularity"][granularity_subclass] + "\n"
            else:
                style_prompt = ""
            if "options" in data_dict["problem"]:
                opt_prompt = "\n".join([f"{chr(ord('A')+idx)}. {opt}"  for idx,opt in enumerate(data_dict["problem"]["options"])])
            else:
                opt_prompt = "\n"
            usr_prompt = f"{style_prompt}{ques_prompt}{opt_prompt}"
            # 合成带有格式的指令
            in_prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": sys_prompt},
                        {"role": "user","content":usr_prompt}], tokenize=False, add_generation_prompt=True)
            out_prompt = f"{ans_prompt}{tokenizer.eos_token}"
            # encode 成tokens ，并且区分出input_ids,attention_mask以及label_mask
            in_tokens = tokenizer.encode(in_prompt, add_special_tokens=False)
            out_tokens = tokenizer.encode(out_prompt, add_special_tokens=False)
            max_length = max(max_length,len(ans_prompt))
            min_length = min(min_length,len(ans_prompt))
            input_ids = in_tokens + out_tokens
            attention_mask = [1]*len(input_ids)
            labels = [-100]*len(in_tokens) + out_tokens
            saved_dict = {
                "style_prompt":style_prompt,
                "ques_prompt":f"{ques_prompt}{opt_prompt}",
                "ans_prompt":ans_prompt,
                "in_prompt":in_prompt,
                "out_prompt":out_prompt,
                "input_ids":input_ids,
                "attention_mask":attention_mask,
                "labels":labels
            }
            saved_data_list.append(saved_dict)
        torch.save(saved_data_list,save_file_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        train_file = "/mnt/data/kw/zg/Task5/train/sft/data/Qwen2.5-7B-Instruct/no_styled.pth"
        dataset = MultipleKnowledgeDataset(train_file,args.model_path)
        ds_loader = DataLoader(dataset,batch_size=10,shuffle=False,collate_fn=dataset.collate_fn)
        saved_data_list = []
        for batch in ds_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            print(f"input_ids shape:{input_ids.shape}, attention_mask shape:{attention_mask.shape}, labels shape:{labels.shape}")
if __name__ == "__main__":
    main()
