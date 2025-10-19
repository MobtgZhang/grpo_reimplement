import os
import json
import random
import torch

from transformers import AutoTokenizer

class GRPORLDataset(torch.utils.data.Dataset):
    def __init__(self,train_file,pretrained_model_path,config_path,styled,**kwargs):
        with open(train_file,mode="r",encoding="utf-8") as rfp:
            all_raw_data_list = json.load(rfp)
        # 训练数据
        with open(train_file,mode="r",encoding="utf-8") as rfp:
            all_raw_data_list = json.load(rfp)
            random.shuffle(all_raw_data_list)
        # 系统提示词
        sys_prompt_file=os.path.join(config_path,"sys_prompt.txt")
        with open(sys_prompt_file,mode="r",encoding="utf-8") as rfp:
            sys_prompt = rfp.read()
            self.sys_prompt = sys_prompt
        # 风格提示词
        style_prompt_file=os.path.join(config_path,"style_prompt.json")
        with open(style_prompt_file,mode="r",encoding="utf-8") as rfp:
            self.style_prompt_dict = json.load(rfp)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.all_data_prompts_list = []
        for data_dict in all_raw_data_list:
            # 风格Prompt 处理
            if styled:
                language = data_dict["language"]
                knowledge_category = data_dict["knowledge_category"]
                granularity_subclass = data_dict["granularity_subclass"]
                style_prompt = self.style_prompt_dict[language][knowledge_category]["Granularity"][granularity_subclass] + "\n"
            else:
                style_prompt = ""
            # 添加sys_prompt 
            if "options" in data_dict["problem"]:
                opt_prompt = "\n" + "\n".join([f"{chr(ord('A')+idx)}. {opt}"  for idx,opt in enumerate(data_dict["problem"]["options"])])
            else:
                opt_prompt = ""
            # 问题
            ques_prompt = data_dict["problem"]["question"]
            # 用户提示词
            usr_prompt = f"{style_prompt}{ques_prompt}{opt_prompt}"
            # 将用户提示词
            main_prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.sys_prompt},
                {"role": "user","content":usr_prompt}], tokenize=False, add_generation_prompt=True)
            data_dict["main_prompt"] = main_prompt
            self.all_data_prompts_list.append(data_dict)
    def __len__(self):
        return len(self.all_data_prompts_list)
    def __getitem__(self,idx):
        return self.all_data_prompts_list[idx]
    def collate_fn(self, batch):
        main_prompt = [item["main_prompt"] for item in batch]
        problem = [item["problem"] for item in batch]
        language_list = [item["language"] for item in batch]
        know_list = [item["knowledge_category"] for item in batch]
        subfine_know_list = [item["granularity_subclass"] for item in batch]
        think_cot_list = [item["think_cot"] for item in batch]
        re_dict = {
            "prompts":main_prompt,
            "problem":problem,
            "think_cot":think_cot_list,
            "language":language_list,
            "knowledge_category":know_list,
            "granularity_subclass":subfine_know_list
        }
        return re_dict
