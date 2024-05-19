import os.path

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
from tqdm import tqdm
import argparse
import jieba
from rouge import Rouge
from datasets import Dataset
import os
import sys
sys.path.append(os.getcwd())
from src.models.modeling_llama import LlamaForCausalLM, _make_causal_mask, _expand_mask

RANDOM_SEED = 2023


def decode_alg_hf_assist(batch_inputs, res_dict):

    ## use greedy decoding from HF
    infer_time = time.perf_counter()
    batch_response_ids = model.generate(**batch_inputs, **GENERATION_CONFIG, assistant_model=assist_model)
    infer_time = time.perf_counter() - infer_time
    # batch_response_ids = [q[i:-1] for i, q in zip(batch_inputs["attention_mask"].sum(axis=-1), batch_response_ids)]
    # print(tokenizer.batch_decode(batch_response_ids, skip_special_tokens=True))
    batch_responses = tokenizer.batch_decode(batch_response_ids[:, batch_inputs["input_ids"].shape[-1]:],
                                             skip_special_tokens=True)
    print(f"Generate greedy decoding with assistant time {infer_time:.3f}: \n{batch_responses[0]}")
    res_dict["infer_time_mask"].append(infer_time)
    res_dict["content_mask"].append(batch_responses[0])

    return res_dict

def decode_alg_hf(batch_inputs, res_dict):
    ## use greedy decoding from HF
    infer_time = time.perf_counter()
    batch_response_ids = model.generate(**batch_inputs, **GENERATION_CONFIG)
    infer_time = time.perf_counter() - infer_time
    # batch_response_ids = [q[i:-1] for i, q in zip(batch_inputs["attention_mask"].sum(axis=-1), batch_response_ids)]
    batch_responses = tokenizer.batch_decode(batch_response_ids[:, batch_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Generate greedy decoding time {infer_time:.3f}: \n{batch_responses[0]}")
    res_dict["infer_time_hf"].append(infer_time)

    return res_dict

def decode_alg_direct(batch_inputs, res_dict, do_sample=False):
    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    past_key_values = None
    attention_mask = batch_inputs["attention_mask"]
    prompt_len = input_ids.shape[-1]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)
    while True:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        if past_key_values:
            next_tokens_ids = next_tokens[:, None]
            position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            next_tokens_ids = input_ids

        with torch.no_grad():
            outputs = model(next_tokens_ids, attention_mask=attention_mask,  position_ids=position_ids, past_key_values=past_key_values, return_dict=True, use_cache=USE_CACHE)
        logits = torch.softmax(outputs.logits, dim=-1)
        next_token_logits = logits[:, -1, :]

        past_key_values = outputs.past_key_values
        if do_sample:
            # torch.random.manual_seed(RANDOM_SEED)
            next_tokens = torch.multinomial(next_token_logits, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        next_tokens =  next_tokens * unfinished_sequences + pad_token_id * (1-unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if unfinished_sequences.max() == 0 or input_ids.shape[-1]-prompt_len>=MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    res_dict["infer_time_direct"].append(infer_time)
    res_dict["content_direct"].append(responses[0])
    print(f"Direct greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    # assert responses == batch_responses
    return res_dict

def decode_alg_mask(batch_inputs, res_dict, token_dict, data_idx, do_sample=False, infer_dtype=torch.float16, save_data=True):
    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    attention_mask = batch_inputs["attention_mask"]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    if args.mask_diff.lower()=='false':
        Lm = MASK_ID * torch.ones((batch_size, MASK_NUM), dtype=input_ids.dtype, device=input_ids.device)
    else:
        Lm = MASK_ID + torch.arange(0, MASK_NUM, dtype=input_ids.dtype, device=input_ids.device).view(batch_size, -1)
    Lc = torch.tensor([MASK_ID for _ in range(MASK_NUM)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
    Pc = torch.tensor([torch.finfo(infer_dtype).max for _ in range(MASK_NUM)], dtype=infer_dtype, device=input_ids.device).repeat(batch_size, 1)

    past_key_values = None
    while True:
        input_ids_idx = input_ids.shape[-1]

        # 构建输入
        tmp = torch.hstack([torch.hstack([Lc[:, i: i + 1], Lm]) for i in range(MASK_NUM)])
        input_ids_extend = torch.hstack([input_ids, Lm, tmp])

        # 构建注意力矩阵和位置编码
        combined_attention_mask = _make_causal_mask(
            input_ids_extend.shape,
            infer_dtype,
            device=input_ids_extend.device,
            past_key_values_length=0,
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(torch.cat([attention_mask, attention_mask.new_ones((batch_size, MASK_NUM * (MASK_NUM + 2)))], dim=-1), infer_dtype, tgt_len=input_ids_extend.shape[-1]).to(
            input_ids_extend.device
        )
        for idx in range(input_ids_idx, expanded_attn_mask.shape[-1], MASK_NUM+1):
            expanded_attn_mask[:, :, idx+MASK_NUM:, idx: idx+MASK_NUM] = torch.finfo(infer_dtype).min
        attention_mask_extend = expanded_attn_mask + combined_attention_mask
        position_ids = (attention_mask_extend==0).sum(axis=-1).squeeze(0) - 1

        # run LLM
        if past_key_values:
            # device = 'cpu'
            kv_cache_idx = torch.tensor([input_ids_idx-new_generate_token+i*(MASK_NUM+1)-1 for i in range(1, new_generate_token)], dtype=int)
            kv_cache_idx = torch.hstack([torch.arange(0, input_ids_idx-new_generate_token, dtype=int), kv_cache_idx])
            # new_past_key_values = []
            # for kv_cache in past_key_values:
            #     new_past_key_values.append((kv_cache[0][:, :, kv_cache_idx, :], kv_cache[1][:, :, kv_cache_idx, :]))
            # past_key_values = new_past_key_values
            if args.model_type=='chatglm':
                past_key_values = [(kv_cache[0][kv_cache_idx, :, :, :], kv_cache[1][kv_cache_idx, :, :, :]) for kv_cache
                                   in past_key_values]
            else:
                past_key_values = [(kv_cache[0][:, :, kv_cache_idx, :], kv_cache[1][:, :, kv_cache_idx, :]) for kv_cache in past_key_values]

            input_ids_extend = input_ids_extend[:, input_ids_idx - 1:]
            position_ids = position_ids[:, input_ids_idx - 1:]
            attention_mask_extend = attention_mask_extend[:, :, input_ids_idx - 1:, :]
            input_ids_idx = 1

        with torch.no_grad():
            if args.model_type == 'chatglm':  # chatglm的推理与其他模型不太一样
                outputs = model(input_ids_extend, full_attention_mask=(attention_mask_extend<0).bool(), position_ids=position_ids, past_key_values=past_key_values,
                                return_dict=True, use_cache=USE_CACHE)
            else:
                outputs = model(input_ids_extend, attention_mask=attention_mask_extend, position_ids=position_ids,
                                past_key_values=past_key_values,
                                return_dict=True, use_cache=USE_CACHE)
        past_key_values = outputs.past_key_values

        logits = torch.softmax(outputs.logits, dim=-1)  # normalized logits

        if save_data:
            token_logits_candidate = logits[:, input_ids_idx: input_ids_idx + MASK_NUM, :]
            # token_candidate = torch.argmax(token_logits_candidate, dim=-1)
            # for i in range(MASK_NUM):
            #     token_dict[f"mask_{i}"].append(token_candidate[0, i].item())
            value, indice = torch.topk(token_logits_candidate, k=10, dim=-1)
            for i in range(MASK_NUM):
                token_dict[f"mask_idx_{i}"].append(json.dumps(indice[0, i, :].tolist()))
                token_dict[f"mask_val_{i}"].append(json.dumps(value[0, i, :].tolist()))

        new_generate_token = 0
        select_idx = input_ids_idx
        next_token_logit = logits[:, input_ids_idx - 1, :]
        for idx in range(MASK_NUM):
            if do_sample:
                condition = np.random.uniform() <= next_token_logit[:, Lc[:, idx]]/Pc[:, idx]
            else:
                condition = torch.argmax(next_token_logit, dim=-1)==Lc[:, idx]

            # condition = False
            if condition:
                next_tokens = Lc[:, idx]
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                                           dim=-1)
                new_generate_token += 1
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(
                        dim=0)
                )
                if unfinished_sequences.max() == 0:
                    break

                next_token_logit = logits[:, input_ids_idx - 1 + (idx + 1) * (MASK_NUM + 1), :]

                select_idx += MASK_NUM + 1
            else:
                break

        if do_sample:
            # torch.random.manual_seed(RANDOM_SEED)
            next_tokens = torch.multinomial(next_token_logit, num_samples=1).squeeze(1)
            Lc, Pc = [], []
            for bs in range(batch_size):
                candidate_tokens = torch.multinomial(logits[bs, select_idx: select_idx + MASK_NUM, :], num_samples=1)

                # TEST ONLY
                # candidate_tokens = []
                # for idx in range(MASK_NUM):
                #     torch.random.manual_seed(RANDOM_SEED)
                #     candidate_tokens.append(torch.multinomial(logits[bs, select_idx + idx: select_idx + idx + 1, :], num_samples=1).squeeze(1))
                # candidate_tokens = torch.tensor(candidate_tokens)

                Lc.append(candidate_tokens.reshape(1, -1))
                Pc.append(torch.tensor([logits[bs, select_idx+i, k] for i, k in enumerate(candidate_tokens)]).reshape(1, -1))
            Lc = torch.cat(Lc).to(device)
            Pc = torch.cat(Pc).to(device)
        else:
            next_tokens = torch.argmax(next_token_logit, dim=-1)
            # generate new candidate tokens
            Pc, Lc = torch.max(logits[:, select_idx: select_idx + MASK_NUM, :], dim=-1)

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        new_generate_token += 1

        if save_data:
            for i in range(MASK_NUM):
                token_dict[f"Lc_{i}"].append(Lc[0, i].item())
            token_dict["token"].append(json.dumps(input_ids[0, -new_generate_token:].cpu().numpy().tolist()))
            token_dict["idx"].append(data_idx)

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if unfinished_sequences.max() == 0 or input_ids.shape[-1]-prompt_len>=MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(f"{data_idx} Mask greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    res_dict["infer_time_mask"].append(infer_time)
    res_dict["content_mask"].append(responses[0])
    return token_dict, res_dict


def get_performance(out_text, df_res, df_token):
    for column in df_res.columns:
        if "time" in column:
            out_text += f"Infer time {column}: {df_res[column].mean():.3f}\n"
    time_percentage = df_res['infer_time_mask'].mean() / df_res['infer_time_direct'].mean()

    df_res['content_direct_token'] = df_res['content_direct'].apply(lambda x: len(tokenizer.encode(x)) + 1)  # +1 for eos token
    df_res['content_mask_token'] = df_res['content_mask'].apply(lambda x: len(tokenizer.encode(x)) + 1)  # +1 for eos token
    speedup_sum = df_res['content_mask_token'].sum() / df_res['infer_time_mask'].sum() / (
                df_res['content_direct_token'].sum() / df_res['infer_time_direct'].sum())
    speedup_avg = (df_res['content_mask_token'] / df_res['infer_time_mask']).mean() / (
        (df_res['content_direct_token'] / df_res['infer_time_direct']).mean())

    out_text += (f"infer time mask {df_res['infer_time_mask'].mean():.3f} direct {df_res['infer_time_direct'].mean():.3f}"
                 f" percentage {time_percentage:.3f} {1 / time_percentage:.3f} speedup_sum {speedup_sum:.3f} speedup_avg {speedup_avg:.3f}\n")

    if df_token is not None:
        num_infer = len(df_token)
        num_token = df_token['token'].apply(lambda x: len(json.loads(x))).sum()
        out_text += f"num infer {num_infer} num tokens {num_token} percentage {num_infer / num_token:.3f} {num_token / num_infer:.3f}\n"

        df_token['token_len'] = df_token['token'].apply(lambda x: len(json.loads(x)))
        for idx, value in df_token['token_len'].value_counts().items():
            out_text += f"{idx} count {value} percentage {value / len(df_token):.3f}\n"
        out_text += f"avg num of tokens {df_token['token_len'].mean():.3f}\n"

    if 'answer' in df_res.columns:
        rouger = Rouge()
        scores = rouger.get_scores(df_res['content_mask'].values, df_res['answer'].values, avg=True)
        for level, data_dict in scores.items():
            for k, v in data_dict.items():
                out_text += f"{level}: {k}: {v:.3f}\n"

    if 'category' in df_res.columns:
        for category, sub_df in df_res.groupby('category'):
            time_percentage = sub_df['infer_time_mask'].mean() / sub_df['infer_time_direct'].mean()
            out_text += f"{category} infer time mask {sub_df['infer_time_mask'].mean():.3f} direct {sub_df['infer_time_direct'].mean():.3f} percentage {time_percentage:.3f} {1 / time_percentage:.3f}\n"
    return out_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval model")
    parser.add_argument("--llm_dir", type=str, default='/data/FM/yihanling/sft/output/llama-2-7b-mask-5-diff-append_11-07-05-55/')
    parser.add_argument("--dataset", type=str, default="test_zh_100_v0.1", help="test_zh_100_v0.1, test_en_100_v0.1, xsum, mt-bench, cip, human_eval")
    parser.add_argument("--mask_id", type=int, default=32002)
    parser.add_argument("--mask_num", type=int, default=5)
    parser.add_argument("--mask_diff", type=str, default='false')
    parser.add_argument("--do_sample", type=str, default='false')
    parser.add_argument("--use_cache", type=str, default='false')
    parser.add_argument("--model_type", type=str, default='llama')
    parser.add_argument("--save_data", type=str, default='true')
    parser.add_argument("--assist_generation", type=str, default='false')
    args = parser.parse_args()
    if args.llm_dir[-1] != '/':
        args.llm_dir += '/'


    # llm_dir = os.path.dirname(os.path.abspath(__file__))
    # MASK_ID = 32002 if 'mask' in llm_dir else 0

    # llm_dir = '/data/FM/sft/output/llama2-70B-SAG_11-01-08-05/tianshu-70b-iter1-safe-24000-SAG/'  # 70B num_mask=5
    # MASK_ID = 76204

    llm_dir = args.llm_dir
    MASK_ID = args.mask_id
    MASK_NUM = args.mask_num
    do_sample = args.do_sample.lower()=='true'  # 是否使用采样
    USE_CACHE = args.use_cache.lower()=='true'

    print(f"Mask num {MASK_NUM} ID {MASK_ID} Do sample {do_sample} Use KVCache {USE_CACHE} Dir {llm_dir}")

    data_list = []
    if args.dataset == 'mt-bench':
        test_file_name = "mt-bench"
        with open(f'data/mt-bench-question.jsonl', 'r') as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": data_dict['turns'], "category": data_dict['category'], "id": data_dict['question_id']})
        additional_col = ["category", "id"]
        MAX_NEW_TOKENS = 1024
    elif args.dataset == 'cip':
        MAX_SAMPLE = 100
        test_file_name = "cip"
        with open(f"data/chatbot_instruction_prompts_test.jsonl", "r") as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": [data_dict['prompt']]})
                if len(data_list) == MAX_SAMPLE:
                    break
        additional_col = []
        MAX_NEW_TOKENS = 512
    elif args.dataset == 'human_eval':
        test_file_name = "human_eval"
        with open(f"data/humaneval-x-python.jsonl", "r") as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": ["Complete the following python code. Give the answer in a code block starts with '```python' and end with '``'\n"+data_dict['prompt']],
                                  "task_id": data_dict['task_id'],
                                  "declaration": data_dict['declaration'],
                                  "canonical_solution": data_dict['canonical_solution'],
                                  "test": data_dict['test'],
                                  "example_test": data_dict['example_test'] })
        additional_col = ["task_id", "declaration", "canonical_solution", "test", "example_test"]
        MAX_NEW_TOKENS = 1024
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    dataset = Dataset.from_list(data_list)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    GENERATION_CONFIG = {"max_new_tokens": MAX_NEW_TOKENS, \
                         "num_beams": 1, \
                         "do_sample": do_sample, \
                         "repetition_penalty": 1.0, \
                         "use_cache": USE_CACHE}

    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True, padding_side='left', add_bos_token=True)

    if 'llama' == args.model_type:
        model = LlamaForCausalLM.from_pretrained(llm_dir, device_map="auto").half()
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    elif 'llama-2-chat' == args.model_type:
        model = LlamaForCausalLM.from_pretrained(llm_dir, device_map="auto").half()
        system_prompt = "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    elif 'qwen' == args.model_type:
        from src.modeling_qwen import QWenLMHeadModel  # 修改代码使其适配半自回归推理
        model = QWenLMHeadModel.from_pretrained(llm_dir, device_map="auto", trust_remote_code=True).half()
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"  # system prompt
    else:
        model = AutoModelForCausalLM.from_pretrained(llm_dir, device_map="auto", trust_remote_code=True).half()
        system_prompt = ""
    model = model.eval()


    # load assistant model
    if args.assist_generation.lower()=='true':
        # assist_model = LlamaForCausalLM.from_pretrained(f"/{data_root}/FM/checkpoints/Llama-2-7b-chat-hf", device_map="auto").half()
        assist_model = LlamaForCausalLM.from_pretrained(f"/{data_root}/FM/checkpoints/TinyLlama-1.1B-Chat-v0.4",
                                                        device_map="auto").half()
        print("Successfully load assist model")

        assist_model = assist_model.eval()
        inputs = tokenizer('who are you?', return_tensors='pt', add_special_tokens=True).to(assist_model.device)
        outputs = assist_model.generate(**inputs)
        print(outputs)
        print(tokenizer.batch_decode(outputs))

    test_file_name += f"_mask_num_{MASK_NUM}_id_{MASK_ID}_do_sample_{do_sample}_use_cache_{USE_CACHE}"

    token_dict = {f"mask_idx_{i}": [] for i in range(MASK_NUM)}
    token_dict.update({f"mask_val_{i}": [] for i in range(MASK_NUM)})
    token_dict.update({"token": [], "idx": []})
    token_dict.update({f"Lc_{i}": [] for i in range(MASK_NUM)})

    res_dict = {"infer_time_hf": [], "infer_time_direct": [], "infer_time_mask": [], "input_text": [],
                "content_hf": [], "content_direct": [], "content_mask": []}
    res_dict.update({col: [] for col in additional_col})

    input_texts_list = []
    data_idx = 0
    if "chatglm" == args.model_type:
        template = "[Round 0]\n\n问：{}\n\n答："
        seperator = None
    elif args.model_type == 'qwen':
        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        seperator = None
    elif args.model_type == 'llama-2-chat':
        template = "[INST]{} [/INST]"
        seperator = " </s><s>"
    else:
        template = "Human:{}\nAssistant:"
        seperator = None
    print(f"using template {template}")

    data_idx = 0
    for data_dict in tqdm(dataset):
        history = system_prompt
        for input_texts in data_dict['input_text']:
            input_texts = template.format(input_texts)
            model_input_texts = '\n'.join([history, input_texts])

            input_tokens = tokenizer(model_input_texts, return_tensors='pt', add_special_tokens=True).to(model.device)
            print(f"model input text:\n{model_input_texts}")

            res_dict['input_text'].append(model_input_texts)
            res_dict = decode_alg_hf(input_tokens, res_dict)
            if args.assist_generation.lower() == 'true':
                res_dict = decode_alg_hf_assist(input_tokens, res_dict)
            # res_dict = decode_alg_direct(input_tokens, res_dict, do_sample=do_sample)
            token_dict, res_dict = decode_alg_mask(input_tokens, res_dict, token_dict, data_idx, do_sample=do_sample, save_data=(args.save_data.lower()=='true'))
            history = model_input_texts + '\n' + res_dict['content_mask'][-1]
            if seperator is not None:
                history += seperator

            for col in additional_col:
                res_dict[col].append(data_dict[col])
            data_idx += 1


    if args.save_data.lower()=='true':
        df_token = pd.DataFrame(token_dict)
        df_token.to_csv(f"{os.path.join(llm_dir, test_file_name)}_token_dict.csv", index=False)
    else:
        df_token = None

    df_res = pd.DataFrame({k: v for k, v in res_dict.items() if len(v)>0})
    df_res.to_csv(f"{os.path.join(llm_dir, test_file_name)}_res_dict.csv", index=False)

    # df_res.dropna(how="any", inplace=True)
    # compare performance
    out_text = ""
    out_text = get_performance(out_text, df_res, df_token)
    out_text += '\n' + '-' * 50 + '\n'

    with open(f"{os.path.join(llm_dir, test_file_name)}.txt", 'w') as f:
        f.writelines(out_text)

    if args.save_data.lower() == 'true':
        # 去除掉重复的样本
        df_res['jieba_num'] = df_res['content_direct'].apply(lambda x: [t for t in jieba.cut(x)]).apply(
            lambda x: len(x) / (len(set(x))+1e-6))
        df_res_filter = df_res[df_res['jieba_num'] <= 3.5]
        index_list = df_res[df_res['jieba_num'] <= 3.5].index
        df_token_filter = df_token[df_token['idx'].apply(lambda x: x in index_list)]
        if len(df_token_filter)>0:
            out_text = get_performance(out_text, df_res_filter, df_token_filter)

        # df_res_error = df_res[df_res['jieba_num'] > 3.5]
        # idx = 0
        # for infer_time_direct, infer_time_mask, content_mask in df_res_error[['infer_time_direct', 'infer_time_mask', 'content_mask']].values:
        #     print(f"{idx} infer time direct {infer_time_direct:.3f} mask {infer_time_mask:.3f} content {content_mask}")
        #     idx += 1

    out_text += f"{args.dataset} Mask num {MASK_NUM} ID {MASK_ID} Do sample {do_sample} Use KVCache {USE_CACHE} Dir {llm_dir}\n"
    out_text += '\n' + '#' * 50 + '\n'

    print(out_text)

    with open(f"{os.path.join(llm_dir, test_file_name)}.txt", 'w') as f:
        f.writelines(out_text)


