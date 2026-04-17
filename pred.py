from __future__ import annotations

import os, csv, json, yaml
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import torch.multiprocessing as mp
import random

from providers import OpenAIProvider, GeminiProvider

MODEL_MAXLEN_PATH = "config/model2maxlen.yaml"
API_KEY_NAME_PATH = "config/api_key_name.yaml"
DIRECT_PROMPT_PATH = "prompts/direct_answer.txt"
EXTRACT_PROMPT_PATH = "prompts/extract_evidence.txt"
ANSWER_FROM_EVIDENCE_PROMPT_PATH = "prompts/answer_from_evidence.txt"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

model2maxlen = load_yaml(MODEL_MAXLEN_PATH)
default_api_key_name = load_yaml(API_KEY_NAME_PATH)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

template_direct_answer = load_text(DIRECT_PROMPT_PATH)
template_extract_evidence = load_text(EXTRACT_PROMPT_PATH)
template_answer_from_evidence = load_text(ANSWER_FROM_EVIDENCE_PROMPT_PATH)


def build_direct_answer_prompt(item, doc_text):
    return (
        template_direct_answer
        .replace('$DOC$', doc_text)
        .replace('$Q$', item['question'].strip())
        .replace('$C_A$', item['choice_A'].strip())
        .replace('$C_B$', item['choice_B'].strip())
        .replace('$C_C$', item['choice_C'].strip())
        .replace('$C_D$', item['choice_D'].strip())
    )

def build_extract_evidence_prompt(item, doc_text):
    return (
        template_extract_evidence
        .replace('$DOC$', doc_text)
        .replace('$Q$', item['question'].strip())
    )

def build_answer_from_evidence_prompt(item, evidence_text):
    return (
        template_answer_from_evidence
        .replace('$EVIDENCE$', evidence_text)
        .replace('$Q$', item['question'].strip())
        .replace('$C_A$', item['choice_A'].strip())
        .replace('$C_B$', item['choice_B'].strip())
        .replace('$C_C$', item['choice_C'].strip())
        .replace('$C_D$', item['choice_D'].strip())
    )


DIRECT_RE = re.compile(r'^\s*"?The correct answer is \(([ABCD])\)"?\s*$', re.I | re.S)

def extract_answer(text):
    """
    Parse the LLM output and extract a direct answer choice.

    Args:
        text (str): The raw output from the LLM

    Returns:
        tuple[str | None, bool]:
            - The extracted answer (A, B, C, or D) if the pattern matches, otherwise None.
            - A boolean indicating whether the LLM followed the expected output pattern.
    """
    m = DIRECT_RE.match(text.strip())
    if not m:
        return None, False
    return m.group(1).upper(), True

def build_provider(args, api_key: str):
    if args.provider == 'openai':
        return OpenAIProvider(
            model=args.model,
            max_input_tokens=args.max_input_tokens,
            tpm_budget=args.tpm_budget,
            retries=args.retries,
            api_key=api_key,
            base_url=args.base_url
        )
    if args.provider == 'gemini':
        return GeminiProvider(
            model=args.model,
            max_input_tokens=args.max_input_tokens,
            tpm_budget=args.tpm_budget,
            retries=args.retries,
            temperature=args.temperature
        )
    raise ValueError(f'Unknown provider: {args.provider}')

def get_pred(data, args, fout):
    # get api_key in the env (not store in args for safety)
    api_key = os.getenv(args.api_key_name)
    if not api_key:
        raise ValueError(f'api_key not found under api_key_name {args.api_key_name}.')
    provider = build_provider(args, api_key)

    for item in tqdm(data):
        context = item['context']
        if args.prompt_variant == 'extract_then_answer':
            # extract
            prompt_evidence = build_extract_evidence_prompt(item, context)
            evidence = provider.generate(prompt_evidence, args.evidence_max_output_tokens)
            item['evidence'] = evidence
            # answer
            prompt_answer = build_answer_from_evidence_prompt(item, evidence)
            output = provider.generate(prompt_answer, args.max_output_tokens)
            pred, follow_instruction = extract_answer(output)
        elif args.prompt_variant == 'direct':
            prompt = build_direct_answer_prompt(item, context)
            output = provider.generate(prompt, args.max_output_tokens)
            pred, follow_instruction = extract_answer(output)
        else:
            raise ValueError('not valid prompt.')
        item['output'] = output.strip()
        item['pred'] = pred
        item['follow_instruction'] = follow_instruction
        item['judge'] = pred == item['answer']
        item['context'] = item['context'][:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    if args.prompt_variant == 'direct':
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_direct.jsonl")
    elif args.prompt_variant == 'extract_then_answer':
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_extract_then_answer.jsonl")
    else:
        raise ValueError('not valid prompt.')

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # which library you want to use while sending the request
    parser.add_argument("--provider", "-p", type=str, choices=list(default_api_key_name.keys()), required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument('--prompt_variant', type=str, default='direct', choices=['direct', 'extract_then_answer'])
    parser.add_argument('--max_output_tokens', type=int, default=128)
    parser.add_argument('--evidence_max_output_tokens', type=int, default=512)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--tpm_budget', type=int, default=1000000)
    parser.add_argument('--api_key_name', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    # Note: n must be set to 1, since we need TPM controller only works per process
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args()

    # check num of process
    if args.n_proc != 1:
        print('Warning: this version is sequential. Use --n_proc 1 for reliable tpm limit handling.')
    
    # check directory
    os.makedirs(args.save_dir, exist_ok=True)

    # get model's max_input_tokens
    if args.model not in model2maxlen:
        raise KeyError(f'model {args.model} not found in {MODEL_MAXLEN_PATH}')
    args.max_input_tokens = model2maxlen[args.model]

    # get default api_key_name if it is not set
    if args.api_key_name is None:
        args.api_key_name = default_api_key_name[args.provider]
    
    print(args)
    main()