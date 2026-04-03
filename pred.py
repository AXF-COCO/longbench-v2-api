import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import random
from collections import deque

# TODO: change TPM_BUDGET as your API needs
# request rate limit
REQUEST_LOG = deque()
TPM_BUDGET = 420000

def estimate_tokens(text, model):
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text, disallowed_special=()))

def throttle_if_needed(prompt, max_output_tokens, model):
    """
    Token-based throttle. Only works per process, which is fine if n_proc=1.
    If the TPM is about to exceed the limit, sleep.
    """
    now = time.time()

    while REQUEST_LOG and now - REQUEST_LOG[0][0] > 60:
        REQUEST_LOG.popleft()

    used = sum(x[1] for x in REQUEST_LOG)
    need = estimate_tokens(prompt, model) + max_output_tokens

    if used + need > TPM_BUDGET:
        wait = 60 - (now - REQUEST_LOG[0][0]) + 1
        time.sleep(max(1, wait))

# TODO: add your model maxmium input token to 'config/model2maxlen.json'
# Find out the acutal max input token of your API
# Even though OpenAI states that GPT-5 supports 400,000 tokens, the actual max input token is 272,000 where the rest of the 400,000 is reserved for the maxmium possible output (128,000 tokens)
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_direct_answer = open('prompts/direct_answer.txt', encoding='utf-8').read()
template_extract_evidence = open('prompts/extract_evidence.txt', encoding='utf-8').read()
template_answer_from_evidence = open('prompts/answer_from_evidence.txt', encoding='utf-8').read()

# TODO: modify client if you need
client = OpenAI()

def call_llm(prompt, model, max_len, max_output_tokens, retries=5):
    prompt = truncate_middle(prompt, model, max_len)
    last_err = None
    for attempt in range(retries):
        try:
            # 
            throttle_if_needed(prompt, max_output_tokens, model)

            # TODO: modify response if you need
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "minimal"},
                max_output_tokens=max_output_tokens
            )

            # add (time, token_used) to log
            REQUEST_LOG.append((time.time(), estimate_tokens(prompt, model) + max_output_tokens))

            return response.output_text or ""
        except Exception as e:
            print(e)
            last_err = e

            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "Error code: 429" in err_str:
                sleep_s = min(30, (2 ** attempt) + random.random())
                time.sleep(sleep_s)
                continue

            time.sleep(1)
    raise RuntimeError(f'OpenAI call failed after {retries} retries: {last_err}')

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

def truncate_middle(prompt, model, max_len):
    tokenizer = tiktoken.encoding_for_model(model)
    prompt = tokenizer.encode(prompt, disallowed_special=())
    if len(prompt) > max_len:
        prompt = prompt[:max_len // 2] + prompt[-max_len // 2:]
    return tokenizer.decode(prompt)

def get_pred(data, args, fout):
    model = args.model
    max_len = maxlen_map[model]
    for item in tqdm(data):
        context = item['context']
        if args.prompt_variant == 'extract_then_answer':
            # extract
            prompt_evidence = build_extract_evidence_prompt(item, context)
            evidence = call_llm(prompt_evidence, model, max_len, args.evidence_max_output_tokens, args.retries)
            item['evidence'] = evidence
            # answer
            prompt_answer = build_answer_from_evidence_prompt(item, evidence)
            output = call_llm(prompt_answer, model, max_len, args.max_output_tokens, args.retries)
            pred, follow_instruction = extract_answer(output)
        else:  # args.prompt_variant == 'direct':
            prompt = build_direct_answer_prompt(item, context)
            output = call_llm(prompt, model, max_len, args.max_output_tokens, args.retries)
            pred, follow_instruction = extract_answer(output)
        item['output'] = output.strip()
        item['pred'] = pred
        item['follow_instruction'] = follow_instruction
        item['judge'] = pred == item['answer']
        item['context'] = item['context'][:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.prompt_variant == 'direct':
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_direct.jsonl")
    elif args.prompt_variant == 'extract_then_answer':
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_extract_then_answer.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

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

# TODO: change values if you need
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini")
    parser.add_argument('--prompt_variant', type=str, default='direct', choices=['direct', 'extract_then_answer'])
    parser.add_argument('--max_output_tokens', type=int, default=64)
    parser.add_argument('--evidence_max_output_tokens', type=int, default=256)
    parser.add_argument('--retries', type=int, default=5)
    # Note: n must be set to 1, since we need TPM controller only works per process
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args()
    main()