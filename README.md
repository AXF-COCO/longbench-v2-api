# LongBench v2 API Version

## Overview
This repository is based on **LongBench v2**.

Compared with the original implementation, this version replaces local model inference with **API-based inference**.  
It currently supports two prompting strategies:

- **direct**: answer the question directly from the full context
- **extract_then_answer**: first extract evidence, then answer from the extracted evidence

To keep the request rate under control, this implementation is currently designed for **single-process execution**. Please keep `-n 1` when running inference.

This is a work-in-progress version and is not yet fully generalized.

If you use a different API provider, you may also need to modify the client initialization, request method, response parsing, and token counting logic in `pred.py`, in addition to the `TODO` sections.

---

## Setup

### 1. Create environment
```bash
conda create -n longbenchv2_api python=3.10
conda activate longbenchv2_api
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export OPENAI_API_KEY="your_key"
```
If you use a different provider, you should modify the client initialization in `pred.py`.

## Usage
#### 1. Run inference
```bash
python pred.py -m your_model_name --prompt_variant direct
python pred.py -m your_model_name --prompt_variant extract_then_answer
```
**Notes:**
- `-m` / `--model` should match the model name configured in `config/model2maxlen.json`
- `--prompt_variant` supports:
  - `direct`
  - `extract_then_answer`
- `-n` / `--n_proc` should stay `1`

#### 2. Evaluate results
```bash
python result.py
```
This will generate `result.tsv` from the files in the `results/` directory.

## Output

The inference script writes one file per run under `results/`, and each record contains:

- metadata fields such as domain, difficulty, and length
- the model output
- the parsed prediction
- whether the output follows the required format
- whether the prediction matches the ground truth answer

## Example
```bash
conda create -n longbenchv2_api python=3.10
conda activate longbenchv2_api
pip install -r requirements.txt

export OPENAI_API_KEY="abcde"

python pred.py -m gpt-5-mini --prompt_variant extract_then_answer
python pred.py -m gpt-5-mini --prompt_variant direct

python result.py
```





# Citation
If you use this repository, please also cite the following papers:

```
@article{bai2024longbench2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks}, 
  author={Yushi Bai and Shangqing Tu and Jiajie Zhang and Hao Peng and Xiaozhi Wang and Xin Lv and Shulin Cao and Jiazheng Xu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
@inproceedings{bai2024longbench,
    title = "{L}ong{B}ench: A Bilingual, Multitask Benchmark for Long Context Understanding",
    author = "Bai, Yushi and Lv, Xin  and Zhang, Jiajie  and Lyu, Hongchang  and
      Tang, Jiankai  and Huang, Zhidian  and Du, Zhengxiao  and Liu, Xiao  and Zeng, Aohan  and Hou, Lei  and Dong, Yuxiao  and Tang, Jie  and Li, Juanzi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.172",
    doi = "10.18653/v1/2024.acl-long.172",
    pages = "3119--3137",
}
```
