# 🌟LMEB: Long-horizon Memory Embedding Benchmark

<p align="center">
  <a href="https://huggingface.co/datasets/KaLM-Embedding/LMEB">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Data-ffbd45.svg" alt="HuggingFace">
  </a>
  <a href="https://huggingface.co/papers/2603.12572">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Paper-90EE90?style=flat" alt="HFPaper">
  </a>
  <a href="https://github.com/KaLM-Embedding/LMEB">
    <img src="https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github&" alt="GitHub Code">
  </a>
  <a href="https://arxiv.org/abs/2603.12572v1">
    <img src="https://img.shields.io/badge/Paper-LMEB-d4333f?logo=arxiv&logoColor=white&colorA=cccccc&colorB=d4333f&style=flat" alt="Paper">
  </a>
</p>

## Enviroment
```bash
conda create -n lmeb python==3.10
conda activate lmeb
pip install -r requirements.txt
```
Note: If you want to evaluate the NV-Embed-v2 model, install the dependencies from requirements_nv.txt instead, as NV-Embed-v2 relies on transformers==4.42.4:
```bash
pip install -r requirements_nv.txt
```

## Data
The LMEB benchmark dataset is required to run the evaluation. Download the dataset to the `eval_data` directory using the following command:
```bash
huggingface-cli download --repo-type dataset --resume-download KaLM-Embedding/LMEB --local-dir ./eval_data
```

## Evaluation
Run LMEB evaluation WITH task-specific instructions (w_inst = with instruction)
```bash
bash ./scripts/run_lmeb_w_inst.sh
```
Run LMEB evaluation WITHOUT task-specific instructions (wo_inst = without instruction)
```bash
bash ./scripts/run_lmeb_wo_inst.sh
```

- **model_path**: Path to the embedding model weights directory (e.g., "./models/KaLM_Embedding_V2.5"). This specifies where the pre-trained model files are stored.
- **tasks**: Name of the evaluation task(s) to run (e.g., "LoCoMo"). Multiple tasks can be specified (separated by commas).
- **benchmark**: Name of the benchmark suite to use (fixed as "LMEB" for this evaluation).
- **batch_size**: Batch size for model inference (e.g., 512). Controls the number of samples processed per iteration, balancing speed and memory usage.
- **output_dir**: Directory path to save evaluation results (e.g., "lmeb_results/").
- **precision**: Precision mode for model inference (e.g., "fp16"). Common values include "fp32" (full precision) and "fp16" (half precision, faster and memory-efficient).
- **model_kwargs**: JSON-formatted keyword arguments passed to the model initialization. Key sub-parameters include:
  - `trust_remote_code`: Whether to trust and execute remote code (set to `true` for custom models).
  - `attn_implementation`: Attention implementation method (e.g., "sdpa" for Scaled Dot-Product Attention).
  - `max_length`: Maximum input sequence length (e.g., 1024), limiting the number of tokens processed per sample.
  - `do_norm`: Whether to normalize the embedding vectors (set to `true` for better retrieval performance).
  - `use_instruction`: Whether to use task-specific instructions during encoding (set to `true` to enable instruction following).
  - `instruction_dict_path`: Path to the JSON file containing task instructions (e.g., "task_instructions.json").
  - `pooler_type`: Pooling method for generating sentence embeddings (e.g., "mean" for mean pooling).
  - `attn_type`: Attention type used in embedding (e.g., "biattn" for bidirectional attention).
  - `instruction_template`: Template for formatting instructions (e.g., "Instruct: {}\nQuery:" where `{}` is replaced with task-specific instructions).
- **encode_kwargs**: JSON-formatted keyword arguments passed to the embedding encoding process. Key sub-parameters include:
  - `normalize_embeddings`: Whether to normalize the final embeddings (consistent with `do_norm` for alignment).
  - `show_progress_bar`: Whether to display a progress bar during encoding (set to `true` for real-time progress tracking).
