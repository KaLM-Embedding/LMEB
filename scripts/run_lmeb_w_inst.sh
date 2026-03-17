export CUDA_VISIBLE_DEVICES=0
export LOCAL_DATA_PREFIX="./eval_data"


### KaLM_Embedding_V2.5 ###
python run_lmeb.py \
    --model_path "./models/KaLM_Embedding_V2.5" \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 512 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": true, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "mean", "attn_type": "biattn", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### Qwen3-Embedding-0.6B ###
python run_lmeb.py \
    --model_path "./models/Qwen3-Embedding-0.6B" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 128 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### bge-m3 ###
python run_lmeb.py \
    --model_path "./models/bge-m3" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 128 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "first", "attn_type": "casual", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### jina-embeddings-v5-text-small-retrieval ###
python run_lmeb.py \
    --model_path "./models/jina-embeddings-v5-text-small-retrieval" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 128 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### jina-embeddings-v5-text-nano-retrieval ###
python run_lmeb.py \
    --model_path "./models/jina-embeddings-v5-text-nano-retrieval" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 128 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": true, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### multilingual-e5-large-instruct ###
python run_lmeb.py \
    --model_path "./models/multilingual-e5-large-instruct" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 256 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 512, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "mean", "attn_type": "biattn", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### KaLM_Embedding_V1.5 ###
python run_lmeb.py \
    --model_path "./models/KaLM_Embedding_V1.5" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 256 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "mean", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### bge-large-en-v1.5 ###
python run_lmeb.py \
    --model_path "./models/bge-large-en-v1.5" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 256 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 512, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "first", "attn_type": "biattn", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

# ### EmbeddingGemma-300M ###
python run_lmeb.py \
    --model_path "./models/EmbeddingGemma-300M" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 256 \
    --output_dir "lmeb_results/" \
    --precision "fp32" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "mean", "attn_type": "biattn", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### Qwen3-Embedding-4B ###
python run_lmeb.py \
    --model_path "./models/Qwen3-Embedding-4B" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 64 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### Qwen3-Embedding-8B ###
python run_lmeb.py \
    --model_path "./models/Qwen3-Embedding-8B" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 32 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### KaLM-Embedding-Gemma3###
python run_lmeb.py \
    --model_path "./models/KaLM-Embedding-Gemma3" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 32 \
    --output_dir "lmeb_results/" \
    --precision "bf16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### NV-Embed-v2###
python run_lmeb.py \
    --model_path "./models/NV-Embed-v2" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 32 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": true, "attn_implementation": "eager", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "mean", "attn_type": "biattn", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### bge-multilingual-gemma2###
python run_lmeb.py \
    --model_path "./models/bge-multilingual-gemma2" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 32 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \

### e5-mistral-7b-instruct###
python run_lmeb.py \
    --model_path "./models/e5-mistral-7b-instruct" \
    --run_kwargs '{"overwrite_results": false}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 32 \
    --output_dir "lmeb_results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": false, "attn_implementation": "sdpa", "max_length": 1024, "do_norm": true, "use_instruction": true, "instruction_dict_path": "task_instructions.json", "pooler_type": "last", "attn_type": "causal", "instruction_template": "Instruct: {}\nQuery:"}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": true}' \