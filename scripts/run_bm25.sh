export LOCAL_DATA_PREFIX="./eval_data"

# bm25
python run_lmeb.py \
    --model_path "bm25" \
    --run_kwargs '{"overwrite_results": true}' \
    --tasks "LoCoMo" \
    --benchmark "LMEB" \
    --batch_size 128 \
    --output_dir "lmeb_results/" \
    --encode_kwargs '{"show_progress_bar": true}' \
