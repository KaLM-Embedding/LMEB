
embedding_model_name="local__KaLM_Embedding_V2.5"
python summary.py ./lmeb_results/${embedding_model_name}/wo_inst "LMEB"
python summary.py ./lmeb_results/${embedding_model_name}/wo_inst "LMEB" "R_cap_at_10"

embedding_model_name="local__KaLM_Embedding_V2.5"
python summary.py ./lmeb_results/${embedding_model_name}/w_inst "LMEB"
python summary.py ./lmeb_results/${embedding_model_name}/w_inst "LMEB" "R_cap_at_10"