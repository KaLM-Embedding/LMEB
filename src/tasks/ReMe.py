from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class ReMe(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="ReMe",
        description=(
            ""
            ""
        ),
        reference="https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "appworld_qwen3_8b/generalized_query": ["eng-Latn"],

            "appworld_qwen3_14b/generalized_query": ["eng-Latn"],

            "appworld_qwen3_32b/generalized_query": ["eng-Latn"],

            "bfcl_qwen3_8b/generalized_query": ["eng-Latn"],
            "bfcl_qwen3_8b/task_query": ["eng-Latn"],

            "bfcl_qwen3_14b/generalized_query": ["eng-Latn"],
            "bfcl_qwen3_14b/task_query": ["eng-Latn"],

            "bfcl_qwen3_32b/generalized_query": ["eng-Latn"],
            "bfcl_qwen3_32b/task_query": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{cao2025remember,
                title={Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution},
                author={Cao, Zouying and Deng, Jiaji and Yu, Li and Zhou, Weikang and Liu, Zhaoyang and Ding, Bolin and Zhao, Hai},
                journal={arXiv preprint arXiv:2512.10696},
                year={2025}
                }""",
        dataset={
            "path": "Procedural/ReMe/",   
            "revision": "1.0",
        },
    )
    query_file_name = "queries.jsonl"
    query_id_field = "id"
    query_text_field = "text"
    corpus_file_name = "corpus.jsonl"
    corpus_id_field = "id"
    corpus_title_field = "title"
    corpus_text_field = "text"
    qrels_dir = ""
    qrels_file_name = "qrels.tsv"
    origin_data_file = None
    k_values = [1, 5, 10, 25, 50]
    MemType = "LMEB_Procedural"
    is_multilingual = True
