from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class MLDR(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="MLDR",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/illuin-conteb/mldr-conteb-eval",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "MLDR": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2402-03216,
                author       = {Jianlv Chen and
                                Shitao Xiao and
                                Peitian Zhang and
                                Kun Luo and
                                Defu Lian and
                                Zheng Liu},
                title        = {{BGE} M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity
                                Text Embeddings Through Self-Knowledge Distillation},
                journal      = {CoRR},
                volume       = {abs/2402.03216},
                year         = {2024}
                }""",
        dataset={
            "path": "Semantic/MLDR/",   
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
    origin_data_file = "candidates.jsonl"
    k_values = [1, 5, 10, 25, 50]
    MemType = "LMEB_Semantic"
    is_multilingual = True
