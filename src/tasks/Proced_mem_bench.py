from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class Proced_mem_bench(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="Proced_mem_bench",
        description=(
            ""
            ""
        ),
        reference="https://github.com/qpiai/Proced_mem_bench/tree/main/procedural_memory_benchmark",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "easy": ["eng-Latn"],
            "medium": ["eng-Latn"],
            "hard": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2511-21730,
                author       = {Ishant Kohar and
                                Aswanth Krishnan},
                title        = {A Benchmark for Procedural Memory Retrieval in Language Agents},
                journal      = {CoRR},
                volume       = {abs/2511.21730},
                year         = {2025}
                }
                """,
        dataset={
            "path": "Procedural/Proced_mem_bench/",   
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
