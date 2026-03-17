from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class Gorilla(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="Gorilla",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "gorilla_huggingface": ["eng-Latn"],
            "gorilla_pytorch": ["eng-Latn"],
            "gorilla_tensor": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/nips/PatilZ0G24,
            author       = {Shishir G. Patil and
                            Tianjun Zhang and
                            Xin Wang and
                            Joseph E. Gonzalez},
            title        = {Gorilla: Large Language Model Connected with Massive APIs},
            booktitle    = {NeurIPS},
            year         = {2024}
            }""",
        dataset={
            "path": "Procedural/Gorilla/",   
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
