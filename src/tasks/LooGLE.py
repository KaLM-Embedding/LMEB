from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class LooGLE(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="LooGLE",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/bigai-nlco/LooGLE",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "LongDepQA": ["eng-Latn"],
            "ShortDepQA": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/acl/LiWZZ24,
                author       = {Jiaqi Li and
                                Mengmeng Wang and
                                Zilong Zheng and
                                Muhan Zhang},
                title        = {LooGLE: Can Long-Context Language Models Understand Long Contexts?},
                booktitle    = {{ACL} {(1)}},
                pages        = {16304--16333},
                publisher    = {Association for Computational Linguistics},
                year         = {2024}
                }""",
        dataset={
            "path": "Semantic/LooGLE/",   
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
