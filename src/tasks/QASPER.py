from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class QASPER(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="QASPER",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/allenai/qasper",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "QASPER": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/naacl/DasigiLBCSG21,
                author       = {Pradeep Dasigi and
                                Kyle Lo and
                                Iz Beltagy and
                                Arman Cohan and
                                Noah A. Smith and
                                Matt Gardner},
                title        = {A Dataset of Information-Seeking Questions and Answers Anchored in
                                Research Papers},
                booktitle    = {{NAACL-HLT}},
                pages        = {4599--4610},
                publisher    = {Association for Computational Linguistics},
                year         = {2021}
                }""",
        dataset={
            "path": "Semantic/QASPER/",   
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
