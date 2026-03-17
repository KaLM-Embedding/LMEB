from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class PeerQA(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="PeerQA",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/UKPLab/PeerQA",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "PeerQA": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/naacl/BaumgartnerBG25,
                author       = {Tim Baumg{\"{a}}rtner and
                                Ted Briscoe and
                                Iryna Gurevych},
                title        = {PeerQA: {A} Scientific Question Answering Dataset from Peer Reviews},
                booktitle    = {{NAACL} (Long Papers)},
                pages        = {508--544},
                publisher    = {Association for Computational Linguistics},
                year         = {2025}
                }""",
        dataset={
            "path": "Semantic/PeerQA/",   
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
