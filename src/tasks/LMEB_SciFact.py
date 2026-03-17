from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class LMEB_SciFact(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="LMEB_SciFact",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/allenai/scifact",
        type="Retrieval",
        category="t2t", 
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "LMEB_SciFact": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/emnlp/WaddenLLWZCH20,
                author       = {David Wadden and
                                Shanchuan Lin and
                                Kyle Lo and
                                Lucy Lu Wang and
                                Madeleine van Zuylen and
                                Arman Cohan and
                                Hannaneh Hajishirzi},
                title        = {Fact or Fiction: Verifying Scientific Claims},
                booktitle    = {{EMNLP} {(1)}},
                pages        = {7534--7550},
                publisher    = {Association for Computational Linguistics},
                year         = {2020}
                }""",
        dataset={
            "path": "Semantic/SciFact/",   
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
    MemType = "LMEB_Semantic"
    is_multilingual = True
