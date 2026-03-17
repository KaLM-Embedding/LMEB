from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class REALTALK(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="REALTALK",
        description=(
            ""
            ""
        ),
        reference="https://github.com/danny911kr/REALTALK/tree/main/data",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "commonsense": ["eng-Latn"],
            "multi_hop": ["eng-Latn"],
            "temporal_reasoning": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2502-13270,
                author       = {Dong{-}Ho Lee and
                                Adyasha Maharana and
                                Jay Pujara and
                                Xiang Ren and
                                Francesco Barbieri},
                title        = {{REALTALK:} {A} 21-Day Real-World Dataset for Long-Term Conversation},
                journal      = {CoRR},
                volume       = {abs/2502.13270},
                year         = {2025}
                }""",
        dataset={
            "path": "Dialogue/REALTALK/",   
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
    MemType = "LMEB_Dialogue"
    is_multilingual = True
