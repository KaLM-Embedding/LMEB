from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class TMD(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="TMD",
        description=(
            ""
            ""
        ),
        reference="https://github.com/Zyphra/TemporalMemoryDataset",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "content_time_qs": ["eng-Latn"],
            "date_span_time_qs": ["eng-Latn"],
            "dates_time_qs": ["eng-Latn"],
            "day_span_time_qs": ["eng-Latn"],
            "earlier_today_time_qs": ["eng-Latn"],
            "last_named_day_time_qs": ["eng-Latn"],
            "month_time_qs": ["eng-Latn"],
            "rel_day_time_qs": ["eng-Latn"],
            "rel_month_time_qs": ["eng-Latn"],
            "rel_session_time_qs": ["eng-Latn"],
            "session_span_time_qs": ["eng-Latn"],
            "session_time_qs": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2406-00057,
            author       = {Nick Alonso and
                            Tomas Figliolia and
                            Anthony Ndirango and
                            Beren Millidge},
            title        = {Toward Conversational Agents with Context and Time Sensitive Long-term
                            Memory},
            journal      = {CoRR},
            volume       = {abs/2406.00057},
            year         = {2024}
            }""",
        dataset={
            "path": "Dialogue/TMD/",   
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
