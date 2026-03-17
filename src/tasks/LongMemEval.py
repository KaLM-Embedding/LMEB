from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LongMemEval(SubsetRetrieval):
    metadata = TaskMetadata(
        name="LongMemEval",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "knowledge_update": ["eng-Latn"],
            "multi_session": ["eng-Latn"],
            "single_session_assistant": ["eng-Latn"],
            "single_session_preference": ["eng-Latn"],
            "single_session_user": ["eng-Latn"],
            "temporal_reasoning": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/iclr/WuWYZCY25,
                author       = {Di Wu and
                                Hongwei Wang and
                                Wenhao Yu and
                                Yuwei Zhang and
                                Kai{-}Wei Chang and
                                Dong Yu},
                title        = {LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
                                Memory},
                booktitle    = {{ICLR}},
                publisher    = {OpenReview.net},
                year         = {2025}
                }""",
        dataset={
            "path": "Dialogue/LongMemEval",   
            "revision": "1.0",
        },
    )
    query_file_name="queries.jsonl"
    query_id_field="id"
    query_text_field="text"
    corpus_file_name="corpus.jsonl"
    corpus_id_field="id"
    corpus_title_field="title"
    corpus_text_field="text"
    qrels_dir=""
    qrels_file_name="qrels.tsv"
    origin_data_file="candidates.jsonl"
    k_values = [1, 5, 10, 25, 50]
    MemType = "LMEB_Dialogue"
    is_multilingual = False
