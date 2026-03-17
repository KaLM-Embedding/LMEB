from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class EPBench(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="EPBench",
        description=(
            ""
            ""
        ),
        reference="https://doi.org/10.6084/m9.figshare.28244480",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "default_claude_long/Entities": ["eng-Latn"],
            "default_claude_long/Event_contents": ["eng-Latn"],
            "default_claude_long/Full_event_details": ["eng-Latn"],
            "default_claude_long/Other_entities": ["eng-Latn"],
            "default_claude_long/Spaces": ["eng-Latn"],
            "default_claude_long/Times": ["eng-Latn"],

            "default_claude_short/Entities": ["eng-Latn"],
            "default_claude_short/Event_contents": ["eng-Latn"],
            "default_claude_short/Full_event_details": ["eng-Latn"],
            "default_claude_short/Other_entities": ["eng-Latn"],
            "default_claude_short/Spaces": ["eng-Latn"],
            "default_claude_short/Times": ["eng-Latn"],

            "default_claude_very_long/Entities": ["eng-Latn"],
            "default_claude_very_long/Event_contents": ["eng-Latn"],
            "default_claude_very_long/Full_event_details": ["eng-Latn"],
            "default_claude_very_long/Other_entities": ["eng-Latn"],
            "default_claude_very_long/Spaces": ["eng-Latn"],
            "default_claude_very_long/Times": ["eng-Latn"],

            "default_gpt4o_long/Entities": ["eng-Latn"],
            "default_gpt4o_long/Event_contents": ["eng-Latn"],
            "default_gpt4o_long/Full_event_details": ["eng-Latn"],
            "default_gpt4o_long/Other_entities": ["eng-Latn"],
            "default_gpt4o_long/Spaces": ["eng-Latn"],
            "default_gpt4o_long/Times": ["eng-Latn"],

            "default_gpt4o_short/Entities": ["eng-Latn"],
            "default_gpt4o_short/Event_contents": ["eng-Latn"],
            "default_gpt4o_short/Full_event_details": ["eng-Latn"],
            "default_gpt4o_short/Other_entities": ["eng-Latn"],
            "default_gpt4o_short/Spaces": ["eng-Latn"],
            "default_gpt4o_short/Times": ["eng-Latn"],

            "sci_fi_claude_long/Entities": ["eng-Latn"],
            "sci_fi_claude_long/Event_contents": ["eng-Latn"],
            "sci_fi_claude_long/Full_event_details": ["eng-Latn"],
            "sci_fi_claude_long/Other_entities": ["eng-Latn"],
            "sci_fi_claude_long/Spaces": ["eng-Latn"],
            "sci_fi_claude_long/Times": ["eng-Latn"],

            "sci_fi_claude_short/Entities": ["eng-Latn"],
            "sci_fi_claude_short/Event_contents": ["eng-Latn"],
            "sci_fi_claude_short/Full_event_details": ["eng-Latn"],
            "sci_fi_claude_short/Other_entities": ["eng-Latn"],
            "sci_fi_claude_short/Spaces": ["eng-Latn"],
            "sci_fi_claude_short/Times": ["eng-Latn"],

            "world_news_claude_long/Entities": ["eng-Latn"],
            "world_news_claude_long/Event_contents": ["eng-Latn"],
            "world_news_claude_long/Full_event_details": ["eng-Latn"],
            "world_news_claude_long/Other_entities": ["eng-Latn"],
            "world_news_claude_long/Spaces": ["eng-Latn"],
            "world_news_claude_long/Times": ["eng-Latn"],

            "world_news_claude_short/Entities": ["eng-Latn"],
            "world_news_claude_short/Event_contents": ["eng-Latn"],
            "world_news_claude_short/Full_event_details": ["eng-Latn"],
            "world_news_claude_short/Other_entities": ["eng-Latn"],
            "world_news_claude_short/Spaces": ["eng-Latn"],
            "world_news_claude_short/Times": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/iclr/HuetB025,
                author = {Alexis Huet and
                                Zied Ben{-}Houidi and
                                Dario Rossi},
                title = {Episodic Memories Generation and Evaluation Benchmark for Large Language
                                Models},
                booktitle = {{ICLR}},
                publisher = {OpenReview.net},
                year = {2025}
                }""",
        dataset={
            "path": "Episodic/EPBench/",   
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
    MemType = "LMEB_Episodic"
    is_multilingual = True
