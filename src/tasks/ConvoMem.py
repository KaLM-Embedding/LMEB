from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class ConvoMem(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="ConvoMem",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/Salesforce/ConvoMem",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "abstention_evidence": ["eng-Latn"],
            "assistant_facts_evidence": ["eng-Latn"],
            "changing_evidence": ["eng-Latn"],
            "implicit_connection_evidence": ["eng-Latn"],
            "preference_evidence": ["eng-Latn"],
            "user_evidence": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2511-10523,
                author = {Egor Pakhomov and
                                Erik Nijkamp and
                                Caiming Xiong},
                title = {Convomem Benchmark: Why Your First 150 Conversations Don't Need
                                {RAG}},
                journal = {CoRR},
                volume = {abs/2511.10523},
                year = {2025}
                }""",
        dataset={
            "path": "Dialogue/ConvoMem/",   
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
