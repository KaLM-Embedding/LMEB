from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class MemBench(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="MemBench",
        description=(
            ""
            ""
        ),
        reference="https://github.com/import-myself/Membench/tree/main/MemData",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "aggregative": ["eng-Latn"],
            "comparative": ["eng-Latn"],
            "emotion": ["eng-Latn"],
            "knowledge_updating": ["eng-Latn"],
            "multi_hop": ["eng-Latn"],
            "multi_session_assistant": ["eng-Latn"],
            "post_processing": ["eng-Latn"],
            "preference": ["eng-Latn"],
            "single_hop": ["eng-Latn"],
            "single_session_assistant": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/acl/Tan000DD25,
                author       = {Haoran Tan and
                                Zeyu Zhang and
                                Chen Ma and
                                Xu Chen and
                                Quanyu Dai and
                                Zhenhua Dong},
                title        = {MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based
                                Agents},
                booktitle    = {{ACL} (Findings)},
                series       = {Findings of {ACL}},
                volume       = {{ACL} 2025},
                pages        = {19336--19352},
                publisher    = {Association for Computational Linguistics},
                year         = {2025}
                }""",
        dataset={
            "path": "Dialogue/MemBench/",   
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
