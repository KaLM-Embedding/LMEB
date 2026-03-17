from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class LoCoMo(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="LoCoMo",
        description=(
            ""
            ""
        ),
        reference="https://github.com/snap-research/locomo/tree/main/data",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "single_hop": ["eng-Latn"],
            "multi_hop": ["eng-Latn"],
            "temporal_reasoning": ["eng-Latn"],
            "open_domain": ["eng-Latn"],
            "adversarial": ["eng-Latn"]
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/acl/MaharanaLTBBF24,
                author       = {Adyasha Maharana and
                                Dong{-}Ho Lee and
                                Sergey Tulyakov and
                                Mohit Bansal and
                                Francesco Barbieri and
                                Yuwei Fang},
                title        = {Evaluating Very Long-Term Conversational Memory of {LLM} Agents},
                booktitle    = {{ACL} {(1)}},
                pages        = {13851--13870},
                publisher    = {Association for Computational Linguistics},
                year         = {2024}
                }""",
        dataset={
            "path": "Dialogue/LoCoMo/",   
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
