from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class KnowMeBench(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="KnowMeBench",
        description=(
            ""
            ""
        ),
        reference="https://github.com/QuantaAlpha/KnowMeBench/tree/main/KnowmeBench",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "event_driven/adversarial_abstention": ["eng-Latn"],
            "event_driven/information_extraction": ["eng-Latn"],
            "event_driven/mind-body_interaction": ["eng-Latn"],
            "event_driven/mnestic_trigger_analysis": ["eng-Latn"],
            "event_driven/temporal_reasoning": ["eng-Latn"],

            "flashback_intensive/adversarial_abstention": ["eng-Latn"],
            "flashback_intensive/information_extraction": ["eng-Latn"],
            "flashback_intensive/mind-body_interaction": ["eng-Latn"],
            "flashback_intensive/mnestic_trigger_analysis": ["eng-Latn"],
            "flashback_intensive/temporal_reasoning": ["eng-Latn"],

            "psychological_depth/adversarial_abstention": ["eng-Latn"],
            "psychological_depth/information_extraction": ["eng-Latn"],
            "psychological_depth/mind-body_interaction": ["eng-Latn"],
            "psychological_depth/mnestic_trigger_analysis": ["eng-Latn"],
            "psychological_depth/temporal_reasoning": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{wu2026knowme,
                title={KnowMe-Bench: Benchmarking Person Understanding for Lifelong Digital Companions},
                author={Wu, Tingyu and Chen, Zhisheng and Weng, Ziyan and Wang, Shuhe and Li, Chenglong and Zhang, Shuo and Hu, Sen and Wu, Silin and Lan, Qizhen and Wang, Huacan and others},
                journal={arXiv preprint arXiv:2601.04745},
                year={2026}
                }""",
        dataset={
            "path": "Episodic/KnowMeBench/",   
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
