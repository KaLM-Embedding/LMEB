from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class DeepPlanning(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="DeepPlanning",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/Qwen/DeepPlanning",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "shopping_level1": ["eng-Latn"],
            "shopping_level2": ["eng-Latn"],
            "shopping_level3": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{zhang2026deepplanning,
            title={DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints},
            author={Zhang, Yinger and Jiang, Shutong and Li, Renhao and Tu, Jianhong and Su, Yang and Deng, Lianghao and Guo, Xudong and Lv, Chenxu and Lin, Junyang},
            journal={arXiv preprint arXiv:2601.18137},
            year={2026}
            }""",
        dataset={
            "path": "Procedural/DeepPlanning/",   
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
    MemType = "LMEB_Procedural"
    is_multilingual = True
