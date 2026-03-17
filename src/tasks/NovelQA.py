from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class NovelQA(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="NovelQA",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/NovelQA/NovelQA",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "Character": ["eng-Latn"],
            "Meaning": ["eng-Latn"],
            "Plot": ["eng-Latn"],
            "Relation": ["eng-Latn"],
            "Setting": ["eng-Latn"],
            "Span": ["eng-Latn"],
            "Times": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/iclr/WangNPWGDBH0WZ25,
                author       = {Cunxiang Wang and
                                Ruoxi Ning and
                                Boqi Pan and
                                Tonghui Wu and
                                Qipeng Guo and
                                Cheng Deng and
                                Guangsheng Bao and
                                Xiangkun Hu and
                                Zheng Zhang and
                                Qian Wang and
                                Yue Zhang},
                title        = {NovelQA: Benchmarking Question Answering on Documents Exceeding 200K
                                Tokens},
                booktitle    = {{ICLR}},
                publisher    = {OpenReview.net},
                year         = {2025}
                }""",
        dataset={
            "path": "Semantic/NovelQA/",   
            "revision": "1.0",
        },
    )
    # 文件字段名声明
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
    MemType = "LMEB_Semantic"
    is_multilingual = True
