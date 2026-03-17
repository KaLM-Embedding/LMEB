from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class ToolBench(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="ToolBench",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "ToolBench": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{DBLP:conf/iclr/QinLYZYLLCTQZHT24,
                author       = {Yujia Qin and
                                Shihao Liang and
                                Yining Ye and
                                Kunlun Zhu and
                                Lan Yan and
                                Yaxi Lu and
                                Yankai Lin and
                                Xin Cong and
                                Xiangru Tang and
                                Bill Qian and
                                Sihan Zhao and
                                Lauren Hong and
                                Runchu Tian and
                                Ruobing Xie and
                                Jie Zhou and
                                Mark Gerstein and
                                Dahai Li and
                                Zhiyuan Liu and
                                Maosong Sun},
                title        = {ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world
                                APIs},
                booktitle    = {{ICLR}},
                publisher    = {OpenReview.net},
                year         = {2024}
                }""",
        dataset={
            "path": "Procedural/ToolBench/",   
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
    MemType = "LMEB_Procedural"
    is_multilingual = True
