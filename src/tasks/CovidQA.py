from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class CovidQA(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="CovidQA",
        description=(
            ""
            ""
        ),
        reference="https://huggingface.co/datasets/illuin-conteb/covid-qa",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "Covid-QA": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{moller-etal-2020-covid,
                title = "{COVID-QA}: A Question Answering Dataset for {COVID}-19",
                author = {M{\"o}ller, Timo  and
                Reina, Anthony  and
                Jayakumar, Raghavan  and
                Pietsch, Malte},
                editor = "Verspoor, Karin  and
                Cohen, Kevin Bretonnel  and
                Dredze, Mark  and
                Ferrara, Emilio  and
                May, Jonathan  and
                Munro, Robert  and
                Paris, Cecile  and
                Wallace, Byron",
                booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID-19} at {ACL} 2020",
                month = jul,
                year = "2020",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2020.nlpcovid19-acl.18/",
            }""",
        dataset={
            "path": "Semantic/Covid-QA/",   
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
    MemType = "LMEB_Semantic"
    is_multilingual = True
