import os
import csv
import numpy as np
from typing import List
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval

from datasets import load_dataset

PREFIX = os.environ.get("LOCAL_DATA_PREFIX", "./")

class LocalRetrieval(AbsTaskRetrieval):
    """
    Retrieval task for local ToolRet dataset with multiple subsets (web, code, customized).
    """
    metadata = TaskMetadata(
        name="UselessTaskName",
        description=(
            "Instruction retrieval benchmark: queries include an instruction + query, "
            "corpus contains tool documentation passages."
        ),
        reference="https://example.com/toolret",
        type="Retrieval",
        category="t2t",  # sentence-to-passage retrieval
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "web": ["eng-Latn"],
            "code": ["eng-Latn"],
            "customized": ["eng-Latn"]
        },
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{your_citation_2024,
            title={ToolRet: A Benchmark for Tool Retrieval},
            author={...},
            booktitle={...},
            year={2024}
            }""",
        dataset={
            "path": "../datasets/mangopy_toolret1",  
            "revision": "1.0"
        },
        
    )

    def load_qrels_from_tsv(self, qrels_path: str):
        qrels_dict = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=["query-id","corpus-id","score"])
            for row in reader:
                qid = row["query-id"]
                doc_id = row["corpus-id"]
                score = int(row["score"])
                qrels_dict.setdefault(qid, {})[doc_id] = score
        return qrels_dict

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        eval_splits = kwargs.get("eval_splits", ["test"])

        subsets = list(self.metadata.eval_langs.keys())
        dataset_name = self.metadata.name

        for hf_subset in subsets:
            self.corpus[hf_subset] = {}
            self.queries[hf_subset] = {}
            self.relevant_docs[hf_subset] = {}
            
            for split in eval_splits:
                # Path of the subset
                if dataset_name in ['CovidQA', "ESGReports", "LMEB_SciFact", "QASPER", "PeerQA", "MLDR", 'ToolBench']:
                    data_folder = os.path.join(PREFIX, self.metadata.dataset["path"])
                else:
                    data_folder = os.path.join(PREFIX, self.metadata.dataset["path"], hf_subset)
                # Path of the dataset
                dataset_folder = os.path.join(PREFIX, self.metadata.dataset["path"])

                queries_path = os.path.join(data_folder, self.query_file_name)
                if dataset_name in ['LoCoMo', 'LongMemEval', 'REALTALK', 'TMD', 'NovelQA', 'Proced_mem_bench']:
                    corpus_path = os.path.join(dataset_folder, self.corpus_file_name)
                elif dataset_name in ['EPBench', 'KnowMeBench', 'ReMe']:
                    corpus_path = os.path.join(os.path.dirname(data_folder.rstrip('/')), self.corpus_file_name)
                else:
                    corpus_path = os.path.join(data_folder, self.corpus_file_name)
                qrels_path = os.path.join(data_folder, self.qrels_dir, self.qrels_file_name)

                # Directly use datasets.load_dataset to read jsonl
                corpus_ds = load_dataset("json", data_files=corpus_path, split="train")
                queries_ds = load_dataset("json", data_files=queries_path, split="train")
                if "dialogue" in dataset_folder.lower():
                    print(f"Loaded {len(corpus_ds)} sess/rounds/turns and {len(queries_ds)} queries for {dataset_name} {hf_subset} {split} split.")
                elif "episodic" in dataset_folder.lower():
                    print(f"Loaded {len(corpus_ds)} events and {len(queries_ds)} queries for {dataset_name} {hf_subset} {split} split.")
                elif "semantic" in dataset_folder.lower():
                    print(f"Loaded {len(corpus_ds)} paragraph/sentence and {len(queries_ds)} queries for {dataset_name} {hf_subset} {split} split.")
                elif "procedural" in dataset_folder.lower():
                    print(f"Loaded {len(corpus_ds)} tool/experience/trajactory/item and {len(queries_ds)} queries for {dataset_name} {hf_subset} {split} split.")

                if "_id" in corpus_ds.column_names:
                    corpus_ds = corpus_ds.remove_columns("_id")
                if "_id" in queries_ds.column_names:
                    queries_ds = queries_ds.remove_columns("_id")

                queries = {str(q[self.query_id_field]): q[self.query_text_field] for q in queries_ds}
                corpus = {
                    d[self.corpus_id_field]: 
                    (d.get(self.corpus_title_field) or "") + " " + 
                    (d.get(self.corpus_text_field) or "") 
                    for d in corpus_ds
                }

                qrels_ds = self.load_qrels_from_tsv(qrels_path)


                self.corpus[hf_subset][split] = corpus
                self.queries[hf_subset][split] = queries
                self.relevant_docs[hf_subset][split] = qrels_ds
        self.data_loaded = True