import os
import json
import logging
from pathlib import Path
from typing import Any
from time import time
from collections import defaultdict
from mteb.abstasks.task_metadata import TaskMetadata
from src.abstasks.LocalRetrieval import LocalRetrieval
from mteb.types import HFSubset, ScoresDict
from mteb._evaluators import RetrievalEvaluator
from mteb._evaluators.retrieval_metrics import make_score_dict
from mteb.models import (
    CrossEncoderProtocol,
    EncoderProtocol,
    MTEBModels,
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
    SearchProtocol,
)
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from metric import recall_cap, evaluate_recall_cap

logger = logging.getLogger(__name__)

PREFIX = os.environ.get("LOCAL_DATA_PREFIX", "./")


class SubsetRetrieval(LocalRetrieval):
    metadata = TaskMetadata(
        name="UselessTaskName",
        description=(
            "Instruction retrieval benchmark: queries include an instruction + query, "
            "corpus contains tool documentation passages."
        ),
        reference="https://example.com/toolret",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "web": ["eng-Latn"],
            "code": ["eng-Latn"],
            "customized": ["eng-Latn"]
        },
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Web"],
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

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> ScoresDict:
        """Rewrite the evaluation logic to fit the special memory retrieval scenario"""
        
        corpus = data_split["corpus"]
        queries = data_split["queries"]
        relevant_docs = data_split["relevant_docs"]
        
        # Dynamically adjust k_values and add corpus length as the maximum value
        original_k_values = list(self.k_values)
        extended_k_values = original_k_values + [len(corpus)]
        
        # Create retriever
        retriever = RetrievalEvaluator(
            corpus=corpus,
            queries=queries,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_ranked=data_split["top_ranked"],
            top_k=extended_k_values[-1],  # Use the maximum top_k
            **kwargs,
        )
        # Wrap the model
        if isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol):
            search_model = SearchEncoderWrapper(model)
        elif isinstance(model, CrossEncoderProtocol):
            search_model = SearchCrossEncoderWrapper(model)
        elif isinstance(model, SearchProtocol):
            search_model = model
        else:
            raise TypeError(
                f"RetrievalEvaluator expects a SearchInterface, Encoder, or CrossEncoder, got {type(model)}"
            )
        
        # Execute retrieval
        logger.info(f"Size of corpus: {len(corpus)}")
        logger.info(f"Size of queries: {len(queries)}")
        
        start_time = time()
        results = retriever(search_model, encode_kwargs=encode_kwargs)
        end_time = time()
        logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")
        
        # Load the mapping of scene to doc_ids
        sce_docids_map = self._load_scene_docids_mapping(hf_subset)
        
        # Filter the results, keeping only the candidate documents corresponding to each query
        filtered_results = self._filter_results_by_mapping(results, sce_docids_map)
        
        # Save the prediction results (if needed)
        if prediction_folder:
            self._save_task_predictions(
                filtered_results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )
        
        # Evaluate using the original k_values (excluding corpus length)
        logger.info("Running retrieval task - Evaluating retrieval scores...")
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
        ) = retriever.evaluate(
            relevant_docs,
            filtered_results,
            original_k_values,  # Use the original k_values
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )

        _, recall_cap = evaluate_recall_cap(
            relevant_docs,
            filtered_results,
            original_k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        
        task_specific_scores = self.task_specific_scores(
            all_scores,
            relevant_docs,
            filtered_results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        # print(recall_cap, task_specific_scores)
        task_specific_scores.update(recall_cap)
        
        logger.info("Running retrieval task - Finished.")
        
        return make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores,
            self._previous_results_model_meta,
        )
    
    def _load_scene_docids_mapping(self, hf_subset: str) -> dict[str, list[str]]:
        """
        Load the mapping from scene_id to candidate doc_ids

        Args:
            hf_subset: The name of the subset being processed

        Returns:
            A dictionary mapping {scene_id: [doc_id1, doc_id2, ...]}
        """
        sce_docids_map = {}
        origin_data_file = self.origin_data_file
        
        if not origin_data_file:
            logger.warning("No origin_data_file specified in metadata, skipping filtering")
            return sce_docids_map
        
        # Build complete paths
        if origin_data_file.endswith(".jsonl"):
            if self.metadata.name in ['LoCoMo', 'LongMemEval', 'REALTALK', 'TMD', 'CovidQA', 'ESGReports', 'QASPER', 'PeerQA', 'MLDR', 'NovelQA']:
                file_path = os.path.join(
                    PREFIX,
                    self.metadata.dataset["path"],
                    origin_data_file
                )
            else:
                file_path = os.path.join(
                    PREFIX,
                    self.metadata.dataset["path"],
                    hf_subset,
                    origin_data_file
                )
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data_point = json.loads(line)
                    sce_docids_map[str(data_point["scene_id"])] = data_point["candidate_doc_ids"]
        else:
            logger.warning("No candidate file")
        
        logger.info(f"Loaded {len(sce_docids_map)} scene-to-docs mappings from {file_path}")
        return sce_docids_map
    
    def _filter_results_by_mapping(
        self,
        results: dict[str, dict[str, float]],
        sce_docids_map: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """
        Filter the retrieval results based on the mapping

        Args:
            results: The original retrieval results {query_id: {doc_id: score}}
            sce_docids_map: The mapping from query to candidate doc_ids

        Returns:
            The filtered results
        """
        if not sce_docids_map:
            return results
        
        filtered_results = {}
        empty_count = 0
        
        for qid, doc_scores in results.items():
            scene_id = ""

            if self.metadata.name in ['CovidQA', 'ESGReports', 'QASPER', 'PeerQA', 'MLDR', 'NovelQA', 'LooGLE', 'DeepPlanning']:
                for can_scene_id in sce_docids_map.keys():
                    if qid==can_scene_id:
                        scene_id = can_scene_id
                        break
            else:
                for can_scene_id in sce_docids_map.keys():
                    if "_".join(qid.split("_")[:2])==can_scene_id:
                        scene_id = can_scene_id
                        break

            if scene_id=="": #
                logger.warning(f"Query ID {qid} not found in mapping, keeping all results")
                filtered_results[qid] = doc_scores
                continue
            
            candidate_docids = set(sce_docids_map[scene_id])
            
            if not candidate_docids:
                logger.warning(f"Query ID {qid} has empty candidate doc list")
                empty_count += 1
            
            # Keep only the documents that are in the candidate list
            filtered_results[qid] = {
                docid: score
                for docid, score in doc_scores.items()
                if docid in candidate_docids
            }
            
            if not filtered_results[qid]:
                logger.warning(f"Query ID {qid} has no matching documents after filtering")
        
        if empty_count > 0:
            logger.warning(f"Found {empty_count} queries with empty candidate lists")
        logger.info(f"Filtered results: {len(filtered_results)} queries")
        return filtered_results