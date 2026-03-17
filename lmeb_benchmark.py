from __future__ import annotations

import logging
from typing import Sequence
import mteb
from mteb.benchmarks.benchmark import Benchmark
from mteb.get_tasks import MTEBTasks

logger = logging.getLogger(__name__)

LMEB_CITATION = r"""
@misc{zhao2026lmeb,
      title={LMEB: Long-horizon Memory Embedding Benchmark}, 
      author={Xinping Zhao and Xinshuo Hu and Jiaxin Xu and Danyu Tang and Xin Zhang and Mengjia Zhou and Yan Zhong and Yao Zhou and Zifei Shan and Meishan Zhang and Baotian Hu and Min Zhang},
      year={2026},
      eprint={2603.12572},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.12572}, 
}
""".strip()


def build_lmeb() -> Benchmark:
    """
    Define LMEB benchmark.
    """
    tasks = mteb.get_tasks(
        tasks=[
            "EPBench",
            "KnowMeBench",
            "LoCoMo",
            "LongMemEval",
            "REALTALK",
            "TMD",
            "MemBench",
            "ConvoMem",
            "QASPER",
            "NovelQA",
            "PeerQA",
            "CovidQA",
            "ESGReports",
            "MLDR",
            "LooGLE",
            "LMEB_SciFact",
            "Gorilla",
            "ToolBench",
            "ReMe",
            "Proced_mem_bench",
            "MemGovern",
            "DeepPlanning",
        ],
        languages=["eng"],
        eval_splits=["test"],
    )

    return Benchmark(
        name="LMEB",
        display_name="LMEB",
        icon=None,
        tasks=MTEBTasks(tasks),
        description="LMEB: Long-horizon Memory Embedding Benchmark",
        reference=None,
        citation=LMEB_CITATION or None,
        contacts=None,
        display_on_leaderboard=True,
    )


LMEB = build_lmeb()


def register_lmeb(clear_cache: bool = True) -> Benchmark:
    """
    Inject LMEB into the mteb.benchmarks.benchmarks module,
    so that get_benchmark/get_benchmarks can find it.

    Usage:
        import lmeb_benchmark
        from mteb.benchmarks import get_benchmark
        b = get_benchmark("LMEB")
    """
    # 1) Inject into mteb.benchmarks.benchmarks module globals
    import mteb.benchmarks.benchmarks as bm

    setattr(bm, "LMEB", LMEB)

    # 2) Clear the cache (important! Otherwise, the _build_registry() might cache old results)
    if clear_cache:
        try:
            from mteb.benchmarks.get_benchmark import (
                # _build_aliases_registry,
                _build_registry,
            )

            _build_registry.cache_clear()
            # _build_aliases_registry.cache_clear()
        except Exception as e:
            logger.warning("Could not clear mteb get_benchmark caches: %s", e)

    return LMEB


register_lmeb(clear_cache=True)