import mteb

from .tasks.EPBench import EPBench
from .tasks.KnowMeBench import KnowMeBench

from .tasks.LongMemEval import LongMemEval
from .tasks.LoCoMo import LoCoMo
from .tasks.REALTALK import REALTALK
from .tasks.TMD import TMD
from .tasks.MemBench import MemBench
from .tasks.ConvoMem import ConvoMem

from .tasks.CovidQA import CovidQA
from .tasks.ESGReports import ESGReports
from .tasks.LMEB_SciFact import LMEB_SciFact
from .tasks.QASPER import QASPER
from .tasks.PeerQA import PeerQA
from .tasks.MLDR import MLDR
from .tasks.NovelQA import NovelQA
from .tasks.LooGLE import LooGLE

from .tasks.DeepPlanning import DeepPlanning
from .tasks.Gorilla import Gorilla
from .tasks.MemGovern import MemGovern
from .tasks.Proced_mem_bench import Proced_mem_bench
from .tasks.ToolBench import ToolBench
from .tasks.ReMe import ReMe

from mteb.get_tasks import _TASKS_REGISTRY

LOCAL_REGISTRY = {
    c.metadata.name: c for c in [
        EPBench,
        KnowMeBench,
        LoCoMo,
        LongMemEval,
        REALTALK,
        TMD,
        MemBench,
        ConvoMem,
        CovidQA,
        ESGReports,
        LMEB_SciFact,
        QASPER,
        PeerQA,
        MLDR,
        NovelQA,
        LooGLE,
        DeepPlanning,
        Gorilla,
        MemGovern,
        Proced_mem_bench,
        ToolBench,
        ReMe,
    ]
}

_builtin_tasks = set(_TASKS_REGISTRY.keys())

assert all(k not in _builtin_tasks for k in LOCAL_REGISTRY.keys()), \
    f"Task name conflict detected: {set(LOCAL_REGISTRY.keys()) & _builtin_tasks}"

# Register custom tasks
_TASKS_REGISTRY.update(LOCAL_REGISTRY)

