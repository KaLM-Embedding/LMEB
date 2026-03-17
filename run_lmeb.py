import json
import logging
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import HfArgumentParser, AutoModel
import mteb
from mteb.models.model_meta import ModelMeta
from mteb.models.get_model_meta import _model_meta_from_sentence_transformers
from mteb.cache import ResultCache
from src.embedding_models_wrapper import STWrapper
from mteb.models.sentence_transformer_wrapper import SentenceTransformerMultimodalEncoderWrapper
import lmeb_benchmark
from mteb.benchmarks import get_benchmark

logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s#%(lineno)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger('run_mteb.py')

@dataclass
class EvalArguments:
    """
    Arguments.
    """
    model_path: Optional[str] = field(
        default='KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    mteb_model: Optional[bool] = field(
        default=False, metadata={"help": "If `True`, use mteb native models."}
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(default="lmeb_results", metadata={"help": "output dir of results"})
    tasks: Optional[str] = field(default="LoCoMo", metadata={"help": "',' seprated"})
    benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name; LMEB"})
    langs: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    only_load: bool = field(default=False, metadata={"help": ""})
    load_model: bool = field(default=False, metadata={"help": "when only_load"})
    batch_size: int = field(default=512, metadata={"help": "Will be set to `encode_kwargs`"})
    precision: str = field(default='fp16', metadata={"help": "amp_fp16,amp_bf16,fp16,bf16,fp32"})
    

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = [s for s in self.tasks.split(',') if s]
        # if isinstance(self.langs, str):
        #     self.langs = [s for s in self.langs.split(',') if s]
        for name in ('model', 'encode', 'run'):
            name = name + '_kwargs'
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))

def get_model(model_path: str, precision: str = 'fp16', **kwargs):
    model = STWrapper(model_path, precision=precision, **kwargs)
    return model

def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks

def run_eval(model, tasks: list, args: EvalArguments, **kwargs):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, 'start'):
        model.start()
        _started = True
    
    mteb_result_cache = ResultCache(cache_path=".")
    
    for t in tasks:
        evaluation = mteb.MTEB(tasks=[t])
        results = evaluation.run(
            model,
            output_folder=args.output_dir,
            encode_kwargs=encode_kwargs,
            **kwargs
        )

    if model is not None and _started and hasattr(model, 'stop'):
        model.stop()
    return results


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.warning(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.warning(f"Args {args}")
    del parser


    if not args.benchmark:
        tasks = mteb.get_tasks(tasks=args.tasks)
    else:
        tasks = get_tasks(names=args.tasks, languages=args.langs, benchmark=args.benchmark)
    logger.warning(f"Selected {len(tasks)} tasks:\n" + '\n'.join(str(t) for t in tasks))

    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            t.load_data()
        if not args.load_model:
            return

    if args.precision == 'fp16':
        args.model_kwargs.update({"torch_dtype": torch.float16})
    elif args.precision == 'bf16':
        args.model_kwargs.update({"torch_dtype": torch.bfloat16})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = 'cuda' if torch.cuda.device_count() >= 1 else 'cpu'
    model = get_model(args.model_path, precision=args.precision, **args.model_kwargs)
    
    if model.mteb_model_meta.name is None:
        if "use_instruction" in args.model_kwargs and not args.model_kwargs['use_instruction']:
            model.mteb_model_meta.revision = "wo_inst"
    if args.only_load:
        return

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(model, tasks, args, **args.run_kwargs)
    logger.warning(f"Done {len(tasks)} tasks.")
    return

if __name__ == "__main__":
    main()
