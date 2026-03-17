import mteb
import json
import os
import sys
from src.embedding_models_wrapper import STWrapper
import lmeb_benchmark
from mteb.benchmarks import get_benchmark

path = sys.argv[1]
results_list = os.listdir(path)
benchmark = "LMEB"
if len(sys.argv) > 2:
    benchmark = sys.argv[2]
metric = "main_score"
if len(sys.argv) > 3:
    metric = sys.argv[3]
results = {}

def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)

    return tasks

tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
names = [t.metadata.name for t in tasks]
tasks = {name: task for name, task in zip(names, tasks)}

print('names', names)
split_tasks = {}
for task in results_list:
    if task.split(".json")[0] not in names:
        continue
    name = task.split(".json")[0]
    meta = tasks[name].metadata 
    with open(os.path.join(path, task)) as f:
        result = json.load(f)
    task_type = tasks[name].MemType
    eval_split = list(result['scores'].keys())[0]

    score = sum([ele[metric] for ele in result['scores'][eval_split]]) / len(result['scores'][eval_split])
    results[name] = score
    if task_type not in split_tasks:
        split_tasks[task_type] = []
    split_tasks[task_type].append(score)

final_scores = sum(results.values()) / len(results)
missed_tasks = [name for name in names if name not in results]
print('metric', metric)
print('missed tasks', missed_tasks)
print('Mean (Dataset)', len(results), round(final_scores*100, 2))
scores = []
for task_type in split_tasks:
    print(task_type, len(split_tasks[task_type]), round(sum(split_tasks[task_type]) / len(split_tasks[task_type]) * 100, 2))
    score = sum(split_tasks[task_type]) / len(split_tasks[task_type])
    scores.append(score)
print('Mean (Type)', round(sum(scores) / len(scores)*100, 2))
for name in results:
    print(name, round(results[name]*100, 2))