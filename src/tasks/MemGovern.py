from src.abstasks.SubsetRetrieval import SubsetRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, ConfigDict, ValidationError


class MemGovern(SubsetRetrieval):
    model_config = ConfigDict(extra='allow')
    metadata = TaskMetadata(
        name="MemGovern",
        description=(
            ""
            ""
        ),
        reference="https://github.com/QuantaAlpha/MemGovern/blob/main/data",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            'Azure_azure-sdk-for-python': ['eng-Latn'], 
            'ClusterHQ_flocker': ['eng-Latn'], 
            'DataDog_integrations-core': ['eng-Latn'], 
            'Microsoft_TypeScript': ['eng-Latn'], 
            'PokemonGoF_PokemonGo-Bot': ['eng-Latn'], 
            'Qiskit_qiskit-terra': ['eng-Latn'], 
            'StackStorm_st2': ['eng-Latn'], 
            'apache_airflow': ['eng-Latn'], 
            'apache_incubator-airflow': ['eng-Latn'],
            'certbot_certbot': ['eng-Latn'], 
            'dask_dask': ['eng-Latn'], 
            'datalad_datalad': ['eng-Latn'],
            'django_django': ['eng-Latn'], 
            'dmwm_WMCore': ['eng-Latn'], 
            'encode_django-rest-framework': ['eng-Latn'], 
            'facebook_react': ['eng-Latn'], 
            'great-expectations_great_expectations': ['eng-Latn'], 
            'home-assistant_core': ['eng-Latn'], 
            'home-assistant_home-assistant': ['eng-Latn'], 
            'huggingface_transformers': ['eng-Latn'], 
            'kubernetes_kubernetes': ['eng-Latn'], 
            'mesonbuild_meson': ['eng-Latn'], 
            'microsoft_vscode': ['eng-Latn'], 
            'mne-tools_mne-python': ['eng-Latn'], 
            'moby_moby': ['eng-Latn'], 
            'napari_napari': ['eng-Latn'], 
            'numpy_numpy': ['eng-Latn'], 
            'optuna_optuna': ['eng-Latn'], 
            'pandas-dev_pandas': ['eng-Latn'], 
            'pydata_xarray': ['eng-Latn'], 
            'pypa_pip': ['eng-Latn'], 
            'pytest-dev_pytest': ['eng-Latn'], 
            'pytorch_pytorch': ['eng-Latn'], 
            'raiden-network_raiden': ['eng-Latn'], 
            'rust-lang_rust': ['eng-Latn'], 
            'saltstack_salt': ['eng-Latn'], 
            'scikit-learn_scikit-learn': ['eng-Latn'], 
            'scipy_scipy': ['eng-Latn'], 
            'scrapy_scrapy': ['eng-Latn'], 
            'spack_spack': ['eng-Latn'], 
            'sphinx-doc_sphinx': ['eng-Latn'], 
            'spotify_luigi': ['eng-Latn'], 
            'spring-projects_spring-framework': ['eng-Latn'], 
            'spyder-ide_spyder': ['eng-Latn'], 
            'sympy_sympy': ['eng-Latn'], 
            'tensorflow_tensorflow': ['eng-Latn'], 
            'webpack_webpack': ['eng-Latn'], 
            'xonsh_xonsh': ['eng-Latn'],
        },
        main_score="ndcg_at_10",
        date=("2026-03-11", "2026-03-11"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{wang2026memgovern,
                title={MemGovern: Enhancing Code Agents through Learning from Governed Human Experiences},
                author={Wang, Qihao and Cheng, Ziming and Zhang, Shuo and Liu, Fan and Xu, Rui and Lian, Heng and Wang, Kunyi and Yu, Xiaoming and Yin, Jianghao and Hu, Sen and others},
                journal={arXiv preprint arXiv:2601.06789},
                year={2026}
                }""",
        dataset={
            "path": "Procedural/MemGovern/",   
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
