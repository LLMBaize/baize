"""
BaiZe — HuggingFace 访问地址配置
================================
huggingface_hub 与 datasets 都只在**被 import 的那一刻**读取一次环境变量 HF_ENDPOINT：
没有 export、在别的终端设置、或在 Python 里 import 之后才改 os.environ，都会导致请求仍然发往
huggingface.co。configure_hf_endpoint 在运行时把地址同时写进两个库的模块常量，不受 import 顺序影响。
"""

import os

DEFAULT_ENDPOINT = "https://huggingface.co"


def configure_hf_endpoint(endpoint: str = None) -> str:
    """设置并返回实际生效的 HF 访问地址。

    endpoint 为空时沿用环境变量 HF_ENDPOINT（未设置则为 huggingface.co）。
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    ep = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

    try:
        import huggingface_hub.constants as hub_constants
        hub_constants.ENDPOINT = ep
        hub_constants.HUGGINGFACE_CO_URL_TEMPLATE = ep + "/{repo_id}/resolve/{revision}/{filename}"
    except ImportError:
        pass
    try:
        import datasets.config as ds_config
        ds_config.HF_ENDPOINT = ep
        ds_config.HUB_DATASETS_URL = ep + "/datasets/{repo_id}/resolve/{revision}/{path}"
    except ImportError:
        pass
    return ep
