"""
BaiZe — HuggingFace 访问地址配置
================================
huggingface_hub 与 datasets 都只在**被 import 的那一刻**读取一次环境变量 HF_ENDPOINT：
没有 export、在别的终端设置、或在 Python 里 import 之后才改 os.environ，都会导致请求仍然发往
huggingface.co。configure_hf_endpoint 在运行时把地址同时写进两个库的模块常量，不受 import 顺序影响。

另外，分页接口的"下一页"地址由服务器在 Link 头里返回；镜像站返回的仍是 huggingface.co，
所以还要把翻页地址的主机改写回当前 endpoint（列出 allenai/c4 multilingual/ 这类上万文件的目录时必然会翻页）。
"""

import os
from urllib.parse import urlsplit, urlunsplit

DEFAULT_ENDPOINT = "https://huggingface.co"


def _rewrite_host(url: str, endpoint: str) -> str:
    """把 url 的 scheme+host 换成 endpoint 的（路径与查询参数保持不变）。"""
    if not url:
        return url
    u, e = urlsplit(url), urlsplit(endpoint)
    if (u.scheme, u.netloc) == (e.scheme, e.netloc):
        return url
    return urlunsplit((e.scheme, e.netloc, u.path, u.query, u.fragment))


def _patch_pagination(endpoint: str):
    """分页接口（如列出数据集目录）的下一页地址来自服务器返回的 Link 头。
    镜像站转发 HF API 时，Link 头里仍是 https://huggingface.co/...，客户端照着翻页就绕回了官方站点。
    这里把下一页地址的主机改写成当前 endpoint。"""
    try:
        from huggingface_hub.utils import _pagination
    except ImportError:
        return
    original = getattr(_pagination, "_baize_original_get_next_page", None) or getattr(
        _pagination, "_get_next_page", None)
    if original is None:
        return
    _pagination._baize_original_get_next_page = original

    def _get_next_page(response):
        nxt = original(response)
        return _rewrite_host(nxt, endpoint) if nxt else nxt

    _pagination._get_next_page = _get_next_page


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
    _patch_pagination(ep)
    try:
        import datasets.config as ds_config
        ds_config.HF_ENDPOINT = ep
        ds_config.HUB_DATASETS_URL = ep + "/datasets/{repo_id}/resolve/{revision}/{path}"
    except ImportError:
        pass
    return ep
