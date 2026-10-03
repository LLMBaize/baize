"""prepare_allenai.py 与数据量控制的测试（不联网：用模拟流 / 本地 json 代替 HF 数据集）。

运行：python tests/test_allenai_data.py   （也兼容 pytest）
"""

import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prepare_allenai as P
from baize.data import iter_documents


def fake_stream(data):
    """按 src['path'] 返回模拟样本流。"""
    def stream_fn(src, seed, shuffle_buffer, skip):
        return iter(data[src["path"]][skip:])
    return stream_fn


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(argv, data):
    args = P.build_parser().parse_args(argv)
    return P.prepare(args, stream_fn=fake_stream(data))


def test_parse_sources():
    srcs = P.parse_sources("c4-zh:0.7,c4-en:0.3", P.PRETRAIN_PRESETS)
    assert [s["key"] for s in srcs] == ["c4-zh", "c4-en"]
    assert [s["weight"] for s in srcs] == [0.7, 0.3]
    assert srcs[0]["data_files"] == "multilingual/c4-zh.*.json.gz"
    custom = P.parse_sources("allenai/foo#bar:2,allenai/c4@multilingual/c4-ja.*.json.gz", {})
    assert custom[0] == dict(path="allenai/foo", name="bar", key="allenai/foo#bar", weight=2.0)
    assert custom[1]["data_files"] == "multilingual/c4-ja.*.json.gz"
    assert custom[1]["weight"] == 1.0


def test_pretrain_mix_quota_and_filters():
    data = {
        "allenai/c4": [{"text": f"中文文档 {i} " + "白泽" * 30} for i in range(1000)],
        "allenai/olmo-mix-1124": [{"text": f"english doc {i} " + "x" * 60} for i in range(1000)]
                                 + [{"text": "too short"}],
    }
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pre.jsonl")
        m = run(["--task", "pretrain", "--sources", "c4-zh:3,olmo-mix-wiki:1",
                 "--max_docs", "400", "--val_docs", "10", "--out", out], data)
        rows = read_jsonl(out)
        assert len(rows) == 400 and m["total_docs"] == 400
        by_src = {s["source"]: s["docs"] for s in m["sources"]}
        assert by_src == {"c4-zh": 300, "olmo-mix-wiki": 100}  # 按 3:1 分配
        assert len(read_jsonl(out.replace(".jsonl", ".val.jsonl"))) == 10
        assert os.path.exists(out.replace(".jsonl", ".manifest.json"))
        # 两个源交错写出，而不是先写完一个再写另一个
        first = [r["source"] for r in rows[:50]]
        assert len(set(first)) == 2


def test_pretrain_char_budget_and_cjk_filter_and_dedup():
    data = {"allenai/c4": [{"text": "中文" * 50}] * 5           # 重复
                          + [{"text": "english only " * 10}] * 5  # 被 cjk 过滤
                          + [{"text": f"第{i}篇" + "中文" * 50} for i in range(100)]}
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pre.jsonl")
        m = run(["--task", "pretrain", "--sources", "c4-zh", "--max_chars", "1000",
                 "--min_cjk_ratio", "0.5", "--out", out], data)
        s = m["sources"][0]
        assert s["duplicates"] == 4 and s["filtered"] == 5
        assert 1000 <= s["chars"] < 1000 + 110  # 达到字符配额即停（最多超一篇）


def test_token_budget_requires_tokenizer_and_counts():
    data = {"allenai/c4": [{"text": f"白泽是神兽 {i} " * 10} for i in range(500)]}
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pre.jsonl")
        try:
            run(["--task", "pretrain", "--sources", "c4-zh", "--max_tokens", "100", "--out", out], data)
            raise AssertionError("应当要求 --tokenizer")
        except ValueError:
            pass
        m = run(["--task", "pretrain", "--sources", "c4-zh", "--max_tokens", "2000",
                 "--tokenizer", os.path.join(ROOT, "tokenizer"), "--out", out], data)
        assert 2000 <= m["total_tokens"] < 2400


def test_sft_normalization_and_filters():
    good = {"messages": [{"role": "system", "content": "你是助手"},
                         {"role": "user", "content": "你好"},
                         {"role": "assistant", "content": "你好！"},
                         {"role": "user", "content": "末尾的 user 轮会被去掉"}],
            "source": "ai2-adapt-dev/flan"}
    other_src = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                 "source": "ai2-adapt-dev/code"}
    tool = {"messages": [{"role": "user", "content": "q"}, {"role": "tool", "content": "x"},
                         {"role": "assistant", "content": "a"}], "source": "ai2-adapt-dev/flan"}
    data = {"allenai/tulu-3-sft-mixture": [good, other_src, tool] * 3}
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "sft.jsonl")
        m = run(["--task", "sft", "--sources", "tulu3", "--max_docs", "100",
                 "--source_filter", "flan", "--out", out], data)
        rows = read_jsonl(out)
        assert len(rows) == 1  # 去重后仅剩 1 条合法的 flan 对话
        assert [x["role"] for x in rows[0]["messages"]] == ["system", "user", "assistant"]
        assert m["sources"][0]["filtered"] == 6


def test_wildchat_language_and_toxic_filter():
    conv = lambda c: [{"role": "user", "content": c, "turn_identifier": 1},
                      {"role": "assistant", "content": "回答" + c, "turn_identifier": 1}]
    data = {"allenai/WildChat-1M": [
        {"conversation": conv("你好"), "language": "Chinese", "toxic": False},
        {"conversation": conv("hello"), "language": "English", "toxic": False},
        {"conversation": conv("坏话"), "language": "Chinese", "toxic": True},
    ]}
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "sft.jsonl")
        run(["--task", "sft", "--sources", "wildchat", "--max_docs", "10",
             "--language", "Chinese", "--out", out], data)
        rows = read_jsonl(out)
        assert len(rows) == 1 and rows[0]["messages"][0] == {"role": "user", "content": "你好"}


def test_dpo_pairs_normalized_and_filtered():
    conv = lambda a: [{"role": "user", "content": "问题"}, {"role": "assistant", "content": a}]
    data = {"allenai/llama-3.1-tulu-3-8b-preference-mixture": [
        {"chosen": conv("好回答"), "rejected": conv("差回答"), "source": "x"},
        {"chosen": conv("相同"), "rejected": conv("相同")},                       # 两边相同 → 丢弃
        {"chosen": conv("a"), "rejected": [{"role": "user", "content": "别的问题"},
                                           {"role": "assistant", "content": "b"}]},  # prompt 不同 → 丢弃
        {"prompt": "字符串问题", "chosen": "好", "rejected": "差"},               # 字符串格式
    ]}
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "dpo.jsonl")
        m = run(["--task", "dpo", "--sources", "tulu3-pref", "--max_docs", "10", "--out", out], data)
        rows = read_jsonl(out)
        assert len(rows) == 2 and m["sources"][0]["filtered"] == 2
        assert rows[0]["chosen"][-1]["content"] == "好回答" and rows[0]["rejected"][-1]["content"] == "差回答"
        assert rows[1]["chosen"][0] == {"role": "user", "content": "字符串问题"}


def test_hf_endpoint_applies_after_import():
    import datasets.config as dc
    import huggingface_hub.constants as hc
    from baize.hub import configure_hf_endpoint
    old = os.environ.get("HF_ENDPOINT")
    try:
        assert configure_hf_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"
        assert hc.ENDPOINT == "https://hf-mirror.com" and dc.HF_ENDPOINT == "https://hf-mirror.com"
        assert dc.HUB_DATASETS_URL.startswith("https://hf-mirror.com/datasets/")
        assert hc.HUGGINGFACE_CO_URL_TEMPLATE.startswith("https://hf-mirror.com/")
    finally:
        if old is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old
        configure_hf_endpoint(old or "https://huggingface.co")


def test_mirror_pagination_next_link_rewritten():
    """镜像站返回的下一页 Link 指向 huggingface.co，必须改写回镜像地址。"""
    from huggingface_hub.utils import _pagination
    from baize.hub import configure_hf_endpoint

    class FakeResponse:
        links = {"next": {"url": "https://huggingface.co/api/datasets/allenai/c4/tree/abc/multilingual?cursor=XYZ&limit=1000"}}
    try:
        configure_hf_endpoint("https://hf-mirror.com")
        assert _pagination._get_next_page(FakeResponse()) == \
            "https://hf-mirror.com/api/datasets/allenai/c4/tree/abc/multilingual?cursor=XYZ&limit=1000"
        configure_hf_endpoint("https://hf-mirror.com")  # 重复调用不会层层包装
        assert _pagination._get_next_page(FakeResponse()).startswith("https://hf-mirror.com/api/")

        class NoNext:
            links = {}
        assert _pagination._get_next_page(NoNext()) is None
    finally:
        configure_hf_endpoint("https://huggingface.co")
    assert _pagination._get_next_page(FakeResponse()).startswith("https://huggingface.co/api/")


def test_source_error_keeps_other_sources():
    def stream_fn(src, seed, shuffle_buffer, skip):
        if src["path"] == "allenai/olmo-mix-1124":
            raise ValueError("BuilderConfig 'wiki' not found")
        return iter([{"text": "中文" * 40 + str(i)} for i in range(100)])
    args = P.build_parser().parse_args(["--sources", "c4-zh,olmo-mix-wiki", "--max_docs", "20",
                                        "--out", os.path.join(tempfile.mkdtemp(), "o.jsonl")])
    m = P.prepare(args, stream_fn=stream_fn)
    st = {s["source"]: s for s in m["sources"]}
    assert "not found" in st["olmo-mix-wiki"]["error"]
    assert st["c4-zh"]["docs"] == 10  # 出错的源不影响其它源按自己的配额继续


def test_open_stream_with_real_datasets_streaming():
    """用 datasets 的本地 json 构建器走一遍真实的 streaming + shuffle + skip 代码路径。"""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "part-0.json")
        with open(fp, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"text": f"doc {i}"}) + "\n")
        src = dict(path="json", data_files=fp)
        rows = list(P.open_stream(src, seed=0, shuffle_buffer=0, skip=5))
        assert [r["text"] for r in rows[:2]] == ["doc 5", "doc 6"] and len(rows) == 45
        shuffled = [r["text"] for r in P.open_stream(src, seed=0, shuffle_buffer=20, skip=0)]
        assert sorted(shuffled) == sorted(f"doc {i}" for i in range(50)) and shuffled[:5] != [f"doc {i}" for i in range(5)]


def test_iter_documents_and_pretrain_caps():
    from pretrain import PretrainDataset
    from baize import BaiZeTokenizer
    tok = BaiZeTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "a.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for i in range(30):
                f.write(json.dumps({"text": f"第一行{i}\n第二行"}, ensure_ascii=False) + "\n")
        docs = list(iter_documents([fp]))
        assert len(docs) == 30 and "\n" in docs[0]  # jsonl 文档可含换行
        assert len(list(iter_documents([fp], max_docs=7))) == 7
        assert PretrainDataset([fp], tok, 16, max_tokens=100).data.__len__() == 100
        ds = PretrainDataset([fp], tok, 16, max_docs=3)
        assert ds.data.count(tok.eos_token_id) == 3


def test_sft_max_samples():
    from sft import SFTDataset
    from baize import BaiZeTokenizer
    tok = BaiZeTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))
    ds = SFTDataset([os.path.join(ROOT, "data", "sft.jsonl")], tok, 256, max_samples=17)
    assert len(ds) == 17


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
