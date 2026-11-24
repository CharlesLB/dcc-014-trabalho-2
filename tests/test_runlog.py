import json

import numpy as np

from mst import runlog


def test_dumps_numpy_and_sets():
    s = runlog.dumps({"a": np.int64(3), "b": np.float32(0.5), "c": np.array([1, 2]),
                      "d": frozenset({2, 1})})
    assert json.loads(s) == {"a": 3, "b": 0.5, "c": [1, 2], "d": [1, 2]}


def test_append_and_read(tmp_path):
    p = tmp_path / "r" / "runs.jsonl"
    runlog.append(p, {"x": 1})
    runlog.append_many(p, [{"x": 2}, {"x": 3}])
    assert [r["x"] for r in runlog.read(p)] == [1, 2, 3]


def test_read_missing_file_empty(tmp_path):
    assert runlog.read(tmp_path / "nada.jsonl") == []


def test_read_skips_blank_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert len(runlog.read(p)) == 2


def test_run_key_stable_and_sensitive():
    k1 = runlog.run_key("MST", "wpbc", "DA", 0, 1, {"a": 1, "b": 2})
    k2 = runlog.run_key("MST", "wpbc", "DA", 0, 1, {"b": 2, "a": 1})
    k3 = runlog.run_key("MST", "wpbc", "DA", 0, 1, {"a": 1, "b": 3})
    assert k1 == k2 != k3
    assert k1.startswith("MST|wpbc|DA|0|1|")


def test_done_keys(tmp_path):
    p = tmp_path / "r.jsonl"
    runlog.append(p, {"run_key": "a"})
    runlog.append(p, {"sem": 1})
    assert runlog.done_keys(p) == {"a"}


def test_new_run_id_unique():
    assert len({runlog.new_run_id() for _ in range(100)}) == 100


def test_environment_fields():
    env = runlog.environment()
    assert {"python", "numpy", "sklearn", "cpu", "git_commit"} <= set(env)


def test_validate_record():
    assert "run_id" in runlog.validate_record({})
    full = {k: None for k in runlog.REQUIRED_FIELDS}
    assert runlog.validate_record(full) == []


def test_write_trace(tmp_path):
    p = tmp_path / "t" / "x.jsonl"
    runlog.write_trace(p, [{"reinicio": 1, "passos": [{"it": 1}, {"it": 2}]},
                           {"reinicio": 2, "passos": [{"it": 1}]}])
    rows = runlog.read(p)
    assert [(r["reinicio"], r["it"]) for r in rows] == [(1, 1), (1, 2), (2, 1)]
