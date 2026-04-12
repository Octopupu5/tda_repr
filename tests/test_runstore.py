import json
import os
import tempfile

from tda_repr.training.results import RunStore


def test_runstore_writes_meta_and_appends_jsonl() -> None:
	with tempfile.TemporaryDirectory() as td:
		run_dir = os.path.join(td, "run")
		store = RunStore(run_dir, unique=False)
		store.write_meta({"name": "test", "x": 1})
		store.log("evt1", {"a": 1})
		store.log("evt2", {"b": 2})

		meta_path = os.path.join(store.run_dir, "meta.json")
		metrics_path = os.path.join(store.run_dir, "metrics.jsonl")
		assert os.path.isfile(meta_path)
		assert os.path.isfile(metrics_path)

		with open(meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
		assert meta["name"] == "test"
		assert meta["x"] == 1
		assert "created_at" in meta

		with open(metrics_path, "r", encoding="utf-8") as f:
			lines = [ln.strip() for ln in f.readlines() if ln.strip()]
		assert len(lines) == 2
		r1 = json.loads(lines[0])
		r2 = json.loads(lines[1])
		assert r1["event"] == "evt1"
		assert r1["a"] == 1
		assert r2["event"] == "evt2"
		assert r2["b"] == 2

