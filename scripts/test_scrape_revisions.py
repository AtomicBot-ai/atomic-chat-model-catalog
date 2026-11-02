"""Offline regressions for pinned discovery snapshots and legacy readers."""
import copy
import gzip
import json
from pathlib import Path
import unittest

from jsonschema import Draft7Validator
from scrape import build_catalog_model

ROOT = Path(__file__).resolve().parent.parent
A = "a" * 40
B = "b" * 40
HASH = "c" * 64


def repository(revision=A):
    return {
        "id": "owner/model", "sha": revision, "author": "owner", "downloads": 123,
        "tags": ["mlx", "gguf"], "library_name": "mlx",
        "siblings": [
            {"rfilename": "model-00001-of-00002.gguf", "size": 10},
            {"rfilename": "model-00002-of-00002.gguf", "size": 20},
            {"rfilename": "mmproj.gguf", "size": 5},
            {"rfilename": "model.safetensors", "lfs": {"size": 30, "sha256": HASH}},
        ],
    }


def refs(model):
    return model["quants"] + model["mmproj_models"] + model["safetensors_files"]


class CatalogRevisions(unittest.TestCase):
    def test_shards_projector_and_mlx_weights_use_the_metadata_commit(self):
        model = build_catalog_model(repository())
        self.assertEqual(model["revision"], A)
        self.assertEqual(len(refs(model)), 4)
        self.assertTrue(all(f"/resolve/{A}/" in file["path"] for file in refs(model)))
        self.assertEqual(model["safetensors_files"][0]["sha256"], HASH)
        self.assertEqual(model["readme"], f"https://huggingface.co/owner/model/resolve/{A}/README.md")

    def test_updating_main_does_not_mutate_the_previous_snapshot(self):
        before = build_catalog_model(repository(A))
        saved = copy.deepcopy(before)
        after = build_catalog_model(repository(B))
        self.assertEqual(before, saved)
        self.assertTrue(all(f"/resolve/{B}/" in file["path"] for file in refs(after)))
        self.assertEqual([f["model_id"] for f in refs(before)], [f["model_id"] for f in refs(after)])
        self.assertEqual([f["file_size"] for f in refs(before)], [f["file_size"] for f in refs(after)])

    def test_missing_or_malformed_sha_never_falls_back_to_main(self):
        for revision in [None, "main", "a" * 39, "A" * 40, "../main"]:
            with self.subTest(revision=revision):
                self.assertIsNone(build_catalog_model(repository(revision)))

    def test_pinned_output_is_additive_for_legacy_readers(self):
        model = build_catalog_model(repository())
        self.assertEqual(model["model_name"], "owner/model")
        self.assertEqual([f["model_id"] for f in model["quants"]], [
            "owner/model-00001-of-00002", "owner/model-00002-of-00002",
        ])
        document = {"manifest_version": 1, "schema_version": 1,
                    "updated_at": "2026-11-02T00:00:00Z", "orgs": ["owner"], "models": [model]}
        Draft7Validator(json.loads((ROOT / "config/schema.catalog.json").read_text())).validate(document)

    def test_existing_published_catalog_still_validates_without_migration(self):
        with gzip.open(ROOT / "dist/catalog.json.gz", "rt") as stream:
            document = json.load(stream)
        count = len(document["models"])
        Draft7Validator(json.loads((ROOT / "config/schema.catalog.json").read_text())).validate(document)
        self.assertEqual(len(document["models"]), count)
        self.assertGreater(count, 1000)


if __name__ == "__main__":
    unittest.main()
