import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vault_scene_tagger import media, sidecar  # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")


class ExampleFixtureTest(unittest.TestCase):
    def test_sample_is_valid(self):
        doc = sidecar.read(os.path.join(EXAMPLES, "sample.scenes.json"))
        self.assertEqual(sidecar.validate(doc), [])

    def test_json_schema_file_parses(self):
        with open(os.path.join(EXAMPLES, "scenes.schema.json"), encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema["properties"]["schema"]["const"], sidecar.SCHEMA)
        self.assertEqual(set(schema["properties"]["markers"]["items"]["properties"]["source"]["enum"]),
                         set(sidecar.SOURCES))


class ValidateTest(unittest.TestCase):
    def base(self, **kw):
        doc = {"schema": "vault-scenes/1", "video": "a.mp4", "duration": 10.0,
               "generator": "x 1", "generated_at": "2026-09-14T10:00:00Z", "markers": []}
        doc.update(kw)
        return doc

    def test_rejects_bad_documents(self):
        cases = [
            self.base(schema="vault-scenes/2"),
            self.base(duration="10"),
            self.base(generated_at="yesterday"),
            self.base(markers=[{"t": 5, "label": "scene", "source": "cut"},
                               {"t": 1, "label": "scene", "source": "cut"}]),
            self.base(markers=[{"t": 5, "end": 5, "label": "sex", "source": "ml"}]),
            self.base(markers=[{"t": 1, "label": "Doggy Style", "source": "ml"}]),
            self.base(markers=[{"t": 1, "label": "sex", "source": "robot"}]),
            self.base(markers=[{"t": 1, "label": "sex", "confidence": 1.5, "source": "ml"}]),
            self.base(markers=[{"t": True, "label": "sex", "source": "ml"}]),
        ]
        for doc in cases:
            self.assertTrue(sidecar.validate(doc), doc)

    def test_accepts_extras_and_overlaps(self):
        doc = self.base(extra={"x": 1}, markers=[
            {"t": 1, "end": 9, "label": "sex", "confidence": 0.5, "source": "ml"},
            {"t": 1, "end": 5, "label": "doggy", "confidence": 0.5, "source": "ml"},
            {"t": 3, "label": "69", "source": "manual"},
        ])
        self.assertEqual(sidecar.validate(doc), [])


class MergeTest(unittest.TestCase):
    def test_keeps_manual_and_foreign_replaces_owned(self):
        existing = {
            "schema": "vault-scenes/1", "video": "a.mp4", "duration": 10, "generator": "vault 2",
            "generated_at": "2026-01-01T00:00:00Z", "reader_note": "keep me",
            "markers": [
                {"t": 1.0, "label": "scene", "source": "cut"},
                {"t": 4.0, "label": "favourite", "source": "manual"},
                {"t": 6.0, "end": 8.0, "label": "sex", "confidence": 0.6, "source": "ml"},
            ],
        }
        new = [{"t": 2.0, "label": "scene", "source": "cut"},
               {"t": 7.0, "label": "scene", "source": "cut"}]
        doc = sidecar.merge(existing, new, {"cut"}, "a.mp4", 10.0, ["a"], {"a": {}})
        self.assertEqual(sidecar.validate(doc), [])
        self.assertEqual([(m["t"], m["source"]) for m in doc["markers"]],
                         [(2.0, "cut"), (4.0, "manual"), (6.0, "ml"), (7.0, "cut")])
        self.assertEqual(doc["reader_note"], "keep me")
        self.assertEqual(doc["tagger"]["phases"], ["a"])
        self.assertEqual(list(doc)[:5], list(sidecar.REQUIRED_KEYS))
        self.assertEqual(list(doc)[-1], "markers")

    def test_manual_never_owned(self):
        existing = {"markers": [{"t": 1.0, "label": "x", "source": "manual"}]}
        doc = sidecar.merge(existing, [], {"cut", "manual"}, "a.mp4", 5, ["a"], {})
        self.assertEqual(len(doc["markers"]), 1)

    def test_phases_accumulate_and_generator_fallback(self):
        prev = {"generator": "vault-scene-tagger 0.0.9", "markers": []}
        self.assertEqual(sidecar.phases_done(prev), {"a"})
        self.assertEqual(sidecar.phases_done({"generator": "vault 1"}), set())
        prev["tagger"] = {"phases": ["a", "c"]}
        doc = sidecar.merge(prev, [], {"cut"}, "a.mp4", 5, ["a"], {})
        self.assertEqual(doc["tagger"]["phases"], ["a", "c"])


class WriteTest(unittest.TestCase):
    def test_atomic_utf8_no_bom_no_leftovers(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "Ünïcode movie.scenes.json")
            doc = sidecar.merge(None, [], set(), "Ünïcode movie.mp4", 1.0, ["a"], {})
            sidecar.write_atomic(path, doc)
            with open(path, "rb") as f:
                raw = f.read()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            self.assertIn("Ünïcode".encode("utf-8"), raw)
            self.assertEqual(os.listdir(d), [os.path.basename(path)])

    def test_read_errors(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(sidecar.read(os.path.join(d, "missing.scenes.json")))
            bad = os.path.join(d, "bad.scenes.json")
            with open(bad, "w") as f:
                f.write("{nope")
            with self.assertRaises(sidecar.SidecarError):
                sidecar.read(bad)


class CutCleaningTest(unittest.TestCase):
    def test_clean_cuts(self):
        raw = [5.0, 0.0, 5.4, 12.0, 30.0, 11.5, 2.0]
        self.assertEqual(media.clean_cuts(raw, min_gap=1.0, duration=30.0), [2.0, 5.0, 11.5])

    def test_sprite_plan_and_vtt(self):
        plan = media.plan_sprites(25.0, 10)
        self.assertEqual((plan["count"], plan["cols"], plan["rows"]), (3, 3, 1))
        vtt = media.sprites_vtt(plan, "a b.sprites.jpg", 25.0)
        self.assertTrue(vtt.startswith("WEBVTT\n"))
        self.assertIn("00:00:20.000 --> 00:00:25.000\na%20b.sprites.jpg#xywh=320,0,160,90", vtt)
        long_plan = media.plan_sprites(20 * 3600.0, 10)
        self.assertLessEqual(long_plan["count"], media.MAX_SPRITE_TILES)


if __name__ == "__main__":
    unittest.main()
