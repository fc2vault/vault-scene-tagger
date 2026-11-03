import json
import os
import py_compile
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vault_scene_tagger import export, sidecar  # noqa: E402


def marker(t, label, source="manual", **extra):
    m = {"t": t, "label": label, "source": source}
    m.update(extra)
    return m


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        # reviewed movie: manual markers, one of them a correction, plus a rejection
        self.write("Actress Ä/CODE-1/CODE-1.mp4", {
            "duration": 600.0,
            "markers": [marker(30, "doggy"),
                        marker(120, "cumshot", was={"t": 118.0, "label": "anal", "source": "ml",
                                                    "confidence": 0.9})],
            "rejected": [{"t": 300.0, "label": "anal", "source": "ml", "confidence": 0.94,
                          "rejected_at": "2026-09-16T10:00:00Z"}]})
        # half-reviewed: one manual + one AI suggestion left
        self.write("Actress Ä/CODE-2/CODE-2.mp4", {
            "duration": 900.0,
            "markers": [marker(60, "blowjob"),
                        marker(400, "doggy", source="ml", confidence=0.8, end=430.0, engine="phar")]})
        # untouched: AI only -> must not be exported
        self.write("Actress Ä/CODE-3/CODE-3.mp4", {
            "duration": 300.0,
            "markers": [marker(10, "fingering", source="ml", confidence=0.7, end=40.0)]})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.out, ignore_errors=True)

    def write(self, rel, doc):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()                      # the "video"
        full = sidecar.merge(None, [], set(), os.path.basename(rel), doc["duration"], ["c"], {})
        full["markers"] = sidecar.sort_markers(doc["markers"])
        if doc.get("rejected"):
            full["rejected"] = doc["rejected"]
        sidecar.write_atomic(os.path.splitext(path)[0] + ".scenes.json", full)

    def test_package_contents_and_counts(self):
        self.assertEqual(export.main([self.root, "--out", self.out]), 0)
        for name in ("dataset.json", "labels.txt", "make_clips.py", "README.md"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        data = json.load(open(os.path.join(self.out, "dataset.json"), encoding="utf-8"))
        c = data["counts"]
        self.assertEqual((c["videos"], c["fully_reviewed"]), (2, 1))       # CODE-3 excluded
        self.assertEqual((c["manual_markers"], c["corrections"], c["rejections"]), (3, 1, 1))
        self.assertEqual(c["unreviewed_ai_markers"], 1)
        self.assertEqual(c["per_label"], {"doggy": 1, "blowjob": 1, "cumshot": 1})
        vids = sorted(v["video"] for v in data["videos"])
        self.assertEqual(vids, ["Actress Ä/CODE-1/CODE-1.mp4", "Actress Ä/CODE-2/CODE-2.mp4"])
        # the correction pair and the engine field survive the export
        code1 = next(v for v in data["videos"] if "CODE-1" in v["video"])
        self.assertEqual(code1["markers"][1]["was"]["label"], "anal")
        self.assertEqual(code1["rejected"][0]["label"], "anal")
        code2 = next(v for v in data["videos"] if "CODE-2" in v["video"])
        self.assertEqual([m.get("engine") for m in code2["markers"]], [None, "phar"])
        # sidecar copies are verbatim
        copied = os.path.join(self.out, "sidecars", "Actress Ä/CODE-1/CODE-1.scenes.json")
        self.assertTrue(os.path.isfile(copied))
        self.assertEqual(json.load(open(copied, encoding="utf-8"))["duration"], 600.0)
        self.assertEqual(open(os.path.join(self.out, "labels.txt"), encoding="utf-8").read().split(),
                         list(c["per_label"]))

    def test_min_manual_filter(self):
        export.main([self.root, "--out", self.out, "--min-manual", "2"])
        data = json.load(open(os.path.join(self.out, "dataset.json"), encoding="utf-8"))
        self.assertEqual([v["video"] for v in data["videos"]], ["Actress Ä/CODE-1/CODE-1.mp4"])

    def test_make_clips_script_is_valid_and_plans_spans(self):
        export.main([self.root, "--out", self.out])
        script = os.path.join(self.out, "make_clips.py")
        py_compile.compile(script, doraise=True)
        ns = {}
        exec(compile(open(script, encoding="utf-8").read(), script, "exec"), ns)
        video = {"duration": 600.0, "markers": [marker(30, "doggy"), marker(120, "cumshot")]}
        spans = ns["spans"](video, window=20.0, min_background=30.0)
        self.assertIn(("doggy", 30.0, 50.0), spans)
        self.assertIn(("cumshot", 120.0, 140.0), spans)
        self.assertTrue(any(s[0] == "background" for s in spans))
        # a marker right after another is truncated at the next one
        video2 = {"duration": 600.0, "markers": [marker(30, "doggy"), marker(40, "cowgirl")]}
        self.assertIn(("doggy", 30.0, 40.0), ns["spans"](video2, 20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
