"""End-to-end: synthetic videos -> CLI -> sidecars. Skipped if ffmpeg isn't installed.

Phase c talks to a fake AI server on 127.0.0.1 that mimics skier233's /v3/process_video/.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vault_scene_tagger import cli, phar, sidecar  # noqa: E402
from vault_scene_tagger.walker import iter_videos  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_video(path):
    """7s test pattern, 6s solid blue, 12s colour bars -> cuts at 7.0 and 13.0."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=7",
           "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=6",
           "-f", "lavfi", "-i", "smptebars=s=320x180:r=25:d=12",
           "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
           "-c:v", "mpeg4", "-q:v", "5", path]
    subprocess.run(cmd, check=True)


class FakeAIHandler(BaseHTTPRequestHandler):
    requests = []
    response = {
        "schema_version": 3, "duration": 25.0, "frame_interval": 2.0, "models": [],
        "timespans": {
            "actions": {"Blowjob": [{"start": 6.0, "end": 10.0}],
                        "Creampie": [{"start": 20.0}]},
            "body": {"Penis": [{"start": 0.0, "end": 24.0}]},
        },
    }

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v3/process_video/":   # an older server: no v4 API -> client falls back
            return self._send({"detail": "Not Found"}, 404)
        FakeAIHandler.requests.append((self.path, body))
        self._send(self.response)

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class WalkerTest(unittest.TestCase):
    def test_reserved_dirs_dotfiles_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            files = [
                "Actress Ä/CODE-1/CODE-1.MP4",
                "Actress Ä/CODE-2/CODE-2 part1.mkv",
                "Actress Ä/CODE-2/CODE-2 part2.mkv",
                "Actress Ä/CODE-2/._CODE-2 part1.mkv",
                "Actress Ä/CODE-2/.hidden.mp4",
                "Actress Ä/CODE-2/CODE-2.nfo",
                "Actress Ä/CODE-2/CODE-2.scenes.json",
                "_vault_trash/old.mp4", "_Duplicates/d.mp4", "_unsorted/u.mp4",
                "_vault/v.mp4", "@eaDir/e.mp4", ".Trashes/t.mp4",
                "ok/@eaDir/nested.mp4",
            ]
            for rel in files:
                p = os.path.join(d, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").close()
            found = [os.path.relpath(p, d) for p in iter_videos(d)]
            self.assertEqual(found, ["Actress Ä/CODE-1/CODE-1.MP4",
                                     "Actress Ä/CODE-2/CODE-2 part1.mkv",
                                     "Actress Ä/CODE-2/CODE-2 part2.mkv"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_dir = tempfile.mkdtemp()
        cls.src = os.path.join(cls.src_dir, "src.mp4")
        make_video(cls.src)
        cls.ai = ThreadingHTTPServer(("127.0.0.1", 0), FakeAIHandler)
        cls.ai_url = "http://127.0.0.1:%d" % cls.ai.server_address[1]
        threading.Thread(target=cls.ai.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.ai.shutdown()
        cls.ai.server_close()
        shutil.rmtree(cls.src_dir, ignore_errors=True)

    def setUp(self):
        FakeAIHandler.requests = []
        self.lib = tempfile.mkdtemp()
        self.part1 = self.place("女優 Ä/FC2-PPV-1/FC2-PPV-1 part1.mp4")
        self.part2 = self.place("女優 Ä/FC2-PPV-1/FC2-PPV-1 part2.mp4")
        self.trash = self.place("_vault_trash/FC2-PPV-9/FC2-PPV-9.mp4")
        open(os.path.join(self.lib, "女優 Ä/FC2-PPV-1/FC2-PPV-1.nfo"), "w").close()
        self.summary_path = self.lib + "-summary.json"

    def tearDown(self):
        shutil.rmtree(self.lib, ignore_errors=True)
        if os.path.exists(self.summary_path):
            os.unlink(self.summary_path)

    def place(self, rel):
        p = os.path.join(self.lib, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(self.src, p)
        return p

    def run_cli(self, *extra):
        extra = list(extra)
        if "--phase" not in extra:
            extra = ["--phase", "a"] + extra
        code = cli.main([self.lib, "-q", "--summary-json", self.summary_path] + extra)
        with open(self.summary_path, encoding="utf-8") as f:
            return code, json.load(f)

    def run_c(self, *extra):
        return self.run_cli("--phase", "c", "--ai-url", self.ai_url, "--min-moment", "4", *extra)

    def sc(self, video):
        return os.path.splitext(video)[0] + ".scenes.json"

    def cuts(self, video):
        return [m["t"] for m in sidecar.read(self.sc(video))["markers"] if m["source"] == "cut"]

    # ---- phase a

    def test_generates_valid_sidecars_per_part(self):
        code, summary = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual((summary["videos_found"], summary["processed"], summary["errors"]), (2, 2, 0))
        for video in (self.part1, self.part2):
            doc = sidecar.read(self.sc(video))
            self.assertEqual(sidecar.validate(doc), [])
            self.assertEqual(doc["video"], os.path.basename(video))
            self.assertAlmostEqual(doc["duration"], 25.0, delta=0.2)
            cuts = self.cuts(video)
            self.assertEqual(len(cuts), 2, cuts)
            self.assertAlmostEqual(cuts[0], 7.0, delta=0.1)
            self.assertAlmostEqual(cuts[1], 13.0, delta=0.1)
            for m in doc["markers"]:
                self.assertEqual((m["label"], m["source"]), ("scene", "cut"))
                self.assertNotIn("end", m)
                self.assertNotIn("confidence", m)
        self.assertFalse(os.path.exists(self.sc(self.trash)))
        leftovers = [n for _, _, fs in os.walk(self.lib) for n in fs if ".tmp-" in n]
        self.assertEqual(leftovers, [])

    def test_rerun_is_noop_and_resumes_changed(self):
        self.run_cli()
        before = os.stat(self.sc(self.part1)).st_mtime_ns
        code, summary = self.run_cli()
        self.assertEqual((code, summary["processed"], summary["skipped"]), (0, 0, 2))
        self.assertEqual(os.stat(self.sc(self.part1)).st_mtime_ns, before)

        future = time.time() + 60
        os.utime(self.part2, (future, future))
        _, summary = self.run_cli()
        self.assertEqual((summary["processed"], summary["skipped"]), (1, 1))

        _, summary = self.run_cli("--dry-run")
        self.assertEqual(summary["processed"], 0)

    def test_manual_marker_survives_force(self):
        self.run_cli()
        path = self.sc(self.part1)
        doc = sidecar.read(path)
        doc["markers"].append({"t": 10.0, "end": 12.0, "label": "favourite", "source": "manual"})
        doc["markers"] = sidecar.sort_markers(doc["markers"])
        doc.pop("tagger")  # simulate a reader that drops unknown fields on write-back
        sidecar.write_atomic(path, doc)

        code, summary = self.run_cli()
        self.assertEqual(summary["processed"], 0)  # generator fallback still counts as done

        code, summary = self.run_cli("--force")
        self.assertEqual((code, summary["processed"]), (0, 2))
        doc = sidecar.read(path)
        self.assertEqual(sidecar.validate(doc), [])
        manual = [m for m in doc["markers"] if m["source"] == "manual"]
        self.assertEqual(manual, [{"t": 10.0, "end": 12.0, "label": "favourite", "source": "manual"}])
        self.assertEqual(len(self.cuts(self.part1)), 2)  # replaced, not duplicated

    def test_unparseable_sidecar_is_not_overwritten(self):
        path = self.sc(self.part1)
        with open(path, "w") as f:
            f.write("{broken")
        code, summary = self.run_cli()
        self.assertEqual((code, summary["errors"], summary["processed"]), (1, 1, 1))
        with open(path) as f:
            self.assertEqual(f.read(), "{broken")

    def test_since_filter(self):
        old = time.time() - 86400 * 10
        os.utime(self.part1, (old, old))
        _, summary = self.run_cli("--since", time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400)))
        self.assertEqual((summary["processed"], summary["skipped"]), (1, 1))

    def test_sprites_both_files(self):
        code, _ = self.run_cli("--sprites", "--sprite-interval", "2")
        self.assertEqual(code, 0)
        base = os.path.splitext(self.part1)[0]
        with open(base + ".sprites.vtt", encoding="utf-8") as f:
            vtt = f.read()
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertEqual(vtt.count("#xywh="), 13)
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", base + ".sprites.jpg"],
                           stdout=subprocess.PIPE, check=True)
        self.assertEqual(r.stdout.decode().strip(), "1600,180")
        _, summary = self.run_cli("--sprites")
        self.assertEqual(summary["processed"], 0)

    def test_unimplemented_phase_refuses(self):
        self.assertEqual(cli.main([self.lib, "-q", "--phase", "b"]), 2)

    # ---- phase c

    def test_key_moments_replace_cuts_and_keep_manual(self):
        self.run_cli()  # phase a leaves raw cuts, like the Amina run
        path = self.sc(self.part1)
        doc = sidecar.read(path)
        doc["markers"] = sidecar.sort_markers(
            doc["markers"] + [{"t": 3.0, "label": "favourite", "source": "manual"}])
        sidecar.write_atomic(path, doc)

        code, summary = self.run_c()
        self.assertEqual((code, summary["processed"], summary["errors"]), (0, 2, 0))
        self.assertEqual(sorted(body["path"] for _, body in FakeAIHandler.requests),
                         sorted([self.part1, self.part2]))
        self.assertTrue(all(p == "/v3/process_video/" and b["frame_interval"] == 2.0
                            for p, b in FakeAIHandler.requests))

        doc = sidecar.read(path)
        self.assertEqual(sidecar.validate(doc), [])
        self.assertEqual(doc["tagger"]["phases"], ["c"])
        got = [(m["t"], m.get("end"), m["label"], m["source"]) for m in doc["markers"]]
        # blowjob 6-12 snaps its start to the (now hidden) cut at 7.0; Penis is unmapped
        self.assertEqual(got, [(3.0, None, "favourite", "manual"),
                               (7.0, 12.0, "blowjob", "ml"),
                               (20.0, 22.0, "creampie", "ml")])
        self.assertTrue(all(0 < m["confidence"] <= 1 for m in doc["markers"] if m["source"] == "ml"))

        code, summary = self.run_c()
        self.assertEqual((summary["processed"], summary["skipped"]), (0, 2))
        self.assertEqual(len(FakeAIHandler.requests), 2)

    def test_user_rejections_are_not_resuggested(self):
        self.run_c()
        path = self.sc(self.part1)
        doc = sidecar.read(path)
        ml = {m["label"]: m for m in doc["markers"] if m["source"] == "ml"}
        # what Vault does: delete the blowjob suggestion, move the creampie one 3 s later
        doc["rejected"] = [dict(ml["blowjob"], rejected_at="2026-09-14T19:00:00Z")]
        cum = ml["creampie"]
        cum["was"] = {k: cum[k] for k in ("t", "end", "label", "source", "confidence")}
        cum.update(t=23.0, source="manual")
        del cum["end"], cum["confidence"]
        doc["markers"] = sidecar.sort_markers([cum])
        sidecar.write_atomic(path, doc)

        code, summary = self.run_c("--force")
        self.assertEqual(code, 0)
        doc = sidecar.read(path)
        self.assertEqual(sidecar.validate(doc), [])
        self.assertEqual([(m["t"], m["label"], m["source"]) for m in doc["markers"]],
                         [(23.0, "creampie", "manual")])
        self.assertEqual(len(doc["rejected"]), 1)

    def test_keep_cuts_option(self):
        self.run_c("--keep-cuts", "--phase", "a,c")
        doc = sidecar.read(self.sc(self.part1))
        self.assertEqual(sorted(doc["tagger"]["phases"]), ["a", "c"])
        self.assertEqual(len(self.cuts(self.part1)), 2)
        self.assertEqual(len([m for m in doc["markers"] if m["source"] == "ml"]), 2)

    # ---- phase c, engine "phar" (the model itself is stubbed: no torch needed here)

    def _stub_phar(self, tag="Doggystyle", conf=0.9):
        spans = [{"tag": tag, "start": float(t), "end": None, "confidence": conf}
                 for t in range(0, 18, 3)]

        class Stub:
            def __init__(self, *a, **kw):
                pass

            def load(self):
                pass

            def analyze(self, video, stop_event=None):
                return {"spans": list(spans), "frame_interval": phar.WINDOW_SECONDS, "duration": None}

        real = phar.PharEngine
        phar.PharEngine = Stub
        self.addCleanup(lambda: setattr(phar, "PharEngine", real))

    def test_phar_engine_writes_moments_and_resumes_per_engine(self):
        self._stub_phar()
        code, summary = self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "5")
        self.assertEqual((code, summary["processed"]), (0, 2))
        doc = sidecar.read(self.sc(self.part1))
        self.assertEqual(sidecar.validate(doc), [])
        self.assertEqual([(m["label"], m["source"], m.get("engine")) for m in doc["markers"]],
                         [("doggy", "ml", "phar")])
        self.assertEqual(doc["tagger"]["params"]["c"]["engines"], ["phar"])

        # same engine again -> nothing to do
        _, summary = self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "5")
        self.assertEqual(summary["processed"], 0)

        # adding the second engine makes them eligible again, and both engines are recorded
        _, summary = self.run_cli("--phase", "c", "--engine", "both", "--ai-url", self.ai_url,
                                  "--min-moment", "4")
        self.assertEqual(summary["processed"], 2)
        doc = sidecar.read(self.sc(self.part1))
        self.assertEqual(doc["tagger"]["params"]["c"]["engines"], ["ai-server", "phar"])
        self.assertEqual(sorted({m.get("engine") for m in doc["markers"]}), ["ai-server", "phar"])
        self.assertEqual(sorted(m["label"] for m in doc["markers"]), ["blowjob", "creampie", "doggy"])

    def test_finetuned_model_is_recorded_and_per_label_cap_applies(self):
        # two separate doggy runs (> 20 s apart, so merge_engines keeps both); the stub reports a
        # fine-tuned model's name
        spans = [{"tag": "doggy", "start": float(t), "end": None, "confidence": 0.9} for t in (0, 22)]

        class Stub:
            name = "phar:vault-r1"

            def __init__(self, *a, **kw):
                pass

            def load(self):
                pass

            def analyze(self, video, stop_event=None):
                return {"spans": list(spans), "frame_interval": 1.0, "duration": None}

        real = phar.PharEngine
        phar.PharEngine = Stub
        self.addCleanup(lambda: setattr(phar, "PharEngine", real))

        self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "1")
        doc = sidecar.read(self.sc(self.part1))
        self.assertEqual(sidecar.validate(doc), [])
        self.assertEqual([(m["label"], m.get("engine")) for m in doc["markers"]],
                         [("doggy", "phar:vault-r1")] * 2)
        self.assertEqual(doc["tagger"]["params"]["c"]["phar_model"], "phar:vault-r1")
        self.assertEqual(doc["tagger"]["params"]["c"]["engines"], ["phar"])  # resume key unchanged

        self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "1", "--force",
                     "--max-per-label", "1")
        doc = sidecar.read(self.sc(self.part1))
        self.assertEqual([m["label"] for m in doc["markers"]], ["doggy"])
        self.assertEqual(doc["tagger"]["params"]["c"]["max_per_label"], 1)

    def test_skip_manual_protects_reviewed_movies(self):
        self._stub_phar()
        self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "5")
        path = self.sc(self.part1)
        doc = sidecar.read(path)
        doc["markers"] = sidecar.sort_markers(
            [m for m in doc["markers"] if m["source"] != "ml"]
            + [{"t": 12.0, "label": "doggy", "source": "manual"}])
        sidecar.write_atomic(path, doc)

        _, summary = self.run_cli("--phase", "c", "--engine", "phar", "--min-moment", "5",
                                  "--skip-manual", "--force")
        self.assertEqual((summary["processed"], summary["skipped"]), (1, 1))   # part1 protected
        doc = sidecar.read(path)
        self.assertEqual([(m["label"], m["source"]) for m in doc["markers"]], [("doggy", "manual")])

    def test_phar_without_torch_fails_before_touching_videos(self):
        code = cli.main([self.lib, "-q", "--phase", "c", "--engine", "phar",
                         "--phar-checkpoint", os.path.join(self.lib, "missing.pth")])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(self.sc(self.part1)))

    def test_ai_server_down_exits_before_work(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        code = cli.main([self.lib, "-q", "--phase", "c", "--ai-url", "http://127.0.0.1:%d" % port])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(self.sc(self.part1)))


if __name__ == "__main__":
    unittest.main()
