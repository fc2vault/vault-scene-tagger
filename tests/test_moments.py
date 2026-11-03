import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vault_scene_tagger import ai_client, moments, phar, sidecar  # noqa: E402


def span(tag, start, end=None, confidence=None):
    return {"tag": tag, "start": start, "end": end, "confidence": confidence}


def as_doc(markers):
    return {"schema": "vault-scenes/1", "video": "a.mp4", "duration": 3600.0, "generator": "x 1",
            "generated_at": "2026-09-14T10:00:00Z", "markers": markers}


class KeyMomentsTest(unittest.TestCase):
    def test_merge_filter_and_map(self):
        spans = [
            span("Blowjob", 60, 88), span("Blowjob", 94, 120),   # 4 s hole -> one moment
            span("Doggystyle", 200, 206),                          # 8 s: too short
            span("Creampie", 300),                                 # single frame, climax: kept
            span("Penis", 0, 3000),                                # unmapped: ignored
            span("Handjob", 400), span("Handjob", 408), span("Handjob", 416),  # patchy: dropped
        ]
        out = moments.key_moments(spans, 3600.0, frame_interval=2.0)
        self.assertEqual(sidecar.validate(as_doc(out)), [])
        self.assertEqual([(m["label"], m["t"], m["end"]) for m in out],
                         [("blowjob", 60.0, 122.0), ("creampie", 300.0, 302.0)])
        self.assertAlmostEqual(out[0]["confidence"], 0.94, places=2)
        self.assertTrue(all(m["source"] == "ml" for m in out))

    def test_cap_per_label_and_total(self):
        spans = []
        labels = ["Titjob", "Missionary", "Cowgirl", "Blowjob", "Doggy", "Kissing", "Anal"]
        t = 0
        for rep in range(3):
            for i, tag in enumerate(labels):
                spans.append(span(tag, t, t + 20 + i * 5))
                t += 60
        out = moments.key_moments(spans, 99999.0, max_moments=10, max_per_label=2)
        self.assertEqual(len(out), 10)
        counts = {}
        for m in out:
            counts[m["label"]] = counts.get(m["label"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(out, sorted(out, key=lambda m: m["t"]))

    def test_creampie_after_position_option(self):
        # on: last act before the creampie is a toy -> dropped; after missionary -> kept;
        # no act before it at all -> kept. Off (default): all four kept.
        spans = [span("Creampie", 30),
                 span("Missionary", 100, 160), span("Creampie", 170),
                 span("Toy", 400, 460), span("Creampie", 470),
                 span("Fingering", 600, 640), span("Doggystyle", 700, 760), span("Creampie", 900)]
        on = moments.key_moments(spans, 3600.0, max_moments=20, max_per_label=10,
                                 climax_needs_position=True)
        self.assertEqual([m["t"] for m in on if m["label"] == "creampie"], [30.0, 170.0, 900.0])
        off = moments.key_moments(spans, 3600.0, max_moments=20, max_per_label=10)
        self.assertEqual(len([m for m in off if m["label"] == "creampie"]), 4)

    def test_climax_ranked_above_long_acts(self):
        spans = [span("Doggystyle", i * 100, i * 100 + 60) for i in range(10)]
        spans.append(span("Creampie", 2000))
        out = moments.key_moments(spans, 5000.0, max_moments=3, max_per_label=10)
        self.assertIn("creampie", [m["label"] for m in out])

    def test_snap_to_cut(self):
        out = moments.key_moments([span("Blowjob", 60, 90)], 600.0, cuts=[20.0, 57.5, 70.0])
        self.assertEqual(out[0]["t"], 57.5)
        out = moments.key_moments([span("Blowjob", 60, 90)], 600.0, cuts=[40.0])
        self.assertEqual(out[0]["t"], 60.0)

    def test_server_confidence_and_clamp_to_duration(self):
        out = moments.key_moments([span("Doggystyle", 0, 30, 0.8)], 20.0)
        self.assertEqual((out[0]["end"], out[0]["confidence"]), (20.0, 0.8))

    def test_label_map_override_and_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.json")
            with open(path, "w") as f:
                json.dump({"Foot Job": "footjob", "Blowjob": None}, f)
            lm = moments.load_label_map(path)
            out = moments.key_moments([span("Foot Job", 0, 40), span("Blowjob", 100, 140)], 999, label_map=lm)
            self.assertEqual([m["label"] for m in out], ["footjob"])
            with open(path, "w") as f:
                json.dump({"x": "Not Kebab"}, f)
            with self.assertRaises(ValueError):
                moments.load_label_map(path)
        out = moments.key_moments([span("Some Weird Tag", 0, 40)], 999, keep_unknown=True)
        self.assertEqual(out[0]["label"], "some-weird-tag")


class MergeEnginesTest(unittest.TestCase):
    def m(self, label, t, conf, engine):
        return {"t": t, "end": t + 30, "label": label, "confidence": conf, "source": "ml",
                "engine": engine}

    def test_dedupes_same_label_keeps_confident_one(self):
        a = [self.m("blowjob", 100, 0.6, "ai-server"), self.m("creampie", 900, 0.9, "ai-server")]
        b = [self.m("blowjob", 108, 0.9, "phar"), self.m("doggy", 400, 0.8, "phar")]
        out = moments.merge_engines([a, b], max_moments=10)
        self.assertEqual([(x["label"], x["t"], x["engine"]) for x in out],
                         [("blowjob", 108, "phar"), ("doggy", 400, "phar"), ("creampie", 900, "ai-server")])

    def test_far_apart_same_label_kept_and_cap_applies(self):
        a = [self.m("doggy", 100, 0.9, "phar"), self.m("doggy", 900, 0.8, "phar")]
        self.assertEqual(len(moments.merge_engines([a], max_moments=10)), 2)
        many = [self.m("doggy", i * 100, 0.5 + i / 100.0, "phar") for i in range(12)]
        self.assertEqual(len(moments.merge_engines([many], max_moments=5)), 5)

    def test_climax_survives_the_cap(self):
        acts = [self.m("blowjob", i * 100, 0.95, "phar") for i in range(6)]
        climax = [self.m("creampie", 5000, 0.5, "ai-server")]
        out = moments.merge_engines([acts, climax], max_moments=3)
        self.assertIn("creampie", [x["label"] for x in out])


class PharTest(unittest.TestCase):
    def test_classes_and_suppression_default(self):
        self.assertEqual(len(phar.CLASSES), 17)
        for dropped in ("anal", "cunnilingus", "handjob", "69"):   # dropped from the label set
            self.assertIn(dropped, phar.SUPPRESS_DEFAULT)
        # "cumshot" is deliberately NOT suppressed: it is re-used as creampie
        self.assertNotIn("cumshot", phar.SUPPRESS_DEFAULT)
        self.assertEqual(moments.map_label("Cumshot", moments.DEFAULT_LABEL_MAP), "creampie")
        self.assertAlmostEqual(phar.WINDOW_SECONDS, 6.4)

    def test_checkpoint_key_mapping(self):
        """The 2022 MMAction2 names must map onto transformers' Timesformer names."""
        sd = {"backbone.cls_token": 1, "backbone.pos_embed": 1, "backbone.time_embed": 1,
              "backbone.patch_embed.projection.weight": 1, "backbone.patch_embed.projection.bias": 1,
              "backbone.norm.weight": 1, "backbone.norm.bias": 1,
              "cls_head.fc_cls.weight": 1, "cls_head.fc_cls.bias": 1}
        for i in range(12):
            p = "backbone.transformer_layers.layers.%d." % i
            for tail in ("attentions.0.norm.weight", "attentions.0.norm.bias",
                         "attentions.0.attn.in_proj_weight", "attentions.0.attn.in_proj_bias",
                         "attentions.0.attn.out_proj.weight", "attentions.0.attn.out_proj.bias",
                         "attentions.0.temporal_fc.weight", "attentions.0.temporal_fc.bias",
                         "attentions.1.norm.weight", "attentions.1.norm.bias",
                         "attentions.1.attn.in_proj_weight", "attentions.1.attn.in_proj_bias",
                         "attentions.1.attn.out_proj.weight", "attentions.1.attn.out_proj.bias",
                         "ffns.0.layers.0.0.weight", "ffns.0.layers.0.0.bias",
                         "ffns.0.layers.1.weight", "ffns.0.layers.1.bias",
                         "ffns.0.norm.weight", "ffns.0.norm.bias"):
                sd[p + tail] = 1
        out = phar.convert_state_dict(sd)
        self.assertEqual(len(out), len(sd))
        self.assertIn("timesformer.embeddings.cls_token", out)
        self.assertIn("classifier.weight", out)
        self.assertIn("timesformer.encoder.layer.0.temporal_attention.attention.qkv.weight", out)
        self.assertIn("timesformer.encoder.layer.11.output.dense.bias", out)
        with self.assertRaises(phar.PharError):
            phar.convert_state_dict({"backbone.surprise": 1})


class EngineResumeTest(unittest.TestCase):
    def test_engines_done(self):
        self.assertEqual(sidecar.engines_done({}), set())
        doc = {"tagger": {"phases": ["c"], "params": {"c": {"engines": ["phar", "ai-server"]}}}}
        self.assertEqual(sidecar.engines_done(doc), {"phar", "ai-server"})
        self.assertEqual(sidecar.engines_done({"tagger": {"params": {"c": {}}}}), set())


class AIClientTest(unittest.TestCase):
    def test_parse_v3(self):
        res = {"schema_version": 3, "duration": 25.0, "frame_interval": 2.0, "models": [],
               "timespans": {"actions": {"Blowjob": [{"start": 0.0, "end": 10.0}, {"start": 14.0}],
                                         "Bad": [{"end": 3}]}}}
        parsed = ai_client.parse_v3(res)
        self.assertEqual(parsed["frame_interval"], 2.0)
        self.assertEqual([(s["tag"], s["start"], s["end"]) for s in parsed["spans"]],
                         [("Blowjob", 0.0, 10.0), ("Blowjob", 14.0, None)])
        self.assertEqual(len(ai_client.parse_v3({"result": res})["spans"]), 2)
        with self.assertRaises(ai_client.AIServerError):
            ai_client.parse_v3({"detail": "nope"})

    def test_parse_v4_frames_to_moments(self):
        def frame(t, tags):
            return {"frame_index": t, "time_seconds": t,
                    "analysis": {"capabilities": {"tagging": {"actions": tags}}}}
        frames = [frame(float(t), []) for t in range(0, 20, 2)]
        frames += [frame(float(t), [["Doggystyle", 0.9]]) for t in range(20, 60, 2)]
        frames += [frame(60.0, [{"tag": "Creampie", "confidence": 0.8}]), frame(62.0, ["Kissing"])]
        res = {"duration_seconds": 64.0, "frame_interval_seconds": 2.0, "frames": frames}
        parsed = ai_client.parse_v4(res)
        self.assertEqual(len(parsed["spans"]), 22)
        self.assertEqual((parsed["frame_interval"], parsed["duration"]), (2.0, 64.0))
        out = moments.key_moments(parsed["spans"], 64.0, frame_interval=2.0)
        self.assertEqual([(m["label"], m["t"], m["end"], m["confidence"]) for m in out],
                         [("doggy", 20.0, 60.0, 0.9), ("creampie", 60.0, 62.0, 0.8)])
        with self.assertRaises(ai_client.AIServerError):
            ai_client.parse_v4({"frames": [{"time_seconds": 0, "errors": ["boom"]}]})

    def test_only_local_servers(self):
        ai_client.StashAIClient("http://127.0.0.1:8000")
        ai_client.StashAIClient("http://192.168.1.20:8000")
        for url in ("http://8.8.8.8:8000", "ftp://127.0.0.1", "not a url"):
            with self.assertRaises(ai_client.AIServerError):
                ai_client.StashAIClient(url)


if __name__ == "__main__":
    unittest.main()
