"""phar engine: which weights it picks, and loading a fine-tuned model directory."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vault_scene_tagger import phar  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

CLASSES = ["missionary", "doggy", "creampie", "background"]


class DefaultModelTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("VAULT_PHAR_MODEL", "VAULT_PHAR_CHECKPOINT")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        # the per-user folder the Vault app downloads models into: empty unless a test fills it
        self.user = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.user, True)
        self.real_user_models_dir = phar.user_models_dir
        up = mock.patch.object(phar, "user_models_dir", lambda: self.user)
        up.start()
        self.addCleanup(up.stop)

    def round_dir(self, name, config=True, root=None):
        d = os.path.join(root or self.dir, name)
        os.makedirs(d)
        if config:
            open(os.path.join(d, "config.json"), "w").close()
        return d

    def test_model_downloaded_by_vault_is_found(self):
        """Vault's Settings → License download lands in user_models_dir(); the newest round wins
        across both folders, so a user never has to pass --phar-model."""
        self.round_dir("vault-r1")
        self.round_dir("vault-r2", root=self.user)
        self.assertEqual(phar.default_checkpoint(self.dir), os.path.join(self.user, "vault-r2"))
        self.round_dir("vault-r3")
        self.assertEqual(phar.default_checkpoint(self.dir), os.path.join(self.dir, "vault-r3"))

    def test_user_models_dir_is_per_platform(self):
        with mock.patch.object(phar.sys, "platform", "darwin"):
            self.assertTrue(self.real_user_models_dir().endswith("Library/Application Support/Vault/models"))
        with mock.patch.object(phar.sys, "platform", "win32"), \
                mock.patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}):
            self.assertEqual(self.real_user_models_dir(), os.path.join(r"C:\Users\x\AppData\Local", "Vault", "models"))
        with mock.patch.object(phar.sys, "platform", "linux"), \
                mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/tmp/xdg"}):
            self.assertEqual(self.real_user_models_dir(), "/tmp/xdg/vault/models")

    def test_falls_back_to_original_checkpoint(self):
        self.assertEqual(phar.default_checkpoint(self.dir), os.path.join(self.dir, "timeSformer.pth"))

    def test_newest_round_wins_numerically(self):
        self.round_dir("vault-r2")
        self.round_dir("vault-r10")
        self.round_dir("vault-r11", config=False)     # half-copied: no config.json yet
        self.round_dir("vault-rx")
        self.assertEqual(phar.default_checkpoint(self.dir), os.path.join(self.dir, "vault-r10"))

    def test_environment_overrides(self):
        self.round_dir("vault-r1")
        with mock.patch.dict(os.environ, {"VAULT_PHAR_CHECKPOINT": "/x/old.pth"}):
            self.assertEqual(phar.default_checkpoint(self.dir), "/x/old.pth")
        with mock.patch.dict(os.environ, {"VAULT_PHAR_MODEL": "/x/vault-r7",
                                          "VAULT_PHAR_CHECKPOINT": "/x/old.pth"}):
            self.assertEqual(phar.default_checkpoint(self.dir), "/x/vault-r7")

    def test_model_name(self):
        self.assertEqual(phar.model_name(self.round_dir("vault-r3") + os.sep), "phar:vault-r3")
        self.assertEqual(phar.model_name(os.path.join(self.dir, "timeSformer.pth")), "phar")

    def test_missing_model_is_a_clear_error(self):
        eng = phar.PharEngine(checkpoint=os.path.join(self.dir, "nope"), device="cpu")
        if not HAVE_TORCH:
            self.skipTest("torch/transformers not installed")
        with self.assertRaises(phar.PharError) as cm:
            eng.load()
        self.assertIn("not found", str(cm.exception))


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class FinetunedLoadTest(unittest.TestCase):
    """A tiny random model saved like train.py saves one: save_pretrained + labels.json."""

    @classmethod
    def setUpClass(cls):
        from transformers import TimesformerConfig, TimesformerForVideoClassification
        cls.dir = tempfile.mkdtemp()
        cls.model = os.path.join(cls.dir, "vault-r3")
        cfg = TimesformerConfig(image_size=phar.SIZE, patch_size=16, num_frames=phar.CLIP_LEN,
                                num_channels=3, hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=64,
                                num_labels=len(CLASSES),
                                id2label={i: c for i, c in enumerate(CLASSES)},
                                label2id={c: i for i, c in enumerate(CLASSES)})
        TimesformerForVideoClassification(cfg).save_pretrained(cls.model)
        cls.meta = {"classes": CLASSES, "clip_len": phar.CLIP_LEN, "sample_fps": phar.SAMPLE_FPS,
                    "size": phar.SIZE, "short_side": 256, "normalize": "(x-127.5)/127.5"}
        cls.write_meta(cls.meta)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def write_meta(cls, meta):
        with open(os.path.join(cls.model, "labels.json"), "w") as f:
            json.dump(meta, f)

    def tearDown(self):
        self.write_meta(self.meta)

    def test_loads_classes_name_and_suppresses_background(self):
        eng = phar.PharEngine(checkpoint=self.model, device="cpu")
        eng.load()
        self.assertEqual(eng.classes, CLASSES)
        self.assertEqual(eng.name, "phar:vault-r3")
        self.assertIn("background", eng.suppress)
        self.assertIn("anal", eng.suppress)            # the P-HAR defaults stay harmless

    def test_refuses_a_different_input_pipeline(self):
        self.write_meta(dict(self.meta, sample_fps=5.0))
        with self.assertRaises(phar.PharError) as cm:
            phar.PharEngine(checkpoint=self.model, device="cpu").load()
        self.assertIn("input pipeline", str(cm.exception))

    def test_refuses_mismatched_class_order(self):
        self.write_meta(dict(self.meta, classes=list(reversed(CLASSES))))
        with self.assertRaises(phar.PharError):
            phar.PharEngine(checkpoint=self.model, device="cpu").load()


if __name__ == "__main__":
    unittest.main()
