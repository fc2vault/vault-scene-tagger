# vault-scene-tagger

A local batch tool that walks a video library and writes a **scene-marker sidecar** next to every
video: `<name>.scenes.json`. A reader such as the Vault media server turns those into markers on
the player's timeline.

The **only** contract with any reader is the file format in [`SCENES_SPEC.md`](SCENES_SPEC.md)
(`vault-scenes/1`). This tool never talks to Vault, its database or its code. It only reads videos
and writes data files next to them.

| Phase | What it emits | Status |
|---|---|---|
| **C** (default) | **AI key moments per video**: ranged `missionary`, `doggy`, `blowjob`, `creampie`… with `confidence`, `source: "ml"` | **shipped**. Two engines: the Vault scene model (`phar`, positions and acts) or skier233's local AI server (`ai-server`, acts only). |
| A | raw shot-change boundaries: `{"t": 12.0, "label": "scene", "source": "cut"}` | shipped. ffmpeg only. |
| B | clothing/nudity states (`source: "heuristic"`) | planned |

---

## Install (Vault Pro users: three commands)

```bash
pip install "git+https://github.com/fc2vault/vault-scene-tagger"
pip install torch transformers numpy      # the model engine; see pytorch.org for a CUDA build
brew install ffmpeg                       # macOS; apt/dnf/winget on Linux/Windows
```

Then download the scene model once in Vault (**Settings → License → Scene model**). The tagger
finds it on its own, and tagging a folder is:

```bash
vault-scene-tagger "/path/to/library/<Actress>" --engine phar
```

Freshly tagged movies appear under **To review** in Vault. That's the whole setup.

## Requirements

- **Python 3.8+**, standard library only. Nothing to `pip install` for the tagger itself.
- **ffmpeg / ffprobe** on `PATH` (or pass `--ffmpeg` / `--ffprobe`).
- **For phase C**, one of: the Vault scene model (`--engine phar`, needs `torch` + `transformers`,
  see [Engines](#engines)), or skier233's **AI server** running locally with a tagging model loaded
  (`--engine ai-server`, see [AI key moments](#ai-key-moments-phase-c)).

## Run

```bash
# see what would happen: walks the library, analyzes and writes nothing
./vault-scene-tagger "/path/to/library" --dry-run

# key moments for one actress with the Vault scene model
./vault-scene-tagger "/path/to/library/Amina" --engine phar

# whole library, logged
./vault-scene-tagger "/path/to/library" --engine phar --log-file tagger.log --summary-json last-run.json

# plain shot cuts only, no AI
./vault-scene-tagger "/path/to/library" --phase a
```

`python3 -m vault_scene_tagger …` is equivalent to `./vault-scene-tagger`. Stop a run any time with
Ctrl+C (or SIGTERM). Sidecars that were already written are complete (writes are atomic), and the next
run resumes where this one stopped.

---

## AI key moments (phase C)

### How it works

1. For each video the tagger sends the **file path** (never the file) to the AI server on
   `127.0.0.1:8000`, via `POST /v3/process_video/`. The server decodes the video itself and
   classifies one frame every 2 s.
2. The server returns, per tag, the time spans where that tag was detected.
3. The tagger reduces that to **at most 10 key moments**:
   - tag names are mapped to the [§4 vocabulary](SCENES_SPEC.md#4-label-vocabulary). **Tags without a mapping are ignored**,
     so body-part or other noise doesn't turn into markers;
   - a label's detections separated by ≤ 8 s are merged into one moment;
   - moments shorter than 15 s are dropped. `cumshot`, `facial` and `creampie` count from a single frame;
   - moments where the label was detected in fewer than 40 % of the sampled frames are dropped;
   - climax moments rank first, then the rest by confidence × log(length). There are at most 2 moments
     per label and 10 in total;
   - if the sidecar already holds shot cuts from an earlier `--phase a` run, each moment's start
     snaps to a cut within 5 s, so jumping lands on a clean edit.
4. **Raw cut markers are removed** from the sidecar. `manual` markers are always kept.

`confidence` = the share of sampled frames inside the moment where the label was detected. When the
server reports per-detection confidence, it's multiplied by the mean of those.

### Setting up the AI server (one-time, by you)

The server and its models are **not** part of this repo and aren't downloaded by it. They're
skier233's project: [media_ai_model_server](https://github.com/skier233/media_ai_model_server)
(formerly `nsfw_ai_model_server`). Its [macOS instructions](https://github.com/skier233/nsfw_ai_model_server/wiki/Installation-Instructions-(MacOS))
in short:

1. Download the newest release and unzip it somewhere, e.g. `~/ai_server`.
2. From the [AI Models page](https://github.com/skier233/nsfw_ai_model_server/wiki/AI-Models), download
   a tagging model and copy its `config` and `models` folders into the server folder:
   - **vivid galaxy**: free, 10 sexual-action tags;
   - **actions_30** (36 action tags) / **positions_20** (28 positions): paid SubscribeStar tiers, the most accurate.
3. `source ./install/install-conda-mac.sh`, then `source ./install/install-mac.sh`
4. `./select_models.sh` to choose the active model(s), then start it with `source ./start.sh`
   (it listens on port 8000).

Then run the tagger. It checks the server is reachable before touching any video.

**Licensing:** the server code and models are licensed for **personal use on your own local
network only**. They may not be modified or redistributed, and the paid models need an active
subscription. This repo only calls the server's HTTP API and bundles nothing from it.

**Hardware:** macOS support in the server is **beta**. On the M1 / 8 GB Mac expect it to be slow
(CPU/MPS, PyTorch) and memory-hungry next to Vault, so run it overnight with `--workers 1`. An
NVIDIA machine on the LAN is much faster: run the server there and pass `--ai-url http://<ip>:8000`
(only loopback/private addresses are accepted). The video path must be valid **on the server
machine**, so a remote server needs the share mounted at the same path.

### Phase C options

| flag | default | meaning |
|---|---|---|
| `--ai-url` | `http://127.0.0.1:8000` | AI server. Must resolve to loopback or a private address. System proxies are never used. |
| `--ai-frame-interval` | `2` | seconds between analyzed frames. Lower is more precise and slower. |
| `--ai-threshold` | `0.5` | per-frame tag confidence the server requires |
| `--ai-timeout` | `14400` | max seconds to wait for one video |
| `--max-moments` | `10` | key moments per video |
| `--max-per-label` | `2` | key moments per label per video. Reviewed movies average ~10 markers, with 20% of them a 3rd+ of the same label, so raise both caps when a run is meant to be measured. |
| `--creampie-after-position` | off | drop a creampie whose last preceding act is toy, fingering, blowjob or titjob: a creampie ends intercourse, so those are close-ups of something else. On round 1 it removed 18 of 29 wrong creampie suggestions and lost 1 of 30 right ones. For everyday tagging; leave it off in training rounds so the review measures the model itself. |
| `--min-moment` | `15` | shortest moment in seconds (climax labels exempt) |
| `--keep-cuts` | off | keep the raw `scene` cut markers too |
| `--label-map JSON` | – | override tag→label mapping, e.g. `{"Doggystyle": "doggy", "Footjob": "footjob", "Kissing": null}` (`null` = ignore) |
| `--keep-unknown-tags` | off | emit unmapped tags (kebab-cased) instead of ignoring them |
| `--engine` | `ai-server` | which analyzer: `ai-server`, `phar`, or `both` (see below) |

### Engines

Phase C can run two analyzers; each moment records which one proposed it (`"engine"` field).

| engine | what it is | covers | speed (M1) |
|---|---|---|---|
| `ai-server` | skier233's local server + "vivid galaxy" model | acts only (8 of our labels) | ~4x realtime |
| `phar` | [P-HAR](https://github.com/rlleshi/phar) TimeSformer, run in plain PyTorch: P-HAR's original weights, or a fine-tuned round `models/vault-r<N>/` | acts **and positions** (doggy, missionary, cowgirl…) | ~3.5x realtime |

`--engine both` runs both and merges them: same-label moments within 20 s are deduplicated, keeping
the more confident one, then the usual cap applies.

**phar weights.** The engine loads either of two:

- **A fine-tuned round:** `models/vault-r1/`, `vault-r2/` … The model is retrained on your reviewed
  markers (training bundle → GPU → `save_pretrained` directory + `labels.json`). Its classes already
  are Vault labels (missionary … scoop-up, plus `background` = no marker, always suppressed). Markers
  record it as `"engine": "phar:vault-r1"`, the run as `tagger.params.c.phar_model`. A model whose
  `labels.json` describes a different input pipeline is refused.
- **P-HAR's original weights:** download `timeSformer.pth` from P-HAR's
  [release v1.0.0](https://github.com/rlleshi/phar/releases/tag/v1.0.0) (Apache-2.0) into `models/`.
  Its 17 classes are mapped to Vault labels (below), and markers record `"engine": "phar"`. Keep it:
  every retraining round starts from it.

Without `--phar-model` (alias `--phar-checkpoint`), the newest `vault-r<N>/` that has a
`config.json` is used — looked for in this project's `models/` **and in the folder the Vault app
downloads models into** (Settings → License → Scene model: `~/Library/Application Support/Vault/models`
on macOS, `%LOCALAPPDATA%\Vault\models` on Windows, `~/.local/share/vault/models` on Linux) — else
`models/timeSformer.pth`. `$VAULT_PHAR_MODEL` overrides both. So a Vault Pro user downloads the model
in Vault once and runs the tagger with no model flag at all. The engine needs `torch`,
`transformers` and `numpy`, which the stdlib-only tool does not require otherwise:

```bash
pip install torch transformers numpy          # once; see pytorch.org for a CUDA build on Linux/Windows
vault-scene-tagger "/path/to/library/<Actress>" --phase c --engine phar
```

It picks CUDA > MPS > CPU automatically (`--phar-device`), streams frames from one ffmpeg pass
(constant memory on a 3-hour movie), and analyzes a 6.4 s window every 3.2 s (`--phar-stride`).

**Known bias (original weights):** P-HAR's `anal` class fires on rear-entry positions (38 of 275 windows in a movie
without any anal), so it is suppressed by default — `--phar-suppress ""` re-enables it. Its
`kissing`, `fondling` and `deepthroat` classes are not in our label set and are ignored;
`facial-cumshot` becomes `cumshot`. `the-snake` (flat on the stomach, from behind) and `scoop-up` are
kept as positions of their own.

Run with `-v` to log how many raw detections each video produced and how many moments were kept. If a
model's tag names don't match, check its tag list and add them with `--label-map`.

---

## General options

| flag | default | meaning |
|---|---|---|
| `--phase` | `c` | phase(s), comma-separated: `c` AI key moments, `a` shot cuts, `b` planned. `a,c` computes fresh cuts for snapping in the same run. |
| `--workers` | `1` | videos in parallel |
| `--force` | off | reprocess everything. Manual markers are still preserved. |
| `--since` | – | only videos modified at/after this time: epoch or ISO (`2026-09-01`, `2026-09-01T18:00`) |
| `--dry-run` | off | list what would be processed and why |
| `--log-file` / `--summary-json` | – | append the log to a file / write the run summary as JSON |
| `-v` / `-q` | – | debug logging / warnings + one-line JSON summary only |

Phase A only: `--scene-threshold 0.3`, `--min-gap 1.0`, `--hwaccel videotoolbox`, `--fast-decode`,
`--sprites` / `--sprite-interval 10` (sprites are optional per §3.3; Vault doesn't use them).

Exit codes: `0` ok, `1` some videos failed (see the summary), `2` bad usage / AI server unreachable,
`130` interrupted.

### Resume rules

A video is **skipped** when its sidecar exists, is at least as new as the video, and records that the
requested phase(s) already ran (the extra `"tagger"` field). So re-running is a no-op, and a replaced
video file is picked up again. Changing thresholds or the model doesn't make files eligible; add `--force`.

### Walking

- Video extensions (case-insensitive): `.mp4 .mkv .avi .wmv .mov .ts .webm .flv .m4v .rmvb .mpg .mpeg`
- Never descends into `_vault_trash`, `_duplicates`, `_unsorted`, `_vault`, `@eaDir` (any case), or any
  dot-directory. Dotfiles and `._*` AppleDouble files are ignored.
- Multi-part movies: each file gets its own sidecar, with timestamps relative to that file.

### Merge & safety

- Markers with `source: "manual"` (written back by the reader) are **never** touched. A run replaces
  only the sources its phases own: C → `ml` (plus `cut`, unless `--keep-cuts`), A → `cut`.
- The sidecar is re-read **right before** writing, so manual markers added in Vault during analysis are kept.
- A sidecar that isn't valid JSON is **never overwritten**. It's reported as an error.
- Writes use a hidden temp file in the same folder, then a rename. Sidecars are never deleted, and
  videos are only ever read.

### Validating output

```bash
python3 -m vault_scene_tagger.validate "/path/to/library"
```

A JSON Schema is at [`examples/scenes.schema.json`](examples/scenes.schema.json).


## Tests

```bash
python3 -m unittest discover -s tests -t .
```

The tests cover moment reduction (merging, caps, climax priority, snapping, label maps), the
AI client (response parsing, the local-only guard), and end to end with synthetic ffmpeg videos and a fake
AI server on 127.0.0.1: key moments replacing cuts while manual markers survive, resume, and
failing fast when the server is unreachable. They also cover phase A cut detection, reserved dirs,
unicode paths, multi-part movies, `--since`, and sprites.

---

## Performance notes (phase A, measured)

On an M1 Mac with the library on an SMB share, ffmpeg 8.0: 1080p HEVC decodes at about
12× realtime on CPU (≈ 4 cores), or with `--hwaccel videotoolbox` at < 1 core. A studio title decoded
at about 26× realtime. The scene score is bimodal on this material. Real edits scored ≥ 0.27, ordinary
motion ≤ 0.08, and a single-shot handheld FC2 clip never went above 0.04, hence the default of 0.3. On
Amina's 1-hour title phase A produced 46 cuts, which is why phase C keeps only 10 key moments and hides the cuts.

## Layout

```
vault-scene-tagger            launcher (no install needed)
vault_scene_tagger/
  cli.py                      argument parsing, walk → filter → worker pool → summary
  tagger.py                   per-video pipeline, resume check, phase registry
  ai_client.py                local AI server client (/v3/process_video/), local-only guard
  moments.py                  detections → ≤ N key moments, label map
  media.py                    ffprobe, ffmpeg cut detection (+ sprites)
  sidecar.py                  read / merge (§5) / validate (§3) / atomic write
  walker.py                   library walking & naming (§7)
  validate.py                 `python3 -m vault_scene_tagger.validate`
SCENES_SPEC.md                the contract (keep identical in the Vault repo)
examples/                     sample sidecar + JSON Schema
tests/                        unittest suite
```
