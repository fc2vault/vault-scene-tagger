"""Export the hand-labelled sidecars as a portable training-data package.

    python3 -m vault_scene_tagger.export <media-root> --out DIR

The package is self-contained apart from the video files themselves, so it can be copied to the
training machine (GPU box / DGX Spark) and turned into clips there:

    DIR/
      dataset.json     every video with its markers, corrections (`was`) and rejections
      labels.txt       the label set, one per line
      sidecars/        verbatim copies of the *.scenes.json (the source of truth)
      make_clips.py    cuts the training clips from the videos (stdlib + ffmpeg only)
      README.md        what the fields mean and how to run make_clips.py

Nothing in the media library is modified.
"""
import argparse
import json
import os
import shutil
import sys

from . import GENERATOR
from .sidecar import SidecarError, read as read_sidecar
from .walker import SCENES_SUFFIX, iter_videos, sidecar_path

MARKER_FIELDS = ("t", "end", "label", "source", "confidence", "engine", "was")


CODE_RE = __import__("re").compile(r"FC2[-_\s]*PPV[-_\s]*(\d{4,})", __import__("re").I)


def _actress_map(catalog):
    """code -> performer, read from Vault's catalog. Optional: without it the package has no
    way to split train/val BY PERSON, and a random clip split leaks the same scene across
    both sides — the single easiest way to get a meaningless validation score here."""
    if not catalog or not os.path.exists(catalog):
        return {}
    import sqlite3
    con = sqlite3.connect(catalog)
    try:
        return {c: (n or "") for c, n in con.execute(
            "SELECT w.code, COALESCE(a.name_en, a.name_jp) FROM works w "
            "LEFT JOIN actresses a ON a.id = w.actress_id")}
    except Exception:
        return {}
    finally:
        con.close()


def collect(root, min_manual=1, catalog=None):
    """Walk the media root and gather every sidecar that carries human labels."""
    who = _actress_map(catalog)
    out, skipped = [], 0
    for video in iter_videos(root):
        sc = sidecar_path(video)
        try:
            doc = read_sidecar(sc)
        except SidecarError:
            skipped += 1
            continue
        if not doc:
            continue
        markers = [m for m in (doc.get("markers") or []) if isinstance(m, dict)]
        manual = [m for m in markers if m.get("source") == "manual"]
        if len(manual) < min_manual:
            continue
        rejected = [r for r in (doc.get("rejected") or []) if isinstance(r, dict)]
        rel = os.path.relpath(video, root)
        m = CODE_RE.search(rel)
        code = "FC2-PPV-" + m.group(1) if m else ""
        out.append({
            "video": rel,
            "sidecar": os.path.relpath(sc, root),
            "code": code,
            # the subject, for a held-out-by-person split. Unknown performers fall back to
            # the code so each such movie is its own subject rather than one giant bucket.
            "actress": who.get(code) or ("~" + (code or rel)),
            "duration": doc.get("duration"),
            "markers": [{k: m[k] for k in MARKER_FIELDS if k in m} for m in markers],
            "rejected": [{k: r[k] for k in ("t", "end", "label", "source", "confidence",
                                            "rejected_at") if k in r} for r in rejected],
        })
    return out, skipped


def label_stats(videos):
    stats = {}
    for v in videos:
        for m in v["markers"]:
            if m.get("source") == "manual":
                stats[m["label"]] = stats.get(m["label"], 0) + 1
    return dict(sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])))


MAKE_CLIPS = '''#!/usr/bin/env python3
"""Cut training clips from the exported markers. Standard library + ffmpeg only.

    python3 make_clips.py --media-root /path/to/library [--window 20] [--out clips]

Every manual marker becomes one clip starting at the marker (truncated by the next marker), and
the gaps between markers become "background" clips. Writes clips/<label>/... and clips.csv.
"""
import argparse, csv, json, os, subprocess, sys

def video_seconds(path, fallback):
    """Length of the VIDEO stream, not the container.

    Four files in this library have an audio track minutes longer than the picture (a bad
    download). The container duration is the audio's, so a clip cut past the end of the
    video stream produced a 261-byte file with no frames."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) if out and out != "N/A" else fallback
    except Exception:
        return fallback

def spans(video, window, min_background, bg_per_video=2, dur=None):
    dur = dur or video.get("duration") or 0
    marks = sorted([m for m in video["markers"] if m.get("source") == "manual"],
                   key=lambda m: m["t"])
    out = []
    for i, m in enumerate(marks):
        end = min(m["t"] + window, dur or m["t"] + window)
        if i + 1 < len(marks):
            end = min(end, marks[i + 1]["t"])
        if end - m["t"] >= 2.0:
            out.append((m["label"], round(m["t"], 2), round(end, 2)))
    # Background. A marker is a POINT, not a segment: the act usually runs well past t+window,
    # so "unlabelled" is NOT reliably "nothing happening". Two guards:
    #   * the gap must be large (default 90 s), and
    #   * we take one fixed-length clip from the MIDDLE of it, furthest from both neighbours,
    #     instead of the whole gap (which would also hand the loader 3-minute clips).
    cursor, gaps = 0.0, []
    for label, start, end in list(out) + [("", dur, dur)]:
        gap = start - cursor
        if gap >= min_background:
            gaps.append((gap, cursor + gap / 2.0))
        cursor = max(cursor, end)
    # Keep only the widest gaps per video. Taking every one made background 46% of the set —
    # a class four times the size of the biggest real one, which teaches the model that the
    # safe answer is always "nothing".
    for _gap, mid in sorted(gaps, reverse=True)[:bg_per_video]:
        out.append(("background", round(mid - window / 2.0, 2), round(mid + window / 2.0, 2)))
    return sorted(out, key=lambda s: s[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--media-root", required=True, help="where the videos live on this machine")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--window", type=float, default=20.0, help="seconds after a marker (default 20)")
    ap.add_argument("--min-background", type=float, default=90.0,
                    help="a gap must be at least this long before a background clip is taken "
                         "from the middle of it (default 90)")
    ap.add_argument("--background-per-video", type=int, default=2,
                    help="background clips taken from each video's widest gaps (default 2)")
    ap.add_argument("--split", default="splits.json",
                    help="held-out-by-performer split; adds a `split` column to clips.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = json.load(open(a.dataset, encoding="utf-8"))
    val = set()
    if a.split and os.path.exists(a.split):
        val = set(json.load(open(a.split, encoding="utf-8")).get("val_videos") or [])
    rows, missing = [], 0
    for v in data["videos"]:
        src = os.path.join(a.media_root, v["video"])
        if not os.path.isfile(src):
            missing += 1
            print("missing video:", src, file=sys.stderr)
            continue
        vdur = video_seconds(src, v.get("duration") or 0)
        for label, start, end in spans(v, a.window, a.min_background,
                                       a.background_per_video, vdur):
            if start >= vdur or end - min(end, vdur) > 0 and min(end, vdur) - start < 2.0:
                continue
            end = min(end, vdur)
            name = "%s_%s.mp4" % (os.path.splitext(os.path.basename(v["video"]))[0], int(start))
            dst = os.path.join(a.out, label, name)
            rows.append({"clip": dst.replace(os.sep, "/"), "label": label,
                         "video": v["video"].replace(os.sep, "/"),
                         "actress": v.get("actress", ""),
                         "split": "val" if v["video"] in val else "train",
                         "start": start, "end": end})
            if a.dry_run or os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", str(start),
                            "-t", str(round(end - start, 2)), "-i", src, "-map", "0:V:0", "-an",
                            "-vf", "scale=-2:256", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", dst], check=True)
            if os.path.getsize(dst) < 1024:        # no frames encoded — do not ship it
                os.remove(dst)
                rows.pop()
                print("empty clip dropped:", dst, file=sys.stderr)
    with open("clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "label", "video", "actress", "split",
                                          "start", "end"])
        w.writeheader()
        w.writerows(rows)
    by_label = {}
    for r in rows:
        key = r["label"]
        by_label.setdefault(key, [0, 0])[0 if r["split"] == "train" else 1] += 1
    print("%d clips%s | videos missing: %d" % (len(rows), " (dry run)" if a.dry_run else "", missing))
    print("  %-16s %7s %7s" % ("label", "train", "val"))
    for k, (tr, va) in sorted(by_label.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print("  %-16s %7d %7d" % (k, tr, va))

if __name__ == "__main__":
    main()
'''

README = """# Training data export

Created by {generator} on {created}.

## What's in here

| file | meaning |
|---|---|
| `dataset.json` | every video with human labels: markers, corrections, rejections |
| `labels.txt` | the label set |
| `sidecars/` | verbatim copies of the `*.scenes.json` files (source of truth) |
| `splits.json` | train/val held out **by performer** — see below |
| `make_clips.py` | cuts the training clips (stdlib + ffmpeg) |
| `clips/` | the cut clips, if they were made here (`clips/<label>/<video>_<start>.mp4`) |
| `clips.csv` | clip, label, video, actress, split, start, end |

`dataset.json` fields per marker:

- `t` — start in seconds (markers are start points), `end` — only on AI suggestions
- `source` — `manual` = verified by a human, `ml`/`cut` = analyzer output, not yet reviewed
- `engine` — which analyzer proposed an `ml` marker (`phar`, `ai-server`)
- `was` — what the AI originally said before the human corrected/confirmed it:
  a **correction pair** (AI wrong → human right), the most valuable training signal
- `rejected` (per video) — AI suggestions the human deleted: **false positives**

Only videos with at least one manual marker are exported. A video counts as fully reviewed when it
has no `ml` markers left; only those are safe to mine for background/negative clips.

## Turning it into clips

Already done if `clips/` is present. Otherwise, on a machine that can see the videos:

```bash
python3 make_clips.py --media-root /path/to/library --dry-run   # count first
python3 make_clips.py --media-root /path/to/library
```

Each manual marker becomes one clip from the marker to +20 s (cut short by the next marker),
scaled to 256 px on the short side. Output: `clips/<label>/<video>_<start>.mp4` + `clips.csv`.

### Two things the defaults are protecting you from

**Background is not "everything unlabelled".** A marker is a *point*, not a segment — the act
usually runs well past `t+20 s`, so unmarked time is not reliably empty. A background clip is
only taken from a gap of at least `--min-background` (90 s), from the **middle** of it, and at
most `--background-per-video` (2) per movie. Without that cap background was 46% of the set:
a class four times the biggest real one, which teaches the model that the safe answer is
always "nothing".

**Split by performer, never by clip.** `splits.json` holds out whole people. Two scenes of the
same act by the same person minutes apart are near-duplicates; a random clip split puts one in
train and one in val and reports a score that means nothing. `clips.csv` carries the `split`
column already — use it, don't re-shuffle.

## Training notes

- The window is 20 s but the marker sits at the **start** of the act for hand-placed markers
  and nearer the middle for confirmed AI ones. Sample a random sub-window at load time
  (temporal jitter) and the difference stops mattering — it also multiplies the data for free.
- Markers repeat within one continuous act (they were placed for training volume). Those
  clips are highly correlated: do not treat them as independent samples, and never let them
  straddle the split. Splitting by performer already prevents that.
- `was` is a correction pair (AI wrong → human right) and `rejected` is a list of confirmed
  false positives. Both are worth more than an equal number of easy positives.
- Class sizes are very uneven. Weight the loss or sample the tail classes up; `scoop-up` and
  `titjob` in particular are too small to trust without it.

## Labels

{labels}
"""


_VAL_FRAC = [0.2]          # set from the CLI before write_package runs


def make_split(videos, val_frac=0.2, seed=20261012):
    """Hold out whole PERFORMERS, never random clips.

    Two scenes of the same act by the same person minutes apart are near-duplicates, and this
    library is dominated by one performer; a random split puts minute 3 in train and minute 4
    in val and reports a score that means nothing. Performers are assigned largest-first to
    whichever side is further below its target, so the rare classes do not all land on one
    side."""
    import random
    by_actress = {}
    for v in videos:
        by_actress.setdefault(v.get("actress") or "~unknown", []).append(v)
    order = sorted(by_actress, key=lambda a: (-len(by_actress[a]), a))
    random.Random(seed).shuffle(order[: max(1, len(order) // 10)])   # jitter the tail only
    total = len(videos)
    want_val = total * val_frac
    val, n_val = set(), 0
    for a in sorted(order, key=lambda a: -len(by_actress[a])):
        if n_val + len(by_actress[a]) <= want_val:
            val.add(a); n_val += len(by_actress[a])
    return {"val_frac": val_frac, "seed": seed,
            "val_actresses": sorted(val),
            "train_videos": [v["video"] for v in videos if v.get("actress") not in val],
            "val_videos": [v["video"] for v in videos if v.get("actress") in val]}


def write_package(root, out_dir, videos, skipped, min_manual):
    os.makedirs(out_dir, exist_ok=True)
    stats = label_stats(videos)
    manual = sum(stats.values())
    ml_left = sum(1 for v in videos for m in v["markers"] if m.get("source") != "manual")
    reviewed = sum(1 for v in videos if all(m.get("source") == "manual" for m in v["markers"]))
    corrections = sum(1 for v in videos for m in v["markers"] if m.get("was"))
    rejections = sum(len(v["rejected"]) for v in videos)

    dataset = {
        "generator": GENERATOR,
        "created": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "media_root": os.path.abspath(root),
        "min_manual": min_manual,
        "labels": list(stats),
        "counts": {"videos": len(videos), "fully_reviewed": reviewed, "manual_markers": manual,
                   "unreviewed_ai_markers": ml_left, "corrections": corrections,
                   "rejections": rejections, "per_label": stats},
        "videos": videos,
    }
    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stats) + ("\n" if stats else ""))
    split = make_split(videos, _VAL_FRAC[0])
    with open(os.path.join(out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
        f.write("\n")
    sc_dir = os.path.join(out_dir, "sidecars")
    for v in videos:
        dst = os.path.join(sc_dir, v["sidecar"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(root, v["sidecar"]), dst)
    script = os.path.join(out_dir, "make_clips.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(MAKE_CLIPS)
    os.chmod(script, 0o755)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(generator=GENERATOR, created=dataset["created"],
                              labels="\n".join("- `%s` — %d marker(s)" % (k, n)
                                               for k, n in stats.items()) or "- (none yet)"))
    return dataset


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m vault_scene_tagger.export",
        description="Export hand-labelled sidecars as a portable training-data package.")
    ap.add_argument("media_root")
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument("--catalog", default=None, metavar="catalog.db",
                    help="Vault catalog, used to label each video with its performer so the "
                         "package can be split train/val BY PERSON")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="share of performers held out for validation (default 0.2)")
    ap.add_argument("--min-manual", type=int, default=1,
                    help="only videos with at least this many manual markers (default 1)")
    opts = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if not os.path.isdir(opts.media_root):
        print("media root not found: %s" % opts.media_root, file=sys.stderr)
        return 2
    _VAL_FRAC[0] = opts.val_frac
    videos, skipped = collect(opts.media_root, opts.min_manual, opts.catalog)
    data = write_package(opts.media_root, opts.out, videos, skipped, opts.min_manual)
    c = data["counts"]
    print("exported %d videos (%d fully reviewed) -> %s" % (c["videos"], c["fully_reviewed"], opts.out))
    print("  %d manual markers, %d corrections, %d rejections, %d unreviewed AI markers"
          % (c["manual_markers"], c["corrections"], c["rejections"], c["unreviewed_ai_markers"]))
    if skipped:
        print("  %d sidecar(s) unreadable and skipped" % skipped)
    for label, n in c["per_label"].items():
        print("    %-16s %d" % (label, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
