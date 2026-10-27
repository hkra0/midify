"""midify: song audio -> separated stems -> MIDI parts, chord chart and MusicXML lead sheet.

    python midify.py song.mp3                # separate + transcribe into out/<song>/
    python midify.py song.mp3 -o result      # choose the output folder
    python midify.py song.mp3 --stems DIR    # reuse stems separated earlier
"""
import argparse
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.ndimage as ndi

SR = 22050
HOP = 512
# Transcription device. MPS (Apple GPU) has hung mid-run with some macOS/torch versions;
# CPU is slower but reliable. Override with --device or MIDIFY_DEVICE.
DEVICE = os.environ.get("MIDIFY_DEVICE", "cpu")
PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"
CACHE = PROJECT / "cache"  # slow-stage results, shared across output folders
SEPARATOR_MODEL = "BS-Roformer-SW.ckpt"  # 6 stems: vocals, drums, bass, guitar, piano, other
STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def stem_path(stem_dir, name):
    """`piano.wav`, or audio-separator's `<song>_(Piano)_<model>.wav`."""
    p = stem_dir / f"{name}.wav"
    if p.exists():
        return p
    hits = [q for q in stem_dir.glob("*.wav") if f"({name.lower()})" in q.name.lower()]
    if not hits:
        raise FileNotFoundError(f"no '{name}' stem in {stem_dir}")
    return hits[0]


def cached(cache_dir, tag, inputs, fn, *args, **kwargs):
    """Run fn once per (tag, input files' size+mtime, args); later runs read the pickle.
    The slow stages (pitch tracking, piano transcription) take minutes on CPU."""
    import hashlib
    import pickle

    # key on the real file (symlinked stems in a de-reverb folder hit the same cache entry)
    real = [Path(os.path.relpath(Path(p).resolve())) for p in inputs]
    sig = repr([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in real]) + repr(args) + repr(kwargs)
    path = Path(cache_dir) / f"{tag}_{hashlib.sha1(sig.encode()).hexdigest()[:10]}.pkl"
    if path.exists():
        print(f"      (cached: {path.name})")
        return pickle.loads(path.read_bytes())
    result = fn(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def load(stem_dir, *names):
    y = None
    for n in names:
        s, _ = librosa.load(stem_path(stem_dir, n), sr=SR, mono=True)
        y = s if y is None else y + s
    return y


# ---------------------------------------------------------------- beats & key

def track_beats(drums, accomp):
    """Beat times from drums (+ accompaniment) onsets; downbeat phase from harmonic change."""
    env = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    env += 0.5 * librosa.onset.onset_strength(y=accomp, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, tightness=200)
    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # the tracker works on 23 ms frames and tends to sit a frame late: calibrate the whole
    # grid against fine-resolution onsets that fall near a beat
    fine = librosa.onset.onset_detect(y=drums + 0.5 * accomp, sr=SR, hop_length=128, units="time")
    if len(fine):
        near = [fine[np.argmin(np.abs(fine - t))] - t for t in beat_times]
        near = [d for d in near if abs(d) < 0.07]
        if len(near) > len(beat_times) // 4:
            beat_times = beat_times + float(np.median(near))
    return float(np.atleast_1d(tempo)[0]), beat_times


def downbeat_phase(beat_chroma, beat_bass):
    """Assume 4/4; pick the beat phase where harmony/bass changes most often."""
    novelty = np.r_[0, np.linalg.norm(np.diff(beat_chroma, axis=1), axis=0)]
    novelty += np.r_[0, np.linalg.norm(np.diff(beat_bass, axis=1), axis=0)]
    scores = [novelty[p::4].mean() for p in range(4)]
    return int(np.argmax(scores))


# Krumhansl-Kessler key profiles
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(chroma):
    prof = chroma.mean(axis=1)
    best = max(
        ((np.corrcoef(prof, np.roll(p, k))[0, 1], k, mode)
         for k in range(12) for p, mode in ((KK_MAJ, "major"), (KK_MIN, "minor"))),
    )
    return best[1], best[2]


# ---------------------------------------------------------------- melody

def mono_pitch(path, fmin=65, fmax=1100, model="full"):
    """Monophonic f0 (torchcrepe) for a single-line stem: vocals or bass."""
    import torch
    import torchcrepe

    audio, sr = librosa.load(path, sr=16000, mono=True)
    x = torch.tensor(audio)[None]
    device = DEVICE
    pitch, periodicity = torchcrepe.predict(
        x, sr, hop_length=160, fmin=fmin, fmax=fmax, model=model,
        decoder=torchcrepe.decode.viterbi, return_periodicity=True,
        batch_size=128, device=device)  # 1024 is ~3.5x slower on CPU
    pitch = pitch[0].cpu().numpy()
    periodicity = torchcrepe.filter.median(periodicity, 3)[0].cpu().numpy()
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=160)[0][: len(pitch)]
    times = np.arange(len(pitch)) * 0.01
    return times, pitch, periodicity, rms[: len(pitch)]


def stem_onsets(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return librosa.onset.onset_detect(y=y, sr=SR, hop_length=128, units="time", backtrack=True, delta=0.08)


def pitch_to_notes(times, f0, periodicity, rms, min_dur=0.09, rel_db=18):
    """Frame-level f0 -> list of (start, end, midi_pitch)."""
    db = librosa.amplitude_to_db(rms, ref=np.max)
    # reverb/echo tails in the vocal stem are pitched but much quieter than the sung line:
    # gate relative to the loudest point in the surrounding ~3 s
    local_max = ndi.maximum_filter1d(db, size=301)
    voiced = (periodicity > 0.45) & (db > -38) & (db > local_max - rel_db)
    voiced = ndi.binary_opening(voiced, np.ones(3))
    midi = librosa.hz_to_midi(np.maximum(f0, 1e-3))
    midi = ndi.median_filter(midi, 7)

    # global tuning offset (singer/recording not exactly A440)
    frac = (midi[voiced] - np.round(midi[voiced]))
    tuning = float(np.median(frac)) if voiced.any() else 0.0
    q = np.round(midi - tuning)
    q[~voiced] = -1

    # runs of constant quantized pitch
    runs, start = [], 0
    for i in range(1, len(q) + 1):
        if i == len(q) or q[i] != q[start]:
            runs.append([start, i, q[start]])
            start = i
    # absorb very short pitched runs (glides, vibrato flicks) into the longer neighbour
    min_frames = int(0.06 / 0.01)
    changed = True
    while changed:
        changed = False
        for j, (s, e, p) in enumerate(runs):
            if p < 0 or e - s >= min_frames:
                continue
            prev = runs[j - 1] if j > 0 and runs[j - 1][2] >= 0 else None
            nxt = runs[j + 1] if j + 1 < len(runs) and runs[j + 1][2] >= 0 else None
            tgt = max((r for r in (prev, nxt) if r), key=lambda r: r[1] - r[0], default=None)
            if tgt is None:
                continue
            p_new = tgt[2]
            runs[j][2] = p_new
            changed = True
        # merge adjacent equal runs
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged

    notes = [(times[s], times[e - 1] + 0.01, int(p)) for s, e, p in runs
             if p >= 0 and times[e - 1] + 0.01 - times[s] >= min_dur]
    notes = fix_octave_jumps(notes)
    return notes, tuning


def align_to_onsets(notes, onsets, before=0.15, after=0.04, gap_merge=0.12, min_part=0.1):
    """Pitch-based note starts lag the sung attack; move them to the vocal onset.
    Also split held notes at onsets inside them (repeated syllables on one pitch) and
    merge same-pitch fragments that have no onset between them."""
    onsets = np.asarray(onsets)
    out = []
    for s, e, p in notes:
        cand = onsets[(onsets >= s - before) & (onsets <= s + after)]
        if len(cand):
            s = float(cand[0])
        if out and out[-1][1] > s:
            out[-1] = (out[-1][0], s, out[-1][2])
        inner = onsets[(onsets > s + min_part) & (onsets < e - min_part)]
        for o in inner:
            out.append((s, float(o), p))
            s = float(o)
        out.append((s, e, p))
    merged = []
    for s, e, p in out:
        if merged and merged[-1][2] == p and s - merged[-1][1] < gap_merge \
                and not ((onsets > merged[-1][1] - 0.03) & (onsets <= s + 0.03)).any():
            merged[-1] = (merged[-1][0], e, p)
        else:
            merged.append((s, e, p))
    return [n for n in merged if n[1] - n[0] >= 0.06]


def fix_octave_jumps(notes):
    out = list(notes)
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1][2], out[i][2], out[i + 1][2]
        if min(abs(b - a), abs(b - c)) <= 9:
            continue
        # leaps away from both neighbours: try the octave that lands closest to them
        best = min((b - 12, b + 12), key=lambda x: min(abs(x - a), abs(x - c)))
        if min(abs(best - a), abs(best - c)) <= 7:
            out[i] = (out[i][0], out[i][1], best)
    return out


# ---------------------------------------------------------------- beat grid

class Grid:
    """Maps seconds <-> beat positions using the tracked beats (extrapolated at the ends)."""

    def __init__(self, beat_times, phase, duration):
        bt = np.asarray(beat_times)
        period = np.median(np.diff(bt))
        # prepend beats so that beat 0 is a downbeat and covers t=0
        pre = []
        t = bt[0]
        n_pre = 0
        while t > 0 or (n_pre + phase) % 4 != 0:  # tracked beat `phase` must land on a bar line
            t -= period
            pre.insert(0, t)
            n_pre += 1
        # beat tracking often stops before a fade-out; extend past the end of the audio,
        # otherwise np.interp clamps every later note onto the last beat
        n_post = int(np.ceil((duration - bt[-1]) / period)) + 8
        self.bt = np.r_[pre, bt, bt[-1] + period * np.arange(1, n_post + 1)]
        self.idx = np.arange(len(self.bt), dtype=float)

    def to_beat(self, t):
        return np.interp(t, self.bt, self.idx)

    def to_time(self, b):
        return np.interp(b, self.idx, self.bt)


def snap(b, step=0.25, eighth_pull=0.16):
    """Round to 16ths, but prefer the 8th grid unless clearly on an off-16th
    (plain rounding would switch at 0.125 beat; this widens the 8th's catchment)."""
    q8 = round(b / 0.5) * 0.5
    return q8 if abs(b - q8) < eighth_pull else round(b / step) * step


def quantize_notes(notes, grid, step=0.25):
    out = []
    for s, e, p in notes:
        bs = snap(grid.to_beat(s), step)
        be = round(grid.to_beat(e) / step) * step
        if be <= bs:
            be = bs + step
        out.append((bs, be, p))
    # legato: close tiny gaps (< 1/8 beat) to the next note
    out.sort()
    for i in range(len(out) - 1):
        if 0 < out[i + 1][0] - out[i][1] <= step / 2 + 1e-9:
            out[i] = (out[i][0], out[i + 1][0], out[i][2])
    # resolve overlaps after quantization
    out.sort()
    clean = []
    for n in out:
        if clean and n[0] < clean[-1][1]:
            if n[0] == clean[-1][0]:
                continue
            clean[-1] = (clean[-1][0], n[0], clean[-1][2])
        clean.append(n)
    return clean


# ---------------------------------------------------------------- chords

QUALITIES = {  # intervals, prior penalty
    "": ([0, 4, 7], 0.0),
    "m": ([0, 3, 7], 0.0),
    "7": ([0, 4, 7, 10], 0.06),
    "maj7": ([0, 4, 7, 11], 0.06),
    "m7": ([0, 3, 7, 10], 0.06),
    "sus4": ([0, 5, 7], 0.08),
}


def chord_vocab():
    names, temps, pens, roots = [], [], [], []
    for r in range(12):
        for q, (iv, pen) in QUALITIES.items():
            t = np.zeros(12)
            t[[(r + i) % 12 for i in iv]] = 1
            t[r] += 0.5  # root emphasis
            names.append(NOTE_NAMES[r] + q)
            temps.append(t / np.linalg.norm(t))
            pens.append(pen)
            roots.append(r)
    return names, np.array(temps), np.array(pens), np.array(roots)


def recognize_chords(beat_chroma, beat_bass, key_root, key_mode, phase=0,
                     stay_by_pos=(0.80, 0.995, 0.93, 0.995)):
    """stay_by_pos: probability of keeping the chord when entering beat 1/2/3/4 of a bar,
    so changes prefer the downbeat, then beat 3."""
    names, temps, pens, roots = chord_vocab()
    c = beat_chroma / (np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9)
    b = beat_bass / (beat_bass.max(axis=0, keepdims=True) + 1e-9)
    score = temps @ c - pens[:, None] + 0.25 * b[roots]

    # diatonic prior
    scale = [0, 2, 4, 5, 7, 9, 11] if key_mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    scale = {(key_root + s) % 12 for s in scale}
    for i, (t, r) in enumerate(zip(temps, roots)):
        tones = set(np.nonzero(t)[0])
        if not tones <= scale:
            score[i] -= 0.05

    # Viterbi over beats: emission = softmax(score * k), sticky transitions
    k = 12.0
    logp = k * score - np.log(np.exp(k * score).sum(axis=0, keepdims=True))
    n, T = logp.shape
    trans_by_pos = []
    for stay in stay_by_pos:
        tr = np.full((n, n), np.log((1 - stay) / (n - 1)))
        np.fill_diagonal(tr, np.log(stay))
        trans_by_pos.append(tr)
    delta = logp[:, 0].copy()
    back = np.zeros((n, T), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + trans_by_pos[(t - phase) % 4]
        back[:, t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logp[:, t]
    path = [int(delta.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(back[path[-1], t])
    path = path[::-1]

    silent = beat_chroma.sum(axis=0) < 1e-3
    return [None if silent[i] else names[j] for i, j in enumerate(path)], (names, temps, roots)


def chord_segments(labels):
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start]:
                segs.append((start, i, labels[start]))
            start = i
    return segs


# ---------------------------------------------------------------- polyphonic accompaniment

PIANO_CKPT = MODELS / "note_F1=0.9677_pedal_F1=0.9186.pth"
PIANO_CKPT_URL = ("https://zenodo.org/record/4034264/files/"
                  "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")


def ensure_piano_checkpoint():
    """Download the piano transcription weights (~170 MB) on first use."""
    if PIANO_CKPT.exists() and PIANO_CKPT.stat().st_size > 1.6e8:
        return
    PIANO_CKPT.parent.mkdir(parents=True, exist_ok=True)
    part = PIANO_CKPT.with_suffix(".part")
    print(f"      downloading piano transcription weights -> {PIANO_CKPT}", flush=True)
    urllib.request.urlretrieve(PIANO_CKPT_URL, part)
    part.rename(PIANO_CKPT)


def transcribe_poly(path, min_vel=20):
    """Piano-style polyphonic transcription (ByteDance piano_transcription) of one stem.
    Returns [(start_s, end_s, pitch, velocity)]."""
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    ensure_piano_checkpoint()
    tr = PianoTranscription(device=DEVICE, checkpoint_path=str(PIANO_CKPT))
    ev = tr.transcribe(y, None)["est_note_events"]
    return [(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
            for e in ev if e["velocity"] >= min_vel]


def quantize_poly(notes, grid, step=0.25):
    out = []
    for s, e, p, v in notes:
        bs = snap(grid.to_beat(s), step)
        be = max(round(grid.to_beat(e) / step) * step, bs + step)
        out.append((bs, be, p, v))
    return out


def sustain_notes(notes, gap=0.2, min_dur=0.3, min_vel=30):
    """The piano model re-strikes held (bowed/blown) notes; join them back into long notes
    and drop the short blips that percussion leaves in the same stem."""
    by_pitch = {}
    for s, e, p, v in sorted(notes):
        cur = by_pitch.setdefault(p, [])
        if cur and s - cur[-1][1] < gap:
            ps, pe, _, pv = cur[-1]
            cur[-1] = (ps, max(pe, e), p, max(pv, v))
        else:
            cur.append((s, e, p, v))
    return sorted(n for ns in by_pitch.values() for n in ns
                  if n[1] - n[0] >= min_dur and n[3] >= min_vel)


# Basic Pitch thresholds (defaults 0.5 / 0.3): stricter = fewer notes but fewer wrong ones
HELD_ONSET, HELD_FRAME = 0.6, 0.4


def transcribe_sustained(path, onset=HELD_ONSET, frame=HELD_FRAME):
    """Basic Pitch (instrument-agnostic) for held-note stems: strings, pads, winds.
    The piano model expects struck notes and fragments bowed/blown ones."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=onset, frame_threshold=frame,
                           minimum_note_length=127.7, multiple_pitch_bends=False, melodia_trick=True)
    return [(float(s), float(e), int(p), int(a * 127)) for s, e, p, a, *_ in events]


def remove_overtones(notes, intervals=(12, 19, 24), min_overlap=0.5):
    """Drop a note when a louder note an octave / twelfth / two octaves below covers most of
    it: transcribers often report a held note's harmonics as extra notes."""
    out = []
    for n in notes:
        s, e, p, v = n
        covered = any(m[2] == p - k and m[3] >= v and min(e, m[1]) - max(s, m[0]) >= min_overlap * (e - s)
                      for m in notes for k in intervals)
        if not covered:
            out.append(n)
    return out


def chord_tone_filter(notes, chords_sec, vocab, scale, keep_long=1.0):
    """Keep held-note parts to the harmony (reverb smear and leakage add stray notes):
    short notes must be in the chord sounding at their midpoint; long notes may also be
    any scale tone. Chord tones outside the key (borrowed chords) are kept either way."""
    names, temps, _ = vocab
    starts = np.array([c[0] for c in chords_sec])
    out = []
    for n in notes:
        mid = (n[0] + n[1]) / 2
        i = np.searchsorted(starts, mid, "right") - 1
        if i < 0 or mid > chords_sec[i][1]:
            out.append(n)
            continue
        tones = set(np.nonzero(temps[names.index(chords_sec[i][2])])[0])
        if n[2] % 12 in tones or (n[1] - n[0] >= keep_long and n[2] % 12 in scale):
            out.append(n)
    return out


def drop_bass_doubles(notes, bass, below=48):
    """Low notes in a held-note part that a bass note of the same pitch class covers are
    bass leakage, not the part itself (strings rarely sit under C3 with the bass)."""
    def doubled(n):
        return n[2] < below and any(b[2] % 12 == n[2] % 12 and b[0] < n[1] and b[1] > n[0] for b in bass)
    return [n for n in notes if not doubled(n)]


# (GM drum note, low Hz, high Hz): kick, snare, closed hi-hat
DRUM_BANDS = [(36, 30, 150), (38, 150, 2500), (42, 6000, 16000)]


def transcribe_drums(path, min_gap=0.06, cluster=0.035, both=2.0, snap=0.05):
    """Drum hits by frequency band: a peak in a band's spectral flux is a hit of that drum.
    One stroke rings in several bands (kick thump reaches the snare band, snare noise the
    hi-hat band), so peaks within `cluster` s count as one stroke: keep the band that stands
    out most, plus any other band at least `both` x over its own threshold. Times snap to
    the nearest fine onset. Coarse (no toms/cymbals), but needs no extra model."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    hop, n_fft = 256, 1024  # short window: a long one sees the hit early (its right edge)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=hop)[0], ref=1.0)
    active = rms_db > np.percentile(rms_db[rms_db > -80], 50) - 20 if (rms_db > -80).any() else rms_db > 0
    cand = []  # (time, gm, strength relative to band threshold, flux)
    for gm, lo, hi in DRUM_BANDS:
        band = librosa.amplitude_to_db(S[(freqs >= lo) & (freqs < hi)].sum(axis=0) + 1e-9, ref=1.0)
        flux = np.maximum(0, np.diff(band, prepend=band[0]))
        flux = np.convolve(flux, np.ones(3) / 3, "same").astype(np.float64)  # peak_pick wants float64
        thresh = float(np.percentile(flux[active], 90)) if active.any() else np.inf
        peaks = librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=20, post_avg=5,
                                       delta=thresh * 0.5, wait=int(min_gap * SR / hop))
        cand += [(p * hop / SR, gm, flux[p] / thresh, flux[p]) for p in peaks if active[p] and flux[p] >= thresh]
    cand.sort()
    fine = librosa.onset.onset_detect(y=y, sr=SR, hop_length=128, units="time")
    top = {gm: np.percentile([c[3] for c in cand if c[1] == gm] or [1], 95) for gm, *_ in DRUM_BANDS}
    hits, i = [], 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and cand[j + 1][0] - cand[i][0] < cluster:
            j += 1
        group = cand[i:j + 1]
        best = max(group, key=lambda c: c[2])
        t = float(np.mean([c[0] for c in group]))
        if len(fine) and np.min(np.abs(fine - t)) < snap:
            t = float(fine[np.argmin(np.abs(fine - t))])
        for c in {c[1]: c for c in sorted(group, key=lambda c: c[2])}.values():  # strongest per band
            if c is best or c[2] >= both:
                hits.append((t, t + 0.1, c[1], int(np.clip(50 + 70 * c[3] / top[c[1]], 30, 120))))
        i = j + 1
    return sorted(hits)


def gate_by_stem_level(notes, path, drop_db=8, win=2.0, keep_vel_pct=75):
    """Keep notes only where the stem is near its typical playing level; quiet stretches
    are reverb/leakage from other instruments (e.g. a strings stem before the strings enter).
    With keep_vel_pct, notes struck hard (top quarter of this stem's velocities) always stay:
    a lone note in a quiet ending is real even though the 2 s average around it is low. Only
    for the piano model, whose velocity is loudness; Basic Pitch's is confidence, and it is
    confident about leaked notes too."""
    if not notes:
        return notes
    y, _ = librosa.load(path, sr=SR, mono=True)
    hop = int(SR * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=int(SR * win), hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)
    thresh = np.percentile(db, 90) - drop_db
    strong = np.percentile([n[3] for n in notes], keep_vel_pct) if keep_vel_pct else np.inf
    loud = lambda t: db[min(int(t * SR / hop), len(db) - 1)] >= thresh
    return [n for n in notes if n[3] >= strong or loud(n[0])]


# ---------------------------------------------------------------- stem audit
# Each separator stem is checked before its notes are used, instead of per-song switches:
#   silent      -> dropped (the instrument is absent)
#   no pitch    -> dropped (a single-line stem without a stable, in-tune pitch is noise)
#   leakage     -> merged into the stem it duplicates (same pitch struck at the same time)
# Drums only feed beat tracking.

MONO_STEMS = {"vocals": dict(),  # mono_pitch defaults: 65-1100 Hz, full model
              "bass": dict(fmin=35, fmax=400, model="tiny")}  # crepe bins start at ~31.7 Hz
POLY_STEMS = ["piano", "guitar", "other"]
SUSTAINED = {"other"}  # strings/pads/winds: held notes, transcribed with Basic Pitch instead
OVERRIDES = {}  # if the audit gets a stem wrong, e.g. {"bass": "drop"}, {"guitar": "keep"}

MIN_ACTIVE = 0.03      # fraction of the song a stem must be audible in
MIN_VOICED = 0.5       # mono stems: share of audible frames with a confident pitch
MIN_IN_TUNE = 0.6      # mono stems: share of pitched frames within 25 cents of a semitone
LEAK_DUP = 0.3         # share of a stem's notes duplicated by a bigger stem -> leakage


def real(p):
    """Cache-stable path: symlinked stems in a de-reverb folder map to the original file."""
    return Path(os.path.relpath(Path(p).resolve()))


def stem_activity(y, mix, win=0.5):
    """Share of windows where the stem is audible: above -60 dBFS and within 25 dB of the mix."""
    hop = int(SR * win)
    db = lambda a: librosa.amplitude_to_db(librosa.feature.rms(y=a, frame_length=hop, hop_length=hop)[0], ref=1.0)
    s, m = db(y), db(mix)
    return float(np.mean((s > -60) & (s > m - 25)))


def pitch_quality(f0, per, rms):
    db = librosa.amplitude_to_db(rms, ref=np.max)
    audible = db > -30
    voiced = audible & (per > 0.45)
    if voiced.sum() < 10:
        return 0.0, 0.0
    midi = librosa.hz_to_midi(np.maximum(f0[voiced], 1e-3))
    frac = midi - np.round(midi)
    dev = np.abs(frac - np.median(frac))
    return float(voiced.sum() / max(audible.sum(), 1)), float(np.mean(np.minimum(dev, 1 - dev) < 0.25))


def duplicate_ratio(notes, ref, tol=0.06):
    """Share of `notes` that `ref` also has: same pitch, onset within `tol` seconds."""
    if not notes or not ref:
        return 0.0
    by_pitch = {}
    for n in ref:
        by_pitch.setdefault(n[2], []).append(n[0])
    by_pitch = {p: np.sort(v) for p, v in by_pitch.items()}

    def dup(n):
        on = by_pitch.get(n[2])
        if on is None:
            return False
        i = np.searchsorted(on, n[0])
        return any(0 <= j < len(on) and abs(on[j] - n[0]) < tol for j in (i - 1, i))
    return float(np.mean([dup(n) for n in notes]))


def merge_notes(ref, extra, tol=0.06):
    """Union of two note lists, skipping `extra` notes that `ref` already has."""
    return sorted(ref + [n for n in extra if duplicate_ratio([n], ref, tol) == 0])


def notes_chroma(notes, beat_times):
    """Beat-synchronous pitch-class weights from transcribed notes (duration x velocity).
    Column i covers beat i -> i+1, matching the audio chroma layout."""
    bt = np.r_[beat_times, beat_times[-1] + np.median(np.diff(beat_times))]
    C = np.zeros((12, len(beat_times)))
    low = np.zeros((12, len(beat_times)))
    lowest = np.full(len(beat_times), 200)
    for n in notes:
        s, e, p = n[:3]
        v = n[3] if len(n) > 3 else 80
        i0, i1 = np.searchsorted(bt, s, "right") - 1, np.searchsorted(bt, e, "left")
        for i in range(max(i0, 0), min(i1, len(beat_times))):
            ov = min(e, bt[i + 1]) - max(s, bt[i])
            if ov > 0:
                C[p % 12, i] += ov * v
                lowest[i] = min(lowest[i], p)
    for i, p in enumerate(lowest):
        if p < 200:
            low[p % 12, i] = 1
    return C, low


# ---------------------------------------------------------------- export

def chord_pitches(name, vocab):
    names, temps, roots = vocab
    t = temps[names.index(name)]
    root = roots[names.index(name)]
    pcs = [int(p) for p in np.nonzero(t)[0]]
    voicing = sorted(48 + ((p - root) % 12) + root % 12 for p in pcs)
    voicing = [v if v < 60 else v - 12 for v in voicing] if root % 12 > 6 else voicing
    return [36 + root % 12] + voicing


def chord_notes(chords, vocab):
    return [(s, e, p, 60) for s, e, name in chords for p in chord_pitches(name, vocab)]


# (track name, GM program); notes are (start, end, pitch[, velocity]) in beats or seconds
TRACKS = {"melody": ("Melody", 73), "piano": ("Piano", 0), "guitar": ("Guitar", 25),
          "other": ("Other (strings/pads)", 48), "bass": ("Bass", 33), "drums": ("Drums", 0),
          "chords": ("Chord pads", 89)}


def write_midi(path, parts, to_sec, tempo):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for key_, notes in parts.items():
        name, program = TRACKS[key_]
        inst = pretty_midi.Instrument(program=program, name=name, is_drum=key_ == "drums")
        for n in notes:
            vel = n[3] if len(n) > 3 else 100
            inst.notes.append(pretty_midi.Note(vel, n[2], to_sec(n[0]), to_sec(n[1])))
        pm.instruments.append(inst)
    pm.write(str(path))


def write_musicxml(path, melody, chords, tempo, key_root, key_mode, title):
    from music21 import harmony, key, metadata, meter, note, stream, tempo as m21tempo

    part = stream.Part()
    part.append(m21tempo.MetronomeMark(number=round(tempo)))
    part.append(key.Key(NOTE_NAMES[key_root], key_mode))
    part.append(meter.TimeSignature("4/4"))
    for s, e, p in melody:
        n = note.Note(p)
        n.quarterLength = e - s
        part.insert(s, n)
    for s, _, name in chords:
        try:
            cs = harmony.ChordSymbol(name.replace("Eb", "E-").replace("Ab", "A-").replace("Bb", "B-"))
        except Exception:
            continue
        part.insert(s, cs)
    sc = stream.Score()
    sc.metadata = metadata.Metadata(title=title)
    sc.insert(0, part)
    sc.makeNotation(inPlace=True)
    sc.write("musicxml", fp=str(path))


def write_instrumental(path, stem_dir, stems=("drums", "bass", "guitar", "piano", "other")):
    """Original-audio backing track: every separated stem except vocals, at full quality."""
    import soundfile as sf

    mix, sr = None, None
    for n in stems:
        y, sr = sf.read(stem_path(stem_dir, n), always_2d=True)
        mix = y if mix is None else mix[: len(y)] + y[: len(mix)]
    peak = np.abs(mix).max()
    if peak > 0.99:  # the stems sum back to the mix, but guard against clipping
        mix *= 0.99 / peak
    sf.write(path, mix, sr)


def bar_label(b):
    return f"{int(b // 4) + 1}.{int(b % 4) + 1}"


def write_chord_chart(path, chords, grid, key_root, key_mode, tempo):
    with open(path, "w") as f:
        f.write(f"# key {NOTE_NAMES[key_root]} {key_mode}, ~{tempo:.1f} BPM, 4/4 (bar.beat  time  chord)\n")
        for s, e, n in chords:
            f.write(f"{bar_label(s):>7}  {grid.to_time(s):7.2f}s  {n}\n")
        f.write("\n# chord chart (one line = 4 bars)\n")
        last_bar = int(chords[-1][1] // 4) + 1
        per_bar = []
        for bar in range(last_bar):
            names = [n for s, e, n in chords if s < (bar + 1) * 4 and e > bar * 4]
            per_bar.append(" ".join(dict.fromkeys(names)) or "-")
        for i in range(0, len(per_bar), 4):
            f.write(f"{i + 1:>4} | " + " | ".join(f"{b:<10}" for b in per_bar[i:i + 4]) + " |\n")


# ---------------------------------------------------------------- main

def separate(audio, stem_dir):
    """Split the song into STEMS with audio-separator (model downloads on first use)."""
    from audio_separator.separator import Separator

    stem_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(stem_dir), model_file_dir=str(MODELS / "separator"), output_format="WAV")
    sep.load_model(model_filename=SEPARATOR_MODEL)
    sep.separate(str(audio))


def has_stems(stem_dir):
    try:
        return all(stem_path(stem_dir, n) for n in STEMS)
    except FileNotFoundError:
        return False


def main(audio, out_dir, stem_dir=None):
    audio, out_dir = Path(audio), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = CACHE
    if stem_dir is None:
        stem_dir = out_dir / "stems"
        if not has_stems(stem_dir):
            print("[0/6] separating stems (GPU if available; several minutes)", flush=True)
            separate(audio, stem_dir)
    stem_dir = Path(stem_dir)

    print("[1/6] loading stems")
    drums = load(stem_dir, "drums")
    bass = load(stem_dir, "bass")
    accomp = load(stem_dir, "bass", "guitar", "piano", "other")

    print("[2/6] beats & key")
    tempo, beat_times = track_beats(drums, accomp)
    harm = librosa.effects.harmonic(accomp)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP)
    bass_chroma = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E1"), n_octaves=3)
    key_root, key_mode = detect_key(chroma)
    beat_frames = librosa.time_to_frames(beat_times, sr=SR, hop_length=HOP)
    bc = librosa.util.sync(chroma, beat_frames, aggregate=np.median)[:, 1:]  # segment i = beat i -> i+1
    bb = librosa.util.sync(bass_chroma, beat_frames, aggregate=np.median)[:, 1:]
    phase = downbeat_phase(bc, bb)
    grid = Grid(beat_times, phase, duration=len(drums) / SR)
    print(f"      tempo ~{tempo:.1f} BPM, {len(beat_times)} beats, key {NOTE_NAMES[key_root]} {key_mode}, downbeat phase {phase}")

    print("[3/6] single-line parts: audit + torchcrepe")
    mix = accomp + drums + load(stem_dir, "vocals")
    report, mono, poly = [], {}, {}

    def note(name, msg):
        report.append(f"{name:7s} {msg}")
        print(f"      {name:7s} {msg}")

    for name, kw in MONO_STEMS.items():
        path = real(stem_path(stem_dir, name))
        act = stem_activity(load(stem_dir, name), mix)
        if OVERRIDES.get(name) == "drop" or (name not in OVERRIDES and act < MIN_ACTIVE):
            note(name, f"audible {act:.0%} -> dropped ({'override' if name in OVERRIDES else 'silent'})")
            continue
        times, f0, per, rms = cached(cache, f"{name}_pitch", [path], mono_pitch, path, **kw)
        voiced, in_tune = pitch_quality(f0, per, rms)
        stats = f"audible {act:.0%}, pitched {voiced:.0%}, in tune {in_tune:.0%}"
        if name not in OVERRIDES and (voiced < MIN_VOICED or in_tune < MIN_IN_TUNE):
            note(name, f"{stats} -> dropped (no stable pitch)")
            continue
        notes, _ = pitch_to_notes(times, f0, per, rms, min_dur=0.09 if name == "vocals" else 0.08)
        mono[name] = align_to_onsets(notes, stem_onsets(path))
        note(name, f"{stats} -> kept, {len(mono[name])} notes")

    print("[4/6] polyphonic parts: audit + piano_transcription / Basic Pitch")
    for name in POLY_STEMS:
        path = real(stem_path(stem_dir, name))
        act = stem_activity(load(stem_dir, name), mix)
        if OVERRIDES.get(name) == "drop" or (name not in OVERRIDES and act < MIN_ACTIVE):
            note(name, f"audible {act:.0%} -> dropped ({'override' if name in OVERRIDES else 'silent'})")
            continue
        if name in SUSTAINED:
            notes = cached(cache, f"held_{name}", [path], transcribe_sustained, path, HELD_ONSET, HELD_FRAME)
            notes = remove_overtones(sustain_notes(notes))
        else:
            notes = cached(cache, f"poly_{name}", [path], transcribe_poly, path)
        poly[name] = gate_by_stem_level(notes, path, keep_vel_pct=None if name in SUSTAINED else 75)
        note(name, f"audible {act:.0%}, {len(poly[name])} notes")
    # leakage: a stem whose notes another (bigger) stem already plays is the same instrument
    for name in sorted(poly, key=lambda n: len(poly[n])):
        bigger = [r for r in poly if r != name and len(poly[r]) >= len(poly[name])]
        dups = {r: duplicate_ratio(poly[name], poly[r]) for r in bigger}
        target = OVERRIDES.get(name)
        if target is None and dups and max(dups.values()) >= LEAK_DUP:
            target = max(dups, key=dups.get)
        if target in poly and target != name:
            why = "override" if name in OVERRIDES else f"{dups.get(target, 0):.0%} of its notes duplicated"
            poly[target] = merge_notes(poly[target], poly.pop(name))
            note(name, f"-> merged into {target} ({why})")
        elif dups:
            note(name, "-> kept (duplicates: " + ", ".join(f"{r} {d:.0%}" for r, d in dups.items()) + ")")
    # struck/plucked stems all go through the same piano model, so in MIDI they sound alike;
    # combine them into one accompaniment part unless an override keeps one apart
    struck = sorted((k for k in poly if k not in SUSTAINED), key=lambda k: -len(poly[k]))
    for name in struck[1:]:
        if OVERRIDES.get(name) != "keep":
            poly[struck[0]] = merge_notes(poly[struck[0]], poly.pop(name))
            note(name, f"-> combined into {struck[0]} (same transcription model; OVERRIDES['{name}'] = 'keep' to split)")
    if "bass" in mono and "bass" not in OVERRIDES:
        dups = {r: duplicate_ratio(mono["bass"], ns) for r, ns in poly.items()}
        if dups and max(dups.values()) >= LEAK_DUP:
            r = max(dups, key=dups.get)
            note("bass", f"-> dropped ({dups[r]:.0%} of its notes duplicated by {r})")
            mono.pop("bass")
        elif dups:
            note("bass", "-> kept (duplicates: " + ", ".join(f"{r} {d:.0%}" for r, d in dups.items()) + ")")

    drums_sec = []
    act = stem_activity(drums, mix)
    if OVERRIDES.get("drums") == "drop" or ("drums" not in OVERRIDES and act < MIN_ACTIVE):
        note("drums", f"audible {act:.0%} -> dropped ({'override' if 'drums' in OVERRIDES else 'silent'})")
    else:
        drums_sec = transcribe_drums(stem_path(stem_dir, "drums"))
        kinds = {gm: sum(1 for h in drums_sec if h[2] == gm) for gm, *_ in DRUM_BANDS}
        note("drums", f"audible {act:.0%} -> kept, kick {kinds[36]}, snare {kinds[38]}, hi-hat {kinds[42]}")
    (out_dir / "stem_report.txt").write_text("\n".join(report) + "\n")

    notes_sec = mono.get("vocals", [])
    bass_sec = mono.get("bass", [])
    melody = quantize_notes(notes_sec, grid) if notes_sec else []

    print("[5/6] chords")
    # chords come from the transcribed accompaniment (+ bass), with the audio chroma as a
    # backstop where the transcription is thin; held pads/strings count half
    harmony = [n if k not in SUSTAINED else (*n[:3], n[3] * 0.5) for k, ns in poly.items() for n in ns]
    pc, plow = notes_chroma(harmony, beat_times)
    bc_notes, bb_notes = notes_chroma([n + (90,) for n in bass_sec], beat_times)
    nrm = lambda m: m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)
    chroma_mix = 0.65 * nrm(pc + 0.5 * bc_notes) + 0.35 * nrm(bc)
    bass_mix = np.maximum(bb_notes, 0.5 * plow) + 0.3 * nrm(bb) if bass_sec else plow
    labels, vocab = recognize_chords(chroma_mix, bass_mix, key_root, key_mode, phase)
    first = grid.to_beat(beat_times[0])  # beat index (in grid) of the first tracked beat
    chords = [(s + first, e + first, n) for s, e, n in chord_segments(labels)]
    print(f"      {len(chords)} chord segments, {len(set(n for *_, n in chords))} distinct chords")
    chords_sec = [(grid.to_time(s), grid.to_time(e), n) for s, e, n in chords]
    steps = [0, 2, 4, 5, 7, 9, 11] if key_mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    scale = {(key_root + st) % 12 for st in steps}
    for k in SUSTAINED & poly.keys():
        before = len(poly[k])
        poly[k] = chord_tone_filter(drop_bass_doubles(poly[k], bass_sec), chords_sec, vocab, scale)
        print(f"      {k}: kept {len(poly[k])} of {before} notes as chord tones or long notes")

    print("[6/6] export")
    realtime = {"melody": notes_sec, "bass": bass_sec, **poly, "drums": drums_sec}
    gridded = {"melody": melody, "bass": quantize_notes(bass_sec, grid) if bass_sec else [],
               "drums": quantize_poly(drums_sec, grid),
               **{k: quantize_poly(ns, grid) for k, ns in poly.items()}}
    realtime = {k: v for k, v in realtime.items() if v}
    gridded = {k: v for k, v in gridded.items() if v}

    sec_per_beat = 60.0 / tempo
    write_midi(out_dir / "song_grid.mid", gridded, lambda b: b * sec_per_beat, tempo)
    write_midi(out_dir / "song_realtime.mid", realtime, lambda t: t, tempo)
    # accompaniment only: every part except the sung melody
    accomp_rt = {k: v for k, v in realtime.items() if k != "melody"}
    accomp_grid = {k: v for k, v in gridded.items() if k != "melody"}
    write_midi(out_dir / "accompaniment_realtime.mid", accomp_rt, lambda t: t, tempo)
    write_midi(out_dir / "accompaniment_grid.mid", accomp_grid, lambda b: b * sec_per_beat, tempo)
    write_instrumental(out_dir / "instrumental.wav", stem_dir)
    # one file per part (grid version) so each can be auditioned on its own;
    # block chords are only a reference, so they stay out of the full-song files
    gridded["chords"] = chord_notes(chords, vocab)
    for f in out_dir.glob("part_*.mid"):
        f.unlink()
    for part, notes in gridded.items():
        write_midi(out_dir / f"part_{part}.mid", {part: notes}, lambda b: b * sec_per_beat, tempo)
    write_musicxml(out_dir / "leadsheet.musicxml", melody, chords, tempo, key_root, key_mode, audio.stem)
    write_chord_chart(out_dir / "chords.txt", chords, grid, key_root, key_mode, tempo)
    print("done ->", out_dir)


def cli():
    global DEVICE
    ap = argparse.ArgumentParser(description="Song audio -> MIDI parts, chords and lead sheet.")
    ap.add_argument("audio", help="song file (mp3, wav, flac, ...)")
    ap.add_argument("-o", "--out", help="output folder (default: out/<song name>)")
    ap.add_argument("--stems", help="folder with already separated stems; skips separation")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], help=f"transcription device (default {DEVICE})")
    args = ap.parse_args()
    if args.device:
        DEVICE = args.device
    main(args.audio, args.out or Path("out") / Path(args.audio).stem, args.stems)


if __name__ == "__main__":
    cli()
