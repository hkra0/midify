# midify

[中文说明](README.zh-CN.md)

Turn a song recording into editable MIDI: the sung melody, piano/accompaniment, held parts
such as strings, bass and drums — plus a chord chart and a MusicXML lead sheet you can open
in MuseScore.

It is meant as a **first draft for transcribing by ear**: melody and piano usually come out
close; expect to fix some notes by hand, especially in dense or reverb-heavy passages.

## How it works

1. **Separate** the mix into six stems (vocals, drums, bass, guitar, piano, other) with
   [BS-Roformer SW](https://github.com/nomadkaraoke/python-audio-separator).
2. **Beats & key** from the drum and accompaniment onsets; the beat grid is calibrated against
   fine onsets and the downbeat is chosen where the harmony changes most.
3. **Audit each stem** before using it (written to `stem_report.txt`): silent stems and
   single-line stems without a stable pitch are dropped; a stem whose notes another stem
   already plays (same pitch, same moment) is merged into it.
4. **Transcribe** each kept stem with a model that fits it:
   - vocals, bass → [CREPE](https://github.com/maxrjones/torchcrepe) pitch tracking, cut into
     notes at the sung onsets, with reverb tails gated out
   - piano, guitar → [piano transcription](https://github.com/qiuqiangkong/piano_transcription_inference)
     (Kong et al.), combined into one piano part
   - other (strings, pads, winds) → [Basic Pitch](https://github.com/spotify/basic-pitch),
     then overtones, bass leakage and short non-chord notes are removed
   - drums → kick / snare / hi-hat by frequency band
5. **Chords** per beat from the transcribed notes and the audio, decoded with a key-aware HMM.
6. **Export** MIDI (original timing and beat-quantized), MusicXML, a chord chart, and an
   instrumental audio track.

## Install

Python 3.11 is recommended (several audio/ML packages do not support newer versions yet).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights (~0.9 GB in total) are downloaded into `models/` on first use.

## Use

```bash
.venv/bin/python midify.py song.mp3                 # results in out/song/
.venv/bin/python midify.py song.mp3 -o my_output    # pick the output folder
.venv/bin/python midify.py song.mp3 --stems DIR     # reuse stems separated earlier
```

The first run of a song takes a while on a laptop (separation ~10 min on an Apple M1 GPU,
transcription ~20 min on CPU). Slow stages are cached in `cache/`, so re-running after a
code or setting change takes well under a minute.

## Output (`out/<song>/`)

| File | Contents |
|---|---|
| `song_realtime.mid` | all parts in the recording's own timing — plays along with the audio |
| `song_grid.mid` | all parts quantized to the beat grid at a fixed tempo — for notation |
| `accompaniment_*.mid` | the same without the melody |
| `part_*.mid` | one file per part (melody, bass, piano, other, drums, block chords) |
| `leadsheet.musicxml` | melody with chord symbols |
| `chords.txt` | chord list with times, and a four-bars-per-line chart |
| `instrumental.wav` | the original audio without vocals |
| `stem_report.txt` | what the stem audit measured and decided |
| `stems/` | the separated stems |

## Settings

At the top of `midify.py`:

- `OVERRIDES` — correct the stem audit, e.g. `{"bass": "drop"}` or `{"guitar": "keep"}`
  (keep guitar as its own part instead of merging it into piano).
- `SUSTAINED` — which stems hold notes (transcribed with Basic Pitch).
- `--device` / `MIDIFY_DEVICE` — transcription device. The default is CPU: on some
  macOS/PyTorch versions the Apple GPU (MPS) backend has hung mid-run.

## Limitations

- Assumes 4/4 and a mostly steady tempo.
- Separation leaks between stems; heavy reverb makes held parts (strings) the least accurate.
- Drums are coarse: three sounds, no toms or cymbals.
- Chord names cover major, minor, 7, maj7, m7 and sus4.

## Copyright

midify's code is MIT-licensed. The recordings you process, and the stems, MIDI and scores
derived from them, remain subject to the rights of the original work — use it with music you
have the right to transcribe, and do not publish the results of copyrighted songs without
permission. The `.gitignore` keeps audio, stems and outputs out of the repository.

## Credits

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT) and
  the BS-Roformer SW model by jarredou
- [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference)
  (MIT) — Kong et al., *High-resolution Piano Transcription with Pedals by Regressing Onset and
  Offset Times*
- [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify
- [torchcrepe](https://github.com/maxrjones/torchcrepe) (MIT), [librosa](https://librosa.org) (ISC),
  [pretty_midi](https://github.com/craffel/pretty-midi) (MIT), [music21](https://web.mit.edu/music21/) (BSD)

Model weights are downloaded from their authors' sources and keep their own licenses.
