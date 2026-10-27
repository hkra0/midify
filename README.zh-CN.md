# midify

[English](README.md)

把一首歌的录音转换成可以编辑的 MIDI：人声旋律、钢琴伴奏、弦乐等持续音声部、贝斯和鼓，同时输出和弦表和可以在 MuseScore 中打开的 MusicXML 旋律谱。

它的定位是**扒谱的初稿**：旋律和钢琴通常比较接近原曲，但仍需要手动修正部分音符，尤其是编配密集或混响很重的段落。

## 工作原理

1. **分离**：用 [BS-Roformer SW](https://github.com/nomadkaraoke/python-audio-separator) 把混音拆成六条分轨：人声、鼓、贝斯、吉他、钢琴、其他。
2. **节拍与调性**：根据鼓和伴奏的起音检测节拍，用精细起音校准网格，并选择和声变化最多的位置作为强拍。
3. **分轨体检**：每条分轨使用前先检查，结果写入 `stem_report.txt`。几乎无声的分轨会被丢弃；单声部分轨如果测不出稳定的音高，也会被丢弃；一条分轨的音如果在另一条分轨里同一时刻、同一音高也出现了，就判定为漏音并合并过去。
4. **转录**：每条分轨用适合它的模型：
   - 人声、贝斯 → [CREPE](https://github.com/maxrjones/torchcrepe) 音高跟踪，按实际起音切分成音符，并过滤混响尾巴
   - 钢琴、吉他 → [钢琴转录模型](https://github.com/qiuqiangkong/piano_transcription_inference)（Kong 等），合并成一个钢琴声部
   - 其他（弦乐、铺底音色、管乐）→ [Basic Pitch](https://github.com/spotify/basic-pitch)，再去掉泛音、贝斯漏音和不属于当前和弦的短音
   - 鼓 → 按频段区分底鼓、军鼓、踩镲
5. **和弦**：结合转录出的音符和原音频，逐拍识别，并用考虑调性的 HMM 平滑。
6. **导出**：MIDI（原始时间版和对齐节拍版）、MusicXML、和弦表，以及去掉人声的伴奏音频。

## 安装

推荐使用 Python 3.11，有几个音频和机器学习包暂时还不支持更新的版本。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

模型权重总共约 0.9 GB，首次运行时会自动下载到 `models/`。

## 使用

```bash
.venv/bin/python midify.py song.mp3                 # 结果在 out/song/
.venv/bin/python midify.py song.mp3 -o my_output    # 指定输出目录
.venv/bin/python midify.py song.mp3 --stems DIR     # 复用之前分离好的分轨
```

在笔记本上，一首歌第一次处理需要一段时间：在 Apple M1 上，分离用 GPU 约 10 分钟，转录用 CPU 约 20 分钟。耗时步骤的结果会缓存到 `cache/`，之后改代码或改设置再重跑，通常不到一分钟。

## 输出（`out/<歌名>/`）

| 文件 | 内容 |
|---|---|
| `song_realtime.mid` | 所有声部，时间和原录音一致，可以跟原曲同步播放 |
| `song_grid.mid` | 所有声部，对齐到节拍网格、固定速度，适合打谱 |
| `accompaniment_*.mid` | 同上，但不含旋律 |
| `part_*.mid` | 每个声部一个文件（旋律、贝斯、钢琴、其他、鼓、和弦块） |
| `leadsheet.musicxml` | 带和弦标记的旋律谱 |
| `chords.txt` | 带时间的和弦列表，以及每行四小节的和弦表 |
| `instrumental.wav` | 去掉人声的原曲音频 |
| `stem_report.txt` | 分轨体检的各项指标和判断结果 |
| `stems/` | 分离出的分轨 |

## 设置

在 `midify.py` 开头：

- `OVERRIDES`：手动修正分轨体检的结果，例如 `{"bass": "drop"}`，或 `{"guitar": "keep"}`（吉他单独保留一轨，不并入钢琴）。
- `SUSTAINED`：哪些分轨是持续音乐器，这些分轨用 Basic Pitch 转录。
- `--device` / `MIDIFY_DEVICE`：转录用的设备，默认是 CPU。在某些 macOS 和 PyTorch 版本组合下，Apple GPU（MPS）会在运行中途卡死。

## 局限

- 默认是 4/4 拍，速度基本稳定。
- 分轨之间会互相漏音；混响很重时，弦乐这类持续音声部最不准确。
- 鼓的转录比较粗略：只分三种声音，没有通鼓和镲片。
- 和弦只识别大三、小三、属七、大七、小七和挂四。

## 版权

midify 的代码采用 MIT 许可。你处理的录音，以及由它生成的分轨、MIDI 和乐谱，仍然受原作品的版权约束。请只用于你有权转录的音乐，未经许可不要公开发布受版权保护歌曲的转录结果。`.gitignore` 已经把音频、分轨和输出排除在仓库之外。

## 致谢

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（MIT），以及 jarredou 训练的 BS-Roformer SW 模型
- [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference)（MIT）：Kong 等，*High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times*
- Spotify 的 [Basic Pitch](https://github.com/spotify/basic-pitch)
- [torchcrepe](https://github.com/maxrjones/torchcrepe)（MIT）、[librosa](https://librosa.org)（ISC）、[pretty_midi](https://github.com/craffel/pretty-midi)（MIT）、[music21](https://web.mit.edu/music21/)（BSD）

模型权重从各自作者发布的地址下载，适用其各自的许可条款。
