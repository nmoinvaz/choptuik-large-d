#!/bin/bash
# Render the scenes of quest_large_d.py and join them, with the voice's sound track, into dist/quest_large_d.mp4.
#   video/build.sh [l|m|h] [Scene ...]     l: 854x480 at 15 fps (default), m: 1280x720 at 30, h: 1920x1080 at 30
#   video/build.sh s [Scene ...]           only list what the voice will say, to read it over (no video, no sound)
# Needs uv, tectonic, ffmpeg and curl. The Python environment is made in video/.venv on the first run, and the
# voice's model (Kokoro, 340 MB) is fetched into media/voice/model and checked against voice_model.sha256.
# QUEST_CPS sets the reading pace of the captions in characters a second (12 by default; a lower number holds them longer).
# QUEST_VOICE names the speaker (am_michael by default, a man's voice; "off" builds the film without sound).
{
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"
shift || true
case "$Q" in
  l|s) RES="854,480"; FPS=15 ;;
  m) RES="1280,720"; FPS=30 ;;
  h) RES="1920,1080"; FPS=30 ;;
  *) echo "quality is l, m or h, or s for the voice's lines"; exit 2 ;;
esac
VOICE="${QUEST_VOICE:-am_michael}"
[ -x .venv/bin/manim ] || { uv venv -q --python 3.13 .venv && uv pip install -q --python .venv/bin/python manim pymupdf; }
.venv/bin/python -c "import kokoro_onnx" 2>/dev/null || uv pip install -q --python .venv/bin/python kokoro-onnx
export PATH="$PWD/tools:$PWD/.venv/bin:$PATH"
ALL=$(python -c "import re; print(' '.join(re.findall(r'^class (S\d\d[a-z]?_\w+)', open('quest_large_d.py').read(), re.M)))")
SCENES="${*:-$ALL}"
if [ "$Q" = s ]; then
  for s in $SCENES; do
    QUEST_LINES=1 nice -n 10 python -I -m manim render -s -v CRITICAL --progress_bar none -r "$RES" --fps "$FPS" --media_dir media quest_large_d.py "$s"
  done
  python -I voice.py lines media/voice/lines
  exit
fi
if [ "$VOICE" != off ]; then
  mkdir -p media/voice/model
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    [ -f "media/voice/model/$f" ] || curl -sSL --fail -o "media/voice/model/$f" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
  done
  (cd media/voice/model && shasum -a 256 -c --quiet ../../../voice_model.sha256)
fi
for s in $SCENES; do      # caching off: a cached animation's length is not counted in frames, and the voice needs the frame count
  nice -n 10 python -I -m manim render --disable_caching -v CRITICAL --progress_bar none -r "$RES" --fps "$FPS" --media_dir media -o "$s" quest_large_d.py "$s"
done
DIR="media/videos/quest_large_d/${RES#*,}p$FPS"
mkdir -p dist
: > dist/list_$Q.txt
FOUND=""
for s in $ALL; do
  [ -f "$DIR/$s.mp4" ] && { echo "file '../$DIR/$s.mp4'" >> dist/list_$Q.txt; FOUND="$FOUND $s"; }
done
OUT="dist/quest_large_d.mp4"
[ "$Q" = h ] || OUT="dist/quest_large_d_$Q.mp4"
if [ "$VOICE" = off ]; then
  ffmpeg -loglevel error -y -f concat -safe 0 -i dist/list_$Q.txt -c copy "$OUT"
else
  python -I voice.py track "media/voice/${RES#*,}p$FPS" "$DIR" "dist/voice_$Q.wav" $FOUND
  ffmpeg -loglevel error -y -f concat -safe 0 -i dist/list_$Q.txt -i "dist/voice_$Q.wav" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 128k -ar 48000 -ac 2 -movflags +faststart "$OUT"
fi
echo "wrote video/$OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -d. -f1) s)"
exit
}
