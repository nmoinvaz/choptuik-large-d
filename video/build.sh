#!/bin/bash
# Render the scenes of quest_large_d.py and join them into dist/quest_large_d.mp4.
#   video/build.sh [l|m|h] [Scene ...]     l: 854x480 at 15 fps (default), m: 1280x720 at 30, h: 1920x1080 at 30
# Needs uv, tectonic and ffmpeg. The Python environment is made in video/.venv on the first run.
{
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"
shift || true
case "$Q" in
  l) RES="854,480"; FPS=15 ;;
  m) RES="1280,720"; FPS=30 ;;
  h) RES="1920,1080"; FPS=30 ;;
  *) echo "quality is l, m or h"; exit 2 ;;
esac
[ -x .venv/bin/manim ] || { uv venv -q --python 3.13 .venv && uv pip install -q --python .venv/bin/python manim pymupdf; }
export PATH="$PWD/tools:$PWD/.venv/bin:$PATH"
ALL=$(python -c "import re; print(' '.join(re.findall(r'^class (S\d\d_\w+)', open('quest_large_d.py').read(), re.M)))")
SCENES="${*:-$ALL}"
for s in $SCENES; do
  nice -n 10 manim render -v CRITICAL --progress_bar none -r "$RES" --fps "$FPS" --media_dir media -o "$s" quest_large_d.py "$s"
done
DIR="media/videos/quest_large_d/${RES#*,}p$FPS"
mkdir -p dist
: > dist/list_$Q.txt
for s in $ALL; do
  [ -f "$DIR/$s.mp4" ] && echo "file '../$DIR/$s.mp4'" >> dist/list_$Q.txt
done
OUT="dist/quest_large_d.mp4"
[ "$Q" = h ] || OUT="dist/quest_large_d_$Q.mp4"
ffmpeg -loglevel error -y -f concat -safe 0 -i dist/list_$Q.txt -c copy "$OUT"
echo "wrote video/$OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -d. -f1) s)"
exit
}
