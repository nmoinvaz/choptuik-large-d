for p in $(pgrep -f "python3 scan_long.py"); do kill $p; done
