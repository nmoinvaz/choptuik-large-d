#!/bin/sh
for p in $(pgrep -f "python3 scan_n[e].py"); do kill $p; done
