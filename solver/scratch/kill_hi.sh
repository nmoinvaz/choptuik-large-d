#!/bin/sh
for p in $(pgrep -f "scan_ne[6].jsonl 19"); do kill $p; done
