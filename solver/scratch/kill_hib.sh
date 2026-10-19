#!/bin/sh
for p in $(pgrep -f "scan_ne6[b].jsonl"); do kill $p; done
