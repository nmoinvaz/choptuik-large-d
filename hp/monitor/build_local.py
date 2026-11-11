"""The chain monitor as one local HTML file: the dashboard's page (index.html) with its data embedded and a small
stand-in for the dashboard runtime (dash.data, dash.calc, dash.draw), so it opens in a browser without signing in.
It reloads itself every 5 minutes, and the coordinator rewrites it after each refresh of the tables.

    cd hp && python3 chain_monitor.py RAWDIR OUT && python3 monitor/build_local.py OUT ~/Source/chain-monitor.html

MONITOR_DIR holds steps.csv and errorbar.csv (chain_monitor.py); the page is monitor/page.html beside this script.
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

d, out = sys.argv[1], sys.argv[2]
rows = {k: list(csv.DictReader(open(os.path.join(d, f"{k}.csv")))) for k in ("steps", "errorbar")}
page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")).read().replace('<meta name="dash-one-writer">', "")
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Large-D chain monitor</title>
<style>
:root {{ --color-fg:#1d1f22; --color-fg-muted:#5d636b; --color-bg:#fbfbfa; --color-panel:#ffffff; --color-border-line:#e3e4e6;
  --color-ok:#1a8a5a; --color-warn:#b8860b; --color-bad:#c2410c;
  --font-anthropic-sans: Inter, system-ui, -apple-system, sans-serif; --font-anthropic-serif: Georgia, "Times New Roman", serif;
  --cds-font-size-caption:12px; --cds-font-size-body:14px; --cds-font-size-heading:17px; --cds-font-size-title:30px; --cds-font-weight-medium:600;
  --cds-gap-xs:4px; --cds-gap-sm:8px; --cds-gap-md:14px; --cds-gap-lg:22px; --cds-gap-xl:34px; --cds-pad-lg:16px; --cds-radius:10px;
  --cds-chart-muted:#9aa0a8; --cds-chart-status-critical:#c2410c; --cds-chart-grid:#ececee; --cds-chart-axis:#c9ccd1; --cds-text-secondary:#5d636b;
  --cds-surface-popover:#ffffff; --cds-border:#d9dbde; --cds-shadow-popover:0 4px 14px rgba(0,0,0,.12); color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root {{ --color-fg:#e8e9eb; --color-fg-muted:#a3a8b0; --color-bg:#16181b; --color-panel:#1e2125;
  --color-border-line:#2e3238; --color-ok:#4cc38a; --color-warn:#e0b341; --color-bad:#f07a4a; --cds-chart-muted:#6f7680; --cds-chart-grid:#26292e;
  --cds-chart-axis:#3c4148; --cds-text-secondary:#a3a8b0; --cds-surface-popover:#23262b; --cds-border:#3a3f46; --cds-chart-status-critical:#f07a4a;
  color-scheme: dark; }} }}
body {{ margin:0; background:var(--color-bg); color:var(--color-fg); font: 14px var(--font-anthropic-sans); }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px max(16px, env(safe-area-inset-left, 0px)) 48px max(16px, env(safe-area-inset-right, 0px)); }}
.stamp {{ color: var(--color-fg-muted); font-size: 12px; margin-bottom: 14px; }}
.dash-select {{ font: inherit; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--color-border-line); background-color: var(--color-panel); color: var(--color-fg); }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
window.DATA = {json.dumps(rows)};
window.dash = (() => {{
  const calcs = {{}}, draws = {{}};
  const api = {{
    colors: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
    calc(id, o) {{ calcs[id] = o; }},
    data(id) {{
      if (window.DATA[id]) return {{ status: 'ok', data: window.DATA[id] }};
      const c = calcs[id];
      if (!c) return {{ status: 'missing', data: null }};
      try {{ return {{ status: 'ok', data: c.fn(...c.inputs.map((i) => api.data(i).data)) }}; }}
      catch (e) {{ console.error(e); return {{ status: 'error', data: null }}; }}
    }},
    draw(name, fn) {{
      if (fn) {{ draws[name] = fn; return; }}
      document.querySelectorAll(`[data-draw="${{name}}"]`).forEach((el) => draws[name](el));
    }},
    run() {{ for (const name of Object.keys(draws)) api.draw(name); }},
  }};
  return api;
}})();
</script>
</head><body><div class="wrap"><div class="stamp">Tables rebuilt {stamp} from the runners' logs. This page reloads every 5 minutes.</div>
"""
tail = """
<script>
  dash.run();
  let w = window.innerWidth;
  window.addEventListener('resize', () => { if (Math.abs(window.innerWidth - w) > 20) { w = window.innerWidth; dash.run(); } });
</script>
</div></body></html>
"""
open(out, "w").write(head + page + tail)
print("wrote", out, os.path.getsize(out), "bytes")
