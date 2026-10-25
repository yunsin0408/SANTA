#!/bin/bash
# Usage: bash demo/stop.sh [server]
# Stops the search demo server and the Cloudflare tunnel / Tailscale Funnel started by demo/start.sh. With "server",
# stops only the server and keeps the tunnel (and so the public link) alive; restart with demo/start.sh (no flag).
# Stopping the Funnel only turns off public serving; tailscaled keeps running so the fixed address stays reserved.
if [[ "$1" != server ]]; then
  pkill -f "[c]loudflared tunnel --no-autoupdate" && echo "tunnel stopped"
  TS="/root/.local/bin/tailscale --socket=${TS_STATE:-/root/.local/state/tailscale}/tailscaled.sock"
  $TS funnel status 2>/dev/null | grep -q "Funnel on" && $TS funnel --https=443 off > /dev/null && echo "funnel stopped"
fi
pkill -f "[p]ython demo/server[.]py" && echo "server stopped"
for _ in $(seq 100); do  # wait until it has really exited, so an immediate start.sh doesn't see it as running
  pgrep -f "[p]ython demo/server[.]py" > /dev/null || break
  read -t 0.2 < /dev/zero 2>/dev/null || true
done
true
