#!/bin/bash
# Usage: GPU=<n> bash demo/start.sh [--tunnel | --funnel]
# Starts demo/server.py in the background on 127.0.0.1:8600; the search is
# open to anyone with the link (rate-limited per client IP).
# With --tunnel also starts a Cloudflare quick tunnel and prints the public https://<random>.trycloudflare.com link
# (a new link on every tunnel start). Stop both: bash demo/stop.sh
# With --funnel instead serves it through Tailscale Funnel at the fixed https://t2xml-demo.<tailnet>.ts.net address
# (same link across restarts and network dropouts; viewers need no account). Needs the one-time `tailscale up` login
# and Funnel enabled for the tailnet; tailscaled runs in userspace mode because the container has no /dev/net/tun.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANTA"
L=demo/logs; mkdir -p $L
CLOUDFLARED=${CLOUDFLARED:-/root/.local/bin/cloudflared}
TS_STATE=${TS_STATE:-/root/.local/state/tailscale}
TS="/root/.local/bin/tailscale --socket=$TS_STATE/tailscaled.sock"

if pgrep -f "[p]ython demo/server[.]py" > /dev/null; then
  echo "server already running"
else
  CUDA_VISIBLE_DEVICES=${GPU:-1} \
    nohup .venv/bin/python demo/server.py --port 8600 > $L/server.log 2>&1 &
  echo "server starting (log: $L/server.log)"
  curl -s --retry 60 --retry-connrefused --retry-delay 3 -o /dev/null http://127.0.0.1:8600/ && echo "server up"
fi

if [[ "$1" == --tunnel ]]; then
  if pgrep -f "[c]loudflared tunnel --no-autoupdate" > /dev/null; then
    echo "tunnel already running"
  else
    nohup $CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:8600 > $L/tunnel.log 2>&1 &
    for _ in $(seq 60); do
      url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L/tunnel.log | head -1)
      [[ -n "$url" ]] && break
      read -t 1 < /dev/zero 2>/dev/null || true
    done
  fi
  url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L/tunnel.log | head -1)
  echo "public link: ${url:-not found yet, see $L/tunnel.log}"
fi

if [[ "$1" == --funnel ]]; then
  if ! pgrep -f "[t]ailscaled --tun=userspace-networking" > /dev/null; then
    mkdir -p $TS_STATE
    nohup /root/.local/bin/tailscaled --tun=userspace-networking --statedir=$TS_STATE --socket=$TS_STATE/tailscaled.sock \
      > $L/tailscaled.log 2>&1 &
    for _ in $(seq 30); do $TS status > /dev/null 2>&1 && break; read -t 1 < /dev/zero 2>/dev/null || true; done
  fi
  $TS funnel --bg 8600 > /dev/null || { echo "funnel failed: run '$TS funnel --bg 8600' to see why"; exit 1; }
  host=$($TS status --json | grep -o '"DNSName": "[^"]*' | head -1 | cut -d'"' -f4)
  echo "public link: https://${host%.}"
fi
