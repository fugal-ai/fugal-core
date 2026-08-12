#!/usr/bin/env bash
# One-time (idempotent) setup of Fugal on an Ubuntu aarch64 box.
# Installs deps, clones the repo to /opt/fugal-core under a dedicated user, creates a CPU
# venv, fetches the backbone, and smoke-tests the router forward pass.
# Run as root: sudo bash deploy/setup_arm.sh
set -euo pipefail

REPO_URL="https://github.com/jtdoherty/fugal-core.git"
DIR=/opt/fugal-core
SVC_USER=fugal

echo "== apt deps =="
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git curl

echo "== service user + checkout =="
id -u $SVC_USER >/dev/null 2>&1 || useradd -r -m -d /var/lib/$SVC_USER -s /usr/sbin/nologin $SVC_USER
if [ -d $DIR/.git ]; then
    git -C $DIR pull --ff-only
else
    git clone "$REPO_URL" $DIR
fi
chown -R $SVC_USER:$SVC_USER $DIR

echo "== python venv (CPU torch) =="
if [ ! -x $DIR/.venv/bin/python ]; then
    sudo -u $SVC_USER python3 -m venv $DIR/.venv
fi
sudo -u $SVC_USER $DIR/.venv/bin/pip install -q --upgrade pip
# CPU-only torch wheel (aarch64 manylinux wheels live on the cpu index)
sudo -u $SVC_USER $DIR/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
sudo -u $SVC_USER $DIR/.venv/bin/pip install -q transformers numpy requests huggingface_hub

echo "== backbone (Qwen3-0.6B, ~1.5 GB) =="
sudo -u $SVC_USER $DIR/.venv/bin/python $DIR/scripts/fetch_backbone.py

echo "== router smoke test (local forward pass, \$0, no API) =="
cd $DIR
sudo -u $SVC_USER $DIR/.venv/bin/python - <<'PY'
import time
t0 = time.time()
from fugal import Fugal
f2 = Fugal()
print(f"  model load: {time.time()-t0:.1f}s")
for q in ["What is 17*23?", "Write a python function that reverses a linked list."]:
    t0 = time.time()
    ranked, probs = f2.route(q)
    print(f"  forward pass {time.time()-t0:.2f}s -> {ranked[0]} (p_solve={probs[0]:.2f})")
print("ROUTER OK")
PY

echo
echo "Done. Next steps (deploy/README.md): /etc/fugal.env secrets -> systemd unit -> Caddy."
