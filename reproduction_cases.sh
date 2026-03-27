ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d "saved_runs/figures_runs" ]; then
  echo "[error] Missing saved_runs/figures_runs. Unpack the Zenodo archive into the repo root."
  exit 2
fi

source .venv/bin/activate

python -m tools.reproduce_pictures \
  --figures-root "saved_runs/figures_runs" \
  --device "${DEVICE:-cpu}" \
  --download
