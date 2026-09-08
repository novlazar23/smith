#!/usr/bin/env bash
# Nächtlicher Evolutions-Zyklus:
#
#   1) Fährt `apps.evolution --cycle` im Backtest-Container (--build: Image
#      nach Code-Änderungen nie stumm veraltet). Der State bleibt auf dem
#      Named Volume `backtest_reports` (/app/backtest_reports/evolution)
#      und überlebt das --rm.
#   2) Kopiert den Auto-Export (State, promoted Mechanismus-Code,
#      Registry-Patch, Digest) vom Volume nach ./evolution auf dem Host —
#      die dauerhafte Heimat; Adoption in den Host-Tree ist eine bewusste
#      Host-Entscheidung (Gates: pyright/ruff/pytest).
#
# Aufruf (Cron/Timer):  ops/evolution-nightly.sh [--with-llm]
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose --profile on-demand run --rm --build backtest \
  python -m apps.evolution --cycle "$@"

# Kopie läuft als Root (bypasst den Privilege-Drop-Entrypoint), danach
# chown auf den Host-User — appuser (uid 999) könnte den Host-Ordner nicht
# beschreiben, und root-eigene Dateien wären im Repo unhandhabbar.
HOST_UID=$(id -u)
HOST_GID=$(id -g)
mkdir -p evolution
docker run --rm --entrypoint /bin/sh \
  -v smith_backtest_reports:/src:ro \
  -v "$(pwd)/evolution":/dst \
  smith-backtest:latest \
  -c "cp -a /src/evolution_export/. /dst/ && chown -R ${HOST_UID}:${HOST_GID} /dst"
echo "Evolutions-Export in ./evolution aktualisiert"
