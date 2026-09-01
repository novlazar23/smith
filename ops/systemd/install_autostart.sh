#!/usr/bin/env bash
# Idempotenter Installer für die Boot-Resilienz des Trading-Orchestra-Stacks.
#
# Installiert die systemd-Unit trading-orchestra.service nach /etc/systemd/system,
# lädt die Unit-Dateien neu und aktiviert + startet die Unit.
# Damit startet der Docker-Compose-Stack automatisch nach einem Host-Reboot.
#
# Aufruf:
#   bash install_autostart.sh           -> Unit installieren (idempotent)
#   bash install_autostart.sh --remove  -> Unit deaktivieren, stoppen und entfernen
#
# Hinweis: `docker compose up -d` ist idempotent — bereits laufende Container
# werden nicht neu gestartet oder gestört.

set -euo pipefail

# Pfad zu dieser Datei und zum Unit-File (unabhängig vom Aufrufverzeichniss)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_FILE="${SCRIPT_DIR}/trading-orchestra.service"
UNIT_NAME="trading-orchestra.service"
TARGET_PATH="/etc/systemd/system/${UNIT_NAME}"

# Fehlerfall: kein systemd vorhanden (z. B. Container ohne systemd)
if ! command -v systemctl >/dev/null 2>&1; then
    echo "FEHLER: systemctl nicht gefunden. Dieses Skript benötigt einen Host mit systemd." >&2
    exit 1
fi

if [[ ! -f "${UNIT_FILE}" ]]; then
    echo "FEHLER: Unit-Datei nicht gefunden: ${UNIT_FILE}" >&2
    exit 1
fi

# Kurzstatus zeigen (ohne Ausfallen, falls die Unit noch nicht existiert)
show_status() {
    echo "--- systemctl status trading-orchestra --no-pager (Auszug) ---"
    systemctl status trading-orchestra --no-pager 2>&1 | tail -n 12 || \
        echo "(Unit derzeit nicht vorhanden — das ist vor der Installation erwartet.)"
    echo "-----------------------------------------------------------------"
}

if [[ "${1:-}" == "--remove" ]]; then
    echo "Entferne trading-orchestra-Unit ..."
    # Deaktivieren ist ohne Fehler, falls die Unit nie aktiviert war
    sudo systemctl disable trading-orchestra 2>/dev/null || true
    # Stoppen (führt ExecStop = docker compose down aus), falls aktiv
    if systemctl is-active --quiet trading-orchestra 2>/dev/null; then
        sudo systemctl stop trading-orchestra
    fi
    sudo rm -f "${TARGET_PATH}"
    sudo systemctl daemon-reload
    show_status
    echo "Fertig. Unit wurde entfernt und deaktiviert."
    exit 0
fi

echo "Installiere trading-orchestra-Unit ..."
# Unit-Datei kopieren (idempotent: vorhandene Datei wird überschrieben)
sudo cp "${UNIT_FILE}" "${TARGET_PATH}"
sudo systemctl daemon-reload
# --now: aktivieren UND starten; idempotent, da `docker compose up -d` kein-
# op für bereits laufende Container ist
sudo systemctl enable --now trading-orchestra
show_status
echo "Fertig. Der Stack startet ab jetzt automatisch nach einem Host-Reboot."
