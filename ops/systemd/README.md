# Boot-Resilienz für den Trading-Orchestra-Stack

## Was macht das?

Der Trading-Orchestra-Stack (13 Docker-Compose-Services) soll **nach einem
Host-Reboot automatisch wieder hochkommen**, ohne dass manuell
`docker compose up -d` ausgeführt werden muss.

Dazu installiert `install_autostart.sh` die systemd-Unit
`trading-orchestra.service` nach `/etc/systemd/system/` und aktiviert sie in
`multi-user.target`. Beim Boot wartet die Unit auf `docker.service` und führt
dann `docker compose up -d` im Projektverzeichnis `/home/til/develop/smith`
aus.

## Warum ist das nötig?

- `restart: unless-stopped` in der `docker-compose.yml` deckt nur **Docker-
  Daemon-Neustarts** ab (z. B. `systemctl restart docker` oder Daemon-Crash).
- Bei einem **vollständigen Host-Reboot** ist das aber nicht der Fall, wenn der
  Docker-Daemon bzw. der Stack beim Herunterfahren bereits gestoppt war:
  Der Daemon startet zwar, aber niemand führt `compose up` aus — die
  Container blieben im Stopp-Zustand.
- Die systemd-Unit schließt genau diese Lücke: Nach jedem Reboot startet der
  Stack automatisch mit.

## Installation

```bash
sudo bash /home/til/develop/smith/ops/systemd/install_autostart.sh
```

Das Skript ist **idempotent**: `docker compose up -d` startet bereits laufende
Container nicht neu, eine erneute Ausführung stört also den laufenden Stack
nicht. Am Ende wird der Zustand der Unit (`systemctl status`) angezeigt.

Nach der Installation prüfen:

```bash
systemctl is-enabled trading-orchestra   # -> enabled
systemctl is-active  trading-orchestra   # -> active
```

## Deinstallation / Entfernen

```bash
sudo bash /home/til/develop/smith/ops/systemd/install_autostart.sh --remove
```

Das Skript deaktiviert die Unit (`systemctl disable`), stoppt sie
(`systemctl stop` → führt `docker compose down` aus) und entfernt die
Unit-Datei inkl. `daemon-reload`. **Achtung:** Damit wird der laufende Stack
stopp. Vorher sicherstellen, dass keine Daten in Arbeit sind.

## Dateien

| Datei | Zweck |
|---|---|
| `trading-orchestra.service` | systemd-Unit (oneshot, `RemainAfterExit=yes`) |
| `install_autostart.sh` | Idempotenter Installer (`--remove` für Entfernung) |
