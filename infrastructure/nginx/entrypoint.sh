#!/bin/sh
# =============================================================================
# Nginx + Certbot — Certificate Sync Entry Point
#
# 1. Wenn ein certbot-Zertifikat existiert → symlink nach ssl/live/
# 2. Ansonsten → self-signed fallback nutzen
# 3. Nginx starten
# =============================================================================

set -e

DOMAIN="${NGINX_DOMAIN:-trading-orchestra.internal}"
CERT_DIR="/etc/letsencrypt/live/${DOMAIN}"
NGINX_SSL_DIR="/etc/nginx/ssl"
NGINX_SSL_LIVE="${NGINX_SSL_DIR}/live/${DOMAIN}"

echo "[entrypoint] Checking for Let's Encrypt certificate..."

# --- Certbot sync ---
if [ -d "$CERT_DIR" ] && [ -f "${CERT_DIR}/fullchain.pem" ] && [ -f "${CERT_DIR}/privkey.pem" ]; then
    echo "[entrypoint] Certbot certificate found — creating symlink."
    mkdir -p "$NGINX_SSL_LIVE"
    # Symlink current cert files (certbot manages these under /etc/letsencrypt/)
    ln -sf "${CERT_DIR}/fullchain.pem" "${NGINX_SSL_LIVE}/fullchain.pem"
    ln -sf "${CERT_DIR}/privkey.pem" "${NGINX_SSL_LIVE}/privkey.pem"
else
    echo "[entrypoint] No certbot certificate — using self-signed fallback."
    mkdir -p "$NGINX_SSL_LIVE"
    if [ ! -f "${NGINX_SSL_LIVE}/fullchain.pem" ]; then
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "${NGINX_SSL_LIVE}/privkey.pem" \
            -out "${NGINX_SSL_LIVE}/fullchain.pem" \
            -days 365 -subj "/CN=${DOMAIN}" 2>/dev/null
        echo "[entrypoint] Self-signed certificate generated for ${DOMAIN}."
    fi
fi

echo "[entrypoint] Starting Nginx..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Ref-Datei für BusyBox-kompatiblen Cert-Change-Detect (keine GNU stat)
touch /tmp/.nginx_cert_ref

# Background cert watcher — reload nginx after cert renewal
watch_certs() {
    NGINX_SSL_LIVE="${NGINX_SSL_DIR}/live/${DOMAIN}"
    while true; do
        sleep 300
        if [ -f "${NGINX_SSL_LIVE}/fullchain.pem" ]; then
            if find "${NGINX_SSL_LIVE}/fullchain.pem" -newer /tmp/.nginx_cert_ref -print | grep -q .; then
                echo "[watcher] Certificate changed — reloading nginx."
                nginx -s reload 2>/dev/null || true
                touch /tmp/.nginx_cert_ref
            fi
        fi
    done
}
watch_certs &
wait $NGINX_PID