#!/bin/sh
# Liest Docker Secrets als Root, degradiert dann auf den nicht-privilegierten App-User.
set -e
if [ -f /run/secrets/postgres_password ]; then
    DB_PASSWORD="$(cat /run/secrets/postgres_password)"
    export DB_PASSWORD
fi
if [ -f /run/secrets/clickhouse_password ]; then
    CH_PASSWORD="$(cat /run/secrets/clickhouse_password)"
    export CH_PASSWORD
fi
if [ -f /run/secrets/minio_password ]; then
    MINIO_SECRET_KEY="$(cat /run/secrets/minio_password)"
    export MINIO_SECRET_KEY
fi
if [ "$(id -u)" = "0" ] && id appuser >/dev/null 2>&1; then
    exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"
fi
exec "$@"
