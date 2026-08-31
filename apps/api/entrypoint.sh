#!/bin/sh
set -e
if [ -f /run/secrets/postgres_password ]; then
    DB_PASSWORD="$(cat /run/secrets/postgres_password)"
    export DB_PASSWORD
fi
if [ -f /run/secrets/clickhouse_password ]; then
    CH_PASSWORD="$(cat /run/secrets/clickhouse_password)"
    export CH_PASSWORD
fi
exec "$@"
