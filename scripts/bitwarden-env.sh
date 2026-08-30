#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ITEM_NAME="${SMITH_BITWARDEN_ITEM:-Smith Autonomous Service Environment}"

usage() {
  cat <<'EOF'
Usage: scripts/bitwarden-env.sh <pull|push> [--force]

Synchronize .env with the Bitwarden secure note named
"Smith Autonomous Service Environment". Authentication is read from
BW_SESSION or the local, git-ignored .bw-session file.

  pull          Restore .env; refuses to overwrite an existing file
  pull --force  Restore and atomically replace an existing .env
  push          Create or update the secure note from the local .env
EOF
}

require_tools() {
  command -v bw >/dev/null || { echo "Bitwarden CLI (bw) is required" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
}

load_session() {
  if [[ -z "${BW_SESSION:-}" && -f "$REPOSITORY_ROOT/.bw-session" ]]; then
    if [[ "$(stat -c '%a' "$REPOSITORY_ROOT/.bw-session")" != "600" ]]; then
      echo ".bw-session must have mode 600" >&2
      exit 1
    fi
    BW_SESSION="$(<"$REPOSITORY_ROOT/.bw-session")"
    export BW_SESSION
  fi
  [[ -n "${BW_SESSION:-}" ]] || {
    echo "Unlock Bitwarden and set BW_SESSION or create mode-600 .bw-session" >&2
    exit 1
  }
  [[ "$(bw status | jq -r '.status')" == "unlocked" ]] || {
    echo "Bitwarden vault is not unlocked" >&2
    exit 1
  }
}

item_id() {
  bw list items --search "$ITEM_NAME" |
    jq -r --arg name "$ITEM_NAME" '[.[] | select(.name == $name)] | if length == 1 then .[0].id elif length == 0 then empty else error("duplicate Bitwarden items") end'
}

pull_env() (
  local force="${1:-}"
  local id tmp
  [[ ! -e "$REPOSITORY_ROOT/.env" || "$force" == "--force" ]] || {
    echo ".env already exists; use pull --force to replace it" >&2
    exit 1
  }
  id="$(item_id)"
  [[ -n "$id" ]] || { echo "Bitwarden item not found: $ITEM_NAME" >&2; exit 1; }
  tmp="$(mktemp "$REPOSITORY_ROOT/.env.XXXXXX")"
  trap 'rm -f -- "$tmp"' EXIT
  chmod 600 "$tmp"
  bw get item "$id" | jq -jer '.notes | select(type == "string" and length > 0)' >"$tmp"
  grep -q '^LIVE_EXECUTION_ENABLED=' "$tmp" || {
    echo "Refusing invalid environment payload" >&2
    exit 1
  }
  mv -- "$tmp" "$REPOSITORY_ROOT/.env"
  trap - EXIT
  echo "Restored .env from Bitwarden"
)

push_env() (
  local id payload
  [[ -f "$REPOSITORY_ROOT/.env" ]] || { echo ".env does not exist" >&2; exit 1; }
  payload="$(mktemp)"
  trap 'rm -f -- "$payload"' EXIT
  chmod 600 "$payload"
  id="$(item_id)"
  if [[ -n "$id" ]]; then
    bw get item "$id" |
      jq --rawfile notes "$REPOSITORY_ROOT/.env" '.notes = $notes' >"$payload"
    bw encode <"$payload" | bw edit item "$id" >/dev/null
    echo "Updated Bitwarden environment item"
  else
    bw get template item |
      jq --rawfile notes "$REPOSITORY_ROOT/.env" --arg name "$ITEM_NAME" \
        '.type = 2 | .name = $name | .notes = $notes | .secureNote = {"type": 0}' >"$payload"
    bw encode <"$payload" | bw create item >/dev/null
    echo "Created Bitwarden environment item"
  fi
  bw sync >/dev/null
)

main() {
  cd "$REPOSITORY_ROOT"
  case "${1:---help}" in
    --help|-h) usage ;;
    pull)
      [[ "${2:-}" == "" || "${2:-}" == "--force" ]] || { usage >&2; exit 2; }
      require_tools
      load_session
      pull_env "${2:-}"
      ;;
    push)
      [[ $# -eq 1 ]] || { usage >&2; exit 2; }
      require_tools
      load_session
      push_env
      ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
