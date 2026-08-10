# Architecture Notes

The initial implementation intentionally uses in-memory stores for agents and snapshots.
The next milestone is PostgreSQL persistence plus immutable run/audit records.

Live execution remains out of scope until shadow and paper evaluation are stable.
The future execution component should be deployed as a separate service with isolated credentials.
