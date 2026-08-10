---
description: Run and repair the complete repository verification gate
agent: build
---

Run `./scripts/bootstrap.sh --check`. Diagnose every failure from its actual output, fix it with the
smallest safe change, and rerun the complete gate until it passes. For behavior defects, add a
failing regression test before changing production code. Do not weaken or skip checks. Summarize the
final test, lint, and type-check results.
