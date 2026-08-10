---
description: Prepare a complete Git-based handoff to another device or agent
agent: build
---

Run the full verification gate and inspect the complete worktree diff. Update @docs/handoff.md with
the achieved state, remaining work, exact verification commands and results, current branch, and any
important decisions or blockers. Ensure no secrets or local runtime data are included. Commit all
task-related files with a Conventional Commit message. Push only when the user explicitly requested
publication; never force-push. Finish with the branch, commit SHA, verification result, and precise
resume command for the next device.
