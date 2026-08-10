---
description: Reconstruct repository state and autonomously continue development
agent: build
---

Reconstruct the current state from Git and the versioned project context:

!`git status -sb`
!`git log --oneline --decorate -10`
!`git diff --stat`

Read @AGENTS.md, @docs/handoff.md, and the relevant sections of @README.md. Synchronize the locked
environment if needed. Continue `$ARGUMENTS` when arguments were supplied; otherwise continue the
active handoff task. If no active task exists, choose the highest-priority safe, incomplete item from
the README development sequence, state that choice, and implement it end to end. Preserve unrelated
changes and all safety boundaries. Verify the result before reporting completion.
