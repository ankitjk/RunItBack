---
name: log-changes
description: Record the changes made in this session as a dated entry in the Changelog of docs/DEVELOPMENT.md - what was modified, and why - then print a short rundown. Use after finishing a change and before committing it, or when asked to log, document, or write up what was changed.
---

# /log-changes - Write the session's changes into DEVELOPMENT.md

`docs/DEVELOPMENT.md` is the project's development chronicle. Its `## Changelog`
section holds one dated entry per working session, newest first. This skill adds
that entry so the record keeps up with the code.

The entry must explain **why**, not just what. The diff already records what
changed; the reasons live in this conversation - the bug that was hit, the
measurement that prompted the change, the approach that was tried and dropped.
Write those down, because they are lost when the session ends.

## Usage

```
/log-changes                 # log uncommitted changes plus this session's commits
/log-changes <commit-range>  # log a specific range, e.g. main~3..main
```

## Instructions

1. **Find what changed.** Use the range if given. Otherwise combine:
   ```bash
   git status --short
   git diff HEAD --stat
   git log --oneline origin/main..HEAD
   ```
   Read the actual diffs for anything non-trivial. Do not describe a file you
   have not looked at.

2. **Find out why.** Take the reasons from this conversation and from commit
   message bodies. If the reason for a change is not known, ask - do not invent
   one. "Unknown - found already modified" is an acceptable entry; a plausible
   guess is not.

3. **Check for an entry from today.** If the newest Changelog entry has today's
   date and covers the same work, extend it instead of adding a second one.

4. **Write the entry** directly under `## Changelog`, above the newest existing
   entry, matching the existing format:

   ```markdown
   ### <Month D, YYYY> - <Short Title>

   **Session Goal:** <one line>

   **Problem:**
   - <what was wrong or missing, with evidence - a failing test, a number, a log line>

   **Changes:**
   - `path/to/file.py` - <what changed> - <why>

   **Results:**
   - <test or benchmark outcome, or "Not measured" - say which>

   **Left Open:**
   - <anything found but deliberately not fixed>

   ---
   ```

   Drop **Left Open** if there is nothing to put in it. Keep **Results**
   honest: if nothing was run, say so.

5. **Update the `**Last Updated:**` date** at the top of the file.

6. **Sync current-state facts if they changed.** If a change makes a statement
   in `CLAUDE.md` or the reference sections of `DEVELOPMENT.md` untrue (a
   parameter value, a flag name, a file path, a "Known Issues" item), fix that
   statement too and mention it in the entry.

7. **Print a rundown** in chat - a table of file / what changed / why, one row
   per file, followed by anything left open. This is the quick summary; the
   DEVELOPMENT.md entry is the permanent one.

## Notes

- Do not commit. Leave the doc change in the working tree so it goes into the
  same commit as the code it describes.
- Group by purpose, not by file, when several files changed for one reason.
- Skip pure formatting or whitespace changes unless they were the point.
- Keep it proportionate: a one-line fix gets a short entry.
