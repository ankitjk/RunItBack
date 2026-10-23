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

## Scale to the change

Every change is recorded, but the record should cost about as much as the
change did. Pick the size first:

| Size | Examples | What to write | Issue line |
|------|----------|---------------|------------|
| **Small** | Typo, comment, formatting, rename, one-line fix with an obvious cause | One bullet under a `**Minor:**` list in today's entry (start the entry with just its heading if there is none) | Usually `Issues: none` |
| **Medium** | A bug fix, a new skill or script, a config change | A short entry: heading, then **Changes** with what and why per file. Skip Session Goal, Problem, and Left Open unless they add something | `Fixes:`/`Found:` if it touches a tracked bug |
| **Large** | Behaviour change in detection, an investigation, anything with measured results | The full template in step 4 | As applicable |

When unsure, go one size down. A short accurate entry beats a long padded one,
and the commit message already carries the detail for small changes. Steps that
do not apply at the chosen size (e.g. syncing CLAUDE.md for a typo) are skipped.

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

5. **Keep `docs/ISSUES.md` in step** (see the `/issues` skill):
   - If the work fixed a tracked issue, move it to the Fixed list there and
     name its ID in the entry, e.g. "`ocr.py` - trim damage ROI - fixes ISSUE-001".
   - Anything under **Left Open** that is a real bug or problem gets added
     there, and the entry cites its new ID. Left Open is the session's note;
     ISSUES.md is the list that persists.

6. **Update the `**Last Updated:**` date** at the top of the file.

7. **Sync current-state facts if they changed.** If a change makes a statement
   in `CLAUDE.md` or the reference sections of `DEVELOPMENT.md` untrue (a
   parameter value, a flag name, a file path, a "Known Issues" item), fix that
   statement too and mention it in the entry.

8. **Print a rundown** in chat - a table of file / what changed / why, one row
   per file, followed by anything left open. This is the quick summary; the
   DEVELOPMENT.md entry is the permanent one.

9. **Give the commit's issue line** - `Fixes: ISSUE-NNN`, `Found: ISSUE-NNN`,
   or `Issues: none` - for whoever commits. `scripts/hooks/commit-msg` rejects
   a commit without one, or with one that does not match `docs/ISSUES.md`.

## Notes

- Do not commit. Leave the doc change in the working tree so it goes into the
  same commit as the code it describes.
- Group by purpose, not by file, when several files changed for one reason.
- Skip pure formatting or whitespace changes unless they were the point.
- Keep it proportionate: a one-line fix gets a short entry.
