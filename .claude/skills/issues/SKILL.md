---
name: issues
description: Track known bugs in docs/ISSUES.md - list what is still open, add a newly found bug, move a fixed one to the Fixed list, or reopen one that came back. Use when a bug is discovered, when a fix lands, before committing, or when asked what still needs work.
---

# /issues - Maintain the open-issues list

`docs/ISSUES.md` lists every known open bug, grouped by severity, with a
one-line **Fixed** list at the bottom. The full record of each fix goes in the
`docs/DEVELOPMENT.md` Changelog. Bugs only - planned features belong in
"Future Improvements" in `docs/DEVELOPMENT.md`.

`scripts/hooks/commit-msg` enforces this file: every commit message must carry
an issue line (see **Commit line** below) that matches it, or the commit is
rejected.

## Usage

```
/issues                      # list open issues, highest severity first
/issues add <description>    # record a new issue
/issues fix <ISSUE-NNN>      # move a fixed issue to the Fixed list
/issues reopen <ISSUE-NNN>   # a fixed bug came back
```

Issues found during other work (an agent report, a failing test, a code review)
should be added the same way, without waiting to be asked.

## Instructions

### List

Read `docs/ISSUES.md` and give a short table: ID, title, severity, one-line
summary. Then recommend which to tackle next and why - usually the highest
severity one, unless a lower one blocks it (e.g. a benchmark that cannot run
blocks measuring a detection fix).

### Add

1. **Check it is not already listed.** If an existing issue covers it, add the
   new evidence to that entry instead of creating a second.
2. **Verify before writing.** Open the code it concerns and confirm the
   `file:line` references. Do not record a guess as fact - if the cause is
   unknown, say "Cause unknown" and record only what was observed.
3. **Take the ID** from the `**Next ID:**` line, then increment that line.
   IDs are never reused.
4. **Pick a severity:**
   - **High** - causes wrong results on real videos
   - **Medium** - blocks measuring or improving accuracy
   - **Low** - tooling, docs, or performance
5. **Write the entry** under its severity heading, matching the format:

   ```markdown
   ### ISSUE-NNN: <short title>
   - **Area:** <component> - **Found:** <YYYY-MM-DD>, <how it was found>
   - **What happens:** <the wrong behaviour, in plain terms>
   - **Evidence:** <video + timestamp, test, number, or log line>
   - **Where:** `path/to/file.py:line`
   - **Suggested fix:** <one or two sentences, if known>
   ```

   Drop **Evidence**, **Where**, or **Suggested fix** if there is nothing true
   to put in them.

### Fix

1. **Confirm it is actually fixed.** Check the change addresses the cause in
   the entry, and that whatever evidence the entry cites now passes - re-run
   the test or benchmark, or use `ocr-investigator` on the cited timestamp. If
   it has not been checked, say so and ask before moving it to Fixed.
2. **Partially fixed?** Edit the entry to describe what remains. Do not move it.
3. **Move it to Fixed.** Delete the full entry from its severity section and
   add one line at the top of `## Fixed` (replacing `_None yet._` if present):
   `- ISSUE-NNN: <title> (fixed YYYY-MM-DD, DEVELOPMENT.md <Month D, YYYY>)`.
   Leave `**Next ID:**` unchanged.
4. **Update linked issues.** If another entry references this ID (e.g. "re-check
   after fixing ISSUE-001"), update that note.
5. **Record it** - the `/log-changes` entry for this work must name the issue
   ID under **Changes**, and the commit takes `Fixes: ISSUE-NNN`.

### Reopen

1. Confirm the bug is the same one - same cause, not just a similar symptom.
   A different cause is a new issue (**Add**).
2. Delete its line from `## Fixed` and restore a full entry under its severity,
   keeping the old ID. Add `- **Reopened:** <YYYY-MM-DD> - <what showed it is
   back>`. Recover the old wording with `git log -p --grep ISSUE-NNN -- docs/ISSUES.md`.
3. The commit takes `Found: ISSUE-NNN`.

### Commit line

Every commit message ends with one or more of these, just above the
`Co-Authored-By` line:

| Line | When | Hook checks |
|------|------|-------------|
| `Issues: none` | No tracked bug fixed or found - features, docs, refactors | Nothing else on the line |
| `Fixes: ISSUE-001` | A bug was fixed | ID is in `## Fixed`, not open |
| `Found: ISSUE-012` | A bug was found or reopened | ID is an open entry |

Several IDs can share a line (`Fixes: ISSUE-001, ISSUE-002`), and `Fixes` and
`Found` can both appear. Stage `docs/ISSUES.md` in the same commit - the hook
reads the staged version. `Issues: none` is a statement, not a skip: only use
it when the work really found nothing.

## Notes

- Do not commit. Leave the file in the working tree with the change it relates to.
- Planned features are not issues - they belong in "Future Improvements" in
  `docs/DEVELOPMENT.md`.
- Keep entries short. The investigation detail belongs in the DEVELOPMENT.md
  Changelog; the issue only needs enough to find and reproduce the problem.
