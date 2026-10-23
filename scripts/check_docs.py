"""Check that commands and code references in Claude's instructions are real.

Skills, agents, CLAUDE.md and docs/ISSUES.md tell Claude what to run. When
they name a flag, function, or file that does not exist, nothing fails until
someone follows them - the /debug-video skill shipped a --debug flag and two
imports that never existed and nobody noticed for nine months. This makes
those mistakes fail at commit time instead.

Checks:
  1. Interpreter    - commands in code blocks use .venv/bin/python, not bare
                      python (not on PATH in Claude's shell)
  2. CLI            - `-m runitback <command> --flags`: the command and every
                      flag exist; `analyze` passes --player (else it blocks
                      on a prompt)
  3. Imports        - `from runitback.x import y`: the module exists and
                      defines y (read with ast, nothing is imported)
  4. Script paths   - a .py/.sh path in a code block exists
  5. Line refs      - `file.py:123` or `file.py:10-20` names a real file
                      with that many lines

Only text Claude acts on is checked - not docs/DEVELOPMENT.md, whose history
deliberately names things that no longer exist. Placeholders like <url> are
skipped, so this proves names exist, not that a command does the right thing.

Usage:
    .venv/bin/python scripts/check_docs.py            # check the repo (exit 1 on any error)
    .venv/bin/python scripts/check_docs.py FILE...    # check specific files as instructions
"""

import ast
import re
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Instructions Claude follows: every check applies.
INSTRUCTION_FILES = ["CLAUDE.md", ".claude/agents/*.md", ".claude/skills/*/SKILL.md"]
# Current-state references: only line refs are checked.
REFERENCE_FILES = ["docs/ISSUES.md"]

SEARCH_DIRS = ["runitback", "tools", "scripts", "tests", ".claude"]

FENCE = re.compile(r"^\s*```")
PLACEHOLDER = re.compile(r"<[^<>]*>")
BARE_PYTHON = re.compile(r"(?:^|[\s;&|(`])(python3?)(?=\s)")
RUNITBACK_CMD = re.compile(r"-m runitback\s+(\S+)(.*)")
IMPORT_FROM = re.compile(r"\bfrom\s+(runitback(?:\.\w+)*)\s+import\s+\(?([\w\s,]+)\)?")
IMPORT_MOD = re.compile(r"^\s*import\s+(runitback(?:\.\w+)*)")
SCRIPT_PATH = re.compile(r"(?:^|[\s\"'`])((?:\.?[\w-]+/)+[\w-]+\.(?:py|sh))\b")
LINE_REF = re.compile(r"(?<![\w/.-])((?:[\w-]+/)*[\w-]+\.(?:py|sh|md|toml|json)):(\d+)(?:-(\d+))?")


def files_to_check(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(ROOT.glob(pattern)))
    return found


def code_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line inside a fenced code block."""
    lines, inside = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside:
            lines.append((number, line))
    return lines


@cache
def cli_commands() -> dict[str, set[str]]:
    """Real `runitback` subcommands and their option strings, from the CLI itself."""
    import typer.main

    sys.path.insert(0, str(ROOT))
    from runitback.cli import app

    group = typer.main.get_command(app)
    return {
        name: {opt for param in cmd.params for opt in param.opts} | {"--help"}
        for name, cmd in group.commands.items()
    }


@cache
def module_names(module: str) -> set[str] | None:
    """Top-level names a runitback module defines, or None if it does not exist."""
    base = ROOT / Path(*module.split("."))
    path = base.with_suffix(".py")
    if not path.exists():
        path = base / "__init__.py"
    if not path.exists():
        return None

    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    if path.name == "__init__.py":  # submodules are importable from a package
        names.update(p.stem for p in path.parent.iterdir() if p.suffix == ".py" or p.is_dir())
    return names


@cache
def resolve(ref: str) -> list[Path]:
    """Files a reference could mean: an exact repo path, else a unique basename."""
    if "/" in ref:
        path = ROOT / ref
        return [path] if path.is_file() else []
    return [p for d in SEARCH_DIRS for p in (ROOT / d).rglob(ref) if ".venv" not in p.parts]


@cache
def line_count(path: Path) -> int:
    return len(path.read_text(errors="replace").splitlines())


def check_line_refs(text: str, errors: list[str], where: str) -> None:
    for number, line in enumerate(text.splitlines(), 1):
        for match in LINE_REF.finditer(line):
            ref, start, end = match.group(1), int(match.group(2)), match.group(3)
            last = int(end) if end else start
            candidates = resolve(ref)
            if not candidates:
                errors.append(f"{where}:{number}: '{match.group(0)}' - no file named {ref}")
            elif len(candidates) > 1:
                errors.append(f"{where}:{number}: '{match.group(0)}' - {ref} is ambiguous, "
                              f"use the full path ({', '.join(str(c.relative_to(ROOT)) for c in candidates)})")
            elif last > line_count(candidates[0]):
                errors.append(f"{where}:{number}: '{match.group(0)}' - "
                              f"{ref} has only {line_count(candidates[0])} lines")


def check_instructions(text: str, errors: list[str], where: str) -> None:
    for number, raw in code_lines(text):
        line = PLACEHOLDER.sub("PLACEHOLDER", raw)
        at = f"{where}:{number}"

        for match in BARE_PYTHON.finditer(line):
            errors.append(f"{at}: bare '{match.group(1)}' - use .venv/bin/python "
                          f"(the venv is not on PATH in Claude's shell)")

        cmd = RUNITBACK_CMD.search(line)
        if cmd:
            name = cmd.group(1)
            args = re.split(r"\s(?:\d?>|\|)", cmd.group(2))[0]  # stop at redirects/pipes
            options = re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", args)
            known = cli_commands()
            if name not in known:
                errors.append(f"{at}: 'runitback {name}' - no such command "
                              f"(commands: {', '.join(sorted(known))})")
            else:
                for opt in options:
                    if opt not in known[name]:
                        errors.append(f"{at}: 'runitback {name} {opt}' - no such option "
                                      f"(options: {', '.join(sorted(o for o in known[name] if o.startswith('--')))})")
                if name == "analyze" and "--help" not in options \
                        and not {"--player", "-p"} & set(options):
                    errors.append(f"{at}: 'runitback analyze' without --player - "
                                  f"it stops to prompt, and Claude cannot answer")

        for match in IMPORT_FROM.finditer(line):
            module = match.group(1)
            names = module_names(module)
            if names is None:
                errors.append(f"{at}: module '{module}' does not exist")
                continue
            for name in (n.strip() for n in match.group(2).split(",")):
                if name and name not in names:
                    errors.append(f"{at}: '{name}' is not defined in {module}")
        for match in IMPORT_MOD.finditer(line):
            if module_names(match.group(1)) is None:
                errors.append(f"{at}: module '{match.group(1)}' does not exist")

        for match in SCRIPT_PATH.finditer(line):
            path = match.group(1)
            if "PLACEHOLDER" in path or path.startswith("path/to/"):  # template examples
                continue
            if not (ROOT / path).is_file():
                errors.append(f"{at}: '{path}' does not exist")


def main(argv: list[str]) -> int:
    errors: list[str] = []
    if argv:
        instructions, references = [Path(a).resolve() for a in argv], []
    else:
        instructions = files_to_check(INSTRUCTION_FILES)
        references = files_to_check(REFERENCE_FILES)

    for path in instructions:
        text, where = path.read_text(), display(path)
        check_instructions(text, errors, where)
        check_line_refs(text, errors, where)
    for path in references:
        check_line_refs(path.read_text(), errors, display(path))

    for error in errors:
        print(error)
    if errors:
        print(f"\ncheck_docs: {len(errors)} problem(s) in Claude's instructions or references.")
        return 1
    return 0


def display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
