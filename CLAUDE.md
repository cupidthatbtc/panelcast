# panelcast — repo conventions

## Commits

Keep them like the early history, not the recent ones.

- **Subject:** conventional prefix (`feat`/`fix`/`docs`/`chore`/`test`/`refactor`) +
  a short imperative summary. Lowercase after the prefix, no trailing period.
- **Body (optional):** one short paragraph — one or two sentences naming *what*
  changed. Only add it when the subject can't carry the change on its own.
- **No AI attribution.** No `Co-Authored-By` / `Generated with` trailers.

Do **not**:
- enumerate every file or consequence as a bullet list,
- pad with parenthetical asides or transitive-dependency trivia,
- add "X is unaffected" reassurances or restate the rationale at length.

### Good (the original style)

```
feat: domain portability — aerospace example, porting guide, and e2e proof

A worked aerospace descriptor (airframes flying scored test flights), the
porting walkthrough, and an end-to-end test proving a new domain runs with no
source changes and that the AOTY defaults stay byte-identical.
```

```
docs: guides, model card, and project README

Getting-started, CLI, extensibility, leakage-control, evaluation-protocol, and
structure guides; the model card with intended use and limitations; and the
general-tool README with the domains table.
```

For a simple removal, one sentence is enough:

```
chore: remove the Graphviz pipeline-diagram generator

Drops the diagrams module, the generate-diagrams CLI command, its tests, and the
graphviz / python-graphviz dependencies.
```

### Avoid (recent slop)

Multi-bullet bodies that list every file, spell out byte counts of transitive
deps, justify the change across several sentences, and end with "the rest is
unaffected." If you're writing bullets, it's too long.

## Workflow

- Run tests through pixi: `pixi run test` (or `pixi run test-fast` for the
  inner loop). `pixi.lock` is the authoritative environment.
- Commit as `cupidthatbtc`; push from the environment where `gh` is
  authenticated.
