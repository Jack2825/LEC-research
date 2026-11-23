# Upload the current code package

Target repository: [Jack2825/LEC-research](https://github.com/Jack2825/LEC-research).
This package was prepared locally. No remote commits, files, settings or
releases were changed, and remote repository contents were not assumed.

Upload the **contents of `GitHub_publication_20261121`** to the repository root,
preserving the directory structure. This is the replacement for the previous
code package. Do not merge its superseded final-assembly directories back in.
If the September package is already committed, review and remove its obsolete
paths in the GitHub working copy when replacing it; keep unrelated repository
files. The original research workspace and old local package remain untouched.

Do not upload the entire research project or the temporary test checkout.
Large inputs, generated figures, Excel workbooks and results are external to
this clean code folder. `.gitignore` protects those paths if you later restore
inputs and run analyses; browser drag-and-drop uploads do not apply `.gitignore`.

Before committing from a Git checkout:

```sh
python tools/validate_release.py
git status --short
git diff --cached --stat
```

Complete the pending metadata/data-access items and selected licence before
calling this the final public manuscript release. Then check:

```sh
python tools/validate_release.py --release
```

Choose the actual version/tag after review and cite the exact commit/release
in the manuscript. A code or data archive must exist before claiming a DOI
or public availability. Details are in `docs/PUBLICATION_ITEMS.md`.
