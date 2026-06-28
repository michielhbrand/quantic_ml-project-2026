# AI Tooling

## Tools Used

### Roo (AI IDE — Claude Sonnet 4.5 / 4.6)
The primary tool used throughout this project. Roo is an AI coding assistant integrated directly into VS Code that can read, write, and refactor files, execute terminal commands, and reason about the codebase holistically.

**Used for:**
- Scaffolding the entire Flask web application (`frontend/app.py`, `frontend/templates/index.html`)
- Writing the full ML training pipeline (`train_models.py`) including cross-validation, scaler fitting, and model serialisation
- Writing all automated tests (`tests/test_unit.py`, `tests/test_integration.py`, `tests/test_smoke.py`)
- Creating GitHub Actions CI/CD workflow files (`.github/workflows/ci-feature.yml`, `.github/workflows/ci-main.yml`)
- Configuring branch protection rules via the GitHub API (`gh api`)
- Debugging deployment failures on Render (OOM crash → gunicorn fix, column alignment fix for scaler)
- All Git operations: branching, committing, PR creation, CI monitoring, merging
- Writing this documentation

---

## What Worked Well

- **End-to-end task execution:** Roo could take a high-level instruction ("set up CI/CD") and produce all the required files, run git commands, watch CI, and merge PRs without manual intervention. This dramatically reduced the time to complete multi-step tasks.
- **Error diagnosis:** When the Render deployment failed with an OOM error, Roo read the logs, identified the root cause (Flask debug reloader doubling memory), and produced the correct fix (gunicorn with `--workers 1`) without any additional prompting.
- **Iterative debugging of test failures:** When a CI test failed because the expected behaviour changed once a real model was loaded, Roo correctly identified that the fix belonged in `app.py` (aligning feature columns to the scaler) rather than simply patching the test assertion.
- **Code consistency:** Roo maintained consistent style across all files without needing explicit style guides.

---

## What Didn't Work as Well

- **Demo data verification:** The initial malware demo button values were not verified against the actual trained model and happened to fall on the goodware side of the decision boundary. This required a separate debugging step to extract confirmed-correct sample rows directly from the dataset using the live model.
- **test_set.csv column names:** The training script originally saved the test set from the scaled numpy array, producing integer column headers. This was caught only when the file was uploaded to the live app, not during code generation. A more rigorous review of the data pipeline at generation time would have caught this.
- **Merge conflicts between parallel PRs:** Two PRs that both touched `test_set.csv` caused a conflict. Coordinating multiple concurrent feature branches requires explicit sequencing that the AI didn't anticipate upfront.
