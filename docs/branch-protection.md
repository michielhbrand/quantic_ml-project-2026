# Branch Protection Rules

The following branch protection rule must be configured **manually** in the GitHub repository settings after the repository has been pushed to GitHub.

---

## Steps to configure

1. Navigate to your repository on GitHub.
2. Go to **Settings** → **Branches**.
3. Under *Branch protection rules*, click **Add rule** (or **Add branch protection rule**).

---

## Rule: `main`

| Setting | Value |
|---------|-------|
| Branch name pattern | `main` |
| Require a pull request before merging | ✅ Enabled |
| Require status checks to pass before merging | ✅ Enabled |
| Status check to require | `test` *(from `ci-main.yml`)* |
| Do not allow bypassing the above settings | ✅ Enabled |

---

## Detailed configuration

### Require a pull request before merging
Enable this option. This prevents anyone (including repository admins when *"Do not allow bypassing"* is active) from pushing directly to `main`. All changes must arrive via a pull request.

### Require status checks to pass before merging
Enable this option and search for the `test` status check in the search box. This check is reported by the **CI — Main Branch (PR & Push)** workflow (`.github/workflows/ci-main.yml`).

> **Note:** The `test` check will only appear in the search box after the workflow has run at least once on the repository.

### Do not allow bypassing the above settings
Enable this option so that the rules apply to everyone, including repository admins and organisation owners.

---

## Effect of this rule

| Action | Result |
|--------|--------|
| Direct push to `main` | ❌ Blocked |
| Pull request to `main` with failing CI | ❌ Cannot be merged |
| Pull request to `main` with passing CI | ✅ Can be merged |
| Push to any feature branch | ✅ Allowed (triggers `ci-feature.yml`) |

---

## CI/CD flow summary

```
feature branch push
        │
        ▼
  ci-feature.yml
  └─ test job (pytest)

pull request → main
        │
        ▼
  ci-main.yml
  └─ test job (pytest)
        │  [must pass before merge is allowed]
        ▼
     merge to main
        │
        ▼
  ci-main.yml (push trigger)
  ├─ test job (pytest)
  └─ deploy job (runs only if test passes)
```
