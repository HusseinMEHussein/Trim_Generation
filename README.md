# MF Table Generator

Small Streamlit app + reference solver to generate modifier (MF) tables for target frequencies using a PuLP mixed-integer model.

## Quickstart

Requirements
- Python 3.11
- Install dependencies:
```bash
pip install -r requirements.txt
```

Run the interactive app (Windows):
```bash
streamlit run streamlit_app.py
```

Run the non-interactive solver:
```bash
python Trim_Generation_PuLP_V01p0.py
```

## What this repo contains
- streamlit_app.py — main Streamlit UI and model mirror
- Trim_Generation_PuLP_V01p0.py — standalone PuLP reference solver
- streamlit_temp_testing.py — small Streamlit toggle test
- requirements.txt — Python dependencies
- .devcontainer/ — devcontainer configuration for VS Code / Codespaces
- data/ — images and assets used by the app

## How it works (high level)
- User provides target frequencies (TFs) and advanced options in the Streamlit UI.
- The solver builds a mixed-integer model:
  - Binary assignment matrix A (which modifiers apply to which TF)
  - Integer modifier vector M (modifier values)
  - AM = A * M enforced via big-M linear constraints
  - Objective: minimize the number of modifiers used
- Results show the generated MF table and feasibility status.

## UI notes
- Default TFs come from `default_TF_List` in streamlit_app.py.
- "MF upper limit" and "All shunts share MF1?" toggles are available under Advanced settings.
- Suggested MF upper limit is computed automatically and displayed.

## Troubleshooting
- If solver is infeasible:
  - Inspect printed arrays in the Streamlit logs/terminal (e.g., `ytarget`, `A_values`, `M_values`, `errors`).
  - Verify `max_shunt_fs`, TF grouping, and bounds (MF upper/lower limits).
- Ensure the TF list contains both series and shunt resonators (the code raises an error otherwise).

## Devcontainer
Open in VS Code devcontainer or GitHub Codespaces. The devcontainer is configured to forward Streamlit's default port (8501).

## License and Attribution
© 2025 Hussein Hussein @ Skyworks Solutions

```// filepath: c:\Users\husseinh\Git\Trim_Generation\README.md

