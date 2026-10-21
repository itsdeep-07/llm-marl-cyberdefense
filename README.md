# LLM-Augmented MARL Cyber Defense

## Run the live dashboard

Use the repository's CC4 virtual environment. Do not use the global
`python` or `streamlit` command, because another installed CybORG package may
be selected instead of CAGE Challenge 4.

```powershell
.\.venv-cc4\Scripts\Activate.ps1
streamlit run .\app.py
```

If the environment has not been created yet, set it up from the repository
root:

```powershell
py -3.11 -m venv .venv-cc4
.\.venv-cc4\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
python -m pip install -e .\third_party\cage-challenge-4
```

The direct launcher is also available without activating the environment:

```powershell
.\.venv-cc4\Scripts\streamlit.exe run .\app.py
```

Verify the CC4 installation:

```powershell
.\.venv-cc4\Scripts\python.exe -c "from CybORG import CybORG; from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent; from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator; print('CC4 import: ok')"
```

The live dashboard uses the real CC4 environment and saves optional
demonstration artifacts under `results/episodes/`. Existing IPPO and MAPPO
checkpoints are under `results/checkpoints/`; selecting seed `42` auto-selects
their matching checkpoint.
