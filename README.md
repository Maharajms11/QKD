# BB84 Quantum Key Distribution Lab

An interactive, step-by-step Streamlit simulator for teaching BB84 with a
1,000-photon exchange. It includes side-by-side no-Eve and intercept-resend
scenarios, configurable channel noise, basis sifting, causal error inspection,
QBER detection, and classroom charts.

This is an idealised educational model, not a production QKD implementation.

## Run locally

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

## Share with students for free

1. Create a GitHub repository and add this project's files.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create an app, select the repository, branch, and `app.py`, then deploy.
4. Share the resulting public URL with students.

Streamlit's official documentation describes Community Cloud as free to use; see
the [Community Cloud overview](https://docs.streamlit.io/deploy/streamlit-community-cloud)
and [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
No API keys, paid services, database, or student login is required by this app.
The simulation runs in the web app process and stores no generated key material.

## Project structure

- `app.py` — Streamlit teaching interface and visualisations.
- `bb84.py` — modular, testable simulation logic.
- `requirements.txt` — Python dependencies.
- `tests/test_bb84.py` — invariant and statistical checks.

## Model boundaries

The app models ideal single-photon BB84 and an intercept-resend attack. Real QKD
requires authenticated classical communication, decoy states, finite-key
analysis, error correction, privacy amplification, device/security assumptions,
loss and detector models, composable security proofs, and secure key handling.
