# BB84 Quantum Key Distribution Lab

An interactive, step-by-step Streamlit simulator for teaching BB84 with a
1,000-photon exchange. It includes side-by-side no-Eve and intercept-resend
scenarios, configurable channel noise, basis sifting, causal error inspection,
QBER detection, and classroom charts.

## Interactive learning workflow

Students may work through the first five photons as optional guided practice. For
each photon they compare Alice's preparation basis with Bob's random measurement
basis and predict either a definite bit or a genuinely random outcome. Incorrect
answers receive progressively clearer hints and can be retried. Students who are
already comfortable with measurement bases can bypass the practice and return to
it later without changing the simulated exchange.

After the remaining measurements are generated, Alice and Bob randomly select a
student-configurable percentage of their sifted key for public comparison.
Students calculate the sample QBER and decide whether it exceeds their selected
acceptance threshold. Every revealed test bit is discarded, and the interface
shows the smaller unrevealed candidate key that would continue to error
correction and privacy amplification. It also explicitly distinguishes the
channel's inherent error probability from the maximum accepted QBER.

The post-checkpoint interface presents four clearly signposted learning modes:

1. **No Eve** establishes the channel-noise baseline.
2. **Eve attack** demonstrates intercept-resend disturbance.
3. **Compare** contrasts both completed experiments and their QBER estimates.
4. **Real QKD** explains the boundaries of the classroom model.

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
