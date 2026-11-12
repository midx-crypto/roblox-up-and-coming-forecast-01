# Roblox Up-and-Coming Forecaster (model-free)

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after the
competition deadline, as one probability per (game, place) plus a top-10
guess, written to `forecast.json`.

## How it works

No model calls, no network, no lookup table. `main.py` (Python 3.13,
standard library only) reads `/input/history.ndjson` and `/input/task.json`
and fits a latent-Gaussian rank-diffusion model to the chart's own history:

- each game gets a latent strength from its current rank;
- strength drifts with a shrunk estimate of the game's recent rank
  velocity and diffuses with a per-rank volatility fitted from the
  history (top ranks are stable, lower ranks wander);
- new games keep joining the chart, modelled as phantom entrants whose
  rate and entry-rank distribution are estimated from the history;
- games with a known creation date leave when they age past 28 days.

Place probabilities are the time-average over 12 points of the 72-hour
window, which is exactly what the log score averages over. Rows with
p < 0.005 stay unnamed so the unnamed share covers surprise entrants.

## Run

```sh
python3 main.py
# or: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

Scored against the public chart archive (Kenny-Dhub/roblox-snapshots)
with a scorer replicating the kit's rules (verified: it reproduces the
house baseline's published practice score of -43.512039 exactly):

- practice window (history to 2026-11-07, scored 2026-11-08–10):
  **-33.65** vs house baseline -43.51, 6.4 top-10 hits
- 30 walk-forward backtests (cutoffs 2026-11-02–10, each scored on the
  next 72 h): mean **-37.0** vs baseline -46.3, better in 29 of 30

Deterministic: two runs on the same history are byte-identical.
Runtime ~0.5 s, well inside the 300 s limit.
