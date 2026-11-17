# gta6-twitch-ccu-forecast-01-v3

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/saifulsidik801-ship-it/gta6-twitch-ccu-forecast-01-v3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/saifulsidik801-ship-it/gta6-twitch-ccu-forecast-01-v3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Family-weighted log-curve forecast of GTA VI's launch-day Twitch viewers, hour by hour (hours 2–24), from the hour-1 print. Fitted at run time from the case's own comparables: a Huber-robust quadratic in log hour-1 viewers blended 50/50 with a family-similarity × size-kernel shape average, where family means console-only + rolling-midnight unlock — the shape console launches actually take (dip at hours 3–4, then rise to a late-day peak) instead of the PC-mixed decay a size-only model gives. The quadratic gains a second, guarded regressor: the channel surge, ln(channels at hour 1 / mean channels in the 14 days before). A launch whose streamer base is already saturated at hour 1 has less room to grow; the surge predicts the peak ratio with r = −0.50 across the 37 training launches, and the guard (used only inside the training range) keeps it from extrapolating the GTA VI peak away. Spreads are the within-training leave-one-out residual quantiles (empirical + Gaussian, scaled 0.5). Python 3.13 stdlib only, deterministic, no network, no clock.

The final case carries live fields the practice cases do not — `streams` (top 20 live streams), `categories_hourly` (top 20 categories) and minute-level `history`. When `streams` is present, v3 turns them into small bounded nudges to the log median (total at most ±0.10 per hour): streamer concentration sustains late hours, fresh streams and a rising category share add early momentum, and the minute-level slope into hour 1 nudges the first hours. On a practice case (`streams` null) none of this fires.

Leave-one-out over the kit's 37 comparable launches (mean pinball): console-rolling family 0.163, big clean launches 0.185. Practice loss 0.179 (v2: 0.187). Synthetic GTA VI forecast at hour 1 = 1,000,000: median peak 1.50M at hour 18, 95% quantile at peak 3.59M.

## Run it

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `$MAKEFUN_OUTPUT_DIR` (else `/output`): one `<id>.out` per `<id>.in`, 23 rows (hours 2–24) × 5 quantiles each.

---

<sub>Entered on [make.fun](https://make.fun/?ref=59gqp20q&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
