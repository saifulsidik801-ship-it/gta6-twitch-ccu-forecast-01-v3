"""gta6-twitch-ccu-forecast-01 v3: family-weighted log-curve forecast with a
channel-surge term and final-only live signals.

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2-24),
given hour 1's print, by fitting at run time from the case's own
`comparables`. Single file, stdlib only, no network, no wall clock,
deterministic.

What v3 adds over v2 (same proven core: family-weighted Huber quadratic on
log hour-1, blended 50/50 with a family-similarity x size-kernel shape
average; spread from within-training leave-one-out residuals):
  1. Channel surge. A launch whose streamer base is already saturated at
     hour 1 (many channels live before the day even starts) has less room
     to grow than one whose streamers are still joining. The log ratio
     ln(channels at hour 1 / mean channels in the 14 days before) predicts
     the peak ratio with r = -0.50 across the 37 training launches, so the
     quadratic gains it as a second regressor -- GUARDED: it is used only
     when the target's surge lies inside the training range; outside it the
     fit falls back to the 1-D size curve. (The guard is what keeps the
     GTA VI forecast honest: the target's pre-release category is nearly
     dead, far outside every training launch on the viewer side, and an
     unguarded second regressor extrapolates the peak away.)
  2. Final-only live signals. The final case (h00) carries fields the
     practice cases do not: `streams` (top 20 live streams at the
     checkpoint), `categories_hourly` (top 20 categories hourly) and
     minute-level `history`. v3 reads them ONLY when `streams` is present
     (it is null in every practice case, so practice runs exactly the
     v2-core + channel-surge model) and turns them into small, bounded
     adjustments to the log median -- total at most +/-0.10 per hour:
       - streamer concentration: top-20 viewers / hour-1 viewers. A
         concentrated audience (big streamers carrying the day) sustains
         late hours; a fragmented one fades. Nudges hours 14-24 by up to
         +/-0.05.
       - stream freshness: the share of top-20 viewers from streams that
         started within 2 h of the checkpoint. Fresh streamers still
         joining means rising momentum. Nudges hours 2-6 by up to +0.04.
       - category share momentum: the target category's share of the
         top-20 total now vs 6 h earlier. A rising share means the game is
         gaining on the platform. Nudges all hours by up to +/-0.05.
       - minute momentum: mean viewers in the last 60 min vs the previous
         60 min of `history` (only when the history is minute-dense, which
         only the final's recorder is). Nudges hours 2-6 by up to +/-0.04.
     None of these is fitted to the comparables -- no training launch has
     per-stream data -- so each is a bounded, mechanism-based nudge, never
     a fitted coefficient. They cannot move the forecast far, and they
     cannot fire at all on a practice case.

Validation (leave-one-out over the kit's 37 comparable launches, pinball):
  console-rolling family 0.163 (v2: 0.163), big clean launches 0.185 (v2:
  0.183), all big launches 0.213 (v2: 0.207), all 37 0.428 (v2: 0.419).
  The median is deliberately conservative: every alternative that scored
  better on practice (size-conditional bias, local-linear, log-log level
  models) did so by steepening the size slope, and every one of them
  dragged the GTA VI synthetic peak below 1.2x hour 1 -- below the
  cleanest big rolling comparable (The Last of Us Part II, 1.34x) and
  against the console-rolling family's late-day rise. v3 keeps the family
  shape (synthetic peak 1.50x hour 1 at h18) and takes its gains where they
  do not cost the final: the channel-surge term (practice 0.179 vs v2
  0.187) and the final-only live signals, which only the final can use.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Blend weights: quad (with channel surge) / family shape average.
W_QUAD, W_AVG = 0.5, 0.5
# Size-kernel bandwidth (standardized ln v_1) for the family average.
SIZE_BW = 1.0
# Spread scale applied to the LOO residual quantiles.
SPREAD_SCALE = 0.5
# Hard cap on the total final-only log-median adjustment per hour.
ADJ_CAP = 0.10


# ---------------------------------------------------------------- utilities

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] = 1e-12
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def wls(X, y, w, ridge=1e-6):
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, w):
        for a in range(p):
            b[a] += wi * xi[a] * yi
            for c in range(p):
                A[a][c] += wi * xi[a] * xi[c]
    for a in range(p):
        A[a][a] += ridge
    return solve(A, b)


def huber_wls(X, y, w, iters=2, c=1.345):
    beta = wls(X, y, w)
    for _ in range(iters):
        res = [yi - sum(b * x for b, x in zip(beta, xi))
               for xi, yi in zip(X, y)]
        srt = sorted(abs(r) for r in res)
        s = 1.4826 * srt[len(srt) // 2] or 1.0
        w2 = []
        for wi, r in zip(w, res):
            u = abs(r) / (c * s)
            w2.append(wi * (1.0 if u <= 1 else 1.0 / u))
        beta = wls(X, y, w2)
    return beta


def predict(beta, x):
    return sum(b * v for b, v in zip(beta, x))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ------------------------------------------------------- feature extraction

def parse_tier(unlock_text):
    m = re.search(r"Tier:\s*([a-z\-]+)", unlock_text or "")
    return m.group(1) if m else "unknown"


def parse_unlock_type(text):
    u = (text or "").lower()
    has_roll = ("midnight local" in u or "rolling" in u)
    has_sim = ("single global" in u or "simultaneous global" in u
               or "single simultaneous" in u or "simultaneous across" in u
               or "simultaneous midnight gmt" in u
               or "single global unlock" in u)
    if has_roll and has_sim:
        return "mixed"
    if has_sim:
        return "simultaneous"
    if has_roll:
        return "rolling"
    return "unknown"


def parse_platforms(text):
    p = (text or "").lower()
    has_pc = ("pc" in p or "steam" in p or "stadia" in p)
    has_ps = ("ps5" in p or "ps4" in p or "playstation" in p)
    has_xbox = ("xbox" in p)
    has_switch = ("switch" in p)
    return {
        "has_pc": has_pc,
        "console_only": (not has_pc) and (has_ps or has_xbox or has_switch),
    }


def parse_target_family(case):
    """(console_only, unlock_type) for the target launch.

    Prefers the target-history comparable (GTA VI's own category history in
    the final), else the case notes' release clause.
    """
    for c in case.get("comparables", []):
        if parse_tier(c.get("unlock", "")) == "target-history":
            return (parse_platforms(c.get("platforms", ""))["console_only"],
                    parse_unlock_type(c.get("unlock", "")))
    notes = case.get("notes", "") or ""
    m = re.search(r"released \d{4}-\d{2}-\d{2} on ([^;(]+)", notes)
    plat_text = m.group(1) if m else notes
    if "console only" in notes.lower() or "ps5 only" in notes.lower():
        console_only = True
    else:
        console_only = parse_platforms(plat_text)["console_only"]
    return (console_only, parse_unlock_type(notes))


def hourly_series(points):
    acc = {}
    for p in points or []:
        if p.get("viewers") is None:
            continue
        t = parse_ts(p["from"]).replace(minute=0, second=0, microsecond=0)
        acc.setdefault(t, []).append(float(p["viewers"]))
    return {t: sum(vs) / len(vs) for t, vs in acc.items()}


def hourly_channels(points):
    acc = {}
    for p in points or []:
        if p.get("channels") is None:
            continue
        t = parse_ts(p["from"]).replace(minute=0, second=0, microsecond=0)
        acc.setdefault(t, []).append(float(p["channels"]))
    return {t: sum(vs) / len(vs) for t, vs in acc.items()}


def channel_surge(points, day_start, ch1):
    """ln(channels at hour 1 / mean channels in the 14 days before).

    Returns None when the channels are missing (the caller then falls back
    to the 1-D size fit). A big surge means the streamer base was already
    live at hour 1, leaving less headroom for the day to grow.
    """
    if not ch1:
        return None
    ch = hourly_channels(points)
    pre = [v for t, v in ch.items()
           if day_start - timedelta(days=14) <= t < day_start]
    if not pre:
        return None
    pre_mean = sum(pre) / len(pre)
    if pre_mean <= 0:
        return None
    return math.log(ch1 / pre_mean)


def launch_record(name, day_start, hourly_pts, platforms="", unlock_text=""):
    series = hourly_series(hourly_pts)
    ds = parse_ts(day_start) if isinstance(day_start, str) else day_start
    day = {}
    for h in range(1, 25):
        t = ds + timedelta(hours=h - 1)
        if t in series:
            day[h] = series[t]
    if 1 not in day or day[1] <= 0:
        return None
    v1 = day[1]
    ch1 = None
    for p in hourly_pts or []:
        try:
            if parse_ts(p["from"]) == ds:
                ch1 = p.get("channels")
                break
        except (KeyError, ValueError):
            continue
    return {
        "name": name,
        "v1": v1,
        "log_v1": math.log(v1),
        "shape": {h: math.log(day[h] / v1) for h in day if h >= 2},
        "plat": parse_platforms(platforms),
        "unlock_type": parse_unlock_type(unlock_text),
        "tier": parse_tier(unlock_text),
        "ch_surge": channel_surge(hourly_pts, ds, ch1),
    }


# ------------------------------------------------------------------- model

def family_weight(tr, tg):
    w = 1.0
    w *= 1.0 if tr["plat"]["console_only"] == tg["plat"]["console_only"] else 0.35
    tu, uu = tg["unlock_type"], tr["unlock_type"]
    if tu == uu:
        w *= 1.0
    elif tu == "unknown" or uu == "unknown":
        w *= 0.6
    elif (tu in ("rolling", "mixed") and uu in ("rolling", "mixed")) or \
         (tu in ("simultaneous", "mixed") and uu in ("simultaneous", "mixed")):
        w *= 0.7
    else:
        w *= 0.3
    return w


def fit_quad(train, target):
    """Huber quadratic in ln(v_1), plus the channel-surge term when the
    target's surge is inside the training range (guarded 2-D fit)."""
    train_f = [r["log_v1"] for r in train]
    mu = sum(train_f) / len(train_f)
    var = sum((x - mu) ** 2 for x in train_f) / max(1, len(train_f) - 1)
    sd = math.sqrt(var) or 1.0
    tg_z = (target["log_v1"] - mu) / sd
    surges = [r["ch_surge"] for r in train if r.get("ch_surge") is not None]
    tg_s = target.get("ch_surge")
    use_2d = (tg_s is not None and len(surges) >= 5
              and min(surges) <= tg_s <= max(surges))
    if use_2d:
        smu = sum(surges) / len(surges)
        svar = sum((x - smu) ** 2 for x in surges) / max(1, len(surges) - 1)
        ssd = math.sqrt(svar) or 1.0
        tg_sz = (tg_s - smu) / ssd
        med_surge = sorted(surges)[len(surges) // 2]
    else:
        smu, ssd, tg_sz, med_surge = 0.0, 1.0, 0.0, 0.0
    w = [family_weight(r, target) for r in train]
    med = {}
    for h in range(2, 25):
        rows = [r for r in train if h in r["shape"]]
        y = [r["shape"][h] for r in rows]
        if use_2d:
            X = [[1.0, (r["log_v1"] - mu) / sd,
                  ((r["log_v1"] - mu) / sd) ** 2,
                  ((r.get("ch_surge") if r.get("ch_surge") is not None
                    else med_surge) - smu) / ssd] for r in rows]
            xq = [1.0, tg_z, tg_z * tg_z, tg_sz]
        else:
            X = [[1.0, (r["log_v1"] - mu) / sd,
                  ((r["log_v1"] - mu) / sd) ** 2] for r in rows]
            xq = [1.0, tg_z, tg_z * tg_z]
        w_h = [wi for wi, r in zip(w, rows)]
        beta = huber_wls(X, y, w_h)
        med[h] = predict(beta, xq)
    return med


def fit_fam_avg(train, target):
    train_f = [r["log_v1"] for r in train]
    mu = sum(train_f) / len(train_f)
    var = sum((x - mu) ** 2 for x in train_f) / max(1, len(train_f) - 1)
    sd = math.sqrt(var) or 1.0
    tg_z = (target["log_v1"] - mu) / sd
    w = []
    for r in train:
        z = (r["log_v1"] - mu) / sd
        wk = math.exp(-0.5 * ((z - tg_z) / SIZE_BW) ** 2)
        w.append(family_weight(r, target) * wk)
    med = {}
    for h in range(2, 25):
        num = sum(wi * r["shape"][h] for wi, r in zip(w, train)
                  if h in r["shape"])
        den = sum(wi for wi, r in zip(w, train) if h in r["shape"]) or 1.0
        med[h] = num / den
    return med


def blend_median(train, target):
    mA = fit_quad(train, target)
    mB = fit_fam_avg(train, target)
    return {h: W_QUAD * mA[h] + W_AVG * mB[h] for h in range(2, 25)}


def loo_residuals(train):
    res = {h: [] for h in range(2, 25)}
    for i, rec in enumerate(train):
        sub = train[:i] + train[i + 1:]
        med = blend_median(sub, rec)
        for h in range(2, 25):
            if h in rec["shape"]:
                res[h].append(rec["shape"][h] - med[h])
    return res


def quantiles_from_residuals(med, res_by_hour):
    out = {}
    for h in range(2, 25):
        res = sorted(res_by_hour[h]) or [0.0]
        n = len(res)
        mean = sum(res) / n
        sd = math.sqrt(sum((r - mean) ** 2 for r in res) / max(1, n - 1)) or 0.15
        qs = []
        for tau, z in zip(LEVELS, Z):
            pos = tau * (n - 1)
            lo = int(math.floor(pos))
            hi = min(n - 1, lo + 1)
            frac = pos - lo
            emp = res[lo] * (1 - frac) + res[hi] * frac
            gau = mean + z * sd
            qs.append(med[h] + SPREAD_SCALE * (0.5 * emp + 0.5 * gau))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out[h] = qs
    return out


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


# ------------------------------------------- final-only live adjustments

def target_ch_surge(case, v1, ch1):
    """Channel surge for the target case.

    The pre-launch window is assembled from `history_hourly` plus the
    target-history comparable (GTA VI's own category, Aug 27 onward): the
    final's `history_hourly` only covers the recorder's window (from
    Nov 17), so the comparable fills the earlier part of the 14 days.
    """
    ds = parse_ts(case["day_start"])
    pts = list(case.get("history_hourly", []) or [])
    for c in case.get("comparables", []):
        if parse_tier(c.get("unlock", "")) == "target-history":
            pts = pts + list(c.get("hourly", []) or [])
    return channel_surge(pts, ds, ch1)


def final_only_adjustments(case, v1):
    """Bounded log-median nudges from the final case's live fields.

    Returns {hour: adjustment} (0.0 everywhere when the case has no live
    fields, i.e. every practice case). Each component is capped, and the
    total per hour is capped at +/-ADJ_CAP. These are mechanism-based
    nudges, not fitted coefficients: no training launch carries
    per-stream data, so nothing here is calibrated on the comparables.
    """
    adj = {h: 0.0 for h in range(2, 25)}
    streams = case.get("streams")
    if not streams:
        return adj
    checkpoint = parse_ts(case["checkpoint"])
    total_top = sum(float(s.get("viewers") or 0) for s in streams)
    if total_top <= 0 or v1 <= 0:
        return adj

    # 1. Streamer concentration: top-20 viewers / hour-1 viewers.
    #    Concentrated (big streamers carry the day) sustains late hours;
    #    fragmented fades. Ramps in over h10-h14.
    conc = total_top / v1
    if conc > 0.7:
        late = 0.05 * clamp((conc - 0.7) / 0.3, 0.0, 1.0)
    elif conc < 0.5:
        late = -0.05 * clamp((0.5 - conc) / 0.5, 0.0, 1.0)
    else:
        late = 0.0
    for h in range(2, 25):
        ramp = clamp((h - 10) / 4.0, 0.0, 1.0)
        adj[h] += late * ramp

    # 2. Stream freshness: share of top-20 viewers from streams that
    #    started within 2 h of the checkpoint. Fresh streamers still
    #    joining means rising momentum into the early hours.
    fresh = 0.0
    for s in streams:
        started = s.get("started_at")
        if not started:
            continue
        try:
            if (checkpoint - parse_ts(started)).total_seconds() <= 2 * 3600:
                fresh += float(s.get("viewers") or 0)
        except (ValueError, TypeError):
            continue
    fresh_share = fresh / total_top if total_top > 0 else 0.0
    if fresh_share > 0.3:
        early = 0.04 * clamp((fresh_share - 0.3) / 0.4, 0.0, 1.0)
        for h in range(2, 25):
            fade = clamp((10 - h) / 4.0, 0.0, 1.0)
            adj[h] += early * fade

    # 3. Category share momentum: the target's share of the top-20 total
    #    now vs 6 h earlier. A rising share means the game is gaining on
    #    the platform; a falling share the opposite.
    cats = case.get("categories_hourly") or []
    game_name = (case.get("game") or {}).get("name", "")
    shares = []
    for row in cats:
        top = row.get("top") or []
        tot = sum(float(t.get("viewers") or 0) for t in top)
        mine = 0.0
        for t in top:
            if t.get("name") == game_name:
                mine = float(t.get("viewers") or 0)
                break
        if tot > 0:
            shares.append((parse_ts(row["from"]), mine / tot))
    if len(shares) >= 2:
        shares.sort(key=lambda x: x[0])
        now_share = shares[-1][1]
        ref_t = shares[-1][0] - timedelta(hours=6)
        ref_share = None
        for t, s in shares:
            if t <= ref_t:
                ref_share = s
        if ref_share is not None:
            drift = now_share - ref_share
            if drift > 0:
                adj_all = 0.05 * clamp(drift / 0.20, 0.0, 1.0)
            else:
                adj_all = -0.05 * clamp(-drift / 0.20, 0.0, 1.0)
            for h in range(2, 25):
                adj[h] += adj_all

    # 4. Minute momentum: mean viewers in the last 60 min vs the previous
    #    60 min of `history`. Only when the history is minute-dense (the
    #    final's recorder); a sparse hourly history skips this.
    hist = case.get("history") or []
    pts = []
    for p in hist:
        if p.get("viewers") is None:
            continue
        try:
            pts.append((parse_ts(p["at"]), float(p["viewers"])))
        except (KeyError, ValueError):
            continue
    pts.sort(key=lambda x: x[0])
    last2h = [v for t, v in pts
              if checkpoint - timedelta(hours=2) <= t < checkpoint]
    if len(last2h) >= 30:
        last1h = [v for t, v in pts
                  if checkpoint - timedelta(hours=1) <= t < checkpoint]
        prev1h = [v for t, v in pts
                  if checkpoint - timedelta(hours=2) <= t
                  < checkpoint - timedelta(hours=1)]
        if last1h and prev1h:
            m_last = sum(last1h) / len(last1h)
            m_prev = sum(prev1h) / len(prev1h)
            if m_prev > 0:
                slope = math.log(m_last / m_prev)
                mom = 0.04 * clamp(slope / 0.30, -1.0, 1.0)
                for h in range(2, 25):
                    fade = clamp((10 - h) / 4.0, 0.0, 1.0)
                    adj[h] += mom * fade

    return {h: clamp(adj[h], -ADJ_CAP, ADJ_CAP) for h in range(2, 25)}


# ------------------------------------------------------------------- cases

def baseline_forecast(case):
    """House-baseline fallback: persistence with a widening random-walk spread."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    p1 = max(1, first["viewers"]) if first else 1
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    v1 = max(1, float(first["viewers"]))
    # training set: every comparable with a real release day
    train = []
    for c in case.get("comparables", []):
        if parse_tier(c.get("unlock", "")) == "target-history":
            continue
        r = launch_record(c["name"], c["day_start"], c.get("hourly", []),
                          c.get("platforms", ""), c.get("unlock", ""))
        if r is not None:
            train.append(r)
    if len(train) < 10:
        return baseline_forecast(case)
    console_only, unlock_type = parse_target_family(case)
    target = {
        "log_v1": math.log(v1),
        "v1": v1,
        "plat": {"console_only": console_only},
        "unlock_type": unlock_type,
        "ch_surge": target_ch_surge(case, v1, first.get("channels")),
    }
    med = blend_median(train, target)
    res = loo_residuals(train)
    adj = final_only_adjustments(case, v1)
    med = {h: med[h] + adj[h] for h in range(2, 25)}
    log_q = quantiles_from_residuals(med, res)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [round2(max(1.0, min(1e9, v1 * math.exp(q))))
              for q in log_q[h]]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
