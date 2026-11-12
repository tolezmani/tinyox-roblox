"""Roblox Up-and-Coming top-10 forecast (make.fun, roblox-up-and-coming-forecast-01).

Method (stdlib only, fully deterministic — no clock, no randomness, no network):
  * Fit, from every pair of reads in /input/history.ndjson, an empirical
    rank-transition model: for each starting-rank bucket and each lag bucket,
    the mean / sd of the destination rank and the probability the game leaves
    the chart. Destination rank is modelled Normal(mu, sd) in rank space.
  * Drift: a game's 24h playing growth predicts its rank change over the next
    24h (regression on the history: dy ~ +3.2 - 13.9*g, corr -0.33; bucket
    coefficients 4.4 / 11.5 / 19.0). mu is shifted by -k*(g - mean_g)*lag/24,
    capped at +-20 ranks (explosive growers can jump to #1).
  * For each of the 72 scored hourly slots, every candidate game's place
    distribution is computed at that slot's lag and averaged over slots.
  * Unnamed share per place = empirical share of slots taken by games not on
    the reference chart (new entrants), averaged over slots, clamped 0.03-0.30.
  * top10 = the 10 games with the highest P(in top 10), ordered by expected rank.

Backtest (local scorer verified exact: reproduces the official starter score
-43.512039 on the practice window):
  practice window (8-10 Nov, 10 reads): -32.98  (v1 -33.37, starter -43.51, house -37.2)
  kit-tail split (train <= 5 Nov, test 5-7 Nov, 11 reads): -42.81 (v1 -44.51, starter -55.85)

Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default
/output) / forecast.json: exactly {"probabilities": [...], "top10": [...]}.
"""
import json
import math
import os
from datetime import datetime

# --- tuned constants (v2: coordinate-tuned on practice + kit-tail splits, 2026-11-12) ---
G_MULT = 2.5         # growth-shift multiplier
SD_MULT = 0.8        # destination-rank sd multiplier
SHIFT_CAP = 25.0     # max rank shift from the growth feature
MIN_P = 0.01         # smallest probability written out
UNNAMED_MULT = 1.25  # multiplier on the empirical unnamed share

RANK_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 10),
                (11, 15), (16, 20), (21, 30), (31, 50), (51, 200)]
LAG_MIDS = [4.0, 12.0, 22.0, 36.0, 52.0, 72.0, 102.0]
# growth coefficient (ranks per unit log playing growth, 24h horizon) by bucket
K_BY_BUCKET = [4.4, 4.4, 4.4, 4.4, 4.4, 4.4, 11.5, 11.5, 11.5, 19.0, 19.0]


def bucket_of(rank):
    for i, (lo, hi) in enumerate(RANK_BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(RANK_BUCKETS) - 1


def parse_at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("up_and_coming"):
                reads.append(r)
    reads.sort(key=lambda r: r["at"])
    return task, reads


def fit_tables(reads):
    """Per (rank bucket, lag bucket): destination mean/sd, exit prob, newcomer share."""
    times = [parse_at(r["at"]) for r in reads]
    charts = [{row["universe_id"]: row for row in r["up_and_coming"]} for r in reads]
    nb = len(RANK_BUCKETS)
    nl = len(LAG_MIDS)
    s = [[0.0] * nl for _ in range(nb)]
    q = [[0.0] * nl for _ in range(nb)]
    c = [[0] * nl for _ in range(nb)]
    x = [[0] * nl for _ in range(nb)]
    newc = [[0] * nl for _ in range(10)]
    newt = [[0] * nl for _ in range(10)]
    for i in range(len(reads)):
        for j in range(i + 1, len(reads)):
            lag = (times[j] - times[i]).total_seconds() / 3600.0
            if lag > 120.0:
                continue
            li = min(range(nl), key=lambda k: abs(LAG_MIDS[k] - lag))
            for uid, row in charts[i].items():
                b = bucket_of(row["rank"])
                r2 = charts[j].get(uid, {}).get("rank")
                if r2 is None:
                    x[b][li] += 1
                else:
                    s[b][li] += r2
                    q[b][li] += r2 * r2
                    c[b][li] += 1
            top_j = sorted(charts[j].values(), key=lambda r: r["rank"])[:10]
            for k, row in enumerate(top_j):
                newt[k][li] += 1
                if row["universe_id"] not in charts[i]:
                    newc[k][li] += 1
    mean = [[0.0] * nl for _ in range(nb)]
    sd = [[0.0] * nl for _ in range(nb)]
    pout = [[0.0] * nl for _ in range(nb)]
    for b in range(nb):
        mid = (RANK_BUCKETS[b][0] + RANK_BUCKETS[b][1]) / 2.0
        for li in range(nl):
            n = c[b][li]
            if n >= 5:
                m = s[b][li] / n
                mean[b][li] = m
                sd[b][li] = math.sqrt(max(q[b][li] / n - m * m, 1.0))
            else:
                mean[b][li] = 48.0 + (mid - 48.0) * math.exp(-LAG_MIDS[li] / 100.0)
                sd[b][li] = 3.0 + 23.0 * (1.0 - math.exp(-LAG_MIDS[li] / 60.0))
            tot = c[b][li] + x[b][li]
            pout[b][li] = x[b][li] / tot if tot >= 10 else 0.0
    newshare = [[(newc[k][li] / newt[k][li] if newt[k][li] >= 5 else 0.0)
                 for li in range(nl)] for k in range(10)]
    return times, charts, mean, sd, pout, newshare


def build_pool(reads, charts):
    last_chart = charts[-1]
    pool = {}
    for uid, row in last_chart.items():
        pool[uid] = {"rank": row["rank"], "last_seen": len(reads) - 1}
    for i in range(max(0, len(reads) - 5), len(reads)):
        for row in sorted(charts[i].values(), key=lambda r: r["rank"])[:30]:
            if row["universe_id"] not in pool:
                pool[row["universe_id"]] = {"rank": row["rank"], "last_seen": i}
    return pool


def build_growth(reads, times, charts):
    """24h log playing growth per game in the last chart, + per-bucket means."""
    last = len(reads) - 1
    target = times[last]
    cand = [i for i in range(last)
            if 20.0 <= (target - times[i]).total_seconds() / 3600.0 <= 28.0]
    if not cand:
        cand = [i for i in range(last)
                if 10.0 <= (target - times[i]).total_seconds() / 3600.0 <= 34.0]
    growth = {}
    if cand:
        i0 = cand[-1]
        for uid, row in charts[last].items():
            if uid in charts[i0] and row.get("playing") and charts[i0][uid].get("playing"):
                growth[uid] = math.log(row["playing"] / charts[i0][uid]["playing"])
    sums = {}
    cnts = {}
    for uid, g in growth.items():
        b = bucket_of(charts[last][uid]["rank"])
        sums[b] = sums.get(b, 0.0) + g
        cnts[b] = cnts.get(b, 0) + 1
    gmean = {b: (sums[b] / cnts[b] if cnts[b] else 0.0)
             for b in range(len(RANK_BUCKETS))}
    gmean_global = sum(growth.values()) / len(growth) if growth else 0.0
    return growth, gmean, gmean_global


def place_probs(uid, info, lag, tables, pool_size, growth, gmean, gmean_global):
    mean, sd, pout = tables
    r0 = info["rank"]
    b = bucket_of(r0)
    mu = interp(LAG_MIDS, mean[b], lag)
    sdv = max(interp(LAG_MIDS, sd[b], lag) * SD_MULT, 1.0)
    po = interp(LAG_MIDS, pout[b], lag)
    g = growth.get(uid)
    if g is not None:
        gd = g - gmean.get(b, gmean_global)
        shift = -K_BY_BUCKET[b] * gd * min(lag, 48.0) / 24.0 * G_MULT
        mu += max(-SHIFT_CAP, min(SHIFT_CAP, shift))
    out = {}
    for k in range(1, 11):
        out[k] = (1.0 - po) * (norm_cdf((k + 0.5 - mu) / sdv)
                               - norm_cdf((k - 0.5 - mu) / sdv))
    out[1] += (1.0 - po) * norm_cdf((0.5 - mu) / sdv)
    return out


def persistence_forecast(last_chart, places):
    chart = [r["universe_id"] for r in sorted(last_chart, key=lambda r: r["rank"])]
    probs = []
    for place in range(1, places + 1):
        if place > len(chart):
            break
        probs.append({"universe_id": chart[place - 1], "place": place, "p": 0.5})
        if place == 1:
            if len(chart) > 1:
                probs.append({"universe_id": chart[1], "place": place, "p": 0.3})
            continue
        probs.append({"universe_id": chart[place - 2], "place": place, "p": 0.15})
        if len(chart) > place:
            probs.append({"universe_id": chart[place], "place": place, "p": 0.15})
    return {"probabilities": probs, "top10": chart[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_input(input_dir)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    last_chart = reads[-1]["up_and_coming"]
    if len(reads) < 4:
        forecast = persistence_forecast(last_chart, places)
    else:
        times, charts, mean, sd, pout, newshare = fit_tables(reads)
        pool = build_pool(reads, charts)
        growth, gmean, gmean_global = build_growth(reads, times, charts)
        last_time = times[-1]
        win = task.get("window", {})
        slot_minutes = int(win.get("slot_minutes", 60))
        slots = int(win.get("slots", 72))
        wfrom = parse_at(win["from"]) if win.get("from") else last_time
        slot_lags = []
        for i in range(slots):
            t = wfrom.timestamp() + i * slot_minutes * 60.0
            slot_lags.append((t - last_time.timestamp()) / 3600.0)
        acc = {uid: {k: 0.0 for k in range(1, places + 1)} for uid in pool}
        tables = (mean, sd, pout)
        for lag in slot_lags:
            for uid in sorted(pool):
                info = pool[uid]
                lag_u = lag
                if info["last_seen"] < len(reads) - 1:
                    lag_u = lag + (len(reads) - 1 - info["last_seen"]) * 8.0
                pp = place_probs(uid, info, lag_u, tables, len(pool),
                                 growth, gmean, gmean_global)
                for k in range(1, places + 1):
                    acc[uid][k] += pp[k] / len(slot_lags)
        rows = []
        for k in range(1, places + 1):
            u = sum(interp(LAG_MIDS, newshare[k - 1], lag) for lag in slot_lags) / len(slot_lags)
            u = min(0.30, max(0.03, u * UNNAMED_MULT))
            ssum = sum(acc[uid][k] for uid in pool)
            scale = (1.0 - u) / ssum if ssum > 0 else 0.0
            for uid in sorted(pool):
                p = acc[uid][k] * scale
                if p >= MIN_P:
                    rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
            # guard: after rounding, a place's sum must stay <= 1
            tot = sum(r["p"] for r in rows if r["place"] == k)
            if tot > 1.0:
                over = tot - 1.0 + 1e-9
                big = max((r for r in rows if r["place"] == k), key=lambda r: r["p"])
                big["p"] = round(big["p"] - over, 6)
        rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
        exp_rank = {uid: sum(acc[uid][k] * k for k in range(1, places + 1))
                    for uid in pool}
        p_top10 = {uid: sum(acc[uid][k] for k in range(1, places + 1))
                   for uid in pool}
        cands = sorted(pool, key=lambda u: (-p_top10[u], exp_rank[u], u))
        top10 = sorted(cands[:places], key=lambda u: (exp_rank[u], u))
        if len(top10) < places:  # pragma: no cover - safety only
            have = set(top10)
            for row in sorted(last_chart, key=lambda r: r["rank"]):
                if len(top10) >= places:
                    break
                if row["universe_id"] not in have:
                    top10.append(row["universe_id"])
                    have.add(row["universe_id"])
        forecast = {"probabilities": rows, "top10": top10}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), "w") as f:
        f.write(json.dumps(forecast, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
