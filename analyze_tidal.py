"""
analyze_tidal.py — Post-run analysis of Tidal Engine effects.
Reads metrics_log.jsonl and checks three hypotheses:
  1. Night predator spawn ~2x day
  2. Seasonal sinusoid in credits (period ~10 epochs)
  3. Food trail rebuilding after tidal rift shifts
"""
import json
import sys

def load_metrics(path="metrics_log.jsonl"):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def analyze(data):
    if not data:
        print("No data found.")
        return

    print(f"=== TIDAL ENGINE ANALYSIS ({len(data)} epochs) ===\n")

    # --- 1. Night vs Day predator spawns ---
    total_day = sum(d.get("tidal_pred_day", 0) for d in data)
    total_night = sum(d.get("tidal_pred_night", 0) for d in data)
    ratio = total_night / total_day if total_day > 0 else float('inf')
    print("1. PREDATOR SPAWNS BY PHASE")
    print(f"   Day total:   {total_day}")
    print(f"   Night total: {total_night}")
    print(f"   Night/Day ratio: {ratio:.2f}x (target: ~2.0x)")
    print(f"   {'PASS' if 1.5 <= ratio <= 3.0 else 'FAIL'}: {'within expected range' if 1.5 <= ratio <= 3.0 else 'outside expected range'}")
    print()

    # Per-epoch detail (first 20 + last 10)
    print("   Epoch breakdown (first 20):")
    for d in data[:20]:
        pd = d.get("tidal_pred_day", 0)
        pn = d.get("tidal_pred_night", 0)
        r = pn / pd if pd > 0 else float('inf')
        bar_d = "#" * pd
        bar_n = "#" * pn
        print(f"   E{d['epoch']:3d}: day={pd:2d} {bar_d:20s} night={pn:2d} {bar_n:20s} ratio={r:.1f}x")
    print()

    # --- 2. Deaths by phase ---
    total_deaths_day = sum(d.get("tidal_deaths_day", 0) for d in data)
    total_deaths_night = sum(d.get("tidal_deaths_night", 0) for d in data)
    d_ratio = total_deaths_night / total_deaths_day if total_deaths_day > 0 else float('inf')
    print("2. DEATHS BY PHASE")
    print(f"   Day deaths:   {total_deaths_day}")
    print(f"   Night deaths: {total_deaths_night}")
    print(f"   Night/Day ratio: {d_ratio:.2f}x")
    print()

    # --- 3. Seasonal sinusoid in credits ---
    print("3. SEASONAL CREDITS SINUSOID")
    print("   Epoch | Credits  | Season Mult | Phase")
    print("   " + "-" * 50)
    min_cred = float('inf')
    max_cred = 0
    min_epoch = 0
    max_epoch = 0
    for d in data:
        c = d.get("tidal_credits", 0)
        sm = d.get("seasonal_mult", 1.0)
        if c < min_cred:
            min_cred = c
            min_epoch = d["epoch"]
        if c > max_cred:
            max_cred = c
            max_epoch = d["epoch"]
        phase = "SUMMER" if sm > 0.85 else "WINTER" if sm < 0.65 else "transition"
        bar = "#" * int(c / 50)
        print(f"   E{d['epoch']:3d}  | {c:8.1f} | {sm:.3f}       | {phase:10s} {bar}")
    print()
    print(f"   Peak credits: {max_cred:.1f} at E{max_epoch}")
    print(f"   Trough credits: {min_cred:.1f} at E{min_epoch}")
    swing = (max_cred - min_cred) / max_cred * 100 if max_cred > 0 else 0
    print(f"   Swing: {swing:.0f}% (expect >30% from 0.5x winter multiplier)")
    print(f"   {'PASS' if swing > 20 else 'WEAK'}: {'seasonal modulation visible' if swing > 20 else 'seasonal effect too weak'}")
    print()

    # Seasonal period check: count summer peaks
    summers = [d["epoch"] for d in data if d.get("seasonal_mult", 1.0) > 0.9]
    winters = [d["epoch"] for d in data if d.get("seasonal_mult", 1.0) < 0.6]
    print(f"   Summer epochs (mult>0.9): {len(summers)} — {summers[:10]}{'...' if len(summers) > 10 else ''}")
    print(f"   Winter epochs (mult<0.6): {len(winters)} — {winters[:10]}{'...' if len(winters) > 10 else ''}")
    print()

    # --- 4. Food trail dynamics ---
    print("4. FOOD TRAIL DYNAMICS (rebuilding after tidal shifts)")
    food_cells = [d.get("food_trail_cells", 0) for d in data]
    noentry_cells = [d.get("noentry_trail_cells", 0) for d in data]
    avg_food = sum(food_cells) / len(food_cells) if food_cells else 0
    avg_noentry = sum(noentry_cells) / len(noentry_cells) if noentry_cells else 0
    food_variance = sum((f - avg_food) ** 2 for f in food_cells) / len(food_cells) if food_cells else 0
    print(f"   Avg food trail cells: {avg_food:.1f} (variance: {food_variance:.1f})")
    print(f"   Avg noentry cells: {avg_noentry:.1f}")
    print(f"   Food trail range: {min(food_cells)} - {max(food_cells)}")
    # High variance = trails being rebuilt (old fade, new form)
    cv = (food_variance ** 0.5) / avg_food if avg_food > 0 else 0
    print(f"   Coefficient of variation: {cv:.2f} (>0.3 = active rebuilding)")
    print(f"   {'PASS' if cv > 0.2 else 'STABLE'}: {'trails actively shifting' if cv > 0.2 else 'trails relatively stable'}")
    print()

    # Food trail per-epoch detail
    print("   Food trail timeline:")
    for d in data:
        f = d.get("food_trail_cells", 0)
        ne = d.get("noentry_trail_cells", 0)
        bar_f = "#" * (f // 3)
        bar_n = "!" * (ne // 3)
        print(f"   E{d['epoch']:3d}: food={f:3d} {bar_f:30s} noentry={ne:3d} {bar_n}")

    # --- 5. System interaction: noentry + tidal ---
    print()
    print("5. SYSTEM INTERACTION: NOENTRY x TIDAL")
    # After rift respawns (every 50 ticks within epoch), noentry should appear
    # then food trails should shift to new positions
    noentry_nonzero = sum(1 for n in noentry_cells if n > 0)
    print(f"   Epochs with noentry > 0: {noentry_nonzero}/{len(data)} ({noentry_nonzero/len(data)*100:.0f}%)")
    # Check correlation: when noentry spikes, does food trail count change next epoch?
    if len(data) > 2:
        shifts = 0
        for i in range(1, len(data) - 1):
            ne_prev = data[i-1].get("noentry_trail_cells", 0)
            ne_curr = data[i].get("noentry_trail_cells", 0)
            food_next = data[i+1].get("food_trail_cells", 0)
            food_curr = data[i].get("food_trail_cells", 0)
            if ne_curr > 20 and abs(food_next - food_curr) > 10:
                shifts += 1
        print(f"   Noentry->food shift events: {shifts} (noentry>20 followed by food change>10)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "metrics_log.jsonl"
    data = load_metrics(path)
    analyze(data)
