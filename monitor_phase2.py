"""Phase 2 monitor — wait for epoch 50 checkpoint, then epoch 100."""
import json, time, os

METRICS = "D:/DEV/AI_Workspace/active/Blub/metrics_log.jsonl"
CHECK_INTERVAL = 30  # seconds between checks

def read_metrics():
    if not os.path.exists(METRICS):
        return []
    with open(METRICS, "r") as f:
        return [json.loads(l) for l in f if l.strip()]

def summarize(metrics, label=""):
    if not metrics:
        print(f"[{label}] No data yet")
        return
    m = metrics[-1]
    ep = m.get("epoch", len(metrics))
    print(f"[{label}] Epoch {ep}: MI={m.get('mutual_info',0):.2f} sMI={m.get('social_mi',0):.2f} "
          f"TS={m.get('top_sim',0):.4f} CSR={m.get('social_csr',0):.1%} "
          f"PCA={m.get('pca',0):.3f} PD={m.get('pos_dis',0):.3f} "
          f"VA={m.get('vocab_argmax',0)} retire={m.get('bottleneck_retirements',0)}")

def avg_last_n(metrics, key, n=5):
    vals = [m.get(key, 0) for m in metrics[-n:]]
    return sum(vals) / len(vals) if vals else 0

print("=== Phase 2 Monitor: waiting for epoch 100 (checkpoint at 50) ===")
checkpoint_done = False

while True:
    metrics = read_metrics()
    n = len(metrics)
    
    if n > 0:
        latest_epoch = metrics[-1].get("epoch", n)
        
        # Checkpoint at epoch 50
        if not checkpoint_done and latest_epoch >= 50:
            checkpoint_done = True
            print("\n" + "="*70)
            print("CHECKPOINT: EPOCH 50")
            print("="*70)
            ts_avg = avg_last_n(metrics, "top_sim", 10)
            mi_avg = avg_last_n(metrics, "mutual_info", 10)
            pd_avg = avg_last_n(metrics, "pos_dis", 10)
            csr_avg = avg_last_n(metrics, "social_csr", 10)
            pca_avg = avg_last_n(metrics, "pca", 10)
            va_avg = avg_last_n(metrics, "vocab_argmax", 10)
            print(f"  TopSim (last 10 avg): {ts_avg:.4f}")
            print(f"  MI (last 10 avg):     {mi_avg:.2f}")
            print(f"  PosDis (last 10 avg): {pd_avg:.3f}")
            print(f"  CSR (last 10 avg):    {csr_avg:.1%}")
            print(f"  PCA (last 10 avg):    {pca_avg:.3f}")
            print(f"  VocabMax (last 10):   {va_avg:.0f}")
            
            if ts_avg < 0.02:
                print("\n  >>> ALERT: TopSim has NOT moved. Recommend observation_rate 0.35 -> 0.25 <<<")
            else:
                print(f"\n  >>> TopSim showing movement ({ts_avg:.4f}). Keep current settings. <<<")
            print("="*70 + "\n")
        
        # Report every 10 epochs
        if latest_epoch % 10 == 0:
            summarize(metrics, f"E{latest_epoch}")
        
        # Done at 100
        if latest_epoch >= 100:
            print("\n" + "="*70)
            print("FINAL REPORT: EPOCH 100")
            print("="*70)
            # Trend analysis
            early = metrics[5:15]   # epochs ~5-15
            mid = metrics[25:35]    # epochs ~25-35  
            late = metrics[-10:]    # last 10 epochs
            
            for label, chunk in [("Early (5-15)", early), ("Mid (25-35)", mid), ("Late (90-100)", late)]:
                if chunk:
                    ts = sum(m.get("top_sim",0) for m in chunk)/len(chunk)
                    mi = sum(m.get("mutual_info",0) for m in chunk)/len(chunk)
                    pd = sum(m.get("pos_dis",0) for m in chunk)/len(chunk)
                    csr = sum(m.get("social_csr",0) for m in chunk)/len(chunk)
                    pca = sum(m.get("pca",0) for m in chunk)/len(chunk)
                    va = sum(m.get("vocab_argmax",0) for m in chunk)/len(chunk)
                    print(f"  {label:15s}: TS={ts:.4f} MI={mi:.2f} PD={pd:.3f} CSR={csr:.1%} PCA={pca:.3f} VA={va:.0f}")
            
            print("="*70)
            break
    
    time.sleep(CHECK_INTERVAL)

print("\nMonitor complete.")
