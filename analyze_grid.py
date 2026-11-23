import glob, os, numpy as np, pandas as pd
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "grid")
def L(s): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{G}/{s}_c*.csv"))], ignore_index=True)
def add(d):
    d["valid"] = ((d["eq"] == 1) & (d["cov"] == 1) & (d["dfr"] == 0) & (d["qviol"] == 0)).astype(float)
    for k in ["dfr", "qviol", "red", "lb", "over"]:
        d[k + "_e"] = np.where(d["eq"] == 1, d[k], np.nan)
    return d
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
if __name__ == "__main__":
    d = add(L("A4")); print("A4 eq/valid/qviol_e"); print(d.pivot_table(index="tau", columns="method", values=["eq", "valid"]).round(3).to_string())
    d = add(L("B2"))
    for v in ["eq", "dfr_e", "valid"]:
        print("B2", v); print(d.pivot_table(index=["m", "w", "cue"], columns="method", values=v).round(3).to_string())
    d = add(L("B3"))
    for v in ["eq", "dfr_e"]:
        print("B3", v); print(d.pivot_table(index=["profile", "ext_bias"], columns="method", values=v).round(3).to_string())
    d = add(L("D2")); print("D2"); print(d.groupby(["lam_I", "lam_C", "lam_Q", "theta"])[["eq", "valid", "dfr_e", "red_e", "lb_e", "over_e", "qviol_e", "time"]].mean().round(3).to_string())
    c = L("C3"); c["opt"] = (c.status == "optimal").astype(float); c["to"] = (c.status == "timeout").astype(float); c["inf"] = (c.status == "infeasible").astype(float)
    print("C3"); print(c.groupby(["N", "nR", "k_max", "T_max"]).agg(t_med=("time", "median"), t_p95=("time", lambda x: np.percentile(x, 95)), opt=("opt", "mean"), to=("to", "mean"), inf=("inf", "mean"), Nv=("Nv", "mean"), conf=("conflict_size", "mean")).round(3).to_string())
    c = L("C4"); print("C4"); print(c.groupby("size").agg(n=("n_tools", "mean"), prep=("prep_s", "mean"), em_med=("emerkit_s", "median"), em_p95=("emerkit_s", lambda x: np.percentile(x, 95)), rag=("rag_s", "median"), zeng=("zeng_s", "median"), eq=("eq", "mean"), real=("real_in_K", "mean"), rss=("maxrss_mb", "max")).round(4).to_string())
    c = L("C2"); c["no_workflow"] = c.get("no_workflow", 0)
    print("C2"); print(c.groupby(["config", "s_max", "density", "nu", "mode", "method"]).agg(eq=("equipped", "mean"), sub=("substituted", "mean"), gaps=("gaps", "mean"), all_eq=("all_equipped", "mean"), dfr=("dfr", "mean"), any_dfr=("any_dfr", "mean"), bt=("backtracks", "mean"), t=("time", "mean"), nowf=("no_workflow", "sum")).round(3).to_string())
    for s in ["E1", "E2"]:
        e = L(s); keys = ["catalogue", "profile"] if s == "E1" else ["remove", "annotate"]
        w = e.groupby(keys + ["cls"]).dem.sum() / e.groupby(keys).dem.sum()
        print(s); print(w.unstack().fillna(0).round(3).to_string())
