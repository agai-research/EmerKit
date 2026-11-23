"""Preparation phase (Algorithm 1): capabilities, behavioural labels, conformal sets, interference index, QoS deficit."""
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .vocab import ATT, ROLES, CAPS
from .data import LABELS

CUE = {
 "out": re.compile(r"sync|share|send|upload|cloud|external|webhook|recipient|email|callback|partner|post", re.I),
 "src_PERF": re.compile(r"score|grade|mark|result", re.I),
 "src_ID": re.compile(r"identifier|student_id|learner_id|user_id|account|identity", re.I),
 "src_BEH": re.compile(r"activity|history|engagement|review log|track", re.I),
 "src_PROF": re.compile(r"profile|preference|prior knowledge", re.I),
 "src_CONT": re.compile(r"output text|execute|run |returns the output|notebook|full output", re.I),
}


def tool_text(t):
    return f"{t['name'].replace('_', ' ')} {t['desc']} {' '.join(p.replace('_', ' ') for p in t['params'])}"


class Prepared:
    """Holds everything the composer needs about a catalogue."""

    def __init__(self, tools, servers, seed=0, cal_frac=0.3, cap_threshold=0.30, cap_vocab=None, keep_caps=False):
        self.tools, self.servers = tools, servers
        self.idx = {t["id"]: i for i, t in enumerate(tools)}
        rng = np.random.default_rng(seed + 31)
        texts = [tool_text(t) for t in tools]
        vocab = cap_vocab or {c: CAPS[c][0] for c in CAPS}
        capnames = list(vocab)
        captexts = [vocab[c] + " " + c.replace("_", " ") for c in capnames]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(texts + captexts)
        self.X = self.vec.transform(texts)
        C = self.vec.transform(captexts)
        sim = (self.X @ C.T).toarray()
        self.capnames, self.capvec, self.capvec_text = capnames, C, captexts
        # capability extraction (pilot: similarity mapper; Claude Code: language model)
        for i, t in enumerate(tools):
            if keep_caps and t.get("caps") is not None:      # capabilities given by an external extractor (LLM)
                continue
            row = sim[i]
            caps = {capnames[j] for j in np.where(row >= cap_threshold)[0]}
            if not caps and row.max() >= 0.15:
                caps = {capnames[int(row.argmax())]}
            t["caps"] = caps
        self.capsim = sim
        # description similarity for the interference index
        dv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([t["desc"] for t in tools])
        self.D = dv.transform([t["desc"] for t in tools])
        # labelled calibration tools: synthetic tools of a random 30% of synthetic servers
        syn_servers = sorted({t["server"] for t in tools if t["synthetic"]})
        cal_servers = set(rng.choice(syn_servers, max(1, int(round(cal_frac * len(syn_servers)))), replace=False)) if syn_servers else set()
        self.cal = [i for i, t in enumerate(tools) if t["synthetic"] and t["server"] in cal_servers]
        self.test_servers = set(syn_servers) - cal_servers
        self.cal_set = set(self.cal)
        self._fit_labels()

    # ------------------------------------------------------------------ labels
    def _features(self, label):
        roles = np.array([[t["role"] == r for r in ROLES] for t in self.tools], dtype=float)
        h = np.array([[0.0 if t["hints"][label] is None else 1.0, float(t["hints"][label] or 0)] for t in self.tools])
        return hstack([self.X, csr_matrix(roles), csr_matrix(h)]).tocsr()

    def _lf_vote(self, t, label):
        """Labelling functions (weak supervision): hints, role priors, lexical cues; majority vote."""
        votes = []
        if t["hints"][label] is not None:
            votes += [t["hints"][label]] * 2
        txt = tool_text(t)
        if CUE[label].search(txt):
            votes.append(1)
        if label == "out":
            votes.append(1 if t["role"] == "PUB" else 0 if t["role"] == "INS" else None)
        votes = [v for v in votes if v is not None]
        if not votes:
            return 0
        return int(np.mean(votes) >= 0.5)

    def _fit_labels(self):
        self.p1 = {}
        self.cal_scores = {}
        for l in LABELS:
            F = self._features(l)
            y = np.array([self._lf_vote(t, l) for t in self.tools])
            if y.min() == y.max():
                p = np.full(len(self.tools), float(y[0]))
            else:
                clf = LogisticRegression(max_iter=2000, C=2.0).fit(F, y)
                p = clf.predict_proba(F)[:, 1]
            p = np.clip(p, 1e-4, 1 - 1e-4)
            self.p1[l] = p
            ytrue = np.array([self.tools[i]["truth"][l] for i in self.cal])
            pc = p[self.cal]
            self.cal_scores[l] = np.sort(1 - np.where(ytrue == 1, pc, 1 - pc))

    def conformal_sets(self, eps):
        """Gamma[label][i] subset of {0,1}; eps=0 gives {0,1} everywhere."""
        G = {}
        for l in LABELS:
            s = self.cal_scores[l]; n = len(s)
            if eps <= 0:
                q = np.inf
            else:
                k = int(np.ceil((n + 1) * (1 - eps)))
                q = np.inf if k > n else s[k - 1]
            p = self.p1[l]
            in1 = (1 - p) <= q
            in0 = p <= q
            sets = []
            for a, b, pi in zip(in0, in1, p):
                st = set()
                if a: st.add(0)
                if b: st.add(1)
                sets.append(st or {int(pi >= 0.5)})
            G[l] = sets
        return G

    def labels_for(self, G, mode):
        """Resolved labels per tool: worst = 1 if 1 in Gamma; best = 1 only if Gamma == {1}; truth = ground truth (None if unknown)."""
        res = []
        for i, t in enumerate(self.tools):
            if mode == "truth":
                res.append(None if t["truth"] is None else dict(t["truth"]))
            elif mode == "declared":   # naive filter: declared hints, absent = safe
                res.append({l: int(t["hints"][l] or 0) for l in LABELS})
            elif i in self.cal_set:     # audited calibration tools: labels known
                res.append(dict(t["truth"]))
            else:
                d = {}
                for l in LABELS:
                    g = G[l][i]
                    d[l] = int(1 in g) if mode == "worst" else int(g == {1})
                res.append(d)
        return res

    # ------------------------------------------------------------------ usage traces (co-usage mining, trace evidence)
    def set_traces(self, traces, k_near=50):
        """traces: list of dict(query=str, tools=[catalogue indices], success=bool). Only successful trajectories count."""
        self.traces = [t for t in traces if t.get("success", True) and t["tools"]]
        self.k_near = k_near
        n = len(self.tools); self.use_count = np.zeros(n)
        self.co_count = {}
        for tr in self.traces:
            ids = sorted(set(tr["tools"]))
            for i in ids:
                self.use_count[i] += 1
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    self.co_count[(ids[a], ids[b])] = self.co_count.get((ids[a], ids[b]), 0) + 1
        self.Q = self.vec.transform([t["query"] for t in self.traces]) if self.traces else None

    def co_usage(self, i, j):
        """co(t,t') = joint successful uses / min(individual uses), in [0,1]; 0 without traces."""
        if not getattr(self, "traces", None):
            return 0.0
        c = self.co_count.get((min(i, j), max(i, j)), 0)
        m = min(self.use_count[i], self.use_count[j])
        return float(c / m) if m > 0 else 0.0

    def evidence(self, i, task_text):
        """ev(t,v): share of the k most similar successful trajectories (by query) that used tool t."""
        if not getattr(self, "traces", None):
            return 0.0
        key = (i, task_text)
        cache = self.__dict__.setdefault("_ev_cache", {})
        if task_text not in cache:
            sims = (self.Q @ self.vec.transform([task_text]).T).toarray().ravel()
            near = np.argsort(-sims)[: self.k_near]
            cache[task_text] = {}
            for k in near:
                for t in set(self.traces[k]["tools"]):
                    cache[task_text][t] = cache[task_text].get(t, 0) + 1 / len(near)
        return float(cache[task_text].get(i, 0.0))

    # ------------------------------------------------------------------ similarity and interference
    def desc_sim(self, i, j):
        return float(self.D[i].multiply(self.D[j]).sum())

    def interference(self, i, j, beta=20.0, omega0=0.75):
        if not (self.tools[i]["caps"] & self.tools[j]["caps"]):
            return 0.0
        s = self.desc_sim(i, j)
        return float(1.0 / (1.0 + np.exp(-beta * (s - omega0)))) * (1.0 - self.co_usage(i, j))   # tools used together do not interfere
