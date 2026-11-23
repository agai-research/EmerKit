/* EmerKit request phase in the browser. Faithful port of emerkit/compose.py (compose_task, forbidden_sets, shortlist, bounds,
   conflict explanation) and emerkit/e2e.py (validation, repair, substitution, backtracking). The composer solves program (27)
   exactly by enumerating the toolkits of the shortlist instead of calling CP-SAT; tests/check_engine.mjs checks that it reaches the
   same optimal objective as the Python implementation. Works in browsers and in Node (export at the end). */
(function (root) {
  const SCALE = 1000;
  const LABELS = ['src_ID', 'src_PROF', 'src_PERF', 'src_BEH', 'src_CONT', 'out'];
  const ATT = ['ID', 'PROF', 'PERF', 'BEH', 'CONT'];
  const median = a => {const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2};
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const inter = (a, b) => a.some(x => b.includes(x));

  function rng(seed) {let a = seed >>> 0; return () => {a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296}}

  class Engine {
    constructor(bundle, registry) {
      this.B = bundle; this.V = bundle.vocab; this.R = bundle.registries[registry]; this.T = this.R.tools; this.S = this.R.servers;
      this.sim = new Map(); for (const [i, j, s] of this.R.sims) this.sim.set(i < j ? i + '|' + j : j + '|' + i, s);
      this.byCap = {}; this.T.forEach((t, i) => t.caps.forEach(c => (this.byCap[c] = this.byCap[c] || []).push(i)));
    }
    /* labels at a level: worst case (a label is 1 when 1 is in its prediction set), 'truth', or 'declared' */
    labels(level, mode = 'worst') {
      const k = this.B.meta.levels.indexOf(level); if (mode === 'worst' && k < 0) throw new Error('level not prepared: ' + level);
      return this.T.map(t => {
        const o = {}; LABELS.forEach((l, b) => {
          if (mode === 'truth') o[l] = t.truth ? t.truth[b] : ((t.m[k][0] >> b) & 1);
          else if (mode === 'declared') o[l] = t.declared[b] === null ? 0 : t.declared[b];
          else o[l] = (t.m[k][0] >> b) & 1;
        }); o.uncertain = mode === 'worst' ? LABELS.filter((l, b) => (t.m[k][1] >> b) & 1) : []; return o;
      });
    }
    src(lab) {return ATT.filter(a => lab['src_' + a])}
    deny(profile) {const n = this.B.norms[profile]; return (ty, a, r) => !!n[ty + '|' + a + '|' + r]}
    candidates(task, exclude = new Set()) {
      const out = new Set(); for (const r of task.R) for (const i of (this.byCap[r] || [])) if (this.T[i].domains.includes(task.domain) && !exclude.has(i)) out.add(i);
      return [...out].sort((a, b) => a - b);
    }
    fit(task, cands) {
      const g = {}; for (const i of cands) for (const r of task.R) if (this.T[i].caps.includes(r)) {
        const phi = (this.V.phi[this.T[i].act] || {})[task.type] ?? 1.0; g[i + '|' + r] = Math.max(0, this.T[i].capsim[r]) * phi;
      } return g;
    }
    shortlist(task, cands, g, N, nmin) {
      const T = this.T, best = {}; for (const i of cands) best[i] = Math.max(0, ...task.R.map(r => g[i + '|' + r] ?? 0));
      const order = [...cands].sort((a, b) => best[b] - best[a] || a - b); const keep = [];
      for (const r of task.R) keep.push(...cands.filter(i => T[i].caps.includes(r)).sort((a, b) => (g[b + '|' + r] ?? 0) - (g[a + '|' + r] ?? 0) || a - b).slice(0, nmin));
      for (const a of task.ho_out || []) keep.push(...order.filter(i => T[i].prod.includes(a)).slice(0, nmin));
      for (const a of task.ho_in || []) keep.push(...order.filter(i => T[i].cons.includes(a)).slice(0, nmin));
      for (const i of order) {if (new Set(keep).size >= Math.max(N, new Set(keep).size)) break; keep.push(i)}
      return {Nv: [...new Set(keep)], best};
    }
    qosDeficit(Nv) {
      const T = this.T, S = this.S; if (!Nv.length) return {};
      const mm = v => {const lo = Math.min(...v), hi = Math.max(...v); return v.map(x => hi - lo < 1e-12 ? 0 : (x - lo) / (hi - lo))};
      const lat = mm(Nv.map(i => T[i].lat)), pr = mm(Nv.map(i => T[i].price)), av = mm(Nv.map(i => 1 - S[T[i].server].avail)), rel = mm(Nv.map(i => 1 - T[i].rel));
      const d = {}; Nv.forEach((i, k) => d[i] = (lat[k] + pr[k] + av[k] + rel[k]) / 4); return d;
    }
    bounds(task, cands, tau) {
      const T = this.T, S = this.S; let lat = 0, pr = 0, dep = 1;
      for (const r of task.R) {const p = cands.filter(i => T[i].caps.includes(r)); if (!p.length) continue;
        lat += median(p.map(i => T[i].lat)); pr += median(p.map(i => T[i].price)); dep *= median(p.map(i => T[i].rel * S[T[i].server].avail))}
      return [tau * lat, tau * pr + 1e-9, Math.max(1e-6, 1 - (1 - dep) * tau)];
    }
    forbidden(cand, labels, task, deny) {
      const T = this.T, ty = task.type, A0 = new Set([...task.D, ...(task.delta_in || [])]), F = new Map(), P = new Map(), ext = this.V.ext_roles;
      for (const i of cand) {
        const t = T[i], lab = labels[i], cause = [];
        for (const a of A0) if (t.acc.includes(a) && deny(ty, a, t.role)) cause.push([a, t.role, 'own']);
        if (lab.out) for (const a of new Set([...A0, ...this.src(lab)])) if (t.acc.includes(a)) for (const r of ext) if (deny(ty, a, r)) cause.push([a, r, 'forward']);
        if (cause.length) F.set(i, cause);
      }
      for (const i of cand) {
        if (F.has(i)) continue; const t = T[i], lab = labels[i];
        for (const j of cand) {
          if (j === i || F.has(j)) continue; const sj = this.src(labels[j]).filter(a => t.acc.includes(a)); if (!sj.length) continue; const cause = [];
          if (T[j].server !== t.server) for (const a of sj) if (deny(ty, a, t.role)) cause.push([a, t.role, 'own']);
          if (lab.out) for (const a of sj) for (const r of ext) if (deny(ty, a, r)) cause.push([a, r, 'forward']);
          if (cause.length) P.set(i + '|' + j, cause);
        }
      }
      return {F, P};
    }
    iota(i, j, beta, omega0) {
      if (!inter(this.T[i].caps, this.T[j].caps)) return 0;
      const s = this.sim.get(i < j ? i + '|' + j : j + '|' + i); return s === undefined ? 0 : sigmoid(beta * (s - omega0));
    }
    /* exact composition of one task; groups switch constraint families on and off (for conflict explanation) */
    solve(task, ctx, groups, wantBest = true) {
      const T = this.T, S = this.S, {Nv, g, dq, bounds, F, P, prm, best} = ctx, R = task.R;
      const on = gr => groups.has(gr), kmax = on('size') ? prm.k_max : Nv.length;
      const [latb, prb, depb] = bounds, LATB = Math.trunc(latb), PRB = Math.trunc(prb * 1e6), DEPB = Math.trunc(-Math.log(depb) * 1e5);
      const allowed = Nv.filter(i => !(on('flow') && F.has(i)));
      const bmax = Math.max(...Nv.map(i => best[i])) || 1, thB = Math.floor(prm.theta * prm.B), Eth = Math.max(1, prm.B - thB);
      let top = null; const K = [];
      const util = (i, r) => {const u = prm.use_qos ? g[i + '|' + r] * (1 - prm.lam_Q * dq[i]) : g[i + '|' + r]; return Math.trunc(SCALE * Math.max(0, u))};
      const evalSet = () => {
        if (on('flow')) for (let a = 0; a < K.length; a++) for (let b = 0; b < K.length; b++) if (a !== b && P.has(K[a] + '|' + K[b])) return;
        const servers = [...new Set(K.map(i => T[i].server))];
        const load = K.reduce((s, i) => s + T[i].cost, 0) + servers.reduce((s, x) => s + S[x].preamble, 0);
        if ((on('budget') || prm.lam_C > 0) && load > prm.B) return;   /* the saturation variable of program (27) is bounded by B */
        for (const a of task.ho_out || []) if (on('ho|' + a) && !K.some(i => T[i].prod.includes(a))) return;
        for (const a of task.ho_in || []) if (on('ho|' + a) && !K.some(i => T[i].cons.includes(a))) return;
        let penI = 0, penC = 0;
        if (prm.lam_I > 0) for (let a = 0; a < K.length; a++) for (let b = a + 1; b < K.length; b++) {
          const io = this.iota(K[a], K[b], prm.beta, prm.omega0); if (io > 1e-3) penI += Math.trunc(SCALE * prm.lam_I * (best[K[a]] / bmax) * (best[K[b]] / bmax) * io)}
        if (prm.lam_C > 0) {const e = Math.min(Eth, Math.max(0, load - thB)); penC = Math.trunc(SCALE * prm.lam_C * (e / Eth) ** 2)}
        const pen = penI + penC;
        const depS = servers.reduce((s, x) => s + Math.trunc(-Math.log(S[x].avail) * 1e5), 0);
        /* assignment of capabilities to providers in K: each covered capability by exactly one provider (at most one when 'cov|r' is off) */
        const opts = R.map(r => {const p = K.filter(i => T[i].caps.includes(r)); return on('cov|' + r) ? p : [null, ...p]});
        if (opts.some(o => !o.length)) return;
        const W = new Array(R.length);
        const rec = (k, u, lat, pr, dep) => {
          if (k === R.length) {
            if (on('qos') && prm.use_qos && (lat > LATB || pr > PRB || depS + dep > DEPB)) return;
            const v = u - pen; if (!top || v > top.v) top = {v, u, penI, penC, lat, pr, dep: depS + dep, K: [...K], W: Object.fromEntries(R.map((r, q) => [r, W[q]]).filter(x => x[1] !== null)), load};
            return;
          }
          for (const i of opts[k]) {
            W[k] = i; if (i === null) {rec(k + 1, u, lat, pr, dep); continue}
            rec(k + 1, u + util(i, R[k]), lat + Math.trunc(T[i].lat), pr + Math.trunc(T[i].price * 1e6), dep + Math.trunc(-Math.log(T[i].rel) * 1e5));
          }
        };
        rec(0, 0, 0, 0, 0);
      };
      const walk = (start) => {
        if (K.length) {evalSet(); if (top && !wantBest) return true}
        if (K.length >= kmax) return false;
        for (let s = start; s < allowed.length; s++) {K.push(allowed[s]); const stop = walk(s + 1); K.pop(); if (stop) return true}
        return false;
      };
      walk(0); return top;
    }
    compose(task, labels, deny, params = {}, exclude = new Set(), explain = true) {
      const prm = {...this.B.meta.defaults, ...params}, t0 = performance.now();
      if (!task.R.length) return {K: [], W: {}, status: 'no_tool_needed', obj: 0, ms: 0};
      const cands = this.candidates(task, exclude);
      const unc = task.R.filter(r => !cands.some(i => this.T[i].caps.includes(r)));
      if (unc.length) return {K: null, status: 'uncovered', uncovered: unc, ms: performance.now() - t0, cands};
      const g = this.fit(task, cands), {Nv, best} = this.shortlist(task, cands, g, prm.N, prm.n_min), dq = this.qosDeficit(Nv);
      const bounds = this.bounds(task, cands, prm.tau), {F, P} = prm.use_flow ? this.forbidden(Nv, labels, task, deny) : {F: new Map(), P: new Map()};
      const groups = new Set(['size', 'flow', 'budget', 'qos', ...task.R.map(r => 'cov|' + r), ...[...(task.ho_out || []), ...(task.ho_in || [])].map(a => 'ho|' + a)]);
      const ctx = {Nv, g, dq, bounds, F, P, prm, best};
      const top = this.solve(task, ctx, groups);
      const base = {cands, Nv, g, dq, bounds, F, P, best, ms: performance.now() - t0};
      if (top) return {...base, K: top.K, W: top.W, obj: top.v / SCALE, load: top.load, parts: {utility: top.u / SCALE, interference: top.penI / SCALE, saturation: top.penC / SCALE},
        qos: {lat: top.lat, latb: Math.trunc(bounds[0]), price: top.pr / 1e6, priceb: Math.trunc(bounds[1] * 1e6) / 1e6, dep: Math.exp(-top.dep / 1e5), depb: Math.exp(-Math.trunc(-Math.log(bounds[2]) * 1e5) / 1e5)},
        params: prm, status: 'optimal', ms: performance.now() - t0};
      let conflict = null;
      if (explain) {
        const rank = gr => ['budget', 'qos', 'size', 'ho', 'flow', 'cov'].indexOf(gr.split('|')[0]);
        const order = [...groups].sort((a, b) => rank(a) - rank(b)); let keep = new Set(groups);
        for (const gr of order) {const trial = new Set(keep); trial.delete(gr); if (!this.solve(task, ctx, trial, false)) keep = trial}
        conflict = [...keep].sort();
      }
      return {...base, K: null, status: 'infeasible', conflict, ms: performance.now() - t0};
    }
    risk(K, labels, task, deny) {const {F, P} = this.forbidden(K, labels, task, deny); return F.size + P.size}

    /* ------------------------------------------------------------- workflows */
    closure(domain, targets) {
      const pre = Object.fromEntries(this.V.concepts[domain]), seen = [];
      const visit = c => {for (const p of pre[c] || []) visit(p); if (!seen.includes(c)) seen.push(c)}; targets.forEach(visit); return seen;
    }
    makeTask(r, domain, ty, concepts, n = null) {
      const V = this.V, allowed = V.type_caps[ty].filter(c => !V.caps[c].domains || V.caps[c].domains.includes(domain));
      const core = V.core_cap[ty + '|' + domain] || V.core_cap[ty + '|None'] || V.core_cap[ty + '|null'];
      const R = new Set(core && allowed.includes(core) ? [core] : []); const want = n ?? 1 + Math.floor(r() * 4);
      const pool = allowed.filter(c => !R.has(c)); while (R.size < want && pool.length) R.add(pool.splice(Math.floor(r() * pool.length), 1)[0]);
      return {type: ty, domain, concepts: [...concepts], level: V.level[ty], R: [...R], D: [...V.default_d[ty]], asm: V.asm[ty]};
    }
    edges(tasks) {
      const V = this.V, out = [];
      for (let i = 0; i < tasks.length - 1; i++) {const ho = [];
        for (const art of Object.keys(V.handoff)) if (tasks[i].R.some(c => V.caps[c].prod.includes(art)) && tasks[i + 1].R.some(c => V.caps[c].cons.includes(art))) ho.push(art);
        out.push([i, i + 1, ho])} return out;
    }
    ruleWorkflow(domain, targets, seed = 1) {
      const r = rng(seed), full = this.closure(domain, targets), keep = new Set([...full.slice(-3), ...targets]), chain = full.filter(c => keep.has(c)), tasks = [];
      for (const c of chain) {if (targets.includes(c) || r() < 0.5) tasks.push(this.makeTask(r, domain, 'acquisition', [c])); tasks.push(this.makeTask(r, domain, 'practice', [c]))}
      tasks.push(this.makeTask(r, domain, 'formative', chain.slice(-2)));
      const x = r(), end = x < 0.5 ? 'summative' : x < 0.8 ? 'production' : 'discussion'; tasks.push(this.makeTask(r, domain, end, targets));
      const ts = tasks.slice(-8); return {domain, targets, tasks: ts, edges: this.edges(ts), source: 'rule-based generator'};
    }
    validate(wf) {
      const V = this.V, errors = [], pre = Object.fromEntries(V.concepts[wf.domain]), first = {};
      wf.tasks.forEach((t, i) => t.concepts.forEach(c => {if (!(c in first)) first[c] = i}));
      wf.tasks.forEach((t, i) => t.concepts.forEach(c => (pre[c] || []).forEach(p => {if (p in first && first[p] > first[c]) errors.push({kind: 'prerequisite', task: i, detail: `${p} comes after ${c}`})})));
      wf.tasks.forEach((t, i) => {const bad = t.R.filter(r => !V.type_caps[t.type].includes(r) || (V.caps[r].domains && !V.caps[r].domains.includes(wf.domain))); if (bad.length) errors.push({kind: 'capability', task: i, detail: bad.join(', ')})});
      const covered = new Set(wf.tasks.flatMap(t => t.concepts)); const miss = wf.targets.filter(c => !covered.has(c)); if (miss.length) errors.push({kind: 'objectives', detail: miss.join(', ')});
      return errors;
    }
    repair(wf) {
      const V = this.V, order = V.concepts[wf.domain].map(x => x[0]), rank = Object.fromEntries(order.map((c, k) => [c, k])), log = []; let tasks = [];
      for (const t0 of wf.tasks) {
        const t = {...t0, concepts: t0.concepts.filter(c => c in rank)}; if (t.concepts.length !== t0.concepts.length) log.push(`removed unknown concepts from a ${t.type} task`);
        if (!t.concepts.length) {log.push(`dropped a ${t.type} task without known concept`); continue}
        let ok = t.R.filter(r => V.type_caps[t.type].includes(r) && (!V.caps[r].domains || V.caps[r].domains.includes(wf.domain)));
        if (!ok.length) {const core = V.core_cap[t.type + '|' + wf.domain] || V.core_cap[t.type + '|None']; ok = core ? [core] : []}
        if (ok.join() !== t.R.join()) log.push(`kept admissible capabilities of a ${t.type} task (${ok.join(', ') || 'none'})`);
        tasks.push({...t, R: ok, D: [...new Set([...(t.D || []), ...V.default_d[t.type]])], level: V.level[t.type], asm: V.asm[t.type]});
      }
      const covered = new Set(tasks.flatMap(t => t.concepts));
      for (const c of wf.targets) if (!covered.has(c) && c in rank) {tasks.push(this.makeTask(rng(7), wf.domain, 'practice', [c], 1)); log.push(`added a practice task for ${c}`)}
      const sorted = [...tasks].sort((a, b) => Math.min(...a.concepts.map(c => rank[c])) - Math.min(...b.concepts.map(c => rank[c])) || this.V.types.indexOf(a.type) - this.V.types.indexOf(b.type));
      if (sorted.some((t, k) => t !== tasks[k])) log.push('reordered tasks so that prerequisites come first');
      return {...wf, tasks: sorted, edges: this.edges(sorted), repairs: log};
    }
    inherited(wf, i) {const out = new Set(); for (const [a, b, ho] of wf.edges) if (b === i) for (const art of ho) this.V.handoff[art].forEach(x => out.add(x)); return [...out]}
    view(wf, i, mode, plan, labels) {
      const t = {...wf.tasks[i]};
      t.ho_out = [...new Set(wf.edges.filter(e => e[0] === i).flatMap(e => e[2]))]; t.ho_in = [...new Set(wf.edges.filter(e => e[1] === i).flatMap(e => e[2]))];
      const d = new Set(this.inherited(wf, i));
      if (mode === 'conservative') for (let u = 0; u < i; u++) {wf.tasks[u].D.forEach(a => d.add(a)); for (const j of plan[u]?.K || []) this.src(labels[j]).forEach(a => d.add(a))}
      t.delta_in = [...d]; return t;
    }
    preserves(v, vp, sub) {
      const V = this.V;
      return v.concepts.every(c => vp.concepts.includes(c)) && V.level[vp.type] >= V.level[v.type] && (sub[v.type] || []).includes(vp.type) && v.asm !== 'summative'
        && (v.ho_out || []).every(a => vp.R.some(r => V.caps[r].prod.includes(a))) && (v.ho_in || []).every(a => vp.R.some(r => V.caps[r].cons.includes(a)));
    }
    substitutes(r, v, sub, ns = 4) {
      const V = this.V, out = [];
      for (const ty of sub[v.type] || []) for (let k = 0; k < ns; k++) {
        const vp = this.makeTask(r, v.domain, ty, v.concepts);
        for (const a of v.ho_out || []) {const p = V.type_caps[ty].find(c => V.caps[c].prod.includes(a) && (!V.caps[c].domains || V.caps[c].domains.includes(v.domain))); if (p && !vp.R.includes(p)) vp.R.push(p)}
        for (const a of v.ho_in || []) {const p = V.type_caps[ty].find(c => V.caps[c].cons.includes(a) && (!V.caps[c].domains || V.caps[c].domains.includes(v.domain))); if (p && !vp.R.includes(p)) vp.R.push(p)}
        Object.assign(vp, {ho_out: v.ho_out, ho_in: v.ho_in, delta_in: v.delta_in}); out.push(vp);
      } return out;
    }
    /* Algorithm 3 on one workflow: per-task composition, one backtracking step in conservative mode, best admissible substitute */
    runWorkflow(wf, labels, deny, params = {}, opts = {}) {
      const mode = opts.mode || 'isolated', smax = opts.s_max ?? 2, sub = opts.sub || this.V.sub, r = rng(opts.seed || 11), trace = [];
      let plan = {}, gaps = [], subs = [], backtracks = 0;
      for (let i = 0; i < wf.tasks.length; i++) {
        let v = this.view(wf, i, mode, plan, labels), res = this.compose(v, labels, deny, params);
        const events = [];
        if (!res.K && mode === 'conservative' && i > 0) {
          const extra = v.delta_in.filter(a => !v.D.includes(a) && !this.inherited(wf, i).includes(a));
          for (let u = i - 1; u >= 0; u--) {
            const bad = (plan[u]?.K || []).filter(j => this.src(labels[j]).some(a => extra.includes(a))); if (!bad.length) continue;
            const uv = this.view(wf, u, mode, plan, labels), ru = this.compose(uv, labels, deny, params, new Set(bad), false);
            if (ru.K) {const trial = {...plan, [u]: {...ru, task: uv}}; const v2 = this.view(wf, i, mode, trial, labels), r2 = this.compose(v2, labels, deny, params);
              if (r2.K) {plan = trial; v = v2; res = r2; backtracks++; events.push(`backtracked to step ${u + 1} and removed ${bad.map(j => this.T[j].name).join(', ')}`)}}
            break;
          }
        }
        if (!res.K && subs.length < smax && v.asm !== 'summative') {
          let cand = null;
          for (const vp of this.substitutes(r, v, sub)) {if (!this.preserves(v, vp, sub)) continue; const rp = this.compose(vp, labels, deny, params, new Set(), false); if (rp.K && (!cand || rp.obj > cand[1].obj)) cand = [vp, rp]}
          if (cand) {subs.push({index: i, original: v, substitute: cand[0]}); events.push(`replaced the ${v.type} task by a ${cand[0].type} task`); res = {...cand[1], original: res}; v = cand[0];
            wf.tasks[i] = {type: v.type, domain: v.domain, concepts: v.concepts, level: v.level, R: v.R, D: v.D, asm: v.asm}}
        }
        if (!res.K) gaps.push(i);
        plan[i] = {...res, task: v, events};
      }
      return {wf, plan, gaps, subs, backtracks};
    }
  }
  const api = {Engine, rng, LABELS, ATT};
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.EmerKitEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
