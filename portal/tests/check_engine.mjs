// Cross-check of the browser engine against the Python implementation: same feasibility, same optimal objective, same conflicts.
import fs from 'fs'; import {createRequire} from 'module';
const require = createRequire(import.meta.url); globalThis.performance ??= {now: () => Date.now()};
const {Engine} = require('../engine.js');
const bundle = JSON.parse(fs.readFileSync(new URL('../bundle.json', import.meta.url))); const cases = JSON.parse(fs.readFileSync(new URL('./cases.json', import.meta.url)));
const E = new Engine(bundle, 'emerkit'); const lab = E.labels(0.05);
let feas = 0, obj = 0, conf = 0, nconf = 0, same = 0, n = 0, worst = 0; const t0 = Date.now();
for (const c of cases) {
  const r = E.compose(c.task, lab, E.deny(c.profile), c.params); n++;
  const pyOk = c.obj !== null && c.obj !== undefined, jsOk = !!r.K;
  if (pyOk === jsOk) feas++;
  if (pyOk && jsOk) {const d = Math.abs(r.obj - c.obj); worst = Math.max(worst, d); if (d < 1e-6) obj++; if (JSON.stringify([...r.K].sort((a, b) => a - b)) === JSON.stringify(c.K)) same++}
  if (c.status === 'infeasible' && r.status === 'infeasible') {nconf++; if (JSON.stringify([...r.conflict].sort()) === JSON.stringify([...c.conflict].sort())) conf++}
}
const both = cases.filter(c => c.obj !== null && c.obj !== undefined).length;
console.log(`cases ${n}; same feasibility ${feas}/${n}; same optimal objective ${obj}/${both} (max gap ${worst.toExponential(2)}); same toolkit ${same}/${both}; same minimal conflict ${conf}/${nconf}; ${(Date.now() - t0) / n} ms per task`);
process.exit(feas === n && obj === both && conf === nconf ? 0 : 1);
