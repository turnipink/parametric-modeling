/* ============================================================
   App: tab navigation + Plotly figures backed by results.json
   ============================================================ */

const STEPS = [
  { id: "overview",  label: "Overview"          },
  { id: "step1",     label: "1. Sample"         },
  { id: "step2",     label: "2. Fit physics"    },
  { id: "step3",     label: "3. Evaluate"       },
  { id: "step4",     label: "4. Optimize"       },
  { id: "tradeoffs", label: "Tradeoffs"         },
  { id: "ai",        label: "AI in the fab"     },
];

// --- nav ---------------------------------------------------------------
function buildNav() {
  const nav = document.getElementById("stepnav");
  STEPS.forEach((s, i) => {
    const b = document.createElement("button");
    b.dataset.target = s.id;
    b.innerHTML = `<span class="num">${i === 0 ? "·" : i}</span>${s.label}`;
    b.addEventListener("click", () => show(s.id));
    nav.appendChild(b);
  });
}
function show(id) {
  document.querySelectorAll(".step")
    .forEach(el => el.classList.toggle("active", el.dataset.step === id));
  document.querySelectorAll(".steps button")
    .forEach(b => b.classList.toggle("active", b.dataset.target === id));
  history.replaceState(null, "", "#" + id);
  window.scrollTo({ top: 0, behavior: "instant" });
  // re-render math in case the tab was never displayed before
  if (window.renderMathInElement) {
    renderMathInElement(document.querySelector(`.step[data-step="${id}"]`),
      { delimiters: [
          {left:'$$',right:'$$',display:true},
          {left:'$', right:'$', display:false}
      ]});
  }
}

// --- formatting helpers ------------------------------------------------
const fmt = {
  sci(x, d=2) {
    if (x === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(x)));
    const m = x / Math.pow(10, e);
    return `${m.toFixed(d)}\u00d710^${e}`;
  },
  num(x, d=2) { return Number(x).toFixed(d); },
};

// --- data loader -------------------------------------------------------
async function loadResults() {
  const r = await fetch("assets/data/results.json", { cache: "no-store" });
  if (!r.ok) throw new Error("results.json not found — run python/generate_results.py first.");
  return await r.json();
}

// --- plot common layout ------------------------------------------------
const PLOT_LAYOUT_BASE = {
  paper_bgcolor: "#161c25",
  plot_bgcolor:  "#0e1217",
  font: { color: "#e7ecf3", family: "Inter, -apple-system, sans-serif", size: 12 },
  margin: { l: 60, r: 24, t: 32, b: 50 },
  legend: { bgcolor: "rgba(0,0,0,0)", bordercolor: "#2a3340", borderwidth: 1 },
};
const CFG = { displayModeBar: false, responsive: true };

// --- step renderers ----------------------------------------------------

function renderOverview(data) {
  const sp = data.process.spec;
  document.getElementById("spec-rs").innerHTML =
    `target <strong>${sp.Rs_target}</strong> &plusmn; ${sp.Rs_tol} &Omega;/sq`;
  document.getElementById("spec-xj").innerHTML =
    `&le; <strong>${sp.Xj_max}</strong> nm`;
  document.getElementById("spec-tb").innerHTML =
    `&le; <strong>${sp.Dt_budget_max.toExponential(1)}</strong> cm&sup2;`;
}

function renderLHSPlot(data) {
  const s = data.training.samples;
  const traces = [
    {
      type: "splom",
      dimensions: [
        { label: "log10(dose)",     values: s.dose.map(v => Math.log10(v)) },
        { label: "T_anneal (K)",    values: s.T_anneal },
        { label: "log10(t_anneal)", values: s.t_anneal.map(v => Math.log10(v)) },
        { label: "log10(pO2)",      values: s.pO2.map(v => Math.log10(v)) },
      ],
      marker: {
        color: s.Rs,
        colorscale: "Viridis",
        size: 7,
        line: { color: "#0e1217", width: 0.5 },
        colorbar: { title: "Rs (Ω/sq)", thickness: 12 },
      },
      diagonal: { visible: false },
      showupperhalf: false,
    },
  ];
  const layout = Object.assign({}, PLOT_LAYOUT_BASE, {
    title: { text: `LHS plan: ${data.training.n_samples} points (color = simulated Rs)`,
             font: { size: 14 } },
    height: 440,
  });
  Plotly.newPlot("plot-lhs", traces, layout, CFG);
}

function renderPhysParams(data) {
  const p = data.physics_params;
  const fi = data.fit_info;
  const rows = [
    ["log10(D0)       (cm²/s)",       p.log10_D0.toFixed(3)],
    ["Ea_diff         (eV)",          p.Ea_diff.toFixed(3)],
    ["log10(A_TED)    (-)",           p.log10_A_TED.toFixed(3)],
    ["p_TED           (-)",           p.p_TED.toFixed(3)],
    ["log10(tau0_311) (s)",           p.log10_tau0_311.toFixed(3)],
    ["Ea_311          (eV)",          p.Ea_311.toFixed(3)],
    ["log10(tau0)     (s)",           p.log10_tau0.toFixed(3)],
    ["Ea_act          (eV)",          p.Ea_act.toFixed(3)],
    ["beta            (-)",           p.beta.toFixed(3)],
    ["log10(mu0)      (cm²/V·s)",     p.log10_mu0.toFixed(3)],
    ["gamma           (-)",           p.gamma.toFixed(3)],
    ["p_exp           (-)",           p.p_exp.toFixed(3)],
  ];
  const w = Math.max(...rows.map(r => r[0].length));
  const body = rows.map(r => `  ${r[0].padEnd(w," ")}   ${r[1]}`).join("\n");
  document.getElementById("phys-params").textContent =
    `least_squares converged: ${fi.success}   cost=${fi.cost.toExponential(3)}   nfev=${fi.n_iter}\n\n` +
    body;
}

function renderParity(data) {
  const h = data.holdout;
  const min = Math.min(...h.truth_Rs), max = Math.max(...h.truth_Rs);
  const diag = { x: [min, max], y: [min, max], mode: "lines",
                 line: { color: "#9aa6b6", dash: "dot", width: 1 },
                 name: "ideal", hoverinfo: "skip" };
  const mark = (name, y, color) => ({
    x: h.truth_Rs, y: y, mode: "markers", name,
    marker: { size: 6, color, opacity: 0.75, line: { width: 0 } },
    hovertemplate: `${name}<br>truth=%{x:.2f} Ω/sq<br>pred=%{y:.2f}<extra></extra>`,
  });
  const traces = [
    diag,
    mark("Physics",   h.physics_Rs, "#6ee7c4"),
    mark("Neural-net",h.nn_Rs,      "#f3b75c"),
    mark("Hybrid",    h.hybrid_Rs,  "#4ea1ff"),
  ];
  const layout = Object.assign({}, PLOT_LAYOUT_BASE, {
    xaxis: { title: "true Rs  (Ω/sq)",       gridcolor: "#2a3340" },
    yaxis: { title: "predicted Rs (Ω/sq)",   gridcolor: "#2a3340" },
    height: 440,
  });
  Plotly.newPlot("plot-parity", traces, layout, CFG);
}

function renderScoreboard(data) {
  const rows = [
    ["Model",       "RMSE Rs",      "MAPE Rs",       "R² Rs",
                    "RMSE Xj",      "MAPE Xj",       "R² Xj",
                    "Extrap Rs MAPE"],
  ];
  const labels = { physics: "Physics", neural_net: "Neural-net", hybrid: "Hybrid" };
  for (const k of ["physics", "neural_net", "hybrid"]) {
    const s  = data.scores[k];
    const e  = data.extrap_scores[k];
    rows.push([
      labels[k],
      s.Rs.rmse.toFixed(2),
      s.Rs.mape_pct.toFixed(2) + "%",
      s.Rs.r2.toFixed(4),
      s.Xj.rmse.toFixed(3),
      s.Xj.mape_pct.toFixed(2) + "%",
      s.Xj.r2.toFixed(4),
      e.Rs.mape_pct.toFixed(2) + "%",
    ]);
  }
  // build HTML table
  const t = document.createElement("table");
  t.className = "compare";
  rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    r.forEach(c => {
      const cell = document.createElement(i === 0 ? "th" : "td");
      cell.textContent = c;
      tr.appendChild(cell);
    });
    t.appendChild(tr);
  });
  const host = document.getElementById("scoreboard");
  host.innerHTML = "";
  host.appendChild(t);
}

function renderTScan(data) {
  const sc = data.temperature_scan;
  const Tmax_train = data.process.bounds.T_anneal[1];
  const Tmin = sc.T[0], Tmax = sc.T[sc.T.length - 1];

  const line = (name, y, color, dash="solid") => ({
    x: sc.T, y, name, mode: "lines",
    line: { color, width: 2.4, dash },
  });
  const traces = [
    line("Truth (CAE)", sc.truth_Rs,   "#e7ecf3"),
    line("Physics",     sc.physics_Rs, "#6ee7c4"),
    line("Neural-net",  sc.nn_Rs,      "#f3b75c"),
    line("Hybrid",      sc.hybrid_Rs,  "#4ea1ff"),
  ];
  const layout = Object.assign({}, PLOT_LAYOUT_BASE, {
    xaxis: { title: "anneal temperature (K)", gridcolor: "#2a3340",
             range: [Tmin, Tmax] },
    yaxis: { title: "Rs (Ω/sq)", gridcolor: "#2a3340" },
    shapes: [{
      type: "rect", xref: "x", yref: "paper",
      x0: Tmax_train, x1: Tmax, y0: 0, y1: 1,
      line: { width: 0 }, fillcolor: "rgba(239,107,107,0.10)",
    }],
    annotations: [{
      x: (Tmax_train + Tmax)/2, y: 1.0, xref: "x", yref: "paper",
      text: "outside training box", showarrow: false,
      font: { color: "#ef6b6b", size: 11 }, yanchor: "bottom",
    }],
    height: 440,
  });
  Plotly.newPlot("plot-tscan", traces, layout, CFG);
}

function renderRecipe(data) {
  const r = data.optimized_recipes.hybrid;
  const sp = data.process.spec;
  const inSpec = Math.abs(r.Rs_truth - sp.Rs_target) <= sp.Rs_tol
                 && r.Xj_truth <= sp.Xj_max
                 && r.Dt_budget <= sp.Dt_budget_max;
  const cls = (ok) => ok ? "ok" : "warn";
  const kpis = [
    { label: "Implant dose",    value: fmt.sci(r.dose, 2),       unit: "cm⁻²"   },
    { label: "Anneal T",        value: r.T_anneal.toFixed(1),    unit: "K"      },
    { label: "Anneal time",     value: r.t_anneal.toFixed(2),    unit: "s"      },
    { label: "O₂ partial press.",value: r.pO2.toFixed(2),         unit: "Torr"   },
    { label: "Predicted Rs",    value: r.Rs_pred.toFixed(2),     unit: "Ω/sq",
      ok: Math.abs(r.Rs_pred - sp.Rs_target) <= sp.Rs_tol },
    { label: "Predicted Xj",    value: r.Xj_pred.toFixed(2),     unit: "nm",
      ok: r.Xj_pred <= sp.Xj_max },
    { label: "Re-simulated Rs", value: r.Rs_truth.toFixed(2),    unit: "Ω/sq",
      ok: Math.abs(r.Rs_truth - sp.Rs_target) <= sp.Rs_tol },
    { label: "Re-simulated Xj", value: r.Xj_truth.toFixed(2),    unit: "nm",
      ok: r.Xj_truth <= sp.Xj_max },
    { label: "Dt budget",       value: r.Dt_budget.toExponential(2),
      unit: "cm²",
      ok: r.Dt_budget <= sp.Dt_budget_max },
    { label: "In yield window", value: inSpec ? "YES" : "NO",    unit: "",
      ok: inSpec },
  ];
  const host = document.getElementById("recipe-card");
  host.innerHTML = "";
  kpis.forEach(k => {
    const div = document.createElement("div");
    div.className = "kpi " + (k.ok === undefined ? "" : cls(k.ok));
    div.innerHTML = `<div class="label">${k.label}</div>
                     <div class="value">${k.value}<span class="unit">${k.unit}</span></div>`;
    host.appendChild(div);
  });
}

// --- boot --------------------------------------------------------------
(async function () {
  buildNav();
  let data;
  try {
    data = await loadResults();
  } catch (e) {
    document.getElementById("main").innerHTML =
      `<section class="step active"><h1>Data missing</h1>
        <p class="lede">${e.message}</p>
        <pre><code>/usr/bin/python3 python/generate_results.py</code></pre>
      </section>`;
    return;
  }

  renderOverview(data);
  renderLHSPlot(data);
  renderPhysParams(data);
  renderParity(data);
  renderScoreboard(data);
  renderTScan(data);
  renderRecipe(data);

  const initial = (location.hash || "#overview").slice(1);
  show(STEPS.find(s => s.id === initial) ? initial : "overview");
})();
