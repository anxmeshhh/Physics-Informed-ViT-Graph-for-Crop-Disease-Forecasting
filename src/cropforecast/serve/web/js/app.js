/* Climate-Adaptive Crop Disease Forecasting - report frontend.
   The literature survey is the entry point: every research gap can be executed
   against the live system from its own row. */

const $ = (id) => document.getElementById(id);
const BAND_COLOUR = { Low: "#16a34a", Medium: "#d97706", High: "#dc2626" };

let riskChart = null, leafletMap = null, mapLayers = null, mapSatellite = false;
let selectedFile = null, demoCatalogue = {}, papers = [];
let samples = [], sampleIdx = -1;
const demoCharts = {};
let pipeTimer = null, pipeStarted = 0;

/* -------------------------------------------------------------- backend */
/* Where the API lives. Served by FastAPI the page and the API share an origin,
   so the base is empty. Deployed as a static site (Netlify) the backend runs on
   the presenter's machine behind a tunnel, so the base comes from ?api=, else
   from the last backend this browser connected to - tried only after the same
   origin fails, so a stale tunnel can never shadow the server the page came from. */
const API_KEY = "cropcast.api";
let API = "";
const api = (path) => API + path;
const cleanUrl = (u) => (u || "").trim().replace(/\/+$/, "");

async function connect() {
  const fromQuery = cleanUrl(new URLSearchParams(location.search).get("api"));
  if (fromQuery) {
    API = fromQuery;
    try { localStorage.setItem(API_KEY, fromQuery); } catch { /* private mode */ }
    return getJSON("/api/health");
  }
  try { return await getJSON("/api/health"); } catch { /* not served by the API */ }
  let saved = "";
  try { saved = cleanUrl(localStorage.getItem(API_KEY)); } catch { /* private mode */ }
  if (!saved) throw new Error("no backend configured");
  API = saved;
  return getJSON("/api/health");
}

/* ------------------------------------------------------------------ utils */
async function getJSON(url) {
  const res = await fetch(api(url));
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = (n) => String(n).replace(/___/g, " — ").replace(/_/g, " ");
const fmt = (v) => (typeof v === "number"
  ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4)) : esc(v ?? "—"));

function table(rows, cols, opts = {}) {
  if (!rows || !rows.length) return `<p class="placeholder">No rows.</p>`;
  const head = cols.map(([, l]) => `<th>${esc(l)}</th>`).join("");
  const body = rows.map((r) => {
    const cls = opts.highlight && opts.highlight(r) ? ' class="best"' : "";
    return `<tr${cls}>` + cols.map(([k], i) =>
      `<td class="${i ? "n" : ""}">${fmt(r[k])}</td>`).join("") + "</tr>";
  }).join("");
  return `<div class="table-scroll"><table class="ruled compact"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/* ------------------------------------------------------------------- boot */
async function boot() {
  try {
    const h = await connect();
    $("docsLink").href = api("/docs");
    const el = $("sysStatus");
    el.className = h.model_trained ? "sys ok" : "sys bad";
    el.innerHTML = `System status: <b>${h.model_trained ? "model loaded" : "model not trained"}</b>`;
    if (API) $("apiHost").textContent = `Backend: ${new URL(API).host}`;
  } catch {
    $("sysStatus").className = "sys bad";
    $("sysStatus").innerHTML = "System status: <b>backend offline</b>";
    // Same origin failing means the server is down; a static host means no
    // backend has been named yet. Either way, offer to point at one.
    $("apiInput").value = API;
    $("apiForm").hidden = false;
    return;
  }
  renderMethod();
  await Promise.all([loadSummary(), loadSurvey(), loadSites(), loadCrops(),
                     loadParameters(), loadProgress()]);
  updateMap();                  // the bulletin is the first thing an officer sees
  await loadSamples();          // needs siteSelect to be populated first
  initTOC();
  bindDemoButtons(document);
}

/* --------------------------------------------------------------- overview */
async function loadSummary() {
  let d = {};
  try { d = await getJSON("/api/summary"); } catch { return; }
  const facts = [
    [(d.observations || 0).toLocaleString(), "Observations"],
    [d.classes ?? "—", "Disease classes"],
    [d.crops ?? "—", "Crops"],
    [d.sites ?? "—", "Farm districts"],
    ["1 / 3 / 5 / 7 d", "Forecast horizons"],
  ];
  $("keyFacts").innerHTML = facts
    .map(([v, l]) => `<div class="kf"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`)
    .join("");

  if (d.backbones) renderBench(d.backbones);
  if (d.ablations) renderAbl(d.ablations);
  try {
    const m = await getJSON("/api/demo/missing_data");
    if (m.rows && m.rows.length) renderMissing(m.rows);
  } catch { /* not generated yet */ }
}

function renderBench(rows) {
  const best = Math.max(...rows.map((r) => r.full_macro_f1 ?? 0));
  $("benchTable").outerHTML = `<div id="benchTable">` + table(rows, [
    ["backbone", "Backbone"], ["dim", "Dim"],
    ["probe_acc", "Probe acc."], ["probe_macro_f1", "Probe macro-F1"],
    ["full_acc", "Full acc."], ["full_macro_f1", "Full macro-F1"],
    ["risk_r2_mean", "Risk R²"],
  ], { highlight: (r) => r.full_macro_f1 === best }) + `</div>`;
}

function renderAbl(rows) {
  $("ablTable").outerHTML = `<div id="ablTable">` + table(rows, [
    ["config", "Configuration"], ["test_acc", "Accuracy"],
    ["test_macro_f1", "Macro-F1"], ["risk_r2_mean", "Risk R²"],
    ["band_acc_mean", "Band acc."],
  ], { highlight: (r) => (r.config || "").startsWith("FULL (") }) + `</div>`;
}

function renderMissing(rows) {
  $("missingTable").outerHTML = `<div id="missingTable">` + table(rows, [
    ["missing_fraction", "Farms without a photo"], ["encoder", "Encoder"],
    ["test_acc", "Accuracy"], ["test_macro_f1", "Macro-F1"],
    ["risk_r2_mean", "Risk R²"],
  ]) + `</div>`;
}

/* -------------------------------------------------------- literature survey */
async function loadSurvey() {
  let d;
  try { d = await getJSON("/api/literature"); }
  catch (e) { $("surveyList").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  papers = d.papers;
  demoCatalogue = d.demos;
  renderSurvey(papers);
  $("surveySearch").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    renderSurvey(!q ? papers : papers.filter((p) =>
      [p.title, p.method, p.gap, p.answer, p.authors].join(" ").toLowerCase().includes(q)));
  });
}

function renderSurvey(list) {
  $("surveyCount").textContent = `${list.length} of ${papers.length} works`;
  $("surveyList").innerHTML = list.map((p) => {
    const demo = demoCatalogue[p.demo] || {};
    return `
    <article class="paper" data-no="${p.no}">
      <div class="paper-head">
        <div class="paper-no">[${p.no}]</div>
        <div>
          <div class="paper-title">${esc(p.title)}</div>
          <div class="paper-authors">${esc(p.authors && p.authors !== "—" ? p.authors + " · " : "")}${esc(p.method)}</div>
        </div>
        <div class="chev">▾</div>
      </div>
      <div class="paper-body">
        <div class="field plain"><div class="k">Key contribution</div>${esc(p.contribution)}</div>
        <div class="grid-2">
          <div class="field gap"><div class="k">Research gap</div>${esc(p.gap)}</div>
          <div class="field ans"><div class="k">Addressed by</div>${esc(p.answer)}
            <div class="modref">${esc(p.module)}</div></div>
        </div>
        <div class="demo-inline" data-demo="${p.demo}">
          <button class="run" data-demo="${p.demo}">Run demonstration — ${esc(demo.title || p.demo)}</button>
          <div class="demo-out"></div>
        </div>
      </div>
    </article>`;
  }).join("");

  $("surveyList").querySelectorAll(".paper-head").forEach((h) => {
    h.addEventListener("click", () => h.parentElement.classList.toggle("open"));
  });
  bindDemoButtons($("surveyList"));
}

/* ------------------------------------------------------------ demonstrations */
function bindDemoButtons(root) {
  root.querySelectorAll("button.run[data-demo]").forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.demo;
      const out = btn.parentElement.querySelector(".demo-out");
      const label = btn.textContent;
      btn.disabled = true; btn.textContent = "Running…";
      try {
        const data = await getJSON(`/api/demo/${id}`);
        out.innerHTML = renderDemo(id, data);
        const cv = out.querySelector("canvas");
        if (cv && data.kind === "epidemiology") drawResponseCurves(cv, data);
        if (cv && data.kind === "forecast") drawTrajectory(cv, data);
      } catch (err) {
        out.innerHTML = `<p class="err">${esc(err.message)}</p>`;
      } finally {
        btn.disabled = false; btn.textContent = label;
      }
    });
  });
}

function demoHeader(id) {
  const d = demoCatalogue[id] || {};
  return `<h4>${esc(d.title || id)}</h4>` +
         (d.blurb ? `<p class="blurb">${esc(d.blurb)}</p>` : "");
}

function renderDemo(id, d) {
  const head = demoHeader(id);
  const note = d.note ? `<div class="demo-note">${esc(d.note)}</div>` : "";

  if (d.kind === "contagion") {
    const pr = d.params;
    return head + `
      <p class="blurb">Epidemic constants — primary inoculum ε = ${pr.epsilon},
        local build-up β = ${pr.beta}, <b>between-farm transmission γ = ${pr.gamma}</b>,
        recovery = ${pr.recovery}, dispersal length ${pr.length_scale_km} km,
        downwind boost ×${pr.wind_boost}.</p>
      <div class="sub">Does a farm's future depend on its neighbours?</div>` +
      table(d.autocorrelation, [
        ["lag_days", "Lag (days)"],
        ["own_pressure_vs_future", "Own state → future"],
        ["neighbour_pressure_vs_future", "Neighbours → future"],
      ]) +
      `<div class="sub">Largest simulated outbreak: ${esc(d.peak.site_id)} on ${esc(d.peak.date)}</div>` +
      table(d.timeline, [
        ["date", "Date"], ["focus_pressure", "Pressure at focus farm"],
        ["highest_elsewhere", "Highest elsewhere"],
      ]) + note;
  }

  if (d.kind === "graph") {
    const s = d.summary, st = d.settings;
    return head + `
      <div class="pairs">
        <div><div class="pk">Nodes</div><div class="pv">${s.num_nodes.toLocaleString()}</div></div>
        <div><div class="pk">Edges</div><div class="pv">${s.num_edges.toLocaleString()}</div></div>
        <div><div class="pk">Downwind</div><div class="pv">${s.downwind_edges.toLocaleString()}</div></div>
        <div><div class="pk">Mean degree</div><div class="pv">${s.mean_in_degree.toFixed(1)}</div></div>
      </div>
      <div class="sub">Settings</div>
      <p class="blurb">k = ${st.k_neighbours}, radius = ${st.radius_km} km,
        time window = ±${st.time_window_days} days, same-crop only = ${st.same_crop_only}</p>
      <div class="sub">Sample edges</div>` +
      table(d.examples, [
        ["from_site", "From"], ["from_date", "Date"], ["to_site", "To"],
        ["crop", "Crop"], ["days_apart", "Δ days"],
        ["wind_alignment", "Wind align."], ["downwind", "Downwind"],
      ]) + note;
  }

  if (d.kind === "physics") {
    return head + table(d.checks, [
      ["constraint", "Constraint"],
      ["legal_case", "Compliant behaviour"], ["legal_penalty", "Penalty"],
      ["illegal_case", "Violating behaviour"], ["illegal_penalty", "Penalty"],
    ]) + `<p class="blurb">λ = ${d.lambda}; term weights ${esc(JSON.stringify(d.weights))}</p>` + note;
  }

  if (d.kind === "epidemiology") {
    const w = d.wet_day, dry = d.dry_day;
    return head + `
      <div class="pairs">
        <div><div class="pk">Wettest day (${esc(w.date)})</div>
             <div class="pv">${w.temperature_C} °C</div>
             <div class="pk">${w.humidity_pct}% RH · ${w.leaf_wetness_h} h wet · VPD ${w.vpd_kPa}</div></div>
        <div><div class="pk">Driest day (${esc(dry.date)})</div>
             <div class="pv">${dry.temperature_C} °C</div>
             <div class="pk">${dry.humidity_pct}% RH · ${dry.leaf_wetness_h} h wet · VPD ${dry.vpd_kPa}</div></div>
      </div>` +
      table(d.rows, [
        ["disease", "Pathogen"], ["t_opt", "T opt (°C)"], ["moisture_mode", "Moisture mode"],
        ["favourability_wet_day", "Favourability, wet day"],
        ["favourability_dry_day", "Favourability, dry day"],
      ]) + `<canvas height="150"></canvas>` + note;
  }

  if (d.kind === "niche") {
    const rows = d.rows.slice(0, 10).concat(d.rows.slice(-6));
    return head + table(rows, [
      ["class_name", "Class"], ["n", "Images"], ["temperature_C", "Temp (°C)"],
      ["humidity_pct", "RH (%)"], ["leaf_wetness_h", "Leaf wetness (h)"],
      ["vpd_kPa", "VPD (kPa)"],
    ]) + note;
  }

  if (d.kind === "forecast") {
    return head + `<p class="blurb">Highest-risk farm for
      ${esc(pretty(d.crop))} on ${esc(d.date)}: <b>${esc(d.site.name)}, ${esc(d.site.state)}</b></p>` +
      table(d.trajectory, [
        ["horizon_days", "Horizon (d)"], ["date", "Date"], ["risk", "Risk"],
        ["temperature_C", "Temp (°C)"], ["humidity_pct", "RH (%)"],
        ["leaf_wetness_h", "Leaf wetness (h)"],
      ]) + `<canvas height="140"></canvas>` + note;
  }

  if (d.kind === "map") {
    return head + table(d.sites.slice(0, 12), [
      ["name", "Farm"], ["state", "State"], ["risk", "Risk"], ["band", "Band"],
      ["temperature_C", "Temp (°C)"], ["leaf_wetness_h", "Leaf wetness (h)"],
    ]) + note;
  }

  if (d.kind === "node") {
    return head + table(d.streams, [
      ["stream", "Stream"], ["dims", "Dimensions"],
      ["source", "Source"], ["example", "Contents"],
    ]) + `<p class="blurb">Fused to ${d.fused_dim} dimensions before message passing.</p>` + note;
  }

  if (d.kind === "table") {
    if (!d.rows || !d.rows.length) {
      return head + `<p class="placeholder">${esc(d.note || "Not generated yet.")}</p>`;
    }
    const cols = d.columns.filter((c) =>
      !["checkpoint", "family", "pool", "masked_stream", "physics", "encoder"].includes(c)
    ).slice(0, 7).map((c) => [c, c.replace(/_/g, " ")]);
    return head + `<p class="blurb">${esc(d.caption || "")}</p>` + table(d.rows, cols);
  }

  return head + `<pre>${esc(JSON.stringify(d, null, 2)).slice(0, 2000)}</pre>`;
}

function drawResponseCurves(canvas, d) {
  new Chart(canvas, {
    type: "line",
    data: {
      labels: d.temps,
      datasets: Object.entries(d.curves).map(([name, vals], i) => ({
        label: name.split("___")[1].replace(/_/g, " ").slice(0, 22),
        data: vals, borderWidth: 2, pointRadius: 0, tension: .3,
        borderColor: ["#15803d", "#ea580c", "#0f766e", "#d97706", "#2563eb"][i % 5],
      })),
    },
    options: {
      plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: "temperature (°C)" },
             ticks: { maxTicksLimit: 10, callback: (v, i) => Math.round(d.temps[i]) } },
        y: { title: { display: true, text: "relative development rate" }, min: 0, max: 1 },
      },
    },
  });
}

function drawTrajectory(canvas, d) {
  new Chart(canvas, {
    type: "line",
    data: {
      labels: d.trajectory.map((t) => `+${t.horizon_days}d`),
      datasets: [{
        label: "agronomic risk", data: d.trajectory.map((t) => t.risk),
        borderColor: "#7a1f2b", backgroundColor: "rgba(122,31,43,.10)",
        fill: true, tension: .3, pointRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { min: 0, max: 1 } },
    },
  });
}

/* ------------------------------------------------------------- methodology */
function renderMethod() {
  const steps = [
    ["Input layer",
     "A leaf photograph, the meteorology actually recorded at that farm on that date, and the farm's location and metadata.",
     ["RGB leaf image", "Open-Meteo ERA5", "latitude / longitude / altitude", "soil, crop, season"]],
    ["Feature extraction and engineering",
     "A frozen vision transformer encodes the leaf; raw weather becomes the quantities plant pathologists use; the spatial graph is built from location and time.",
     ["DINOv2-S (frozen)", "VPD · leaf wetness · GDD", "3/7/14-day windows", "KNN + radius search"]],
    ["Feature fusion",
     "The three streams are projected to a common width and fused, with climate modulating the visual stream multiplicatively.",
     ["FiLM gating", "vision ⊕ climate ⊕ metadata"]],
    ["Graph neural network",
     "Message passing over the spatio-temporal farm graph yields spatially aware embeddings; directed downwind edges encode wind-borne spore transport.",
     ["GraphSAGE / GCN / GAT", "residual + layer norm", "downwind edges"]],
    ["Prediction and physics-informed loss",
     "A 38-class disease head for the current leaf, plus an autoregressive forecaster for 1, 3, 5 and 7 days, trained under agronomic constraints.",
     ["classification head", "multi-horizon forecaster", "CE + λ · physics"]],
    ["Explainability",
     "Attention rollout and Grad-CAM over the leaf, integrated gradients and SHAP over the weather variables, gradient attribution over graph edges.",
     ["Grad-CAM", "attention rollout", "integrated gradients", "SHAP", "graph attribution"]],
  ];
  $("methodFlow").innerHTML = steps.map(([t, b, tags], i) => `
    <div class="mstep">
      <div class="mn">${i + 1}</div>
      <div><h4>${esc(t)}</h4><p>${esc(b)}</p>
        <div class="tags">${tags.map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
      </div>
    </div>`).join("");
}

/* -------------------------------------------------------------- parameters */
async function loadParameters() {
  let d;
  try { d = await getJSON("/api/parameters"); } catch {
    $("paramGroups").innerHTML =
      `<p class="placeholder">Parameters unavailable &mdash; backend offline.</p>`;
    return;
  }
  /* Collapsed by default: 34 rows of justification is the right depth for a
     question from the floor, and the wrong amount to scroll past to reach the
     live demo. The first group stays open so the shape is obvious. */
  $("paramGroups").innerHTML = d.groups.map((g, i) => `
    <details class="param-group" id="p-${esc(g.id)}"${i === 0 ? " open" : ""}>
      <summary>
        <span class="pg-title">${esc(g.title)}</span>
        <span class="pg-count">${g.rows.length}</span>
      </summary>
      <div class="pg-body">
        <p class="micro">${esc(g.lead)}</p>
        <div class="table-scroll">
          <table class="ruled compact param-table">
            <thead><tr><th>Parameter</th><th>Value</th><th>Why this value</th></tr></thead>
            <tbody>${g.rows.map(([k, v, why]) => `
              <tr><td>${esc(k)}</td><td class="pv">${fmt(v)}</td>
                  <td class="pw">${esc(why)}</td></tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    </details>`).join("");
}

/* ---------------------------------------------------------------- progress */
/* The percentage is the backend's, computed from artefacts on disk. Nothing
   here invents or rounds it up - this renders what /api/progress reports. */
async function loadProgress() {
  let d;
  try { d = await getJSON("/api/progress"); } catch {
    $("moduleList").innerHTML =
      `<p class="placeholder">Status unavailable &mdash; backend offline.</p>`;
    return;
  }

  $("progPct").textContent = `${d.percent}%`;
  $("progSub").textContent = `${d.modules_done} of ${d.modules_total} modules`;

  // r = 52 in the viewBox, so the full sweep is 2*pi*52.
  const circ = 2 * Math.PI * 52;
  const fill = $("donutFill");
  fill.style.strokeDasharray = `${circ}`;
  fill.style.strokeDashoffset = `${circ}`;
  requestAnimationFrame(() => {
    fill.style.strokeDashoffset = `${circ * (1 - d.percent / 100)}`;
  });

  const mast = $("mastPct");
  mast.hidden = false;
  mast.innerHTML = `Prototype complete: <b>${d.percent}%</b>`;

  $("phaseBars").innerHTML = d.phases.map((p) => `
    <div class="pbar">
      <div class="pbar-head">
        <span>${esc(p.phase)}</span>
        <span class="pbar-n">${p.done}/${p.total} &middot; ${p.percent}%</span>
      </div>
      <div class="pbar-track">
        <div class="pbar-fill${p.percent === 0 ? " none" : ""}"
             style="width:${p.percent}%"></div>
      </div>
    </div>`).join("");

  $("moduleList").innerHTML = d.modules.map((m) => `
    <div class="mod ${esc(m.status)}">
      <div class="mod-mark" aria-hidden="true">${m.done ? "&check;" : m.status === "roadmap" ? "&rarr;" : ""}</div>
      <div class="mod-main">
        <div class="mod-top">
          <span class="mod-name">${esc(m.name)}</span>
          <span class="mod-phase">${esc(m.phase)}</span>
          <span class="mod-share">${m.share == null ? "next" : `${m.share}%`}</span>
        </div>
        <p class="mod-what">${esc(m.what)}</p>
        <code class="mod-files">${esc(m.modules)}</code>
      </div>
    </div>`).join("");
}

/* ------------------------------------------------------------------- TOC */
function initTOC() {
  const links = [...document.querySelectorAll(".toc a")];
  const sections = links.map((a) => document.querySelector(a.getAttribute("href")))
                        .filter(Boolean);
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      links.forEach((l) => l.classList.toggle("active",
        l.getAttribute("href") === `#${en.target.id}`));
    });
  }, { rootMargin: "-15% 0px -70% 0px" });
  sections.forEach((s) => obs.observe(s));
}

/* --------------------------------------------------------------- selectors */
async function loadSites() {
  try {
    const d = await getJSON("/api/sites");
    $("siteSelect").innerHTML = d.sites
      .map((s) => `<option value="${esc(s.site_id)}">${esc(s.name)}, ${esc(s.state)}</option>`)
      .join("");
  } catch { /* offline */ }
}

async function loadCrops() {
  let d;
  try { d = await getJSON("/api/crops"); } catch { return; }
  const opts = d.crops.map((c) =>
    `<option value="${esc(c.crop)}">${esc(c.label || pretty(c.crop))}</option>`).join("");
  $("cropSelect").innerHTML = `<option value="">Infer from image</option>` + opts;
  $("mapCrop").innerHTML = opts;
  // Pre-monsoon maize: a realistic mix of districts to act on, watch and leave.
  $("mapCrop").value = "Corn_(maize)";
}

/* ------------------------------------------------------------------ upload */
const dz = $("dropzone");
dz.addEventListener("click", () => $("fileInput").click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
$("fileInput").addEventListener("change", (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});
function setFile(file) {
  selectedFile = file;
  const r = new FileReader();
  r.onload = (e) => {
    $("preview").src = e.target.result;
    $("preview").classList.add("show");
    $("dropHint").style.display = "none";
  };
  r.readAsDataURL(file);
}

/* ------------------------------------------------------------ sample input */
/* One button that fills every field from tests/index.csv and runs the pipeline.
   Each click advances to the next curated image, so the demo never repeats. */
async function loadSamples() {
  try {
    const d = await getJSON("/api/samples");
    samples = d.samples || [];
  } catch { samples = []; }
  if (samples.length) $("sampleBar").hidden = false;
}

function describeSample(s, n) {
  const conf = s.confidence == null ? "" :
    ` &middot; expected <b>${(s.confidence * 100).toFixed(1)}%</b>`;
  return `Sample ${n}/${samples.length} &middot; <b>${esc(pretty(s.true_class))}</b>` +
         `<br>${esc(s.farm)} &middot; ${esc(s.date)}${conf}`;
}

$("sampleBtn").addEventListener("click", async () => {
  if (!samples.length) return;
  const btn = $("sampleBtn");
  btn.disabled = true; btn.textContent = "Loading…";
  $("predictError").textContent = "";

  try {
    sampleIdx = (sampleIdx + 1) % samples.length;
    const s = samples[sampleIdx];

    const res = await fetch(api(`/api/samples/${encodeURIComponent(s.file)}`));
    if (!res.ok) throw new Error(`Could not load ${s.file}`);
    const blob = await res.blob();
    setFile(new File([blob], s.file, { type: blob.type || "image/jpeg" }));

    $("siteSelect").value = s.site_id;
    if (!$("siteSelect").value) throw new Error(`Farm ${s.site_id} is not in the registry`);
    $("dateInput").value = s.date;
    // Left on "infer from image" deliberately: that is how tests/index.csv
    // measured its confidences, so the number on screen matches the table.
    $("cropSelect").value = "";
    $("sampleNote").innerHTML = describeSample(s, sampleIdx + 1);

    // Deliberately does not run: the operator presses "Run full pipeline" when
    // the room is watching, so the trace animates on cue rather than on load.
    resetRun();
    $("predictBtn").classList.add("ready");
  } catch (err) {
    $("predictError").textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Next sample";
  }
});

/* Clear the previous run so a loaded sample never sits next to a stale
   diagnosis from the sample before it. */
function resetRun() {
  $("resultBody").hidden = true;
  $("resultEmpty").hidden = false;
  $("resultEmpty").textContent = "Press Run full pipeline.";
  $("pipeline").innerHTML =
    `<p class="placeholder">Run the pipeline to trace it.</p>`;
  if (riskChart) { riskChart.destroy(); riskChart = null; }
}

/* ----------------------------------------------------------------- predict */
$("predictBtn").addEventListener("click", runPredict);

async function runPredict() {
  $("predictError").textContent = "";
  if (!selectedFile) { $("predictError").textContent = "Select a leaf image first."; return; }
  const btn = $("predictBtn");
  btn.disabled = true; btn.textContent = "Running…";
  btn.classList.remove("ready");
  resetRun();
  $("resultEmpty").textContent = "Running…";

  const fd = new FormData();
  fd.append("image", selectedFile);
  fd.append("site_id", $("siteSelect").value);
  fd.append("date", $("dateInput").value);
  if ($("cropSelect").value) fd.append("crop", $("cropSelect").value);

  try {
    const res = await fetch(api("/api/predict/stream"), { method: "POST", body: fd });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Prediction failed");
    }
    const data = await consumeRun(res);
    showResult(data);
    syncMapToRun(data);       // not awaited: the diagnosis should not wait on the map
  } catch (err) {
    $("predictError").textContent = err.message;
    $("resultEmpty").textContent = "Awaiting input.";
    failPipeline();
  } finally {
    btn.disabled = false; btn.textContent = "Run full pipeline";
  }
}

/* Read the NDJSON stream, drawing each stage the moment the server reports it.
   The model finishes a warm run in about 20 ms, which would flash past before
   anyone could read it, so each stage is held on screen for a minimum dwell.
   The pacing is presentation; the durations printed on each card are the real
   measurements the server sent. */
const MIN_DWELL_MS = 190;

async function consumeRun(res) {
  beginPipeline();
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", result = null, shownAt = 0;

  const paceIn = async (step) => {
    const wait = Math.max(0, shownAt + MIN_DWELL_MS - performance.now());
    if (wait) await new Promise((r) => setTimeout(r, wait));
    pushStage(step);
    shownAt = performance.now();
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });

    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.type === "step") await paceIn(msg.step);
      else if (msg.type === "result") result = msg.result;
      else if (msg.type === "error") throw new Error(msg.detail);
    }
  }
  if (!result) throw new Error("Stream ended before the forecast arrived");

  // Let the last stage hold its dwell, then close the trace out.
  const wait = Math.max(0, shownAt + MIN_DWELL_MS - performance.now());
  if (wait) await new Promise((r) => setTimeout(r, wait));
  finishPipeline(result.trace);
  return result;
}

/* The regional map answers "where else is this crop at risk on this date?", so
   it should never be left showing a different crop or date than the run above
   it. Anything that fails here is reported in the map's own table, never on
   top of the diagnosis. */
async function syncMapToRun(d) {
  try {
    const crop = $("mapCrop");
    if (d.crop && [...crop.options].some((o) => o.value === d.crop)) crop.value = d.crop;
    if (d.date) $("mapDate").value = d.date;
    await updateMap();
  } catch (err) {
    // initMap() runs outside updateMap's own guard, so a CDN-less Leaflet
    // would otherwise reject here unseen.
    $("alertList").innerHTML = `<p class="err">Bulletin unavailable: ${esc(err.message)}</p>`;
  }
}

function showResult(d) {
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;
  const worst = d.risk_band.reduce((a, b) => (RISK_ORDER[b] > RISK_ORDER[a] ? b : a), "Low");
  $("diagCard").className = `diag-card ${worst}`;
  $("diagName").textContent = d.disease_label || pretty(d.disease);
  $("diagCrop").textContent = pretty(d.crop);
  $("diagOrganism").textContent = d.organism || "";
  $("diagConf").textContent = `${(d.confidence * 100).toFixed(1)}%`;
  $("sevPill").className = `sev-pill ${worst}`;
  $("sevPill").textContent = `Spread risk: ${worst} · ${ALERT_WORD[worst]}`;
  $("diagWhere").textContent =
    `${d.site.name}, ${d.site.state} · ${fmtDate(d.date)} · ${d.site.agro_zone || ""}`;
  const items = (xs) => xs.map((a) => `<li>${esc(a)}</li>`).join("");
  $("farmerActions").innerHTML = d.farmer_actions && d.farmer_actions.length
    ? items(d.farmer_actions)
    : "<li>No disease found on this leaf &mdash; continue routine scouting</li>";
  $("officerActions").innerHTML = items(d.officer_actions || []);

  $("horizonTable").innerHTML =
    `<thead><tr><th>Horizon</th><th>Risk</th><th>Band</th><th>± σ</th></tr></thead><tbody>` +
    d.horizons.map((h, i) => `<tr>
      <td>+${h} day${h > 1 ? "s" : ""}</td>
      <td class="n">${(d.risk[i] * 100).toFixed(0)}%</td>
      <td><span class="band ${d.risk_band[i]}">${d.risk_band[i]}</span></td>
      <td class="n">${(d.uncertainty[i] * 100).toFixed(1)}</td></tr>`).join("") + "</tbody>";

  if (riskChart) riskChart.destroy();
  riskChart = new Chart($("riskChart"), {
    type: "line",
    data: {
      labels: d.horizons.map((h) => `+${h}d`),
      datasets: [{
        label: "risk", data: d.risk,
        borderColor: "#15803d", backgroundColor: "rgba(21,128,61,.10)",
        fill: true, tension: .3, pointRadius: 5,
        pointBackgroundColor: d.risk_band.map((b) => BAND_COLOUR[b]),
      }],
    },
    options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } },
  });

  const labels = {
    temperature_C: "Temperature (°C)", humidity_pct: "Relative humidity (%)",
    rainfall_mm: "Rainfall (mm)", wind_kmh: "Max wind (km/h)",
    leaf_wetness_h: "Leaf wetness (h)", vpd_kPa: "VPD (kPa)", gdd: "Growing degree days",
  };
  $("climateTable").innerHTML = "<tbody>" + Object.entries(d.climate)
    .map(([k, v]) => `<tr><td>${esc(labels[k] || k)}</td><td class="n">${v}</td></tr>`)
    .join("") + "</tbody>";

  $("advisory").textContent = d.advisory;
  $("topk").innerHTML =
    `<thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>` +
    d.top_k.map((t) => `<tr><td>${esc(pretty(t.class))}</td>
      <td class="n">${(t.probability * 100).toFixed(2)}%</td></tr>`).join("") + "</tbody>";
}

/* ---------------------------------------------------------------- pipeline */
/* Built incrementally: beginPipeline lays out the shell, pushStage appends one
   card as the server reports that block finishing, finishPipeline closes the
   total once the forecast lands. Each card shows the duration the server
   measured for that block. */
function beginPipeline() {
  $("pipeline").innerHTML = `
    <div class="pipe-total running">
      <span>Running</span><b id="pipeElapsed">0 ms</b>
      <span class="pipe-n" id="pipeCount">stage 1&hellip;</span>
    </div>
    <ol class="pipe-list" id="pipeList"></ol>`;
  pipeStarted = performance.now();
  clearInterval(pipeTimer);
  pipeTimer = setInterval(() => {
    const el = $("pipeElapsed");
    if (el) el.textContent = `${Math.round(performance.now() - pipeStarted)} ms`;
  }, 60);
}

function pushStage(t) {
  const list = $("pipeList");
  if (!list) return;

  // The stage that was live becomes settled as soon as the next one starts.
  const prev = list.querySelector(".pipe-step.live");
  if (prev) prev.classList.remove("live");

  const li = document.createElement("li");
  li.className = "pipe-step live";
  li.innerHTML = `
    <div class="pipe-rail"><span class="pipe-dot">${t.index + 1}</span></div>
    <div class="pipe-card">
      <div class="pipe-head">
        <span class="pipe-block">${esc(t.block)}</span>
        <span class="pipe-name">${esc(t.name)}</span>
        <span class="pipe-ms">${t.ms.toFixed(1)} ms</span>
      </div>
      <p class="pipe-detail">${esc(t.detail)}</p>
      <div class="pipe-out"><span>&rarr;</span><code>${esc(t.output)}</code></div>
      <div class="pipe-bar"><i data-ms="${t.ms}"></i></div>
    </div>`;
  list.appendChild(li);
  requestAnimationFrame(() => li.classList.add("in"));

  const count = $("pipeCount");
  if (count) count.textContent = `stage ${t.index + 1}…`;
}

function finishPipeline(trace) {
  clearInterval(pipeTimer);
  const live = document.querySelector(".pipe-step.live");
  if (live) live.classList.remove("live");

  const total = (trace || []).reduce((a, t) => a + t.ms, 0);
  const head = document.querySelector(".pipe-total");
  if (head) {
    head.classList.remove("running");
    head.classList.add("done");
    head.innerHTML = `<span>End to end</span><b>${total.toFixed(0)} ms</b>
      <span class="pipe-n">${(trace || []).length} stages &middot; measured server-side</span>`;
  }

  // Scale the per-stage bars now that the slowest block is known.
  const peak = Math.max(...(trace || []).map((t) => t.ms), 1);
  document.querySelectorAll(".pipe-bar i").forEach((b) => {
    b.style.width = `${(Number(b.dataset.ms) / peak) * 100}%`;
  });
}

function failPipeline() {
  clearInterval(pipeTimer);
  const head = document.querySelector(".pipe-total");
  if (head) { head.classList.remove("running"); head.classList.add("failed"); }
  const live = document.querySelector(".pipe-step.live");
  if (live) live.classList.remove("live");
}

/* --------------------------------------------------------------------- map */
function initMap() {
  if (leafletMap) return;
  leafletMap = L.map("leaflet").setView([22.5, 79], 5);
  // Esri tiles need no API key; CARTO's basemaps now watermark every tile
  // "API KEY REQUIRED" without one. Base and labels are separate layers so the
  // risk markers sit between the land and the place names.
  const esri = "https://server.arcgisonline.com/ArcGIS/rest/services";
  const tiles = (path, opts) => L.tileLayer(`${esri}/${path}/MapServer/tile/{z}/{y}/{x}`, opts);
  const labels = (path) => tiles(path, { maxZoom: 18, pane: "shadowPane" });
  const basemaps = {
    "Map": L.layerGroup([
      tiles("Canvas/World_Light_Gray_Base", { maxZoom: 16,
        attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" }),
      labels("Canvas/World_Light_Gray_Reference"),
    ]),
    "Satellite": L.layerGroup([
      tiles("World_Imagery", { maxZoom: 18,
        attribution: "Imagery &copy; Esri &mdash; Maxar, Earthstar Geographics, GIS User Community" }),
      labels("Reference/World_Boundaries_and_Places"),
    ]),
  };
  basemaps["Map"].addTo(leafletMap);
  L.control.layers(basemaps, null, { collapsed: false }).addTo(leafletMap);
  leafletMap.on("baselayerchange", (e) => {
    mapSatellite = e.name === "Satellite";
    // The light map tops out at z16; imagery goes deeper.
    leafletMap.setMaxZoom(mapSatellite ? 18 : 16);
    mapLayers.eachLayer(restyle);
  });
  mapLayers = L.layerGroup().addTo(leafletMap);

  // Leaflet sizes its tile grid once and does not re-measure on its own, so a
  // window resize leaves tiles hanging past the new viewport width.
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => leafletMap.invalidateSize(), 160);
  });
}

/* Hairlines vanish on dark imagery, so on satellite the edges turn white
   and each marker gains a white ring; the alert colour stays in the fill. */
function edgeStyle() {
  return mapSatellite ? { color: "#ffffff", weight: 1.5, opacity: .7 }
                      : { color: "#15803d", weight: 1, opacity: .35 };
}
function markerStyle(colour) {
  return mapSatellite ? { color: "#ffffff", weight: 2, fillColor: colour, fillOpacity: .85 }
                      : { color: colour, weight: 2, fillColor: colour, fillOpacity: .5 };
}
function restyle(layer) {
  if (layer instanceof L.CircleMarker) layer.setStyle(markerStyle(layer.options.band));
  else if (layer instanceof L.Polyline) layer.setStyle(edgeStyle());
}

/* ------------------------------------------------------- district bulletin */
/* The regional view restated for an officer: which districts need action this
   week, what the threat is, which way it is moving, and what to do. The map and
   the cards are two views of one bulletin and stay linked in both directions. */
let bulletin = null;
const districtMarkers = {};
const RISK_ORDER = { Low: 0, Medium: 1, High: 2 };
const ALERT_WORD = { High: "Act now", Medium: "Watch", Low: "Normal" };
const pct = (v) => `${Math.round(v * 100)}%`;
const days = (n) => `${n} day${n === 1 ? "" : "s"}`;
const fmtDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString(
  "en-IN", { day: "numeric", month: "short", year: "numeric" });

$("mapBtn").addEventListener("click", updateMap);
$("showEdges").addEventListener("change", () => { if (bulletin) drawBulletinMap(bulletin, false); });
$("printBtn").addEventListener("click", () => {
  document.body.classList.add("print-bulletin");
  window.print();
});
window.addEventListener("afterprint", () => document.body.classList.remove("print-bulletin"));

async function updateMap() {
  initMap();
  setTimeout(() => leafletMap.invalidateSize(), 50);
  const btn = $("mapBtn");
  btn.disabled = true; btn.textContent = "Generating…";
  try {
    bulletin = await getJSON(
      `/api/bulletin?crop=${encodeURIComponent($("mapCrop").value)}&date=${$("mapDate").value}`);
    renderBulletin(bulletin);
    drawBulletinMap(bulletin, true);
  } catch (err) {
    $("bulletinHead").hidden = true;
    $("alertList").innerHTML = `<p class="err">${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Generate bulletin";
  }
}

function outlookText(r) {
  if (r.peak_day === 0) return r.alert === "Low" ? "Low all week" : "Worst today";
  if (RISK_ORDER[r.peak_band] > RISK_ORDER[r.band]) return `${r.peak_band} in ${days(r.peak_day)}`;
  return `Peaks in ${days(r.peak_day)}`;
}

/* Past week solid, coming week dashed, today marked - so "rising" and "about to
   rise" are visibly different things. */
function sparkline(past, ahead, colour) {
  const vals = [...past, ...ahead];
  const W = 120, H = 34, P = 3;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(vals.length - 1, 1);
  const y = (v) => H - P - v * (H - 2 * P);
  const pts = (arr, off) => arr.map((v, i) => `${x(i + off).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const ti = past.length - 1;
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Risk over the past 7 days and the next 7 days">
    <title>Past 7 days (solid) and next 7 days (dashed)</title>
    <line x1="${x(ti)}" x2="${x(ti)}" y1="${P}" y2="${H - P}" stroke="#cfd8d2" stroke-dasharray="2 2"/>
    <polyline points="${pts(past, 0)}" fill="none" stroke="${colour}" stroke-width="2"/>
    ${ahead.length ? `<polyline points="${pts([past[ti], ...ahead], ti)}" fill="none"
      stroke="${colour}" stroke-width="2" stroke-dasharray="3 3"/>` : ""}
    <circle cx="${x(ti)}" cy="${y(past[ti])}" r="3" fill="${colour}"/></svg>`;
}

function renderBulletin(b) {
  const s = b.summary;
  const calm = !s.counts.High;
  const states = s.states_on_alert.length ? ` · States on alert: ${s.states_on_alert.join(", ")}` : "";
  $("bHeadline").className = `headline${calm ? " calm" : ""}`;
  $("bHeadline").innerHTML = `${esc(b.headline)}
    <small>${esc(b.crop_label)} · as of ${fmtDate(b.date)} · outlook to ${fmtDate(b.outlook_until)}${esc(states)}</small>`;
  $("bKpis").innerHTML = [
    ["High", s.counts.High, "Act now"], ["Medium", s.counts.Medium, "Watch"],
    ["Low", s.counts.Low, "Normal"], ["rise", s.rising, "Rising over the past week"],
  ].map(([c, v, l]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  $("bulletinHead").hidden = false;

  $("alertList").innerHTML = b.districts.map((r) => {
    const colour = BAND_COLOUR[r.alert];
    const arrow = { Rising: "↑", Falling: "↓", Steady: "→" }[r.direction];
    const threat = r.threat
      ? `<b>${esc(r.threat.name)}</b> <span class="org">· ${esc(r.threat.organism)}</span>`
      : `<span class="org">No significant threat this week</span>`;
    const spread = r.neighbours_high.length
      ? `<div class="acard-spread">⚠ ${r.neighbours_high.length} neighbouring district${r.neighbours_high.length > 1 ? "s" : ""} on alert: ${
          r.neighbours_high.map((n) => `${esc(n.name)} (${Math.round(n.distance_km)} km)`).join(", ")}</div>`
      : "";
    const farmer = r.threat && r.alert !== "Low"
      ? `<div><h5>For farmers</h5><ul>${r.threat.farmer_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>`
      : "";
    return `<article class="acard ${r.alert}" data-site="${esc(r.site_id)}">
      <div class="acard-top">
        <span class="acard-name">${esc(r.name)}</span>
        <span class="acard-state">${esc(r.state)}</span>
        <span class="pill ${r.alert}">${ALERT_WORD[r.alert]}</span>
      </div>
      <div class="acard-row">
        <div class="acard-stat">Risk today<b>${pct(r.risk)}</b></div>
        <div class="acard-stat">Next 7 days<b>${esc(outlookText(r))}</b></div>
        <div class="acard-stat">Past week<b class="dir-${r.direction}">${arrow} ${r.direction}</b></div>
        ${sparkline(r.trend, r.outlook, colour)}
      </div>
      <div class="acard-threat">${threat}</div>
      <div class="acard-drivers">${esc(r.drivers)}</div>
      ${spread}
      <div class="acard-more">Show recommended actions ▾</div>
      <div class="acard-actions">
        <div><h5>For officials</h5><ul>${r.officer_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>
        ${farmer}
      </div>
    </article>`;
  }).join("");

  $("alertList").querySelectorAll(".acard").forEach((card) => {
    card.addEventListener("click", () => {
      card.classList.toggle("open");
      const m = districtMarkers[card.dataset.site];
      if (m) { leafletMap.panTo(m.getLatLng()); m.openPopup(); }
      markCard(card);
    });
  });
}

function markCard(card) {
  $("alertList").querySelectorAll(".acard.focus").forEach((c) => c.classList.remove("focus"));
  card.classList.add("focus");
}

function focusCard(siteId) {
  const list = $("alertList");
  const card = list.querySelector(`.acard[data-site="${CSS.escape(siteId)}"]`);
  if (!card) return;
  markCard(card);
  card.classList.add("open");
  // Scroll the list itself, never the page, so the map stays in view.
  list.scrollTo({ top: card.offsetTop - list.offsetTop - 6, behavior: "smooth" });
}

function drawBulletinMap(b, fit) {
  mapLayers.clearLayers();
  Object.keys(districtMarkers).forEach((k) => delete districtMarkers[k]);
  const byId = Object.fromEntries(b.districts.map((r) => [r.site_id, r]));

  if ($("showEdges").checked) {
    b.edges.forEach((e) => {
      const a = byId[e.source], t = byId[e.target];
      if (!a || !t) return;
      L.polyline([[a.lat, a.lon], [t.lat, t.lon]], edgeStyle())
        .bindTooltip(`${a.name} → ${t.name}: ${Math.round(e.distance_km)} km`).addTo(mapLayers);
    });
  }
  // Drawn calm-first so the districts that need action sit on top.
  [...b.districts].reverse().forEach((r) => {
    const colour = BAND_COLOUR[r.alert];
    const m = L.circleMarker([r.lat, r.lon], {
      radius: 7 + Math.max(r.risk, r.peak_risk) * 11, band: colour, ...markerStyle(colour),
    }).bindPopup(`<b>${esc(r.name)}</b>, ${esc(r.state)}<br>
      <span class="pill ${r.alert}">${ALERT_WORD[r.alert]}</span><br>
      Risk today <b>${pct(r.risk)}</b> · next 7 days: ${esc(outlookText(r))}<br>
      ${r.threat ? `Threat: <b>${esc(r.threat.name)}</b><br>` : ""}${esc(r.drivers)}`)
      .addTo(mapLayers);
    m.on("click", () => focusCard(r.site_id));
    districtMarkers[r.site_id] = m;
  });
  if (fit && b.districts.length) {
    // Measure first: fitting a map that has not sized itself yet centres it wrong.
    leafletMap.invalidateSize();
    leafletMap.fitBounds(L.latLngBounds(b.districts.map((r) => [r.lat, r.lon])).pad(0.25));
  }
}

/* Reload with ?api= so the URL on screen is the link to share. */
$("apiForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const u = cleanUrl($("apiInput").value);
  if (!u) return;
  const next = new URL(location.href);
  next.searchParams.set("api", u);
  location.href = next.toString();
});

boot();
