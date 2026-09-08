/* Climate-Adaptive Crop Disease Forecasting - report frontend.
   The literature survey is the entry point: every research gap can be executed
   against the live system from its own row. */

const $ = (id) => document.getElementById(id);
const BAND_COLOUR = { Low: "#1c6ea4", Medium: "#b07d0a", High: "#9d2235" };

let riskChart = null, leafletMap = null, mapLayers = null;
let selectedFile = null, demoCatalogue = {}, papers = [];
const demoCharts = {};

/* ------------------------------------------------------------------ utils */
async function getJSON(url) {
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = (n) => String(n).replace(/_/g, " ").replace(/___/g, " — ");
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
    const h = await getJSON("/api/health");
    const el = $("sysStatus");
    el.className = h.model_trained ? "sys ok" : "sys bad";
    el.innerHTML = `System status: <b>${h.model_trained ? "model loaded" : "model not trained"}</b>`;
  } catch {
    $("sysStatus").className = "sys bad";
    $("sysStatus").innerHTML = "System status: <b>backend offline</b>";
    return;
  }
  renderMethod();
  await Promise.all([loadSummary(), loadSurvey(), loadSites(), loadCrops()]);
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
        borderColor: ["#14315c", "#7a1f2b", "#1f6b3a", "#b07d0a", "#1c6ea4"][i % 5],
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
    options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } },
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
    `<option value="${esc(c.crop)}">${esc(pretty(c.crop))}</option>`).join("");
  $("cropSelect").innerHTML = `<option value="">Infer from image</option>` + opts;
  $("mapCrop").innerHTML = opts;
  $("mapCrop").value = "Tomato";
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

/* ----------------------------------------------------------------- predict */
$("predictBtn").addEventListener("click", async () => {
  $("predictError").textContent = "";
  if (!selectedFile) { $("predictError").textContent = "Select a leaf image first."; return; }
  const btn = $("predictBtn");
  btn.disabled = true; btn.textContent = "Running…";

  const fd = new FormData();
  fd.append("image", selectedFile);
  fd.append("site_id", $("siteSelect").value);
  fd.append("date", $("dateInput").value);
  if ($("cropSelect").value) fd.append("crop", $("cropSelect").value);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");
    showResult(data);
  } catch (err) {
    $("predictError").textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Run full pipeline";
  }
});

function showResult(d) {
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;
  $("diagName").textContent = pretty(d.disease);
  $("diagCrop").textContent = pretty(d.crop);
  $("diagConf").textContent = `${(d.confidence * 100).toFixed(1)}%`;

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
        borderColor: "#14315c", backgroundColor: "rgba(20,49,92,.10)",
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

/* --------------------------------------------------------------------- map */
function initMap() {
  if (leafletMap) return;
  leafletMap = L.map("leaflet").setView([22.5, 79], 5);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap, &copy; CARTO", maxZoom: 18,
  }).addTo(leafletMap);
  mapLayers = L.layerGroup().addTo(leafletMap);
}

$("mapBtn").addEventListener("click", updateMap);

async function updateMap() {
  initMap();
  setTimeout(() => leafletMap.invalidateSize(), 50);
  const btn = $("mapBtn");
  btn.disabled = true; btn.textContent = "Loading…";
  try {
    const d = await getJSON(
      `/api/graph?crop=${encodeURIComponent($("mapCrop").value)}&date=${$("mapDate").value}`);
    mapLayers.clearLayers();

    if ($("showEdges").checked) {
      const pos = Object.fromEntries(d.nodes.map((n) => [n.site_id, [n.lat, n.lon]]));
      d.edges.forEach((e) => {
        if (!pos[e.source] || !pos[e.target]) return;
        L.polyline([pos[e.source], pos[e.target]],
          { color: "#14315c", weight: 1, opacity: .3 })
          .bindTooltip(`${e.source} → ${e.target}: ${e.distance_km} km`).addTo(mapLayers);
      });
    }
    d.nodes.forEach((n) => {
      const colour = BAND_COLOUR[n.band] || "#7b838c";
      const risk = n.risk ?? 0;
      L.circleMarker([n.lat, n.lon], {
        radius: 6 + risk * 10, color: colour, weight: 2,
        fillColor: colour, fillOpacity: .5,
      }).bindPopup(`<b>${esc(n.name)}</b><br>${esc(n.state)}<br>
        Risk <b>${(risk * 100).toFixed(0)}%</b> (${esc(n.band || "n/a")})<br>
        ${n.temperature_C ?? "—"} °C · ${n.humidity_pct ?? "—"}% RH<br>
        Leaf wetness ${n.leaf_wetness_h ?? "—"} h`).addTo(mapLayers);
    });

    const ranked = d.nodes.filter((n) => n.risk != null).sort((a, b) => b.risk - a.risk);
    $("mapTable").innerHTML =
      `<thead><tr><th>Farm</th><th>State</th><th>Risk</th><th>Band</th>
        <th>Temp (°C)</th><th>RH (%)</th><th>Leaf wetness (h)</th></tr></thead><tbody>` +
      ranked.map((n) => `<tr><td>${esc(n.name)}</td><td>${esc(n.state)}</td>
        <td class="n">${(n.risk * 100).toFixed(0)}%</td>
        <td><span class="band ${n.band}">${n.band}</span></td>
        <td class="n">${n.temperature_C}</td><td class="n">${n.humidity_pct}</td>
        <td class="n">${n.leaf_wetness_h}</td></tr>`).join("") + "</tbody>";
  } catch (err) {
    $("mapTable").innerHTML = `<tbody><tr><td class="err">${esc(err.message)}</td></tr></tbody>`;
  } finally {
    btn.disabled = false; btn.textContent = "Update map";
  }
}

boot();
