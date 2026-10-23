/* Climate-Adaptive Crop Disease Forecasting - report frontend.
   The literature survey is the entry point: every research gap can be executed
   against the live system from its own row. */

const $ = (id) => document.getElementById(id);
const BAND_COLOUR = { Low: "#16a34a", Medium: "#d97706", High: "#dc2626" };

let riskChart = null, leafletMap = null, mapLayers = null, mapSatellite = false;
let selectedFile = null;
let samples = [], sampleIdx = -1;
let pipeTimer = null, pipeStarted = 0;

/* -------------------------------------------------------------- backend */
/* Where the API lives. Served by FastAPI the page and the API share an origin,
   so the base is empty. Deployed as a static site (Netlify) the backend runs on
   the presenter's machine behind a tunnel, so the base comes from ?api=, else
   from the last backend this browser connected to - tried only after the same
   origin fails, so a stale tunnel can never shadow the server the page came from. */
const API_KEY = "cropcast.api";
const API_LEVEL = 4;            // the server this page needs; see api.py
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

/* ------------------------------------------------------------------- boot */
async function boot() {
  try {
    const h = await connect();
    $("docsLink").href = api("/docs");
    const el = $("sysStatus");
    el.className = h.model_trained ? "sys ok" : "sys bad";
    el.innerHTML = `System status: <b>${h.model_trained ? "model loaded" : "model not trained"}</b>`;
    if ((h.api_level || 0) < API_LEVEL) {
      el.className = "sys bad";
      el.innerHTML = "Backend out of date: <b>restart scripts/09_go_live.py</b>";
    }
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
  await Promise.all([loadSummary(), loadSites(), loadCrops(), loadProgress()]);
  updateMap();                  // the bulletin is the first thing an officer sees
  await loadSamples();          // needs siteSelect to be populated first
  initTOC();
  layoutWires();
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
    if (viewMode !== "country") setMode("country");
    await updateMap();
    // Open this run's own farm, so the leaf and its district read as one story.
    if (d.site && districtMarkers[d.site.site_id]) {
      focusCard(d.site.site_id);
      districtMarkers[d.site.site_id].openPopup();
    }
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
  renderWhy(d);

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
  leafletMap.createPane("heat");
  leafletMap.getPane("heat").style.zIndex = 350;      // above tiles, below markers and labels
  leafletMap.getPane("heat").classList.add("heat-pane");
  mapLayers = L.layerGroup().addTo(leafletMap);
  // Anywhere on the map, in either view: the heatmap invites "what about here?"
  leafletMap.on("click", (e) => choosePoint(e.latlng.lat, e.latlng.lng, "Pinned location"));
  leafletMap.on("mousemove", readHeat);

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
// Only district markers and graph edges follow the basemap; the location pin,
// its radius ring and the spread lines keep their own colours.
function restyle(layer) {
  if (layer.options.band) layer.setStyle(markerStyle(layer.options.band));
  else if (layer.options.edge) layer.setStyle(edgeStyle());
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
$("showEdges").addEventListener("change", () => {
  if (bulletin && viewMode === "country") drawBulletinMap(bulletin, false);
});
$("printBtn").addEventListener("click", () => {
  document.body.classList.add("print-bulletin");
  window.print();
});
window.addEventListener("afterprint", () => document.body.classList.remove("print-bulletin"));

async function updateMap() {
  if (viewMode === "near") {
    if (point) return assessPoint();
    $("nearError").textContent = "Choose a location first: GPS, a place search, or a click on the map.";
    return;
  }
  initMap();
  setTimeout(() => leafletMap.invalidateSize(), 50);
  const btn = $("mapBtn");
  btn.disabled = true; btn.textContent = "Generating…";
  try {
    bulletin = await getJSON(
      `/api/bulletin?crop=${encodeURIComponent($("mapCrop").value)}&date=${$("mapDate").value}`);
    renderBulletin(bulletin);
    drawBulletinMap(bulletin, true);
    refreshHeat();
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

  $("alertList").innerHTML = b.districts.map((r) => alertCard(r)).join("");
  bindAlertCards();
}

/* One place's alert card - a monitored district, or the location being looked at. */
function alertCard(r, extraClass = "") {
  const colour = BAND_COLOUR[r.alert];
  const arrow = { Rising: "↑", Falling: "↓", Steady: "→" }[r.direction];
  const threat = r.threat
    ? `<b>${esc(r.threat.name)}</b> <span class="org">· ${esc(r.threat.organism)}</span>`
    : `<span class="org">No significant threat this week</span>`;
  const spread = r.neighbours_high && r.neighbours_high.length
    ? `<div class="acard-spread">⚠ ${r.neighbours_high.length} neighbouring district${r.neighbours_high.length > 1 ? "s" : ""} on alert: ${
        r.neighbours_high.map((n) => `${esc(n.name)} (${Math.round(n.distance_km)} km)`).join(", ")}</div>`
    : "";
  const farmer = r.threat && r.alert !== "Low"
    ? `<div><h5>For farmers</h5><ul>${r.threat.farmer_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>`
    : "";
  return `<article class="acard ${r.alert} ${extraClass}" data-site="${esc(r.site_id)}">
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
}

function bindAlertCards() {
  $("alertList").querySelectorAll(".acard, .src-card").forEach((card) => {
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
      L.polyline([[a.lat, a.lon], [t.lat, t.lon]], { ...edgeStyle(), edge: true, bubblingMouseEvents: false })
        .bindTooltip(`${a.name} → ${t.name}: ${Math.round(e.distance_km)} km`).addTo(mapLayers);
    });
  }
  // Drawn calm-first so the districts that need action sit on top.
  [...b.districts].reverse().forEach((r) => {
    const colour = BAND_COLOUR[r.alert];
    const m = L.circleMarker([r.lat, r.lon], {
      radius: 7 + Math.max(r.risk, r.peak_risk) * 11, band: colour, ...markerStyle(colour),
      bubblingMouseEvents: false,
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

/* ----------------------------------------------------- near a location */
/* The same alert, threat and actions as a district, for any point: the weather
   recorded at that exact spot, plus disease arriving from the districts around
   it on the wind. Chosen by GPS, by place search, or by clicking the map. */
let viewMode = "country";
let point = null;               // { lat, lon, label }
const SOURCE_STATE = (x) => (x > 0.5 ? "Active outbreak" : x > 0.1 ? "Outbreak building" : "Quiet");

function setMode(mode) {
  viewMode = mode;
  $("modeCountry").classList.toggle("on", mode === "country");
  $("modeNear").classList.toggle("on", mode === "near");
  $("modeCountry").setAttribute("aria-pressed", String(mode === "country"));
  $("modeNear").setAttribute("aria-pressed", String(mode === "near"));
  $("nearBar").hidden = mode !== "near";
  $("mapBtn").textContent = mode === "near" ? "Assess location" : "Generate bulletin";
  $("printBtn").textContent = mode === "near" ? "Print report" : "Print bulletin";
  if (mode === "near" && !point) {
    initMap();
    mapLayers.clearLayers();
    $("bulletinHead").hidden = true;
    $("alertList").innerHTML = `<p class="placeholder">Choose a location: use your GPS,
      search for a place, or click anywhere on the map.</p>`;
    return;
  }
  updateMap();
}
$("modeCountry").addEventListener("click", () => setMode("country"));
$("modeNear").addEventListener("click", () => setMode("near"));
$("radiusSel").addEventListener("change", () => { if (point) assessPoint(); });

function choosePoint(lat, lon, label) {
  point = { lat, lon, label };
  $("placeResults").innerHTML = "";
  $("nearError").textContent = "";
  if (viewMode !== "near") setMode("near"); else assessPoint();
}

$("gpsBtn").addEventListener("click", () => {
  $("nearError").textContent = "";
  if (!navigator.geolocation) {
    $("nearError").textContent = "This browser cannot share its location - search for a place instead.";
    return;
  }
  const btn = $("gpsBtn");
  btn.disabled = true; btn.textContent = "Locating…";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      btn.disabled = false; btn.innerHTML = "&#128205; Use my location";
      choosePoint(pos.coords.latitude, pos.coords.longitude, "Your location");
    },
    (err) => {
      btn.disabled = false; btn.innerHTML = "&#128205; Use my location";
      $("nearError").textContent = err.code === 1
        ? "Location permission was refused - search for a place or click the map instead."
        : "Could not get a GPS fix - search for a place or click the map instead.";
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 },
  );
});

$("placeForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("placeInput").value.trim();
  if (q.length < 2) return;
  $("nearError").textContent = "";
  $("placeResults").innerHTML = `<span class="hint">Searching…</span>`;
  try {
    const d = await getJSON(`/api/geocode?q=${encodeURIComponent(q)}`);
    if (!d.results.length) {
      $("placeResults").innerHTML = `<span class="hint">No place called “${esc(q)}” found.</span>`;
      return;
    }
    $("placeResults").innerHTML = d.results.map((h, i) =>
      `<button type="button" class="place-hit" data-i="${i}">${esc(h.name)} <span>${esc(h.region)}</span></button>`).join("");
    $("placeResults").querySelectorAll(".place-hit").forEach((b) => b.addEventListener("click", () => {
      const h = d.results[Number(b.dataset.i)];
      choosePoint(h.lat, h.lon, `${h.name}, ${h.region.split(",").slice(-2, -1)[0]?.trim() || h.region}`);
    }));
  } catch (err) {
    $("placeResults").innerHTML = "";
    $("nearError").textContent = err.message;
  }
});

/* A real case from the record: on 22 June 2023 Bhopal is not a monitored
   district and its own weather is only moderately favourable, but an outbreak
   is starting in Ujjain and Indore to the west and the wind is blowing east. */
$("nearExample").addEventListener("click", () => {
  $("mapCrop").value = "Corn_(maize)";
  $("mapDate").value = "2023-06-22";
  choosePoint(23.2599, 77.4126, "Bhopal, Madhya Pradesh");
});

async function assessPoint() {
  initMap();
  setTimeout(() => leafletMap.invalidateSize(), 50);
  const btn = $("mapBtn");
  btn.disabled = true; btn.textContent = "Assessing…";
  $("nearError").textContent = "";
  try {
    const q = new URLSearchParams({
      lat: point.lat.toFixed(5), lon: point.lon.toFixed(5), crop: $("mapCrop").value,
      date: $("mapDate").value, radius_km: $("radiusSel").value,
    });
    const r = await getJSON(`/api/nearby?${q}`);
    renderNear(r);
    drawNearMap(r);
    refreshHeat();
  } catch (err) {
    $("bulletinHead").hidden = true;
    $("alertList").innerHTML = `<p class="err">${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Assess location";
  }
}

function placeName(r) {
  if (point.label !== "Pinned location") return point.label;
  return r.nearest.distance_km < 40 ? `Near ${r.nearest.name}` : "Pinned location";
}

function renderNear(r) {
  const name = placeName(r);
  const calm = r.alert === "Low";
  const top = r.sources.find((s) => s.share > 0.05);
  const lead = top
    ? `Disease arriving from nearby farms: ${pct(r.incoming)}, mostly from ${top.name} ` +
      `(${Math.round(top.distance_km)} km ${top.direction}${top.wind_toward_you ? ", wind blowing toward you" : ""}).`
    : "No outbreak nearby is sending disease this way.";
  const notes = [r.note, r.in_study_area ? null
    : "This point is outside the Indian study area; treat the result as indicative only."].filter(Boolean);
  $("bHeadline").className = `headline${calm ? " calm" : ""}`;
  $("bHeadline").innerHTML = `${esc(name)}: ${ALERT_WORD[r.alert]} - risk ${pct(r.risk)} today, ${
    esc(outlookText(r).replace(/^(Low|Worst|Peaks)/, (w) => w.toLowerCase()))}. ${esc(lead)}
    <small>${esc(r.crop_label)} · as of ${fmtDate(r.date)} · ${r.lat.toFixed(3)}, ${r.lon.toFixed(3)} ·
      nearest monitored district: ${esc(r.nearest.name)}, ${Math.round(r.nearest.distance_km)} km ${r.nearest.direction}
      ${notes.map((n) => `<br>${esc(n)}`).join("")}</small>`;
  const bandOf = (v) => (v > 0.55 ? "High" : v > 0.25 ? "Medium" : "Low");
  $("bKpis").innerHTML = [
    [r.alert, pct(r.risk), "Overall risk here today"],
    [bandOf(r.local), pct(r.local), "Weather here favours disease"],
    [bandOf(r.incoming), pct(r.incoming), "Disease arriving from nearby farms"],
    [r.peak_band, pct(r.peak_risk), r.peak_day ? `Worst in the next 7 days (day ${r.peak_day})` : "Worst in the next 7 days (today)"],
  ].map(([c, v, l]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  $("bulletinHead").hidden = false;

  const here = alertCard({
    ...r, site_id: "here", name: `📍 ${name}`,
    state: r.weather_source,
  }, "here open");
  const sources = r.sources.map((s) => `
    <div class="src-card" data-site="${esc(s.site_id)}">
      <div class="src-top">
        <span class="acard-name">${esc(s.name)}</span>
        <span class="acard-state">${esc(s.state)} · ${Math.round(s.distance_km)} km ${s.direction} of here</span>
        <span class="pill ${s.alert}">${ALERT_WORD[s.alert]}</span>
      </div>
      <div class="src-meta">${SOURCE_STATE(s.infection)} ·
        <span class="src-wind ${s.wind_toward_you ? "toward" : ""}">${s.wind_toward_you
          ? `wind blowing toward you (${Math.round(s.wind_kmh)} km/h)` : "wind not toward you"}</span></div>
      <div class="src-meta">${pct(s.share)} of the disease arriving here today</div>
      <div class="share-bar"><i style="width:${Math.max(s.share * 100, s.share > 0 ? 2 : 0)}%"></i></div>
    </div>`).join("");
  $("alertList").innerHTML = here +
    `<div class="list-head">Where disease could come from · within ${esc($("radiusSel").value)} km</div>` +
    (sources || `<p class="placeholder">No monitored ${esc(r.crop_label.toLowerCase())} district within
      ${esc($("radiusSel").value)} km. The nearest is ${esc(r.nearest.name)}, ${Math.round(r.nearest.distance_km)} km away.</p>`);
  bindAlertCards();
}

function drawNearMap(r) {
  mapLayers.clearLayers();
  Object.keys(districtMarkers).forEach((k) => delete districtMarkers[k]);
  const here = [r.lat, r.lon];
  const ring = L.circle(here, {
    radius: r.radius_km * 1000, color: "#f97316", weight: 1.5, dashArray: "6 6", fill: false,
  }).addTo(mapLayers);

  // Spread lines: thicker for a larger share, dashed when the wind is not
  // carrying spores this way.
  r.sources.forEach((s) => {
    L.polyline([[s.lat, s.lon], here], {
      color: BAND_COLOUR[s.alert], weight: 1.5 + s.share * 10, opacity: .75,
      dashArray: s.wind_toward_you ? null : "4 6", bubblingMouseEvents: false,
    }).bindTooltip(`${s.name} → here: ${Math.round(s.distance_km)} km, ${pct(s.share)} of incoming`)
      .addTo(mapLayers);
    const colour = BAND_COLOUR[s.alert];
    const m = L.circleMarker([s.lat, s.lon], { radius: 9, band: colour, ...markerStyle(colour),
      bubblingMouseEvents: false })
      .bindPopup(`<b>${esc(s.name)}</b>, ${esc(s.state)}<br>
        <span class="pill ${s.alert}">${ALERT_WORD[s.alert]}</span><br>
        ${SOURCE_STATE(s.infection)} · ${Math.round(s.distance_km)} km ${s.direction} of here<br>
        ${s.wind_toward_you ? "Wind blowing toward you" : "Wind not toward you"} · ${pct(s.share)} of incoming`)
      .addTo(mapLayers);
    m.on("click", () => focusCard(s.site_id));
    districtMarkers[s.site_id] = m;
  });

  const pin = L.circleMarker(here, {
    radius: 11, color: "#ffffff", weight: 3, fillColor: "#f97316", fillOpacity: 1,
    bubblingMouseEvents: false,
  }).bindPopup(`<span class="pin-label">${esc(placeName(r))}</span><br>
    <span class="pill ${r.alert}">${ALERT_WORD[r.alert]}</span> risk ${pct(r.risk)} today`)
    .addTo(mapLayers);
  districtMarkers.here = pin;
  leafletMap.invalidateSize();
  leafletMap.fitBounds(ring.getBounds().pad(0.05));
}

/* ----------------------------------------------------------------- heatmap */
/* Every grid cell is computed server-side exactly as a clicked location is:
   the weather recorded there through the pathogen physics, plus disease
   arriving on the wind. The page only interpolates between cells, pins the
   surface to each district's own reading, and clips it to land. */
let heatData = null, heatKey = "", heatOverlay = null, landMask = null;
const HEAT_PX = 18;                    // canvas pixels per degree of longitude
const ANCHOR_REACH_KM = 50;            // how far a district's own reading reaches
const GRID_WEIGHT_KM = 35;             // the grid counts as a reading this far away
const HEAT_STOPS = [[0, [22, 163, 74]], [0.25, [234, 179, 8]], [0.55, [249, 115, 22]],
                    [0.8, [220, 38, 38]], [1, [153, 27, 27]]];
const HEAT_NAME = { weather: "Disease-favourable weather", spread: "Disease arriving",
                    risk: "Combined risk" };

function heatColour(v) {
  for (let k = 1; k < HEAT_STOPS.length; k++) {
    const [b, cb] = HEAT_STOPS[k];
    if (v <= b) {
      const [a, ca] = HEAT_STOPS[k - 1];
      const t = (v - a) / (b - a);
      return ca.map((c, i) => Math.round(c + t * (cb[i] - c)));
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

// Web Mercator, which is how Leaflet stretches an image overlay.
const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const mercLat = (y) => (360 / Math.PI) * Math.atan(Math.exp(y)) - 90;
const kmBetween = (a1, o1, a2, o2) => {
  const r = Math.PI / 180, dA = (a2 - a1) * r, dO = (o2 - o1) * r;
  const h = Math.sin(dA / 2) ** 2 + Math.cos(a1 * r) * Math.cos(a2 * r) * Math.sin(dO / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(Math.min(1, h)));
};

/* [value, cover] at any point: bilinear between the four surrounding cells
   (empty cells dropped and the weights renormalised), then pinned to nearby
   districts for the weather layer - the only layer the districts measure
   directly. cover is how much of the neighbourhood holds data, which fades the
   shading out over the last cell instead of ending in a 1 degree staircase. */
function heatSampler() {
  const d = heatData, layer = $("heatLayer").value, when = $("heatWhen").value;
  const grid = d.layers[layer][when];
  const anchors = layer === "weather" ? d.anchors : [];
  return (lat, lon) => {
    const fi = (lat - d.lats[0]) / d.step, fj = (lon - d.lons[0]) / d.step;
    const i0 = Math.floor(fi), j0 = Math.floor(fj);
    let sw = 0, sv = 0;
    for (const [i, j, w] of [[i0, j0, (1 - (fi - i0)) * (1 - (fj - j0))], [i0 + 1, j0, (fi - i0) * (1 - (fj - j0))],
                             [i0, j0 + 1, (1 - (fi - i0)) * (fj - j0)], [i0 + 1, j0 + 1, (fi - i0) * (fj - j0)]]) {
      const v = grid[i] && grid[i][j];
      if (v != null && w > 0) { sw += w; sv += w * v; }
    }
    const g = sw > 0 ? sv / sw : NaN;
    // Inverse-distance pinning: a district's reading is reproduced exactly at
    // the district, dominates within ~20 km, and has faded into the grid by
    // ~60 km. Two close districts (Indore and Ujjain, 50 km) each stay exact.
    let U = 0, A = 0, reach = 0;
    for (const a of anchors) {
      if (Math.abs(a.lat - lat) > 1 || Math.abs(a.lon - lon) > 1) continue;
      const d = kmBetween(lat, lon, a.lat, a.lon);
      if (d < 0.5) return [a[when], 1];
      const taper = Math.exp(-((d / ANCHOR_REACH_KM) ** 2));
      U += taper / (d * d); A += (taper / (d * d)) * a[when]; reach += taper;
    }
    if (U === 0) return [g, sw];
    const cover = Math.max(sw, Math.min(1, reach));
    if (Number.isNaN(g)) return reach > 0.05 ? [A / U, cover] : [NaN, 0];
    const ug = 1 / (GRID_WEIGHT_KM * GRID_WEIGHT_KM);
    return [(A + ug * g) / (U + ug), cover];
  };
}

function heatBounds() {
  const d = heatData, h = d.step / 2;
  return { s: d.lats[0] - h, n: d.lats[d.lats.length - 1] + h,
           w: d.lons[0] - h, e: d.lons[d.lons.length - 1] + h };
}

/* Land outline, drawn once into a mask in the overlay's own projection, so the
   heat never bleeds into the sea. No country borders are drawn. */
async function loadLand() {
  if (landMask !== null) return landMask;
  try {
    const topo = await (await fetch("https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json")).json();
    landMask = topojson.feature(topo, topo.objects.land);
  } catch { landMask = false; }        // offline: draw unclipped rather than not at all
  return landMask;
}

function drawHeat() {
  if (heatOverlay) { leafletMap.removeLayer(heatOverlay); heatOverlay = null; }
  const off = $("heatLayer").value === "off";
  $("heatWhen").disabled = off;
  $("heatLegend").hidden = off;
  if (off || !heatData) return;

  const b = heatBounds();
  const W = Math.round((b.e - b.w) * HEAT_PX);
  const yN = mercY(b.n), yS = mercY(b.s);
  const H = Math.round(W * (yN - yS) / ((b.e - b.w) * Math.PI / 180));
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(W, H);
  const heatValue = heatSampler();
  for (let y = 0; y < H; y++) {
    const lat = mercLat(yN - ((y + 0.5) / H) * (yN - yS));
    for (let x = 0; x < W; x++) {
      const [v, cover] = heatValue(lat, b.w + ((x + 0.5) / W) * (b.e - b.w));
      if (Number.isNaN(v) || cover <= 0) continue;
      const c = Math.max(0, Math.min(1, v));
      const [r, g, bl] = heatColour(c);
      const k = (y * W + x) * 4;
      const fade = cover * cover * (3 - 2 * cover);          // smoothstep
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = bl;
      img.data[k + 3] = Math.round(255 * (0.22 + 0.48 * c) * fade);
    }
  }
  ctx.putImageData(img, 0, 0);

  if (landMask) {
    const m = document.createElement("canvas");
    m.width = W; m.height = H;
    const mc = m.getContext("2d");
    const px = ([lon, lat]) => [((lon - b.w) / (b.e - b.w)) * W, ((yN - mercY(lat)) / (yN - yS)) * H];
    mc.beginPath();
    const polys = landMask.geometry ? [landMask.geometry] : landMask.features.map((f) => f.geometry);
    for (const geom of polys) {
      const list = geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates;
      for (const poly of list) for (const ring of poly) {
        if (!ring.some(([lo, la]) => lo > b.w - 5 && lo < b.e + 5 && la > b.s - 5 && la < b.n + 5)) continue;
        ring.forEach((p, i) => { const [x, y] = px(p); i ? mc.lineTo(x, y) : mc.moveTo(x, y); });
        mc.closePath();
      }
    }
    mc.fill("evenodd");
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(m, 0, 0);
  }

  heatOverlay = L.imageOverlay(cv.toDataURL(), [[b.s, b.w], [b.n, b.e]], {
    pane: "heat", interactive: false,
  }).addTo(leafletMap);
}

async function refreshHeat() {
  if ($("heatLayer").value === "off") { drawHeat(); return; }
  const key = `${$("mapCrop").value}|${$("mapDate").value}`;
  try {
    if (key !== heatKey) {
      $("heatRead").textContent = "Computing the risk surface from recorded weather…";
      const [d] = await Promise.all([
        getJSON(`/api/heatmap?crop=${encodeURIComponent($("mapCrop").value)}&date=${$("mapDate").value}`),
        loadLand(),
      ]);
      heatData = d; heatKey = key;
    }
    drawHeat();
    $("heatRead").textContent = `${heatData.cells} grid cells, each from the weather recorded there` +
      " · hover to read a value · click to assess that spot";
  } catch (err) {
    heatData = null; heatKey = ""; drawHeat();
    if (/rate-limit/i.test(err.message)) {
      // The free weather archive allows ~600 lookups a minute; one new date
      // uses 341. Wait it out once rather than leave the map bare.
      $("heatRead").textContent = "The weather archive is busy - retrying in 30 seconds…";
      setTimeout(() => {
        if (key === `${$("mapCrop").value}|${$("mapDate").value}` && !heatData) refreshHeat();
      }, 30000);
    } else {
      $("heatRead").textContent = err.message;
    }
  }
}
$("heatLayer").addEventListener("change", refreshHeat);
$("heatWhen").addEventListener("change", drawHeat);

function readHeat(e) {
  if (!heatData || $("heatLayer").value === "off") return;
  const [v] = heatSampler()(e.latlng.lat, e.latlng.lng);
  const when = $("heatWhen").value === "week" ? "worst in next 7 days" : "today";
  $("heatRead").textContent = Number.isNaN(v)
    ? "Outside the monitored region"
    : `${HEAT_NAME[$("heatLayer").value]} at ${e.latlng.lat.toFixed(2)}, ${e.latlng.lng.toFixed(2)}: ` +
      `${pct(v)} (${when}) · ${ALERT_WORD[v > 0.55 ? "High" : v > 0.25 ? "Medium" : "Low"]}`;
}

/* ------------------------------------------------------- why this result */
/* The chain from this leaf to this forecast: what the disease needs, what the
   weather at this farm actually was, whether the disease is here or arriving,
   and why the district map can say "act now" while this farm's forecast is
   lower. Every link is a number the server computed. */
// Graded from the farmer's side: weather that suits the disease is the warning.
const GRADE_TEXT = { favourable: "⚠ suits the disease", "partly favourable": "≈ partly suits it",
                     unfavourable: "✓ does not suit it" };

function whyRow(icon, label, value, grade) {
  const pill = grade ? `<span class="why-grade g-${grade.split(" ")[0]}">${GRADE_TEXT[grade]}</span>` : "";
  return `<li><span class="why-ico">${icon}</span><div><div class="why-label">${esc(label)}</div>
    <div class="why-val">${value}</div></div>${pill}</li>`;
}

function renderWhy(d) {
  const c = d.context || {};
  const td = c.this_disease;
  const farm = `${esc(d.site.name)}, ${fmtDate(d.date)}`;
  const worst = d.risk_band.reduce((a, b) => (RISK_ORDER[b] > RISK_ORDER[a] ? b : a), "Low");
  const rows = [whyRow("🍃", "This leaf", `<b>${esc(d.disease_label)}</b> · ${(d.confidence * 100).toFixed(1)}% sure`)];

  if (td) {
    rows.push(whyRow("🧫", `What ${td.name.toLowerCase()} needs`,
      `${esc(td.needs.temperature)} · ${esc(td.needs.moisture)}`));
    rows.push(whyRow("🌡", `Temperature at ${farm}`,
      `${esc(td.observed.temperature)} - ${esc(td.temperature_note)}`, td.temperature_grade));
    rows.push(whyRow("💧", "Moisture that day", esc(td.observed.moisture), td.moisture_grade));
    rows.push(whyRow("⚖", `How much this weather suits ${td.name.toLowerCase()}`,
      `${pct(td.today)} today${td.peak_day ? `, rising to ${pct(td.peak)} in ${days(td.peak_day)}` : ""}`,
      td.peak >= 0.55 ? "favourable" : td.peak >= 0.25 ? "partly favourable" : "unfavourable"));
  } else {
    rows.push(whyRow("🌦", `Weather threat for ${esc((c.crop_label || pretty(d.crop)).toLowerCase())} at ${farm}`,
      c.district ? `${ALERT_WORD[c.district.alert]}${c.district.threat ? ` - main threat ${esc(c.district.threat)}` : ""}` : "-"));
  }
  const from = c.arriving_from && c.disease_arriving > 0.01
    ? ` (mostly from ${esc(c.arriving_from.name)}, ${Math.round(c.arriving_from.distance_km)} km)` : "";
  rows.push(whyRow("🦠", "Is the disease spreading here?",
    `Present at this farm: <b>${pct(c.disease_present || 0)}</b> · arriving from neighbours: <b>${pct(c.disease_arriving || 0)}</b>${from}`));
  rows.push(whyRow("📈", "So the 7-day spread forecast is",
    `<b>${worst}</b> - ${d.horizons.map((h, i) => `${pct(d.risk[i])} in ${days(h)}`).join(", ")}`));
  $("whyChain").innerHTML = rows.join("");

  // Reconcile with the district map, which the run has just moved to.
  // How much disease is about: none, a trace, or an active outbreak. The
  // presence scale saturates quickly, so a trace must not read like an outbreak.
  const spread = Math.max(c.disease_present || 0, c.disease_arriving || 0);
  const level = spread < 0.05 ? "none" : spread < 0.3 ? "little" : "much";
  const dist = c.district;
  const crop = esc((c.crop_label || pretty(d.crop)).toLowerCase());
  let note = "";
  if (dist) {
    const mapSays = `The district map marks ${esc(d.site.name)} <b>${ALERT_WORD[dist.alert]}</b>` +
      (dist.alert === "Low" ? ` - this week's weather does not favour ${crop} diseases.`
        : ` because this week's weather favours ${crop} diseases${dist.threat ? ` (leading: ${esc(dist.threat)})` : ""}.`);
    const where = (c.disease_present || 0) >= 0.3 && (c.disease_arriving || 0) >= 0.3
      ? "present at this farm and arriving from its neighbours"
      : (c.disease_present || 0) >= (c.disease_arriving || 0) ? "already present at this farm" : "arriving from neighbouring farms";
    if (RISK_ORDER[worst] < RISK_ORDER[dist.alert]) {
      note = level === "much"
        ? `${mapSays} The disease is ${where}, but the model expects ${worst === "Low" ? "little" : "moderate"}
           spread here this week, so this farm is <b>${worst}</b> - keep it under watch.`
        : `${mapSays} This farm's forecast is lower because ${level === "none" ? "no disease is"
           : "only a little disease is"} present here or arriving from neighbours - spread needs
           <b>both</b> favourable weather and the disease nearby.`;
    } else {
      note = level === "much"
        ? `${mapSays} The disease is ${where}, so this farm's forecast is <b>${worst}</b>.`
        : `${mapSays} This week's weather alone is enough to put this farm's forecast at <b>${worst}</b>.`;
    }
  }
  $("whyNote").innerHTML = note;
}

/* ------------------------------------------------------------ how it works */
/* One held-out test leaf, sent through the live pipeline, with each block of
   the diagram lighting up in turn and showing what it really produced. The
   model finishes in about 20 ms, so the steps are paced for people to follow;
   every value and timing on screen is the one the server measured. */
const FLOW_WIRES = [
  ["leaf", "vision"], ["weather", "climate"], ["farms", "graph"],
  ["vision", "fusion"], ["climate", "fusion"], ["fusion", "gnn"], ["graph", "gnn"],
  ["gnn", "diagnosis"], ["gnn", "forecast"], ["diagnosis", "advice"], ["forecast", "advice"],
];
const HOW_DWELL_MS = 2600;        // long enough to read a caption aloud
const wireState = {};
let howIdx = -1, howBusy = false;
const flowNode = (id) => document.querySelector(`.fnode[data-node="${id}"]`);
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

document.querySelectorAll(".fnode .fn-out").forEach((el) => { el.dataset.idle = el.innerHTML; });
document.querySelectorAll(".fnode .fn-top").forEach((el) => el.insertAdjacentHTML("afterend", '<div class="fn-tech"></div>'));

const millions = (n) => `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`;
/* Short technical labels per block, from what the server measured. */
function fillTech(e) {
  const tech = {
    leaf: "JPEG → 224×224 RGB",
    weather: "Open-Meteo ERA5 archive",
    farms: `${e.graph_k} nearest · ≤${e.graph_radius_km} km · wind-aware`,
    vision: `DINOv2 ViT-S/14 · ${millions(e.backbone_params)} params · frozen`,
    climate: `${e.climate_dim} weather + ${e.meta_dim} farm features`,
    graph: `KNN + radius graph`,
    fusion: `FiLM gating → ${e.fusion_dim}-d`,
    gnn: `GraphSAGE · ${e.gnn_layers} layers · ${e.gnn_hidden}-d`,
    diagnosis: `MLP + softmax · ${e.classes} classes`,
    forecast: `GRU decoder · ${e.horizons.length} horizons`,
    advice: "Rules on pathogen knowledge base",
  };
  Object.entries(tech).forEach(([id, txt]) => { flowNode(id).querySelector(".fn-tech").textContent = txt; });
  $("engineStrip").innerHTML = `<span><b>Engine</b> ${esc(e.framework)} on ${esc(e.device)}</span>
    <span><b>Server</b> ${esc(e.server)}</span>
    <span><b>Model size</b> ${millions(e.backbone_params)} frozen + ${millions(e.model_params)} trained parameters</span>`;
}

/* One line of "what ran here" under each caption. */
function techLine(t, e) {
  const line = {
    Input: `Data: Open-Meteo ERA5 reanalysis · ${t.output}`,
    Vision: `Model: ${e.backbone} (ViT-S/14, ${millions(e.backbone_params)} parameters, frozen) · 224×224 in → ${e.vision_dim} numbers out`,
    Climate: `Engine: pandas + NumPy · VPD, leaf wetness, GDD, 3/7/14-day windows · ${e.climate_dim} + ${e.meta_dim} features, standardised`,
    Fusion: `Model: FiLM (feature-wise linear modulation) · 3 streams → ${e.fusion_dim} numbers`,
    Graph: `Model: GraphSAGE, ${e.gnn_layers} layers, ${e.gnn_hidden}-d · graph: ${e.graph_k} nearest within ${e.graph_radius_km} km`,
    Diagnosis: `Model: 2-layer MLP + softmax over ${e.classes} classes`,
    Consistency: "Engine: re-run with corrected crop metadata",
    Forecast: `Model: GRU-cell decoder · +${e.horizons.join(", +")} days · risk, band and uncertainty`,
    Advisory: "Engine: rule-based advisor on the pathogen knowledge base",
  }[t.block];
  return line ? `<div class="how-tech">${esc(line)} · ${t.ms.toFixed(1)} ms on ${esc(e.device.split(" (")[0])}</div>` : "";
}

function layoutWires() {
  const flow = $("flow"), svg = $("flowWires");
  // Stacked on a phone: the order of the blocks carries the flow, not wires.
  if (getComputedStyle(flow).gridTemplateColumns.trim().split(/\s+/).length < 2) { svg.innerHTML = ""; return; }
  const box = flow.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  svg.innerHTML = FLOW_WIRES.map(([a, b]) => {
    const ra = flowNode(a).getBoundingClientRect(), rb = flowNode(b).getBoundingClientRect();
    const id = `w-${a}-${b}`, state = wireState[`${a}>${b}`] || "";
    let d;
    if (rb.left >= ra.right - 4) {          // next column: right edge to left edge
      const x1 = ra.right - box.left, y1 = ra.top + ra.height / 2 - box.top;
      const x2 = rb.left - box.left, y2 = rb.top + rb.height / 2 - box.top;
      const mx = (x1 + x2) / 2;
      d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    } else {                                // same column: straight down
      const x = ra.left + ra.width / 2 - box.left;
      d = `M${x},${ra.bottom - box.top} L${x},${rb.top - box.top}`;
    }
    const packet = state === "flow"
      ? `<circle class="packet" r="5"><animateMotion dur="0.9s" repeatCount="indefinite"><mpath href="#${id}"/></animateMotion></circle>`
      : "";
    return `<path id="${id}" class="wire ${state}" d="${d}"/>${packet}`;
  }).join("");
}
let wireTimer;
window.addEventListener("resize", () => { clearTimeout(wireTimer); wireTimer = setTimeout(layoutWires, 120); });
window.addEventListener("load", layoutWires);

function resetFlow() {
  Object.keys(wireState).forEach((k) => delete wireState[k]);
  document.querySelectorAll(".fnode").forEach((n) => {
    n.classList.remove("live", "done");
    const out = n.querySelector(".fn-out");
    out.innerHTML = out.dataset.idle;
  });
  $("howThumb").classList.remove("show");
  layoutWires();
}

/* Close the step before: live blocks become done, flowing wires settle. */
function settle() {
  document.querySelectorAll(".fnode.live").forEach((n) => { n.classList.remove("live"); n.classList.add("done"); });
  Object.keys(wireState).forEach((k) => { if (wireState[k] === "flow") wireState[k] = "done"; });
}

function stage(wires, nodes, caption, n, total) {
  settle();
  wires.forEach(([a, b]) => { wireState[`${a}>${b}`] = "flow"; });
  nodes.forEach(([id, html]) => {
    const el = flowNode(id);
    el.classList.remove("done"); el.classList.add("live");
    if (html != null) el.querySelector(".fn-out").innerHTML = html;
  });
  $("howCaption").innerHTML = caption;
  $("howStep").textContent = `Step ${n} of ${total}`;
  layoutWires();
}

/* What each block of the pipeline means, in words, with its real output. */
function describe(t, r, s) {
  const ms = `<span class="ms">${t.ms.toFixed(1)} ms</span>`;
  const c = r.climate;
  const dims = (t.output.match(/\d+/) || [""])[0];
  switch (t.block) {
    case "Input": return [[], [["weather", `${c.temperature_C} °C · ${c.humidity_pct}% humidity · ${c.rainfall_mm} mm rain`]],
      `<b>Weather lookup.</b> The system fetches the weather actually recorded at ${esc(r.site.name)} on
       ${fmtDate(r.date)}: ${c.temperature_C} °C, ${c.humidity_pct}% humidity, ${c.rainfall_mm} mm of rain. ${ms}`];
    case "Vision": return [[["leaf", "vision"]], [["vision", `Leaf summarised as ${dims} numbers`]],
      `<b>Vision transformer.</b> A transformer pre-trained on millions of images reads the photo and
       describes the leaf as ${dims} numbers - its spots, colours and textures. ${ms}`];
    case "Climate": return [[["weather", "climate"]], [["climate", `${c.leaf_wetness_h} h leaf wetness · VPD ${c.vpd_kPa} kPa`]],
      `<b>Disease weather.</b> Raw weather becomes what plant pathologists use: hours the leaf stays wet
       (${c.leaf_wetness_h} h), how fast the air dries it (${c.vpd_kPa} kPa), and heat build-up. ${ms}`];
    case "Fusion": return [[["vision", "fusion"], ["climate", "fusion"]], [["fusion", `One ${dims}-number summary of this farm`]],
      `<b>Fusion.</b> Leaf, weather and farm details become one summary. The weather decides which leaf
       features matter - the same spots mean more in wet weather than in dry. ${ms}`];
    case "Graph": {
      const edges = Number((t.output.match(/(\d+) edges?/) || [0, 0])[1]);
      return [[["farms", "graph"], ["graph", "gnn"], ["fusion", "gnn"]],
        [["graph", edges ? `${edges} links to neighbouring farms` : "No neighbour reports in this run"],
         ["gnn", "Farm summary refined"]],
        `<b>Graph neural network.</b> Each farm is combined with its linked neighbours, along wind-aware
         links up to 300 km. ${edges ? `This run had ${edges} links.` : `This test sends a single photo, so
         there are no neighbour reports and the farm stands alone - in <b>District Alerts</b>, neighbours
         drive the spread.`} ${ms}`];
    }
    case "Diagnosis": {
      const right = r.disease === s.true_class;
      if (r.disease.endsWith("___healthy")) {
        return [[["gnn", "diagnosis"]], [["diagnosis", `<b>Healthy</b> · ${(r.confidence * 100).toFixed(1)}% sure`]],
          `<b>Diagnosis.</b> Checked against 38 crop diseases, the leaf is <b>healthy</b>,
           ${(r.confidence * 100).toFixed(1)}% confident. ${right ? "Correct: an expert labelled it healthy too." :
           `An expert labelled it ${esc(pretty(s.true_class))}.`} ${ms}`];
      }
      return [[["gnn", "diagnosis"]], [["diagnosis", `<b>${esc(r.disease_label)}</b> · ${(r.confidence * 100).toFixed(1)}% sure`]],
        `<b>Diagnosis.</b> Out of 38 crop diseases, the most likely is <b>${esc(r.disease_label)}</b>,
         ${(r.confidence * 100).toFixed(1)}% confident. ${right ? "Correct: an expert labelled this leaf the same way." :
         `An expert labelled it ${esc(pretty(s.true_class))}.`} ${ms}`];
    }
    case "Consistency": return [[], [["fusion", "Re-checked with the right crop"]],
      `<b>Self-check.</b> ${esc(t.output)} - the farm details are corrected and the pass is repeated. ${ms}`];
    case "Forecast": return [[["gnn", "forecast"]],
      [["forecast", r.horizons.map((h, i) => `+${h}d ${pct(r.risk[i])}`).join(" · ")]],
      `<b>7-day forecast.</b> The chance the disease spreads at this farm: ${r.horizons.map((h, i) =>
        `${pct(r.risk[i])} in ${days(h)}`).join(", ")} - ${esc(r.risk_band.every((b) => b === "Low") ? "low throughout" :
        `peaking at ${r.risk_band.reduce((a, b) => (RISK_ORDER[b] > RISK_ORDER[a] ? b : a))}`)}. ${ms}`];
    case "Advisory": return [[["diagnosis", "advice"], ["forecast", "advice"]],
      [["advice", esc((r.farmer_actions && r.farmer_actions[0]) || r.advisory)]],
      `<b>What to do.</b> ${esc(r.advisory)} ${ms}`];
    default: return [[], [], `<b>${esc(t.name)}.</b> ${esc(t.output)} ${ms}`];
  }
}

$("howPlay").addEventListener("click", playHow);

async function playHow() {
  if (howBusy) return;
  if (!samples.length) { $("howCaption").textContent = "Test leaves are unavailable - is the backend running?"; return; }
  howBusy = true;
  const btn = $("howPlay");
  btn.disabled = true; btn.textContent = "Playing…";
  resetFlow();
  howIdx = (howIdx + 1) % samples.length;
  const s = samples[howIdx];
  try {
    const blob = await (await fetch(api(`/api/samples/${encodeURIComponent(s.file)}`))).blob();
    const fd = new FormData();
    fd.append("image", new File([blob], s.file, { type: blob.type || "image/jpeg" }));
    fd.append("site_id", s.site_id);
    fd.append("date", s.date);
    const res = await fetch(api("/api/predict/stream"), { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "The pipeline did not run");
    const msgs = (await res.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const err = msgs.find((m) => m.type === "error");
    if (err) throw new Error(err.detail);
    const steps = msgs.filter((m) => m.type === "step").map((m) => m.step);
    const r = msgs.find((m) => m.type === "result").result;
    const total = steps.length + 2;

    if (r.engine) fillTech(r.engine);
    $("howThumb").src = URL.createObjectURL(blob);
    $("howThumb").classList.add("show");
    stage([], [["leaf", `${esc(pretty(s.crop))} leaf · test photo ${howIdx + 1} of ${samples.length}`],
               ["farms", `${esc(s.farm)} and the farms around it`]],
      `<b>The inputs.</b> A photo of a ${esc(pretty(s.crop).toLowerCase())} leaf from ${esc(s.farm)}, taken on
       ${fmtDate(s.date)}, and the network of farms around it. The model never saw this photo while learning.
       <div class="how-tech">Inputs: one RGB photo · farm ID + date · registry of 37 districts</div>`, 1, total);
    await pause(HOW_DWELL_MS);

    for (const [k, t] of steps.entries()) {
      const [wires, nodes, caption] = describe(t, r, s);
      stage(wires, nodes, caption + (r.engine ? techLine(t, r.engine) : ""), k + 2, total);
      await pause(HOW_DWELL_MS);
    }
    settle();
    const totalMs = steps.reduce((a, t) => a + t.ms, 0);
    $("howCaption").innerHTML = `<b>Done in ${totalMs.toFixed(0)} ms.</b> One photo became a diagnosis
      (${esc(r.disease_label)}), a 7-day risk forecast and advice for the farmer and officials.
      Press <b>Next leaf</b> to run another.`;
    $("howStep").textContent = `Step ${total} of ${total}`;
    layoutWires();
  } catch (e) {
    $("howCaption").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally {
    howBusy = false;
    btn.disabled = false; btn.innerHTML = "&#9654; Next leaf";
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
