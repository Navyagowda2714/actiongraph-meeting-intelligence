"use strict";
// ActionGraph dashboard: vanilla JS, no build step. All transcript-derived text is escaped.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { meetings: [], current: null, data: null, asOf: null };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 2600);
}

const PALETTE = ["#4f46e5", "#0891b2", "#c2410c", "#7c3aed", "#059669", "#db2777", "#2563eb", "#ca8a04"];
function colorFor(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
function initials(name) {
  return name.split(/\s+/).map((p) => p[0]).slice(0, 2).join("").toUpperCase();
}
function ownerHtml(owner) {
  if (!owner) return `<span class="owner"><span class="avatar none">?</span>Unassigned</span>`;
  return `<span class="owner"><span class="avatar" style="background:${colorFor(owner)}">${esc(initials(owner))}</span>${esc(owner)}</span>`;
}
function daysBetween(a, b) {
  return Math.round((new Date(b + "T00:00:00") - new Date(a + "T00:00:00")) / 86400000);
}
function fmtDate(iso) {
  return new Date(iso + "T00:00:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

// ---------------------------------------------------------------- sidebar
async function loadMeetings(selectId) {
  state.meetings = await api("/api/meetings");
  const nav = $("#meeting-list");
  if (!state.meetings.length) {
    nav.innerHTML = `<div class="muted small" style="padding:0 8px">No meetings yet.</div>`;
    return;
  }
  const bySeries = {};
  for (const m of state.meetings) (bySeries[m.series] ||= []).push(m);
  nav.innerHTML = Object.entries(bySeries).map(([series, ms]) => `
    <div class="series-label">${esc(series)}</div>
    ${ms.map((m) => `
      <a class="meeting-link" data-id="${esc(m.id)}">
        <div class="t">${esc(m.title)}</div>
        <div class="s"><span>${esc(m.meeting_date)}</span><span>${m.open}/${m.actions} open</span>
          ${m.review ? `<span style="color:var(--med)">${m.review} review</span>` : ""}</div>
      </a>`).join("")}`).join("");
  $$(".meeting-link").forEach((a) => a.addEventListener("click", () => selectMeeting(a.dataset.id)));
  const target = selectId || state.current || state.meetings[state.meetings.length - 1].id;
  await selectMeeting(target);
}

async function loadNotifications() {
  const asOf = state.asOf || new Date().toISOString().slice(0, 10);
  const items = await api(`/api/notifications?today=${asOf}&days=7`);
  const ul = $("#notifications");
  const head = `<li class="muted small">As of <input type="date" id="asof" value="${asOf}" style="font:inherit;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:2px 4px"></li>`;
  ul.innerHTML = head + (items.length ? items.map((t) => `
    <li><div>${esc(t.item.description)}</div>
      <div class="when ${t.overdue ? "overdue" : ""}">${t.overdue ? `${-t.days_left}d overdue` : t.days_left === 0 ? "due today" : `in ${t.days_left}d`} · ${esc(t.item.owner || "unassigned")}</div></li>`).join("")
    : `<li class="muted small">Nothing due.</li>`);
  $("#asof").addEventListener("change", (e) => { state.asOf = e.target.value; loadNotifications(); });
}

// ---------------------------------------------------------------- meeting view
async function selectMeeting(id) {
  state.current = id;
  $$(".meeting-link").forEach((a) => a.classList.toggle("active", a.dataset.id === id));
  state.data = await api(`/api/meetings/${encodeURIComponent(id)}`);
  render();
}

function render() {
  const { analysis: a, transcript: t, tracking } = state.data;
  $("#empty-state").hidden = true;
  $("#meeting-view").hidden = false;
  $("#m-series").textContent = a.series;
  $("#m-title").textContent = a.title;
  $("#m-date").textContent = fmtDate(a.meeting_date);
  $("#m-participants").textContent = a.participants.join(", ");
  for (const fmt of ["md", "json", "ics"]) $(`#exp-${fmt}`).href = `/api/meetings/${encodeURIComponent(a.meeting_id)}/export/${fmt}`;

  const open = a.action_items.filter((i) => i.status === "open");
  $("#k-open").textContent = open.length;
  $("#k-due").textContent = open.filter((i) => i.deadline && daysBetween(a.meeting_date, i.deadline) <= 3).length;
  $("#k-review").textContent = a.action_items.filter((i) => i.needs_review).length;
  $("#k-decisions").textContent = a.decisions.length;
  $("#k-risks").textContent = a.risks.length;
  $("#c-actions").textContent = a.action_items.length;
  $("#c-decisions").textContent = a.decisions.length;
  $("#c-risks").textContent = a.risks.length;
  $("#c-turns").textContent = t.turns.length;

  $("#summary-text").textContent = a.summary;
  const tr = tracking || { completed: [], still_open: [], reraised: [] };
  $("#tr-completed").innerHTML = tr.completed.length ? tr.completed.map((e) => `
    <li>✓ ${esc(e.description)} <span class="muted">(${esc(e.owner || "unassigned")})</span>
      <span class="q">“${esc(e.quote)}” <a data-turn="${esc(e.turn_id)}" class="jump">${esc(e.turn_id)}</a></span></li>`).join("")
    : `<li class="muted">Nothing closed (first meeting in the series, or no updates).</li>`;
  $("#tr-open").innerHTML = tr.still_open.length || tr.reraised.length ? [
    ...tr.reraised.map((e) => `<li>↻ ${esc(e.description)} <span class="chip carry">re-raised</span><span class="q">“${esc(e.quote)}”</span></li>`),
    ...tr.still_open.map((e) => `<li>• ${esc(e.description)} <span class="muted">(${esc(e.owner || "unassigned")})</span></li>`),
  ].join("") : `<li class="muted">No carried-over items.</li>`;
  const u = a.usage;
  $("#usage").innerHTML = `<span>Model: <b>${esc(u.model)}</b></span><span>LLM calls: ${u.llm_calls}</span>
    <span>Chunks: ${u.chunks}</span><span>Tokens in/out: ${u.input_tokens}/${u.output_tokens}</span>
    <span>Evidence verified: ${u.raw_evidence_valid}/${u.raw_evidence_total}</span>
    <span>Unsupported items dropped: ${u.dropped_unsupported}</span>`;

  const owners = [...new Set(a.action_items.map((i) => i.owner).filter(Boolean))].sort();
  const sel = $("#f-owner");
  const prev = sel.value;
  sel.innerHTML = `<option value="">All owners</option>` + owners.map((o) => `<option>${esc(o)}</option>`).join("");
  sel.value = owners.includes(prev) ? prev : "";
  renderActions();
  renderDecisions();
  renderRisks();
  renderTranscript();
  bindJumps();
}

function evidenceHtml(ev) {
  return `<div class="evidence">${ev.map((e) => `
    <div class="ev"><a class="jump" data-turn="${esc(e.turn_id)}">${esc(e.turn_id)}</a> <b>${esc(e.speaker)}:</b> “${esc(e.quote)}”</div>`).join("")}</div>`;
}

function confHtml(c) {
  return `<div class="conf">confidence ${Math.round(c * 100)}%<div class="bar"><span style="width:${Math.round(c * 100)}%"></span></div></div>`;
}

function renderActions() {
  const { analysis: a } = state.data;
  const owner = $("#f-owner").value;
  const reviewOnly = $("#f-review").checked;
  const hideDone = $("#f-hide-done").checked;
  const items = a.action_items.filter((i) =>
    (!owner || i.owner === owner) && (!reviewOnly || i.needs_review) && (!hideDone || i.status !== "completed"));
  $("#action-list").innerHTML = items.length ? items.map((i) => {
    const done = i.status === "completed";
    let due = `<div class="due none">No deadline</div>`;
    if (i.deadline) {
      const d = daysBetween(a.meeting_date, i.deadline);
      due = `<div class="due">${esc(fmtDate(i.deadline))}</div><div class="muted small">${d === 0 ? "same day" : `+${d}d from meeting`}</div>`;
    } else if (i.deadline_text) {
      due = `<div class="due none">“${esc(i.deadline_text)}”</div>`;
    }
    return `
    <div class="item ${done ? "done" : ""} ${i.needs_review ? "review" : ""}">
      <button class="check ${done ? "on" : ""}" title="${done ? "Reopen" : "Mark complete"}" data-item="${esc(i.id)}" data-done="${done}"></button>
      <div>
        <div class="desc">${esc(i.description)}</div>
        <div class="chips">
          ${ownerHtml(i.owner)}
          <span class="chip ${esc(i.priority)}">${esc(i.priority)}</span>
          <span class="chip ${i.kind === "implicit" ? "implicit" : ""}">${esc(i.kind)}</span>
          ${i.carried_over_from ? `<span class="chip carry">carried over</span>` : ""}
          ${i.needs_review ? `<span class="chip review" title="${esc(i.review_reasons.join("; "))}">needs review</span>` : ""}
        </div>
        ${i.needs_review ? `<div class="reasons">${esc(i.review_reasons.join(" · "))}</div>` : ""}
        ${evidenceHtml(i.evidence)}
      </div>
      <div class="side">${due}${confHtml(i.confidence)}</div>
    </div>`;
  }).join("") : `<div class="muted">No action items match the filters.</div>`;
  $$(".check").forEach((b) => b.addEventListener("click", () => toggleItem(b.dataset.item, b.dataset.done === "true")));
  bindJumps();
}

function renderDecisions() {
  const { analysis: a } = state.data;
  $("#decision-list").innerHTML = a.decisions.length ? a.decisions.map((d) => `
    <div class="item" style="grid-template-columns:1fr auto">
      <div><div class="desc">${esc(d.summary)}</div>${evidenceHtml(d.evidence)}</div>
      <div class="side">${confHtml(d.confidence)}</div>
    </div>`).join("") : `<div class="muted">No decisions recorded.</div>`;
}

function renderRisks() {
  const { analysis: a } = state.data;
  $("#risk-list").innerHTML = a.risks.length ? a.risks.map((r) => `
    <div class="item" style="grid-template-columns:1fr auto">
      <div><div class="desc">${esc(r.description)}</div>
        <div class="chips"><span class="chip ${r.kind === "blocker" ? "blocker" : ""}">${esc(r.kind)}</span>
          <span class="chip ${esc(r.severity)}">${esc(r.severity)} severity</span></div>
        ${evidenceHtml(r.evidence)}</div>
      <div class="side">${confHtml(r.confidence)}</div>
    </div>`).join("") : `<div class="muted">No risks or blockers recorded.</div>`;
}

function renderTranscript() {
  const { analysis: a, transcript: t } = state.data;
  const tags = {};
  const tag = (ev, label, cls) => ev.forEach((e) => (tags[e.turn_id] ||= []).push(`<span class="chip ${cls}">${label}</span>`));
  a.action_items.forEach((i) => tag(i.evidence, "action", ""));
  a.decisions.forEach((d) => tag(d.evidence, "decision", "carry"));
  a.risks.forEach((r) => tag(r.evidence, r.kind, r.kind === "blocker" ? "blocker" : "medium"));
  ((state.data.tracking || {}).completed || []).forEach((e) => tag([{ turn_id: e.turn_id }], "closes earlier item", "done-tag"));
  const ts = (s) => s == null ? "" : `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  $("#transcript").innerHTML = t.turns.map((turn) => `
    <div class="turn ${tags[turn.id] ? "cited" : ""}" id="turn-${esc(turn.id)}">
      <div class="tid">${esc(turn.id)}</div>
      <div class="who">${esc(turn.speaker)} <div class="ts">${ts(turn.start)}</div></div>
      <div><div class="txt">${esc(turn.text)}</div>
        ${tags[turn.id] ? `<div class="tags">${[...new Set(tags[turn.id])].join("")}</div>` : ""}</div>
    </div>`).join("");
}

function bindJumps() {
  $$(".jump").forEach((a) => {
    a.onclick = () => {
      switchTab("transcript");
      const el = document.getElementById(`turn-${a.dataset.turn}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.classList.remove("flash");
        void el.offsetWidth;
        el.classList.add("flash");
      }
    };
  });
}

async function toggleItem(id, done) {
  try {
    await api(`/api/items/${encodeURIComponent(id)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: done ? "open" : "completed" }),
    });
    toast(done ? "Reopened" : "Marked complete");
    await loadMeetings(state.current);
    await loadNotifications();
  } catch (e) { toast(`Update failed: ${e.message}`); }
}

function switchTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
}

// ---------------------------------------------------------------- upload
function setupUpload() {
  const dlg = $("#upload-dialog");
  $("#open-upload").addEventListener("click", () => { $("#upload-error").hidden = true; dlg.showModal(); });
  $("#cancel-upload").addEventListener("click", () => dlg.close());
  $("#upload-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("#submit-upload");
    btn.disabled = true;
    btn.textContent = "Analyzing…";
    try {
      const fd = new FormData(ev.target);
      for (const k of ["title", "meeting_date", "series"]) if (!fd.get(k)) fd.delete(k);
      const res = await api("/api/meetings", { method: "POST", body: fd });
      dlg.close();
      ev.target.reset();
      toast(`Extracted ${res.analysis.action_items.length} action items`);
      await loadMeetings(res.meeting_id);
      await loadNotifications();
    } catch (e) {
      $("#upload-error").textContent = e.message;
      $("#upload-error").hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze";
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  ["#f-owner", "#f-review", "#f-hide-done"].forEach((s) => $(s).addEventListener("change", renderActions));
  setupUpload();
  state.asOf = new URLSearchParams(location.search).get("asof");
  try {
    const h = await api("/healthz");
    $("#model-pill").textContent = `model: ${h.model}`;
    await loadMeetings();
    await loadNotifications();
  } catch (e) { toast(`Failed to load: ${e.message}`); }
});
