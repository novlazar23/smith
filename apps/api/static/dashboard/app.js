"use strict";

const POLL_MS = 5000;

const nf = {
  price: new Intl.NumberFormat("de-DE", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  qty: new Intl.NumberFormat("de-DE", { minimumFractionDigits: 0, maximumFractionDigits: 8 }),
  int: new Intl.NumberFormat("de-DE", { maximumFractionDigits: 0 }),
  pct: new Intl.NumberFormat("de-DE", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
};

function $(sel) {
  return document.querySelector(sel);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function fmtPrice(v) {
  return isNum(v) ? nf.price.format(v) : "—";
}

function fmtQty(v) {
  return isNum(v) ? nf.qty.format(v) : "—";
}

function fmtInt(v) {
  return isNum(v) ? nf.int.format(v) : "—";
}

function fmtPct(v) {
  return isNum(v) ? nf.pct.format(v) : "—";
}

function fmtSigned(v) {
  if (!isNum(v)) return "—";
  if (v < 0) return `−${nf.price.format(Math.abs(v))}`;
  if (v > 0) return `+${nf.price.format(v)}`;
  return nf.price.format(v);
}

function pnlClass(v) {
  if (!isNum(v) || v === 0) return "flat";
  return v > 0 ? "pos" : "neg";
}

function timeStr(ts) {
  const d = ts ? new Date(ts) : null;
  return d && !Number.isNaN(d.getTime())
    ? d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";
}

function dateTimeStr(ts) {
  const d = ts ? new Date(ts) : null;
  if (!d || Number.isNaN(d.getTime())) return "—";
  const date = d.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit" });
  const time = d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
  return `${date} ${time}`;
}

function uptimeStr(sec) {
  if (!isNum(sec)) return "—";
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h} h ${String(m).padStart(2, "0")} m`;
  if (m > 0) return `${m} m ${String(r).padStart(2, "0")} s`;
  return `${r} s`;
}

function emptyState(text) {
  return `<div class="empty">` +
    `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M12 7.5V12l3 2.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>` +
    `<span>${esc(text)}</span></div>`;
}

function tri(dir) {
  if (dir === "up") return `<svg class="tri" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 1.2 7.3 6.8H.7Z"/></svg>`;
  if (dir === "down") return `<svg class="tri" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 6.8.7 1.2h6.6Z"/></svg>`;
  return "";
}

function changeDir(pct) {
  if (!isNum(pct)) return "flat";
  return pct > 0 ? "up" : pct < 0 ? "down" : "flat";
}

function dirClass(dir) {
  return dir === "up" ? "ch-up" : dir === "down" ? "ch-down" : "ch-flat";
}

function sparkline(candles) {
  const W = 100;
  const H = 40;
  const PAD = 4;
  const pts = [];
  candles.forEach((c, i) => {
    if (c && isNum(c.c)) {
      pts.push({ x: candles.length > 1 ? (i / (candles.length - 1)) * W : 0, v: c.c });
    }
  });
  if (pts.length < 2) return { svg: "", dot: null };
  let min = Infinity;
  let max = -Infinity;
  for (const p of pts) {
    if (p.v < min) min = p.v;
    if (p.v > max) max = p.v;
  }
  const span = max - min || Math.abs(max) || 1;
  const mapped = pts.map((p) => ({ x: p.x, y: H - PAD - ((p.v - min) / span) * (H - 2 * PAD) }));
  const line = mapped.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
  const first = mapped[0];
  const last = mapped[mapped.length - 1];
  const area = `${line} L${last.x.toFixed(2)} ${H} L${first.x.toFixed(2)} ${H} Z`;
  return {
    svg: `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">` +
      `<path class="spark-area" d="${area}"/><path class="spark-line" d="${line}"/></svg>`,
    dot: { left: (last.x / W) * 100, top: (last.y / H) * 100 },
  };
}

function flashClass(current, previous) {
  if (!isNum(current) || !isNum(previous)) return "";
  if (current > previous) return "flash-up";
  if (current < previous) return "flash-down";
  return "";
}

/* ── Topbar ─────────────────────────────────────────────────────────────── */

function renderTopbar(sys) {
  const status = String((sys && sys.status) || "degraded");
  const pill = $("#sys-pill");
  pill.classList.toggle("ok", status === "running");
  pill.classList.toggle("warn", status === "degraded");
  pill.classList.toggle("err", status !== "running" && status !== "degraded");
  $("#sys-status").textContent =
    status === "running" ? "Running" : status === "degraded" ? "Degraded" : "Fehler";

  const src = String((sys && sys.data_source) || "synthetic");
  const badge = $("#badge-src");
  badge.textContent = src === "live" ? "LIVE" : "SYNTHETIC";
  badge.classList.toggle("src-live", src === "live");
  badge.classList.toggle("src-syn", src !== "live");

  $("#uptime").textContent = `Uptime ${uptimeStr(sys ? sys.uptime_seconds : null)}`;
  if (sys && isNum(sys.candles_total)) {
    $("#market-meta").textContent = `${nf.int.format(sys.candles_total)} Kerzen gesamt`;
  }
}

/* ── Markt ──────────────────────────────────────────────────────────────── */

const prevPrices = new Map();

function renderMarket(market) {
  const wrap = $("#market-cards");
  const items = market.filter((m) => m && typeof m.instrument === "string");
  if (items.length === 0 || items.every((m) => !isNum(m.last_price))) {
    wrap.innerHTML = emptyState("Noch keine Marktdaten");
    return;
  }
  wrap.innerHTML = items.map((m) => marketCard(m, prevPrices.get(m.instrument))).join("");
  for (const m of items) {
    if (isNum(m.last_price)) prevPrices.set(m.instrument, m.last_price);
  }
}

function marketCard(m, previous) {
  const candles = Array.isArray(m.candles) ? m.candles : [];
  const pct = isNum(m.change_pct) ? m.change_pct : null;
  const dir = changeDir(pct);
  const spark = sparkline(candles);
  const flash = flashClass(m.last_price, previous);
  const sparkHtml = spark.svg
    ? `<div class="spark-wrap ${dir}">${spark.svg}` +
      `<span class="spark-dot" style="left:${spark.dot.left.toFixed(2)}%;top:${spark.dot.top.toFixed(2)}%"></span></div>`
    : `<span class="mc-nodata">keine Kerzendaten</span>`;
  return `<article class="market-card">` +
    `<div class="mc-head"><span class="mc-instr">${esc(m.instrument)}</span>` +
    `<span class="mc-meta">${candles.length} Kerzen</span></div>` +
    `<div class="mc-price-row"><span class="mc-price ${flash}">${fmtPrice(m.last_price)}</span>` +
    `<span class="chip ${dirClass(dir)}">${tri(dir)}${fmtPct(pct)}&thinsp;%</span></div>` +
    `<div class="mc-spark">${sparkHtml}</div>` +
    `</article>`;
}

/* ── Konto ──────────────────────────────────────────────────────────────── */

function renderAccount(acct) {
  const body = $("#account-body");
  if (!acct || typeof acct !== "object") {
    body.innerHTML = emptyState("Noch keine Kontodaten");
    return;
  }
  const pnl = isNum(acct.total_pnl) ? acct.total_pnl : null;
  const pct = isNum(acct.pnl_pct) ? acct.pnl_pct : null;
  const pnlText = pct == null ? "" : ` (${fmtSigned(pct)} %)`;
  body.innerHTML =
    `<div class="acct-demo">Demo-Modus — imaginäres Geld</div>` +
    `<div class="acct-equity"><span class="label">Equity</span>` +
    `<span class="value big">${fmtPrice(acct.equity)}</span></div>` +
    `<dl class="acct-rows">` +
    `<div><dt>Barstand</dt><dd>${fmtPrice(acct.cash)}</dd></div>` +
    `<div><dt>Startkapital</dt><dd>${fmtPrice(acct.initial_cash)}</dd></div>` +
    `<div><dt>Gesamt-P&amp;L</dt><dd class="${pnlClass(pnl)}">${fmtSigned(pnl)}${pnlText}</dd></div>` +
    `<div><dt>Trades</dt><dd>${fmtInt(acct.total_trades)}</dd></div>` +
    `<div><dt>Kommission</dt><dd>${fmtPrice(acct.total_commission)}</dd></div>` +
    `</dl>` +
    `<div class="acct-updated">Aktualisiert ${timeStr(acct.updated_at)}</div>`;
}

/* ── Positionen ─────────────────────────────────────────────────────────── */

function renderPositions(positions) {
  const body = $("#positions-body");
  const meta = $("#positions-meta");
  if (positions.length === 0) {
    meta.textContent = "";
    body.innerHTML = emptyState("Keine offenen Positionen");
    return;
  }
  meta.textContent = `${positions.length} offen`;
  const rows = positions.map((p) =>
    `<tr><td class="td-instr">${esc(p.instrument)}</td>` +
    `<td class="num">${fmtQty(p.quantity)}</td>` +
    `<td class="num">${fmtPrice(p.avg_price)}</td>` +
    `<td class="num">${fmtPrice(p.market_price)}</td>` +
    `<td class="num ${pnlClass(p.unrealized_pnl)}">${fmtSigned(p.unrealized_pnl)}</td>` +
    `<td class="td-date">${dateTimeStr(p.opened_at)}</td></tr>`
  ).join("");
  body.innerHTML =
    `<div class="table-wrap"><table class="pos-table">` +
    `<thead><tr><th>Instrument</th><th class="num">Menge</th><th class="num">Ø-Preis</th>` +
    `<th class="num">Markt</th><th class="num">u. P&amp;L</th><th>Offen seit</th></tr></thead>` +
    `<tbody>${rows}</tbody></table></div>`;
}

/* ── Trades ─────────────────────────────────────────────────────────────── */

function renderTrades(trades) {
  const body = $("#trades-body");
  if (trades.length === 0) {
    body.innerHTML = emptyState("Noch keine Trades");
    return;
  }
  body.innerHTML = trades.map((t) => {
    const direction = String(t.direction || "").toUpperCase();
    const side = direction === "BUY" ? "buy" : direction === "SELL" ? "sell" : "flat";
    return `<div class="trade-row">` +
      `<span class="tr-time">${timeStr(t.ts)}</span>` +
      `<span class="tr-instr">${esc(t.instrument)}</span>` +
      `<span class="chip side-${side}">${esc(direction || "—")}</span>` +
      `<span class="tr-qty">${fmtQty(t.quantity)}</span>` +
      `<span class="tr-price">${fmtPrice(t.price)}</span>` +
      `<span class="tr-comm">${fmtPrice(t.commission)}</span>` +
      `<span class="tr-status">${esc(t.status || "")}</span>` +
      `</div>`;
  }).join("");
}

/* ── Entscheidungen ─────────────────────────────────────────────────────── */

const DECISION_CLASS = { NO_TRADE: "d-neutral", LONG_BIAS: "d-long", SHORT_BIAS: "d-short" };

function renderDecisions(decisions) {
  const body = $("#decisions-body");
  if (decisions.length === 0) {
    body.innerHTML = emptyState("Noch keine Entscheidungen");
    return;
  }
  body.innerHTML = decisions.map((d) => {
    const decision = String(d.decision || "UNKNOWN").toUpperCase();
    const cls = DECISION_CLASS[decision] || "d-other";
    const conf = isNum(d.confidence) ? Math.min(1, Math.max(0, d.confidence)) : null;
    const width = conf == null ? "0" : (conf * 100).toFixed(1);
    const confText = conf == null ? "—" : `${fmtPct(conf * 100)} %`;
    return `<div class="dec-item">` +
      `<div class="dec-head"><span class="dec-time">${timeStr(d.ts)}</span>` +
      `<span class="dec-instr">${esc(d.instrument)}</span>` +
      `<span class="chip ${cls}">${esc(decision.replaceAll("_", " "))}</span></div>` +
      `<div class="dec-reason" title="${esc(d.reason)}">${esc(d.reason || "—")}</div>` +
      `<div class="dec-conf"><span class="dec-conf-bar">` +
      `<span class="dec-conf-fill ${cls}" style="width:${width}%"></span></span>` +
      `<span class="dec-conf-val">${confText}</span></div>` +
      `</div>`;
  }).join("");
}

/* ── News ───────────────────────────────────────────────────────────────── */

function categoryClass(cat) {
  switch (String(cat || "").toUpperCase()) {
    case "INITIAL": return "initial";
    case "UPDATE": return "update";
    case "CONFIRMATION": return "confirm";
    case "RUMOR": return "rumor";
    case "RETRACTION": return "retract";
    default: return "other";
  }
}

function renderNews(news) {
  const body = $("#news-body");
  if (news.length === 0) {
    body.innerHTML = emptyState("Noch keine News");
    return;
  }
  body.innerHTML = news.map((n) =>
    `<div class="news-item">` +
    `<div class="news-head"><span class="news-time">${timeStr(n.ts)}</span>` +
    `<span class="news-source">${esc(n.source)}</span>` +
    `<span class="chip cat-${categoryClass(n.category)}">${esc(n.category || "—")}</span></div>` +
    `<div class="news-title" title="${esc(n.title)}">${esc(n.title || "—")}</div>` +
    `</div>`
  ).join("");
}

/* ── Evolved Agents ─────────────────────────────────────────────────────── */

function renderEvolved(agents, lastRun) {
  const body = $("#evolved-body");
  const items = (agents || []).filter((a) => a && typeof a.name === "string");
  let html = items.map((a) =>
    `<div class="ev-item">` +
    `<div class="ev-head"><span class="ev-name">${esc(a.name)}</span>` +
    `<span class="chip d-neutral">SHADOW</span>` +
    `<span class="ev-ver">v${esc(a.version ?? 1)}</span></div>` +
    `<div class="ev-claim" title="${esc(a.claim)}">${esc(a.claim || "—")}</div>` +
    `<div class="ev-meta">OOS ${isNum(a.score) ? a.score.toFixed(4) : "—"} · seit ${dateTimeStr(a.admitted_at)}</div>` +
    `</div>`
  ).join("");
  const verdicts = lastRun && Array.isArray(lastRun.verdicts) ? lastRun.verdicts : [];
  if (verdicts.length > 0) {
    html +=
      `<div class="evl-head">Letzter Lauf${lastRun.run_at ? " · " + dateTimeStr(lastRun.run_at) : ""}</div>` +
      verdicts.map((v) =>
        `<div class="evl-item">` +
        `<span class="evl-name" title="${esc(v.name)}">${esc(v.name)}</span>` +
        `<span class="chip ${v.admitted ? "d-long" : "d-short"}">${v.admitted ? "ZUGELASSEN" : "ABGELEHNT"}</span>` +
        `<span class="evl-score">OOS ${isNum(v.score) ? v.score.toFixed(4) : "—"}</span>` +
        `</div>` +
        ((v.reasons || []).length ? `<div class="evl-reason">${esc(v.reasons.join("; "))}</div>` : "")
      ).join("");
  }
  body.innerHTML = html || emptyState("Noch keine zugelassenen Agenten");
}

/* ── Evolution (Stufe 2) ─────────────────────────────────────────────────── */

const PROMOTION_DAYS = 14; // Promotions-Schwelle: 14 durchgehende tägliche Re-Checks

function evoDate(ts) {
  const d = ts ? new Date(ts) : null;
  return d && !Number.isNaN(d.getTime())
    ? d.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric" })
    : "—";
}

function evoTruncate(str, max) {
  const s = String(str || "");
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

function renderEvolution(data) {
  const body = $("#evolution-body");
  const shadow = data && typeof data.evolved_agents_shadow === "object" && data.evolved_agents_shadow !== null
    ? data.evolved_agents_shadow
    : {};
  const lastRun = data && typeof data.evolved_agents_last_run === "object" && data.evolved_agents_last_run !== null
    ? data.evolved_agents_last_run
    : null;
  const shadowMeta = lastRun && typeof lastRun.shadow === "object" && lastRun.shadow !== null
    ? lastRun.shadow
    : {};
  const archive = data && Array.isArray(data.evolved_agents_archive) ? data.evolved_agents_archive : [];

  const names = Object.keys(shadow).filter((n) => shadow[n] && typeof shadow[n] === "object");
  let html = `<div class="evl-head">Shadow-Status</div>`;
  html += names.length === 0
    ? emptyState("Noch keine zugelassenen Agenten im Shadow-Tracking")
    : names.map((name) => {
        const s = shadow[name];
        const hasPasses = isNum(s.consecutive_passes);
        const passes = hasPasses ? Math.max(0, Math.floor(s.consecutive_passes)) : null;
        // ready: Wert aus der Shadow-Datei; Fallback auf last_run.shadow[name].ready
        const meta = shadowMeta[name];
        const ready = hasPasses ? passes >= PROMOTION_DAYS : !!(meta && meta.ready === true);
        const hist = Array.isArray(s.score_history) ? s.score_history.filter(isNum) : [];
        const spread = hist.length < 2
          ? "–"
          : `${Math.min(...hist).toFixed(2)}–${Math.max(...hist).toFixed(2)}`;
        return `<div class="ev-item">` +
          `<div class="ev-head"><span class="ev-name" title="${esc(name)}">${esc(name)}</span>` +
          (ready ? `<span class="chip d-long">BEREIT</span>` : "") +
          `<span class="ev-ver">${passes == null ? "—" : passes}/${PROMOTION_DAYS} Tage</span></div>` +
          `<div class="ev-meta">Score ${spread} · geprüft ${dateTimeStr(s.last_checked)}</div>` +
          `</div>`;
      }).join("");

  const rejected = archive.filter((c) => c && typeof c === "object").slice(-15);
  html += `<div class="evl-head">Letzte Ablehnungen${archive.length > 15 ? " · letzte 15" : ""}</div>`;
  html += rejected.length === 0
    ? emptyState("Noch keine abgelehnten Kandidaten archiviert")
    : rejected.map((c) => {
        const persona = typeof c.persona === "string" && c.persona.trim() !== "" ? c.persona : null;
        const reason = Array.isArray(c.reasons)
          ? String(c.reasons.find((r) => typeof r === "string" && r.trim() !== "") || "")
          : "";
        return `<div class="evl-item">` +
          `<span class="evl-name" title="${esc(c.name)}">${esc(c.name || "—")}</span>` +
          (persona ? `<span class="chip p-persona">${esc(persona)}</span>` : "") +
          `<span class="evl-score">${isNum(c.score) ? c.score.toFixed(2) : "—"}</span>` +
          `<span class="evo-date">${evoDate(c.run_at)}</span>` +
          `</div>` +
          (reason ? `<div class="evl-reason" title="${esc(reason)}">${esc(evoTruncate(reason, 60))}</div>` : "");
      }).join("");

  body.innerHTML = html;
}

/* ── Overfitting-Validierung (CPCV/PBO/DSR) ──────────────────────────────── */

const DSR_MIN = 0.95; // README-Heuristik: DSR > 0,95 übersteht die Trial-Deflation
const PBO_MAX = 0.5; // README-Heuristik: PBO < 0,5 = Overfitting dominiert nicht

function renderCpcv(data) {
  const body = $("#cpcv-body");
  const meta = $("#cpcv-meta");
  const raw = data && typeof data.cpcv_validation === "object" && data.cpcv_validation !== null
    ? data.cpcv_validation
    : null;
  if (!raw || Object.keys(raw).length === 0) {
    meta.textContent = "CPCV · PBO · DSR";
    body.innerHTML = emptyState("Noch kein CPCV-Lauf (backtest --cpcv)");
    return;
  }
  const v = raw;
  const params = v.params && typeof v.params === "object"
    ? Object.entries(v.params).map(([k, val]) => `${k}=${val}`).join(", ")
    : "";
  meta.textContent = v.strategy || "CPCV · PBO · DSR";
  const windowText = v.window_start && v.window_end
    ? `${evoDate(v.window_start)} → ${evoDate(v.window_end)}`
    : "—";
  const folds = isNum(v.n_folds)
    ? (isNum(v.expected_folds) ? `${v.n_folds}/${v.expected_folds}` : String(v.n_folds))
    : "—";
  const pbo = isNum(v.pbo) ? v.pbo : null;
  const dsr = isNum(v.dsr) ? v.dsr : null;
  const pboChip = pbo == null ? "" : `<span class="chip ${pbo < PBO_MAX ? "ch-up" : "ch-down"}">${pbo < PBO_MAX ? "OK" : "NG"}</span>`;
  const dsrChip = dsr == null ? "" : `<span class="chip ${dsr > DSR_MIN ? "ch-up" : "ch-down"}">${dsr > DSR_MIN ? "OK" : "NG"}</span>`;
  const ch = v.champion && typeof v.champion === "object" ? v.champion : {};
  body.innerHTML =
    `<dl class="acct-rows">` +
    `<div><dt>Strategie</dt><dd>${esc(v.strategy || "—")}</dd></div>` +
    (params ? `<div><dt>Parameter</dt><dd>${esc(params)}</dd></div>` : "") +
    `<div><dt>Instrument</dt><dd>${esc(v.instrument || "—")} · ${esc(v.timeframe || "—")}</dd></div>` +
    `<div><dt>Fenster</dt><dd>${esc(windowText)}</dd></div>` +
    `<div><dt>Kerzen</dt><dd>${isNum(v.n_candles) ? fmtInt(v.n_candles) : "—"}</dd></div>` +
    `<div><dt>Folds (CPCV)</dt><dd>${esc(folds)}${isNum(v.n_skipped_folds) && v.n_skipped_folds > 0 ? ` · ${v.n_skipped_folds} übersprungen` : ""}</dd></div>` +
    `<div><dt>Trials</dt><dd>${isNum(v.n_trials) ? fmtInt(v.n_trials) : "—"}${isNum(v.n_zoo) ? ` · Zoo ${v.n_zoo}` : ""}</dd></div>` +
    `<div><dt>PBO (max. 0,50)</dt><dd>${pbo == null ? "—" : `${pbo.toFixed(2)} ${pboChip}`}</dd></div>` +
    `<div><dt>DSR (min. 0,95)</dt><dd>${dsr == null ? "—" : `${dsr.toFixed(2)} ${dsrChip}`}</dd></div>` +
    `<div><dt>Champion-Sharpe</dt><dd class="${pnlClass(ch.sharpe_ratio)}">${isNum(ch.sharpe_ratio) ? ch.sharpe_ratio.toFixed(2) : "—"}</dd></div>` +
    `<div><dt>Champion-Return</dt><dd class="${pnlClass(ch.total_return_pct)}">${isNum(ch.total_return_pct) ? `${fmtSigned(ch.total_return_pct)} %` : "—"}</dd></div>` +
    `<div><dt>Champion-Trades</dt><dd>${isNum(ch.total_trades) ? fmtInt(ch.total_trades) : "—"}</dd></div>` +
    `</dl>`;
}

/* ── Tabs / Embeds ───────────────────────────────────────────────────────── */

const TAB_DEFAULT = "trading";

const EMBED_SRC = {
  monitoring: "/proxy/grafana/d/trading-orchestra?kiosk&from=now-1h&to=now",
  metriken: "/proxy/prometheus/graph?h=1",
  alerts: "/proxy/alertmanager/",
  ml: "/proxy/mlflow/",
  storage: "/proxy/minio/",
};

let activeTab = TAB_DEFAULT;
const embedFrames = new Map();

function tabFromHash() {
  const raw = String(window.location.hash || "").replace(/^#\/?/, "");
  if (raw === TAB_DEFAULT) return TAB_DEFAULT;
  if (Object.prototype.hasOwnProperty.call(EMBED_SRC, raw)) return raw;
  return TAB_DEFAULT;
}

function writeHash(tabId, push) {
  const target = `#/${tabId}`;
  if (window.location.hash === target) return;
  try {
    if (push) history.pushState(null, "", target);
    else history.replaceState(null, "", target);
  } catch (err) {
    window.location.hash = target; // Fallback: feuert hashchange, activateTab ist idempotent
  }
}

function ensureEmbed(tabId) {
  if (embedFrames.has(tabId)) return;
  const frame = document.createElement("iframe");
  frame.className = "embed-frame";
  const label = document.getElementById(tabId);
  frame.title = label && label.textContent ? label.textContent.trim() : tabId;
  frame.src = EMBED_SRC[tabId];
  $("#embed-wrap").appendChild(frame);
  embedFrames.set(tabId, frame);
}

function activateTab(tabId) {
  activeTab = tabId;
  const isTrading = tabId === TAB_DEFAULT;

  document.querySelectorAll("#tabbar .tab").forEach((btn) => {
    const selected = btn.id === tabId;
    btn.classList.toggle("active", selected);
    if (selected) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });

  if (!isTrading) ensureEmbed(tabId);

  $("main.grid").hidden = !isTrading;
  $("#embed-wrap").hidden = isTrading;

  for (const [id, frame] of embedFrames) {
    frame.classList.toggle("is-active", !isTrading && id === tabId);
  }

  // Hash-Invariante: URL spiegelt immer den aktiven Tab (kein eigener History-Eintrag)
  writeHash(tabId, false);
}

function onTabClick(evt) {
  const btn = evt.target && typeof evt.target.closest === "function"
    ? evt.target.closest(".tab")
    : null;
  if (!btn || !btn.id || btn.id === activeTab) return;
  writeHash(btn.id, true);
  activateTab(btn.id);
}

function initTabs() {
  const bar = $("#tabbar");
  if (bar) bar.addEventListener("click", onTabClick);
  window.onhashchange = () => activateTab(tabFromHash());
  activateTab(tabFromHash());
}

/* ── Polling ────────────────────────────────────────────────────────────── */

function render(data) {
  renderTopbar(data.system || {});
  renderMarket(Array.isArray(data.market) ? data.market : []);
  renderAccount(data.demo_account);
  renderPositions(Array.isArray(data.positions) ? data.positions : []);
  renderTrades(Array.isArray(data.recent_trades) ? data.recent_trades : []);
  renderDecisions(Array.isArray(data.recent_decisions) ? data.recent_decisions : []);
  renderNews(Array.isArray(data.recent_news) ? data.recent_news : []);
  renderEvolved(Array.isArray(data.evolved_agents) ? data.evolved_agents : [], data.evolved_agents_last_run);
  renderEvolution(data);
  renderCpcv(data);
}

function setConnected(ok) {
  $("#conn-banner").hidden = ok;
  document.body.classList.toggle("offline", !ok);
}

function updateClock() {
  $("#clock").textContent = new Date().toLocaleTimeString("de-DE", { hour12: false });
}

async function poll() {
  try {
    const res = await fetch("/v1/dashboard", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data || typeof data !== "object") throw new Error("Ungültige Antwort");
    render(data);
    setConnected(true);
  } catch (err) {
    console.warn("[dashboard] Poll fehlgeschlagen:", err);
    setConnected(false);
  } finally {
    updateClock();
  }
}

function init() {
  initTabs();
  updateClock();
  setInterval(updateClock, 1000);
  poll();
  setInterval(poll, POLL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
