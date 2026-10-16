// LastCall demo: legge data.json (generato da churn/export_demo.py).
// Le decisioni del negozio (approva / modifica / rifiuta) restano nel browser (localStorage) e si esportano in CSV.
(async function () {
  "use strict";

  const app = document.getElementById("app");
  let DATA = null;
  try {
    const r = await fetch("data.json", { cache: "no-store" });
    if (r.ok) DATA = await r.json();
  } catch (e) { /* gestito sotto */ }
  const STORE_KEY = "lastcall-decisions-v1";
  const OFFER_LABEL = { discount: "Discount", free_shipping: "Free shipping", loyalty_points: "Loyalty points", similar_products: "Similar products" };
  const REASON_LABEL = { price: "Price", lost_interest: "Lost interest", seasonal: "Seasonal", bad_experience: "Bad experience", other: "Other" };
  const STATUS = {
    review: { label: "To review", color: "var(--accent)" },
    approved: { label: "Approved", color: "var(--good)" },
    edited: { label: "Edited", color: "var(--violet)" },
    rejected: { label: "Rejected", color: "#A1A1AA" },
  };

  const ui = { offer: "all", status: "review", query: "", selected: new Set() };
  // Pagina Customers: filtro principale (attivi / a rischio) e sottofiltri del rischio
  const CUI_DEFAULT = { main: "active", level: "all", delay: "any", aov: "any", orders: "any", cancel: "any", spend: "any", season: "any", q: "", sort: "risk", limit: 60 };
  const cui = { ...CUI_DEFAULT };
  const AT_RISK = 50; // come nella selezione delle offerte: rischio sopra 50

  // ---------- utilità ----------
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-GB"));
  const gbp = (n) => (n == null ? "–" : "£" + Math.round(n).toLocaleString("en-GB"));
  const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
  const offerText = (c) => (c.offer === "discount" ? `${c.discount_pct}% discount` : OFFER_LABEL[c.offer] || "–");

  function loadDecisions() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { return {}; }
  }
  let decisions = loadDecisions();
  function saveDecisions() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(decisions)); } catch (e) { /* solo in memoria */ }
  }
  const statusOf = (c) => (decisions[c.id] && decisions[c.id].status) || "review";
  const messageOf = (c) => (decisions[c.id] && decisions[c.id].message) || c.message || "";

  function decide(id, status, message) {
    decisions[id] = { status, message: message ?? (decisions[id] && decisions[id].message), at: new Date().toISOString() };
    saveDecisions();
  }

  // Offerte generate dalla pagina Customers (con il server locale): salvate nel browser e aggiunte alla lista
  const GEN_KEY = "lastcall-generated-v1";
  let generated = {};
  try { generated = JSON.parse(localStorage.getItem(GEN_KEY) || "{}"); } catch (e) { generated = {}; }
  function saveGenerated() {
    try { localStorage.setItem(GEN_KEY, JSON.stringify(generated)); } catch (e) { /* solo in memoria */ }
  }
  if (DATA) DATA.customers = DATA.customers.concat(Object.values(generated).filter((g) => !DATA.customers.some((c) => c.id === g.id)));
  const hasOffer = (id) => DATA.customers.some((c) => c.id === id);

  function toast(text) {
    const t = document.createElement("div");
    t.className = "toast";
    t.setAttribute("role", "status");
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2200);
  }

  function counts() {
    const out = { review: 0, approved: 0, edited: 0, rejected: 0 };
    DATA.customers.forEach((c) => out[statusOf(c)]++);
    return out;
  }

  function statusPill(c) {
    const s = STATUS[statusOf(c)];
    return `<span class="row" style="gap:6px"><span class="dot" style="background:${s.color}" aria-hidden="true"></span>${s.label}</span>`;
  }

  // Spiegazioni al passaggio del mouse (e al focus da tastiera): vedi tooltip in fondo al file
  const tip = (text) => `data-tip="${esc(text)}" tabindex="0"`;
  const RISK_TIP = "Risk score from 0 to 100: how likely this customer is to stop buying, i.e. not order again within about twice their usual time between orders. The higher, the more likely.";
  function bandTip(b) {
    const what = { Low: "unlikely to stop buying", Medium: "may stop buying", High: "very likely to stop buying" }[b.name];
    return `${b.name} risk (score ${b.range}): ${fmt(b.customers)} customers ${what}. In the following months ${pct(b.left, b.customers)}% of them really stopped.`;
  }

  function riskChip(r) {
    const level = r >= 70 ? "high" : r >= 40 ? "medium" : "low";
    return `<span class="chip ${r >= 70 ? "blue" : "grey"} num" ${tip(`Risk ${r}/100 (${level}). ${RISK_TIP}`)}>${r}</span>`;
  }

  // ---------- pagine ----------
  function overview() {
    const bands = DATA.bands;
    const maxBand = Math.max(...bands.map((b) => b.customers));
    const mix = {};
    DATA.customers.forEach((c) => { mix[c.offer] = (mix[c.offer] || 0) + 1; });
    const n = counts();
    const top = DATA.customers.slice(0, 6);
    const discounts = DATA.customers.filter((c) => c.offer === "discount").map((c) => c.discount_pct);
    const avgDiscount = discounts.length ? discounts.reduce((a, b) => a + b, 0) / discounts.length : null;

    return `
      <div>
        <h1>Good morning, shop owner</h1>
      </div>

      <section class="grid-12">
        <div class="card span-4 stack">
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
            <div ${tip(`Everyone who bought at least once before ${DATA.reference_date}.`)}>
              <div class="label">Total customers</div>
              <div class="big num">${fmt(DATA.total_customers)}</div>
            </div>
            <div ${tip("Customers who bought at least once in the 12 months before the reference date. These are the ones we can still win back.")}>
              <div class="label">Active customers</div>
              <div class="big num">${fmt(DATA.active_customers)}</div>
            </div>
          </div>
          <div class="row">
            <a class="btn dark" style="flex:1" href="#/offers">Review offers</a>
            <a class="btn light" style="flex:1" href="#/how">How it works</a>
          </div>
          <div class="soft" style="padding:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px">
            ${bands.map((b) => `
              <div style="background:#fff;border-radius:12px;padding:10px" ${tip(bandTip(b))}>
                <div class="sub">${esc(b.name)} risk</div>
                <div class="num" style="font-weight:700;margin-top:4px">${fmt(b.customers)}</div>
                <span class="chip ${b.name === "High" ? "blue" : "grey"}" style="margin-top:4px">${pct(b.left, b.customers)}% left</span>
              </div>`).join("")}
          </div>
        </div>

        <div class="card span-4" style="padding:14px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
          <div class="tile accent" ${tip("Active customers likely to stop buying (risk above 50) who are also worth keeping: they order regularly or have spent more than £500.")}>
            <span class="label">Worth saving</span>
            <div><div class="big num">${fmt(DATA.worth_saving)}</div><span class="chip on-accent">at risk · valuable</span></div>
          </div>
          <div class="soft tile" ${tip("Suggested offers still waiting for your decision.")}>
            <span class="label">Offers to review</span>
            <div><div class="big num">${fmt(n.review)}</div><span class="chip blue">of ${fmt(DATA.customers.length)} suggested</span></div>
          </div>
          <div class="soft tile" ${tip("Offers you approved, as they are or after editing the message.")}>
            <span class="label">Approved</span>
            <div><div class="big num">${fmt(n.approved + n.edited)}</div><span class="chip good">ready to send</span></div>
          </div>
          <div class="soft tile" ${tip("Average size of the suggested discounts.")}>
            <span class="label">Average discount</span>
            <div><div class="big num">${avgDiscount == null ? "–" : avgDiscount.toFixed(1) + "%"}</div><span class="chip violet">${fmt(discounts.length)} discount offers</span></div>
          </div>
        </div>

        <div class="card span-4 stack">
          <div><h2>Who really left</h2><div class="sub">Customers per risk level: who stopped buying in the following months</div></div>
          <div class="soft stack" style="padding:14px;flex:1">
            <div class="legend"><span><i style="background:var(--accent)"></i>Left</span><span><i style="background:var(--black)"></i>Stayed</span></div>
            <div class="bars" role="img" aria-label="${bands.map((b) => `${b.name}: ${b.left} left, ${b.stayed} stayed`).join("; ")}">
              ${bands.map((b) => `
                <div class="bar-col">
                  <div class="bar-left" style="height:${(100 * b.left) / maxBand}%" ${tip(`${b.name} risk: ${fmt(b.left)} customers stopped buying (${pct(b.left, b.customers)}% of ${fmt(b.customers)}).`)}></div>
                  <div class="bar-stay" style="height:${(100 * b.stayed) / maxBand}%" ${tip(`${b.name} risk: ${fmt(b.stayed)} customers kept buying (${pct(b.stayed, b.customers)}% of ${fmt(b.customers)}).`)}></div>
                </div>`).join("")}
            </div>
            <div class="row" style="gap:22px;padding:0 8px;font-size:12px;color:var(--muted)">
              ${bands.map((b) => `<span style="flex:1;text-align:center">${esc(b.name)}</span>`).join("")}
            </div>
          </div>
        </div>
      </section>

      <section class="card stack">
        <div class="row spread"><h2>Offer mix</h2><span class="sub">${fmt(DATA.customers.length)} suggested offers · click a type to see them</span></div>
        <div class="row">
          ${Object.keys(OFFER_LABEL).map((k) => `<button type="button" class="btn light" data-mix="${k}" ${mix[k] ? "" : "disabled"} aria-label="Show ${OFFER_LABEL[k]} offers (${mix[k] || 0})">${OFFER_LABEL[k]} <strong class="num">${fmt(mix[k] || 0)}</strong> <span aria-hidden="true">→</span></button>`).join("")}
        </div>
      </section>

      <section class="card stack">
        <div class="row spread"><h2>Top priority customers</h2><a class="btn outline" href="#/offers">See all</a></div>
        <div class="table-box">
          <table>
            <thead><tr><th>Customer</th><th>Why</th><th>Net spend</th><th><span ${tip(RISK_TIP)}>Risk ⓘ</span></th><th>Offer</th><th>Status</th><th>Silent for</th></tr></thead>
            <tbody>
              ${top.map((c) => `
                <tr>
                  <td><button type="button" class="link num" data-open="${esc(c.id)}">#${esc(c.id)}</button></td>
                  <td>${esc(REASON_LABEL[c.reason] || c.reason)}</td>
                  <td class="num" style="font-weight:600">${gbp(c.net_spend)}</td>
                  <td>${riskChip(c.risk)}</td>
                  <td>${esc(offerText(c))}</td>
                  <td>${statusPill(c)}</td>
                  <td style="color:var(--ink-2)">${fmt(c.days_silent)} days${c.usual_gap ? ` · ${c.delay_ratio}× usual` : ""}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </section>`;
  }

  function filtered() {
    const q = ui.query.trim().toLowerCase();
    return DATA.customers.filter((c) =>
      (ui.offer === "all" || c.offer === ui.offer) &&
      (ui.status === "all" || statusOf(c) === ui.status) &&
      (!q || c.id.includes(q) || (c.favourite || "").toLowerCase().includes(q)));
  }

  function offers() {
    const n = counts();
    const list = filtered();
    const statusCard = (key, extra) => {
      const active = ui.status === key;
      return `<button type="button" class="${active ? "tile accent" : "card tile"}" data-status="${key}" aria-pressed="${active}" style="border:0;text-align:left;cursor:pointer;${active ? "" : "border-radius:22px"}">
        <span class="label">${STATUS[key].label}</span>
        <span class="big num">${fmt(n[key])}</span>
        <span class="chip ${active ? "on-accent" : extra}">${{ review: "waiting for you", approved: "ready to send", edited: "changed by you", rejected: "no offer" }[key]}</span>
      </button>`;
    };
    return `
      <div class="head-row">
        <div>
          <h1>Offers to review</h1>
          <p class="lead">A personal offer for each customer at risk. Nothing is sent until you approve it.</p>
        </div>
        <div class="row">
          <button type="button" class="btn white lg" data-action="export">Export CSV</button>
          <button type="button" class="btn dark lg" data-action="approve-selected" ${ui.selected.size ? "" : "disabled"}>Approve selected${ui.selected.size ? ` (${ui.selected.size})` : ""}</button>
        </div>
      </div>

      <section class="grid-12" aria-label="Status">
        <div class="span-3">${statusCard("review", "blue")}</div>
        <div class="span-3">${statusCard("approved", "good")}</div>
        <div class="span-3">${statusCard("edited", "violet")}</div>
        <div class="span-3">${statusCard("rejected", "grey")}</div>
      </section>

      <section class="card stack">
        <div class="row spread">
          <div class="row" style="gap:6px" role="group" aria-label="Offer type">
            ${[["all", "All"], ...Object.entries(OFFER_LABEL)].map(([k, l]) =>
              `<button type="button" class="btn ${ui.offer === k ? "dark" : "light"}" data-offer="${k}" aria-pressed="${ui.offer === k}">${l}</button>`).join("")}
          </div>
          <div class="row">
            <label class="search">
              <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
              <span class="sr">Search customers</span>
              <input type="search" id="q" placeholder="Customer or product" value="${esc(ui.query)}">
            </label>
            <button type="button" class="btn outline" data-status="all" aria-pressed="${ui.status === "all"}">Show all statuses</button>
          </div>
        </div>
        <div class="table-box">
          <table>
            <thead><tr>
              <th style="width:24px"><input type="checkbox" id="all" aria-label="Select all shown" ${list.length && list.every((c) => ui.selected.has(c.id)) ? "checked" : ""}></th>
              <th>Customer</th><th><span ${tip(RISK_TIP)}>Risk ⓘ</span></th><th>Net spend</th><th>Why</th><th>Offer</th><th>Message</th><th>Status</th><th style="text-align:right">Actions</th>
            </tr></thead>
            <tbody>
              ${list.length ? list.map((c) => `
                <tr class="${ui.selected.has(c.id) ? "selected" : ""}">
                  <td><input type="checkbox" data-select="${esc(c.id)}" aria-label="Select ${esc(c.id)}" ${ui.selected.has(c.id) ? "checked" : ""}></td>
                  <td><button type="button" class="link num" data-open="${esc(c.id)}">#${esc(c.id)}</button></td>
                  <td>${riskChip(c.risk)}</td>
                  <td class="num" style="font-weight:600">${gbp(c.net_spend)}</td>
                  <td>${esc(REASON_LABEL[c.reason] || c.reason)}</td>
                  <td>${esc(offerText(c))}</td>
                  <td class="msg-cell" title="${esc(messageOf(c))}">${esc(messageOf(c))}</td>
                  <td>${statusPill(c)}</td>
                  <td style="text-align:right;white-space:nowrap">
                    <button type="button" class="btn dark" data-approve="${esc(c.id)}" ${statusOf(c) === "approved" ? "disabled" : ""}>Approve</button>
                    <button type="button" class="btn light" data-open="${esc(c.id)}">Open</button>
                    <button type="button" class="btn ghost" data-reject="${esc(c.id)}" aria-label="Reject offer for ${esc(c.id)}">Reject</button>
                  </td>
                </tr>`).join("") : `<tr><td colspan="9" class="empty">No offers match these filters.</td></tr>`}
            </tbody>
          </table>
        </div>
      </section>`;
  }

  function customer(id) {
    // Frecce avanti/indietro: dentro le offerte filtrate se il popup è aperto dalla pagina Offers
    const shown = location.hash.startsWith("#/offers") ? filtered() : [];
    const list = shown.some((x) => x.id === id) ? shown : DATA.customers;
    const c = list.find((x) => x.id === id) || list[0];
    if (!c) return `<p class="lead">No customers yet: run the pipeline first.</p>`;
    const i = list.indexOf(c);
    const next = list[(i + 1) % list.length];
    const prev = list[(i - 1 + list.length) % list.length];
    const msg = messageOf(c);
    const late = c.usual_gap ? c.delay_ratio : null;
    const aovChange = c.aov_change_pct == null ? "no comparison" : `${c.aov_change_pct > 0 ? "+" : ""}${c.aov_change_pct}% vs last year`;
    return `
      <div class="head-row">
        <div>
          <h1 id="detail-title">Customer <span class="num">#${esc(c.id)}</span></h1>
          <p class="lead">${esc(c.country)} · ${fmt(c.orders)} orders · ${fmt(c.products)} different products · loves ${esc(c.favourite)}</p>
        </div>
        <div class="row">
          ${statusPill(c)}
          <button type="button" class="btn white" data-open="${esc(prev.id)}" aria-label="Previous offer">←</button>
          <span class="chip grey num" style="font-size:12px;padding:8px 12px">${i + 1} / ${list.length}</span>
          <button type="button" class="btn white" data-open="${esc(next.id)}" aria-label="Next offer">→</button>
          <button type="button" class="btn dark" data-close aria-label="Close">✕</button>
        </div>
      </div>

      <section class="grid-12" style="align-items:start">
        <div class="span-5 stack" style="gap:16px">
          <div class="card" style="padding:14px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
            <div class="tile accent" ${tip(RISK_TIP)}>
              <span class="label">Risk of leaving ⓘ</span>
              <div><span class="big num">${c.risk}</span> / 100<div><span class="chip on-accent">${c.risk >= 70 ? "high" : c.risk >= 40 ? "medium" : "low"}</span></div></div>
            </div>
            <div class="soft tile">
              <span class="label">Silent for</span>
              <div><span class="big num">${fmt(c.days_silent)} days</span><div>${late ? `<span class="chip blue">${late}× its ${c.usual_gap}-day rhythm</span>` : `<span class="chip grey">bought on one day only</span>`}</div></div>
            </div>
            <div class="soft tile">
              <span class="label">Net spend</span>
              <div><span class="big num" style="font-size:26px">${gbp(c.net_spend)}</span><div><span class="chip grey">${fmt(c.orders)} orders · ${fmt(c.cancelled)} cancelled</span></div></div>
            </div>
            <div class="soft tile">
              <span class="label">Average order</span>
              <div><span class="big num" style="font-size:26px">${gbp(c.aov_last_12m)}</span><div><span class="chip grey">${esc(aovChange)}</span></div></div>
            </div>
          </div>

          <div class="card stack">
            <h2>Orders over time</h2>
            <div class="soft" style="padding:14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;text-align:center">
              <div><div class="sub">Previous year</div><div class="num" style="font-size:22px;font-weight:700">${fmt(c.orders_prev_12m)}</div></div>
              <div><div class="sub">Last year</div><div class="num" style="font-size:22px;font-weight:700">${fmt(c.orders_last_12m)}</div></div>
              <div><div class="sub">Same season last year</div><div style="font-size:22px;font-weight:700;color:var(--accent-text)">${c.same_season_last_year ? "Yes" : "No"}</div></div>
            </div>
            <div class="sub" style="font-size:13px">${c.usual_gap ? `Usually buys every ${c.usual_gap} days; now silent ${c.delay_ratio}× longer.` : "Bought on a single day before the reference date."}</div>
          </div>
        </div>

        <div class="card span-7 stack" style="gap:16px;padding:22px">
          <div class="row spread"><h2 style="font-size:20px">Suggested last call</h2></div>
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
            <div class="soft" style="padding:16px"><div class="sub">Likely reason</div><div style="font-size:20px;font-weight:700;margin-top:4px">${esc(REASON_LABEL[c.reason] || c.reason)}</div></div>
            <div class="tile dark" style="min-height:0"><div class="label">Offer</div><div style="font-size:20px;font-weight:700">${esc(offerText(c))}</div></div>
          </div>
          <p style="margin:0;line-height:1.6;color:var(--ink-2)">${esc(c.explanation)}</p>
          <div class="stack" style="gap:8px">
            <label for="msg" style="font-weight:700">Email to the customer</label>
            <textarea id="msg" class="email">${esc(msg)}</textarea>
            <div class="row" style="gap:8px">
              ${c.checks_ok ? `<span class="chip good">✓ names the favourite product</span><span class="chip good">✓ only the chosen offer</span>` : `<span class="chip grey">Automatic checks not passed: review carefully</span>`}
              <span class="chip grey num" id="words">${msg.trim().split(/\s+/).filter(Boolean).length} words</span>
            </div>
          </div>
          <div class="row">
            <button type="button" class="btn dark lg" data-approve="${esc(c.id)}" data-next="${esc(next.id)}">Approve</button>
            <button type="button" class="btn light lg" data-save="${esc(c.id)}">Save edit</button>
            <button type="button" class="btn light lg" data-reset="${esc(c.id)}">Restore original</button>
            <button type="button" class="btn ghost lg" style="margin-left:auto" data-reject="${esc(c.id)}" data-next="${esc(next.id)}">Reject</button>
          </div>
        </div>
      </section>`;
  }

  // ---------- pagina Customers ----------
  const SUBFILTERS = [
    { key: "level", label: "Risk level", tip: "How likely the customer is to stop buying.", options: [["all", "Any"], ["medium", "Medium (51–69)"], ["high", "High (70–100)"]] },
    { key: "delay", label: "Silent vs usual rhythm", tip: "How much longer than usual the customer has gone without ordering. 3× means three times their normal gap between orders.", options: [["any", "Any"], ["2", "2× or more"], ["5", "5× or more"], ["10", "10× or more"]] },
    { key: "aov", label: "Average order", tip: "Change in the average order value compared with the previous year.", options: [["any", "Any"], ["drop", "Dropped 20%+"], ["up", "Grew"]] },
    { key: "orders", label: "Orders this year", tip: "Number of orders in the last 12 months compared with the 12 months before.", options: [["any", "Any"], ["fewer", "Fewer than last year"], ["none_before", "New this year"]] },
    { key: "cancel", label: "Cancelled orders", tip: "Orders the customer cancelled: many cancellations can mean a bad experience.", options: [["any", "Any"], ["1", "1 or more"], ["3", "3 or more"]] },
    { key: "spend", label: "Total spend", tip: "Everything the customer has bought so far.", options: [["any", "Any"], ["500", "Over £500"], ["2000", "Over £2,000"]] },
    { key: "season", label: "Same season last year", tip: "Whether the customer bought in the same months one year earlier: a sign of seasonal buying.", options: [["any", "Any"], ["yes", "Yes"], ["no", "No"]] },
  ];

  function pool() {
    const q = cui.q.trim().toLowerCase();
    const list = DATA.pool.filter((c) => {
      if (hasOffer(c.id)) return false;
      if (q && !c.id.includes(q) && !(c.favourite || "").toLowerCase().includes(q)) return false;
      if (cui.main === "active") return true;
      if (c.risk <= AT_RISK) return false;
      if (cui.level === "medium" && c.risk >= 70) return false;
      if (cui.level === "high" && c.risk < 70) return false;
      if (cui.delay !== "any" && !(c.delay_ratio >= Number(cui.delay))) return false;
      if (cui.aov === "drop" && !(c.aov_change_pct != null && c.aov_change_pct <= -20)) return false;
      if (cui.aov === "up" && !(c.aov_change_pct != null && c.aov_change_pct > 0)) return false;
      if (cui.orders === "fewer" && !(c.orders_last_12m < c.orders_prev_12m)) return false;
      if (cui.orders === "none_before" && c.orders_prev_12m !== 0) return false;
      if (cui.cancel !== "any" && c.cancelled < Number(cui.cancel)) return false;
      if (cui.spend !== "any" && c.total_spend <= Number(cui.spend)) return false;
      if (cui.season === "yes" && !c.same_season_last_year) return false;
      if (cui.season === "no" && c.same_season_last_year) return false;
      return true;
    });
    const by = { risk: (a, b) => b.risk - a.risk, spend: (a, b) => b.total_spend - a.total_spend, silent: (a, b) => b.days_silent - a.days_silent }[cui.sort];
    return list.sort(by);
  }

  function customersPage() {
    const all = DATA.pool.filter((c) => !hasOffer(c.id));
    const atRisk = all.filter((c) => c.risk > AT_RISK).length;
    const list = pool();
    const shown = list.slice(0, cui.limit);
    const mainBtn = (key, label, n, help) => `
      <button type="button" class="${cui.main === key ? "tile accent" : "card tile"}" data-cmain="${key}" aria-pressed="${cui.main === key}" style="border:0;text-align:left;cursor:pointer;min-height:110px" ${tip(help)}>
        <span class="label">${label}</span><span class="big num">${fmt(n)}</span>
      </button>`;
    const sub = cui.main === "risk" ? `
      <section class="card stack" aria-label="Risk filters">
        <div class="row spread"><h2>Filter customers at risk</h2><button type="button" class="btn light" data-creset>Clear filters</button></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px">
          ${SUBFILTERS.map((f) => `
            <div class="stack" style="gap:6px" role="group" aria-label="${esc(f.label)}">
              <span class="label" style="font-weight:600" ${tip(f.tip)}>${esc(f.label)} ⓘ</span>
              <div class="row" style="gap:6px">
                ${f.options.map(([v, l]) => `<button type="button" class="btn ${cui[f.key] === v ? "dark" : "light"}" style="min-height:34px;padding:0 12px;font-size:13px" data-cf="${f.key}" data-cv="${v}" aria-pressed="${cui[f.key] === v}">${l}</button>`).join("")}
              </div>
            </div>`).join("")}
        </div>
      </section>` : "";
    return `
      <div class="head-row">
        <div>
          <h1>Customers</h1>
          <p class="lead">Customers who don't have an offer yet. Open one to see their numbers and ask the AI for an offer.</p>
        </div>
      </div>

      <section class="grid-12" aria-label="Customer groups">
        <div class="span-6">${mainBtn("active", "Active customers", all.length, "Customers who bought in the last 12 months and don't have an offer yet.")}</div>
        <div class="span-6">${mainBtn("risk", "Customers at risk", atRisk, "Active customers with a risk score above 50: more likely than not to stop buying.")}</div>
      </section>
      ${sub}

      <section class="card stack">
        <div class="row spread">
          <h2>${fmt(list.length)} customers</h2>
          <div class="row">
            <label class="search">
              <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
              <span class="sr">Search customers</span>
              <input type="search" id="cq" placeholder="Customer or product" value="${esc(cui.q)}">
            </label>
            <label class="search" style="padding-right:6px"><span>Sort</span>
              <select id="csort" style="border:0;background:transparent;outline:none">
                ${[["risk", "Risk"], ["spend", "Total spend"], ["silent", "Days silent"]].map(([v, l]) => `<option value="${v}" ${cui.sort === v ? "selected" : ""}>${l}</option>`).join("")}
              </select>
            </label>
          </div>
        </div>
        <div class="table-box">
          <table>
            <thead><tr><th>Customer</th><th><span ${tip(RISK_TIP)}>Risk ⓘ</span></th><th>Total spend</th><th>Orders (year / before)</th><th>Silent for</th><th>Average order</th><th>Favourite product</th><th style="text-align:right"></th></tr></thead>
            <tbody>
              ${shown.length ? shown.map((c) => `
                <tr>
                  <td><button type="button" class="link num" data-pool="${esc(c.id)}">#${esc(c.id)}</button></td>
                  <td>${riskChip(c.risk)}</td>
                  <td class="num" style="font-weight:600">${gbp(c.total_spend)}</td>
                  <td class="num">${fmt(c.orders_last_12m)} / ${fmt(c.orders_prev_12m)}</td>
                  <td style="color:var(--ink-2)">${fmt(c.days_silent)} days${c.usual_gap ? ` · ${c.delay_ratio}× usual` : ""}</td>
                  <td class="num">${c.aov_change_pct == null ? "–" : `${c.aov_change_pct > 0 ? "+" : ""}${c.aov_change_pct}%`}</td>
                  <td class="msg-cell" style="max-width:220px">${esc(c.favourite)}</td>
                  <td style="text-align:right"><button type="button" class="btn light" data-pool="${esc(c.id)}">Open</button></td>
                </tr>`).join("") : `<tr><td colspan="8" class="empty">No customers match these filters.</td></tr>`}
            </tbody>
          </table>
        </div>
        ${list.length > shown.length ? `<button type="button" class="btn outline" style="align-self:center" data-cmore>Show more (${fmt(list.length - shown.length)} left)</button>` : ""}
      </section>`;
  }

  function poolDetail(id, state) {
    const c = DATA.pool.find((x) => x.id === id);
    if (!c) return "";
    const aovChange = c.aov_change_pct == null ? "no comparison" : `${c.aov_change_pct > 0 ? "+" : ""}${c.aov_change_pct}% vs last year`;
    const level = c.risk >= 70 ? "high" : c.risk >= 40 ? "medium" : "low";
    const action = state === "loading"
      ? `<button type="button" class="btn dark lg" disabled>Writing the offer…</button><span class="sub">This takes about 10–30 seconds.</span>`
      : `<button type="button" class="btn dark lg" data-generate="${esc(c.id)}">✦ Generate offer with AI</button>`;
    return `
      <div class="head-row">
        <div>
          <h1 id="detail-title">Customer <span class="num">#${esc(c.id)}</span></h1>
          <p class="lead">${esc(c.country)} · ${fmt(c.orders_last_12m + c.orders_prev_12m)} orders in 2 years · ${fmt(c.products)} different products · loves ${esc(c.favourite)}</p>
        </div>
        <div class="row">
          <span class="row" style="gap:6px"><span class="dot" style="background:#A1A1AA" aria-hidden="true"></span>No offer yet</span>
          <button type="button" class="btn dark" data-close aria-label="Close">✕</button>
        </div>
      </div>
      <section class="grid-12" style="align-items:start">
        <div class="span-5 stack" style="gap:16px">
          <div class="card" style="padding:14px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
            <div class="tile accent" ${tip(RISK_TIP)}>
              <span class="label">Risk of leaving ⓘ</span>
              <div><span class="big num">${c.risk}</span> / 100<div><span class="chip on-accent">${level}</span></div></div>
            </div>
            <div class="soft tile">
              <span class="label">Silent for</span>
              <div><span class="big num">${fmt(c.days_silent)} days</span><div>${c.usual_gap ? `<span class="chip blue">${c.delay_ratio}× its ${c.usual_gap}-day rhythm</span>` : `<span class="chip grey">bought on one day only</span>`}</div></div>
            </div>
            <div class="soft tile">
              <span class="label">Total spend</span>
              <div><span class="big num" style="font-size:26px">${gbp(c.total_spend)}</span><div><span class="chip grey">${fmt(c.cancelled)} cancelled</span></div></div>
            </div>
            <div class="soft tile">
              <span class="label">Average order</span>
              <div><span class="big num" style="font-size:26px">${gbp(c.aov_last_12m)}</span><div><span class="chip grey">${esc(aovChange)}</span></div></div>
            </div>
          </div>

          <div class="card stack">
            <h2>Orders over time</h2>
            <div class="soft" style="padding:14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;text-align:center">
              <div><div class="sub">Previous year</div><div class="num" style="font-size:22px;font-weight:700">${fmt(c.orders_prev_12m)}</div></div>
              <div><div class="sub">Last year</div><div class="num" style="font-size:22px;font-weight:700">${fmt(c.orders_last_12m)}</div></div>
              <div><div class="sub">Same season last year</div><div style="font-size:22px;font-weight:700;color:var(--accent-text)">${c.same_season_last_year ? "Yes" : "No"}</div></div>
            </div>
            <div class="sub" style="font-size:13px">${c.usual_gap ? `Usually buys every ${c.usual_gap} days; now silent ${c.delay_ratio}× longer.` : "Bought on a single day before the reference date."}</div>
          </div>
        </div>

        <div class="card span-7 stack" style="gap:16px;padding:22px">
          <div class="row spread"><h2 style="font-size:20px">Suggested last call</h2></div>
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">
            <div class="soft" style="padding:16px"><div class="sub">Likely reason</div><div style="font-size:20px;font-weight:700;margin-top:4px;color:var(--muted)">Not yet</div></div>
            <div class="tile dark" style="min-height:0"><div class="label">Offer</div><div style="font-size:20px;font-weight:700;color:#BDBDC4">Not yet</div></div>
          </div>
          <p style="margin:0;line-height:1.6;color:var(--ink-2)">This customer has no offer yet. The AI can suggest why they are slowing down, the offer that fits and a ready-to-send email. You review it before anything is sent.</p>
          <div class="stack" style="gap:8px">
            <span style="font-weight:700">Email to the customer</span>
            <div class="email" style="border:1px dashed #D4D4D8;border-radius:18px;padding:16px;color:var(--muted);min-height:150px;display:flex;align-items:center;justify-content:center;text-align:center">The email will appear here.</div>
          </div>
          <div class="row" id="gen-area">${action}</div>
        </div>
      </section>`;
  }

  async function generateOffer(id) {
    dialog.innerHTML = `<div class="detail-body">${poolDetail(id, "loading")}</div>`;
    try {
      const r = await fetch("api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ customer_id: id }) });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || (r.status === 404 || r.status === 405 || r.status === 501 ? "offline" : "HTTP " + r.status));
      const p = DATA.pool.find((x) => x.id === id);
      const offer = {
        ...p, band: p.risk >= 70 ? "alto" : p.risk >= 40 ? "medio" : "basso",
        net_spend: body.net_spend, orders: body.orders,
        reason: body.reason, offer: body.offer, discount_pct: body.discount_pct,
        explanation: body.explanation, message: body.message, checks_ok: body.checks_ok, generated: true,
      };
      generated[id] = offer; saveGenerated();
      DATA.customers.push(offer);
      toast(`Offer for #${id} ready to review`);
      render();
      ui.offer = "all"; ui.status = "all";
      openDetail(id);
    } catch (e) {
      const offline = String(e.message) === "offline" || e instanceof TypeError;
      dialog.innerHTML = `<div class="detail-body">${poolDetail(id)}</div>`;
      const area = document.getElementById("gen-area");
      if (area) area.insertAdjacentHTML("beforeend", `<span class="sub" style="color:var(--danger);max-width:320px">${offline ? "AI generation needs the local server: run <code>python demo/server.py</code>." : "The AI could not write the offer: " + esc(e.message)}</span>`);
    }
  }

  function how() {
    const high = DATA.bands.find((b) => b.name === "High");
    return `
      <div>
        <div class="row" style="gap:12px"><h1>How LastCall works</h1><span class="chip violet" style="font-size:12px;padding:5px 11px">AI on AMD</span></div>
        <p class="lead">We spot who is about to stop buying, suggest the right offer and write the message. You decide.</p>
      </div>
      <section class="grid-12">
        <div class="card span-3 stack"><span class="chip grey" style="align-self:start">Step 1</span><h2>Your sales history</h2><p class="label" style="margin:0;line-height:1.6">We look at how each customer has bought so far: how often, how much and what.</p></div>
        <div class="tile accent span-3" style="border-radius:22px;padding:20px;justify-content:start;gap:10px"><span class="chip on-accent" style="align-self:start">Step 2</span><h2 style="margin:0;font-size:17px">Who is drifting away</h2><p style="margin:0;line-height:1.6;font-size:13px">Every customer gets a risk score from 0 to 100: how likely they are to stop buying, compared with their own usual rhythm.</p></div>
        <div class="card span-3 stack"><span class="chip violet" style="align-self:start">Step 3</span><h2>The right offer</h2><p class="label" style="margin:0;line-height:1.6">For valuable customers at risk we suggest why they are slowing down, the offer that fits and a ready-to-send email.</p></div>
        <div class="tile dark span-3" style="border-radius:22px;padding:20px;justify-content:start;gap:10px"><span class="chip" style="background:#2E2E33;color:#fff;align-self:start">Step 4</span><h2 style="margin:0;font-size:17px">You approve</h2><p style="margin:0;line-height:1.6;font-size:13px;color:#D4D4D8">Nothing is sent automatically: approve, edit or reject each offer.</p></div>
      </section>
      <section class="grid-12">
        <div class="card span-6 stack">
          <h2>Does the risk score work?</h2>
          <p class="label" style="margin:0;line-height:1.6">We checked it on past data: of the customers marked as high risk, <strong>${pct(high.left, high.customers)}%</strong> really stopped buying in the following months. Among low-risk customers it was only <strong>${pct(DATA.bands[0].left, DATA.bands[0].customers)}%</strong>.</p>
          <div class="soft stack" style="padding:16px;font-size:13px">
            ${DATA.bands.map((b) => `<div ${tip(bandTip(b))}><div class="row spread"><span>${esc(b.name)} risk</span><span class="num" style="font-weight:700">${pct(b.left, b.customers)}% left</span></div>
              <div class="meter" style="height:12px"><div style="height:12px;width:${pct(b.left, b.customers)}%;background:${b.name === "High" ? "var(--accent)" : "var(--black)"}"></div></div></div>`).join("")}
          </div>
        </div>
        <div class="card span-6 stack">
          <h2>What every message respects</h2>
          ${["It mentions the product the customer buys most",
             "It contains only the offer we suggest, nothing else",
             "It is short, friendly and in plain English",
             "Nothing is sent without your approval, and you can always edit it"].map((t) =>
            `<div class="soft row" style="padding:12px 14px;flex-wrap:nowrap"><span class="chip good">✓</span><span style="font-size:13px">${t}</span></div>`).join("")}
        </div>
      </section>`;
  }

  // ---------- router ----------
  function render() {
    const parts = (location.hash.replace(/^#\/?/, "") || "overview").split("/");
    let route = parts[0];
    if (route === "customer" && parts[1]) {
      // vecchi link #/customer/ID: apri la scheda come popup sopra la pagina delle offerte
      history.replaceState(null, "", "#/offers");
      route = "offers";
      if (parts[1]) setTimeout(() => openDetail(parts[1]));
    }
    document.querySelectorAll("[data-route]").forEach((a) => {
      if (a.dataset.route === route) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const page = { overview, offers, customers: customersPage, how }[route] || overview;
    app.innerHTML = page();
    document.title = "LastCall · " + ({ overview: "Overview", offers: "Offers", customers: "Customers", how: "How it works" }[route] || "Overview");
  }

  // ---------- scheda cliente in popup ----------
  const dialog = document.createElement("dialog");
  dialog.className = "detail";
  dialog.setAttribute("aria-labelledby", "detail-title");
  document.body.appendChild(dialog);
  let openId = null;

  function openDetail(id) {
    openId = id;
    dialog.innerHTML = `<div class="detail-body">${customer(id)}</div>`;
    if (!dialog.open) dialog.showModal();
    dialog.scrollTop = 0;
  }
  // Cliente senza offerta: solo statistiche e il pulsante per chiedere un'offerta all'AI
  function openPool(id) {
    openId = null;
    dialog.innerHTML = `<div class="detail-body">${poolDetail(id)}</div>`;
    if (!dialog.open) dialog.showModal();
    dialog.scrollTop = 0;
  }
  function closeDetail() {
    if (dialog.open) dialog.close();
  }
  dialog.addEventListener("close", () => { openId = null; render(); });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) closeDetail(); });

  function exportCsv() {
    const head = ["customer_id", "risk", "net_spend_gbp", "reason", "offer", "discount_pct", "status", "message"];
    const rows = DATA.customers.map((c) => [c.id, c.risk, c.net_spend, c.reason, c.offer, c.discount_pct, statusOf(c), messageOf(c)]);
    const csv = [head, ...rows].map((r) => r.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "lastcall_offers_reviewed.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function onClick(e) {
    const t = e.target.closest("button, input[type=checkbox]");
    if (!t) return;
    const d = t.dataset;
    if (d.open) openDetail(d.open);
    else if (d.pool) openPool(d.pool);
    else if (d.generate) generateOffer(d.generate);
    else if (d.cmain) { cui.main = d.cmain; cui.limit = CUI_DEFAULT.limit; render(); }
    else if (d.cf) { cui[d.cf] = d.cv; cui.limit = CUI_DEFAULT.limit; render(); }
    else if (d.creset !== undefined) { Object.assign(cui, CUI_DEFAULT, { main: "risk", q: cui.q, sort: cui.sort }); render(); }
    else if (d.cmore !== undefined) { cui.limit += 60; render(); }
    else if (d.close !== undefined) closeDetail();
    else if (d.mix) { ui.offer = d.mix; ui.status = "all"; ui.query = ""; ui.selected.clear(); location.hash = "#/offers"; }
    else if (d.status) { ui.status = d.status; ui.selected.clear(); render(); }
    else if (d.offer) { ui.offer = d.offer; ui.selected.clear(); render(); }
    else if (d.select) { t.checked ? ui.selected.add(d.select) : ui.selected.delete(d.select); render(); }
    else if (t.id === "all") { filtered().forEach((c) => (t.checked ? ui.selected.add(c.id) : ui.selected.delete(c.id))); render(); }
    else if (d.approve) {
      const box = document.getElementById("msg");
      const c = DATA.customers.find((x) => x.id === d.approve);
      const edited = box && box.value.trim() !== (c.message || "").trim();
      decide(d.approve, edited ? "edited" : "approved", box ? box.value : undefined);
      toast(`Offer for #${d.approve} ${edited ? "edited and approved" : "approved"}`);
      afterDecision(d.next);
    } else if (d.reject) {
      decide(d.reject, "rejected");
      toast(`Offer for #${d.reject} rejected`);
      afterDecision(d.next);
    } else if (d.save) {
      decide(d.save, "edited", document.getElementById("msg").value);
      toast("Edit saved"); refresh();
    } else if (d.reset) {
      delete decisions[d.reset]; saveDecisions(); toast("Original message restored"); refresh();
    } else if (d.action === "export") exportCsv();
    else if (d.action === "approve-selected") {
      ui.selected.forEach((id) => decide(id, "approved"));
      toast(`${ui.selected.size} offers approved`);
      ui.selected.clear(); render();
    }
  }

  // Dopo una decisione nel popup passa all'offerta successiva; nella tabella aggiorna solo la pagina
  function afterDecision(nextId) {
    if (dialog.open && nextId) { render(); openDetail(nextId); } else refresh();
  }
  function refresh() {
    render();
    if (dialog.open && openId) openDetail(openId);
  }

  app.addEventListener("click", onClick);
  dialog.addEventListener("click", onClick);
  dialog.addEventListener("input", (e) => {
    if (e.target.id === "msg") {
      document.getElementById("words").textContent = e.target.value.trim().split(/\s+/).filter(Boolean).length + " words";
    }
  });

  app.addEventListener("change", (e) => {
    if (e.target.id === "csort") { cui.sort = e.target.value; render(); }
  });

  app.addEventListener("input", (e) => {
    if (e.target.id === "cq") {
      cui.q = e.target.value;
      cui.limit = CUI_DEFAULT.limit;
      const pos = e.target.selectionStart;
      render();
      const q = document.getElementById("cq");
      q.focus(); q.setSelectionRange(pos, pos);
    } else if (e.target.id === "q") {
      ui.query = e.target.value;
      const pos = e.target.selectionStart;
      render();
      const q = document.getElementById("q");
      q.focus(); q.setSelectionRange(pos, pos);
    } else if (e.target.id === "msg") {
      document.getElementById("words").textContent = e.target.value.trim().split(/\s+/).filter(Boolean).length + " words";
    }
  });

  // ---------- tooltip: un solo riquadro che segue l'elemento con data-tip (mouse o tastiera) ----------
  const tipBox = document.createElement("div");
  tipBox.className = "tip";
  tipBox.setAttribute("role", "tooltip");
  tipBox.hidden = true;
  document.body.appendChild(tipBox);
  function showTip(el) {
    const host = el.closest("dialog") || document.body; // il popup sta sopra a tutto: il tooltip va dentro
    if (tipBox.parentNode !== host) host.appendChild(tipBox);
    tipBox.textContent = el.dataset.tip;
    tipBox.hidden = false;
    const r = el.getBoundingClientRect();
    const w = tipBox.offsetWidth, h = tipBox.offsetHeight;
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
    let top = r.top - h - 8;
    if (top < 8) top = r.bottom + 8;
    tipBox.style.left = left + "px";
    tipBox.style.top = top + "px";
  }
  const hideTip = () => { tipBox.hidden = true; };
  document.addEventListener("mouseover", (e) => { const el = e.target.closest("[data-tip]"); el ? showTip(el) : hideTip(); });
  document.addEventListener("focusin", (e) => { const el = e.target.closest("[data-tip]"); el ? showTip(el) : hideTip(); });
  document.addEventListener("scroll", hideTip, true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });

  window.addEventListener("hashchange", render);
  if (!DATA) app.innerHTML = '<p class="lead">data.json could not be loaded: run <code>python churn/export_demo.py</code> and open the page through a web server (e.g. <code>python -m http.server --directory demo</code>).</p>';
  else render();
})();
