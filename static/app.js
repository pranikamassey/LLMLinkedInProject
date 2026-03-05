function escapeHtml(str) {
  return (str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    return false;
  }
}

function tableHtml(title, items) {
  const rows = (items ?? []).map((p, idx) => {
    const url = p.profile_url || "";
    const msg = p.message_300 || "";
    const why = (p.why_matched || []).slice(0, 5).join(", ");

    return `
      <tr>
        <td class="num">${idx + 1}</td>
        <td>${escapeHtml(p.name || "")}</td>
        <td class="num">${escapeHtml(String(p.confidence ?? ""))}</td>
        <td>
          ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">Open</a>` : ""}
        </td>
        <td class="why">${escapeHtml(why)}</td>
        <td class="msgcell">
          <textarea class="msg" readonly>${escapeHtml(msg)}</textarea>
          <button class="copyBtn" data-msg="${escapeHtml(msg)}">Copy</button>
        </td>
      </tr>
    `;
  }).join("");

  return `
    <div class="tableCard">
      <h3>${escapeHtml(title)}</h3>
      <div class="tableWrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Conf</th>
              <th>LinkedIn</th>
              <th>Why</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  `;
}

function companyBlock(rep) {
  return `
    <div class="companyBlock">
      <div class="companyHeader">
        <h2>${escapeHtml(rep.company)} <span class="muted">— ${escapeHtml(rep.location)}</span></h2>
      </div>

      <div class="grid">
        ${tableHtml("Part 1 — Recruiting (Rules-only)", rep.part1_recruiting_rules)}
        ${tableHtml("Part 1 — Recruiting (LLM-on-top)", rep.part1_recruiting_llm)}
        ${tableHtml("Part 2 — Senior Engineers (Rules-only)", rep.part2_senior_engineers_rules)}
        ${tableHtml("Part 2 — Senior Engineers (LLM-on-top)", rep.part2_senior_engineers_llm)}
      </div>
    </div>
  `;
}

function wireCopyButtons(root) {
  root.querySelectorAll(".copyBtn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const msg = btn.getAttribute("data-msg") || "";
      const ok = await copyText(msg);
      btn.textContent = ok ? "Copied" : "Copy failed";
      setTimeout(() => (btn.textContent = "Copy"), 1200);
    });
  });
}

async function runSearch() {
  const companies = document.getElementById("companies").value;
  const location = document.getElementById("location").value || "United States";
  const role_focus = document.getElementById("role_focus").value || null;

  const status = document.getElementById("status");
  const results = document.getElementById("results");

  status.textContent = "Running...";
  results.innerHTML = "";

  const payload = {
    companies_text: companies,
    location,
    role_focus,
    per_query_count: 10,
    seed_n: 25,
    rules_k: 2,
    llm_k: 2
  };

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const reports = data.reports || [];

    if (reports.length === 0) {
      status.textContent = "No companies provided.";
      return;
    }

    status.textContent = `Done. Companies: ${reports.length}`;
    results.innerHTML = reports.map(companyBlock).join("");
    wireCopyButtons(results);

  } catch (err) {
    status.textContent = `Error: ${err.message || err}`;
  }
}

function clearAll() {
  document.getElementById("companies").value = "";
  document.getElementById("role_focus").value = "";
  document.getElementById("location").value = "United States";
  document.getElementById("results").innerHTML = "";
  document.getElementById("status").textContent = "";
}

document.getElementById("runBtn").addEventListener("click", runSearch);
document.getElementById("clearBtn").addEventListener("click", clearAll);