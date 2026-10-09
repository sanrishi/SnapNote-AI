import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js";
import {
  getAuth,
  signInWithPopup,
  GoogleAuthProvider,
  onAuthStateChanged,
  signOut,
} from "https://www.gstatic.com/firebasejs/10.14.0/firebase-auth.js";

const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// DOM refs
const loginBtn = document.getElementById("login-btn");
const logoutBtn = document.getElementById("logout-btn");
const authSection = document.getElementById("auth-section");
const mainSection = document.getElementById("main-section");
const creditsBadge = document.getElementById("credits-badge");
const lastExtraction = document.getElementById("last-extraction");
const markdownPreview = document.getElementById("markdown-preview");
const tagsContainer = document.getElementById("tags-container");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const jeeCard = document.getElementById("jee-card");

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function renderJeeCard(jee) {
  if (!jee || (jee.match_status !== "matched" && jee.match_status !== "ambiguous")) {
    jeeCard.style.display = "none";
    jeeCard.innerHTML = "";
    return;
  }
  let html = "";
  if (jee.match_status === "ambiguous") {
    html += `<h4>🎓 JEE Context — ambiguous</h4><p>This could be more than one JEE concept:</p><ul>`;
    for (const cand of jee.candidates || []) {
      html += `<li>${escapeHtml(cand.name)}</li>`;
    }
    html += `</ul>`;
  } else {
    html += `<h4>🎓 JEE Context</h4>`;
    html += `<p><strong>${escapeHtml(jee.canonical_name || "")}</strong> (${escapeHtml(jee.subject || "")} · ${escapeHtml(jee.chapter || "")})</p>`;
    if (jee.syllabus && jee.syllabus.subtopic) {
      html += `<p class="jee-syllabus">Syllabus: ${escapeHtml(jee.syllabus.subtopic)}</p>`;
    }
    if ((jee.pyqs || []).length > 0) {
      html += `<p>Verified PYQs (${jee.pyq_count}):</p><ul>`;
      for (const pyq of jee.pyqs) {
        const label = `${escapeHtml(pyq.exam)} ${pyq.year} Q${pyq.paper_question_number} (${escapeHtml(pyq.question_type)})`;
        const answer = pyq.answer_status === "confirmed" ? escapeHtml(pyq.answer_key) : "answer unconfirmed";
        const prov = pyq.source_class === "OFFICIAL" ? "official NTA paper" : "third-party transcription";
        const link = pyq.source_url
          ? ` <a href="${escapeHtml(pyq.source_url)}" target="_blank" rel="noopener">[${prov}]</a>`
          : ` [${prov}]`;
        html += `<li>${label}: ${escapeHtml(pyq.question_summary || "see source")} — ${answer}${link}</li>`;
      }
      html += `</ul>`;
      if (jee.pyqs_truncated) {
        html += `<p class="jee-note">Showing ${jee.pyqs.length} of ${jee.pyq_count} verified PYQs.</p>`;
      }
    }
    for (const pattern of jee.patterns || []) {
      html += `<p><strong>Pattern: ${escapeHtml(pattern.name)}</strong> (${escapeHtml(pattern.sub_concept)})<br>${escapeHtml(pattern.method)}<br><span class="jee-note">${escapeHtml(pattern.coverage_note)}</span></p>`;
    }
    if (jee.coverage_note) {
      html += `<p class="jee-note">${escapeHtml(jee.coverage_note)}</p>`;
    }
  }
  jeeCard.innerHTML = html;
  jeeCard.style.display = "block";
}

onAuthStateChanged(auth, async (user) => {
  if (user) {
    authSection.style.display = "none";
    mainSection.style.display = "block";
    const token = await user.getIdToken();
    await chrome.storage.local.set({ idToken: token });
    creditsBadge.textContent = "50 credits";
    loadLastExtraction();
  } else {
    authSection.style.display = "block";
    mainSection.style.display = "none";
    await chrome.storage.local.remove("idToken");
  }
});

loginBtn.addEventListener("click", () => {
  signInWithPopup(auth, provider).catch((err) => {
    console.error("Auth error:", err);
    alert("Sign-in failed: " + err.message);
  });
});

logoutBtn.addEventListener("click", async () => {
  await signOut(auth);
});

async function loadLastExtraction() {
  const { lastExtraction: data } = await chrome.storage.local.get(
    "lastExtraction"
  );
  if (!data) return;

  lastExtraction.style.display = "block";
  markdownPreview.textContent = data.markdown;

  tagsContainer.innerHTML = data.tags
    .map((t) => `<span>${t}</span>`)
    .join("");

  renderJeeCard(data.jee);

  copyBtn.onclick = () => {
    navigator.clipboard.writeText(data.markdown).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy Markdown"), 2000);
    });
  };

  downloadBtn.onclick = () => {
    const blob = new Blob([data.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `snapnote-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };
}
