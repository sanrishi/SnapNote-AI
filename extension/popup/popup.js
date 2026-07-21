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
