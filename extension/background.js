chrome.commands.onCommand.addListener(async (command) => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  if (command === "capture-text" || command === "capture-diagram") {
    chrome.tabs.sendMessage(tab.id, {
      action: "capture",
      mode: command === "capture-text" ? "text" : "diagram",
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "extract") {
    extractNotes(message.imageData, message.mode, message.context)
      .then(sendResponse)
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }
});

async function extractNotes(imageData, mode, context) {
  const token = (await chrome.storage.local.get("idToken")).idToken;
  if (!token) {
    return { error: "Not authenticated. Click the extension icon to log in." };
  }

  const endpoint =
    mode === "text"
      ? "https://api.snapnote.ai/api/extract/text"
      : "https://api.snapnote.ai/api/extract/diagram";

  const blob = dataURLToBlob(imageData);
  const formData = new FormData();
  formData.append("image", blob, "screenshot.png");
  formData.append("context", JSON.stringify(context));

  const resp = await fetch(endpoint, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    return { error: err.detail || `Server error (${resp.status})` };
  }

  return await resp.json();
}

function dataURLToBlob(dataUrl) {
  const [header, data] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)[1];
  const bytes = atob(data);
  const buf = new ArrayBuffer(bytes.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) view[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}
