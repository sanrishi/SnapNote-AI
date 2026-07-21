chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "capture") {
    captureFrame(message.mode);
  }
});

async function captureFrame(mode) {
  try {
    const video = document.querySelector("video");
    let imageData;

    if (video && video.readyState >= 2) {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      imageData = canvas.toDataURL("image/png");
    } else {
      imageData = await captureVisibleTab();
    }

    const context = {
      title: document.title,
      url: window.location.href,
      week: extractWeekInfo(document.title, window.location.href),
    };

    chrome.runtime.sendMessage(
      { action: "extract", imageData, mode, context },
      (response) => {
        if (chrome.runtime.lastError) {
          showToast("Error: " + chrome.runtime.lastError.message, "error");
          return;
        }
        if (response.error) {
          showToast(response.error, "error");
          return;
        }
        handleResult(response, mode);
      }
    );
  } catch (err) {
    showToast("Capture failed: " + err.message, "error");
  }
}

function handleResult(result, mode) {
  if (mode === "text") {
    copyToClipboard(result.markdown);
    showToast(
      result.type === "table" ? "Table copied to clipboard!" : "Text copied to clipboard!",
      "success"
    );
  } else {
    showToast("Diagram extracted! Click extension to copy.", "success");
  }

  chrome.storage.local.set({
    lastExtraction: {
      markdown: result.markdown,
      imageUrl: result.imageUrl,
      tags: result.tags,
      timestamp: Date.now(),
    },
  });
}

function captureVisibleTab() {
  return new Promise((resolve, reject) => {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
      if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
      else resolve(dataUrl);
    });
  });
}

function extractWeekInfo(title, url) {
  const weekMatch =
    title.match(/week[_\s]?(\d+)/i) || url.match(/week[_\s]?(\d+)/i);
  return weekMatch ? `Week ${weekMatch[1]}` : "";
}

function copyToClipboard(text) {
  const el = document.createElement("textarea");
  el.value = text;
  el.style.position = "fixed";
  el.style.left = "-9999px";
  document.body.appendChild(el);
  el.select();
  document.execCommand("copy");
  document.body.removeChild(el);
}

function showToast(message, type) {
  const existing = document.getElementById("snapnote-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "snapnote-toast";
  toast.textContent = message;
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "24px",
    right: "24px",
    padding: "12px 20px",
    borderRadius: "8px",
    fontSize: "14px",
    fontWeight: "500",
    zIndex: "999999",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    transition: "opacity 0.3s",
    background: type === "error" ? "#ef4444" : "#22c55e",
    color: "#fff",
    fontFamily: "system-ui, sans-serif",
    maxWidth: "400px",
  });
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
