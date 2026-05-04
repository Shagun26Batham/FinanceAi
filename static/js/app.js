/**
 * app.js – FinanceAI Light-Blue SaaS UI (v3)
 * ─ Dark-mode toggle with localStorage persistence
 * ─ SSE token streaming from /chat
 * ─ Markdown rendering via marked.js
 * ─ Clear Chat, typing indicator, auto-resize textarea
 */

(() => {
  "use strict";

  /* ════════════════════════════════════════════════════════════════
     THEME  –  dark / light toggle
     ════════════════════════════════════════════════════════════════ */

  const THEME_KEY   = "financeai-theme";
  const htmlEl      = document.documentElement;
  const themeToggle = document.getElementById("btn-theme-toggle");
  const iconSun     = themeToggle.querySelector(".icon-sun");
  const iconMoon    = themeToggle.querySelector(".icon-moon");

  /** Apply theme to <html data-theme="..."> and persist it */
  function applyTheme(theme) {
    const isDark = theme === "dark";
    htmlEl.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
    themeToggle.setAttribute("aria-pressed", String(isDark));
    themeToggle.title     = isDark ? "Switch to light mode" : "Switch to dark mode";
    themeToggle.ariaLabel = themeToggle.title;
    // Directly control icon visibility (immune to CSS transition interference)
    iconSun.style.display  = isDark ? "none"  : "block";
    iconMoon.style.display = isDark ? "block" : "none";
  }

  /** Read stored preference (default = light) */
  function initTheme() {
    const stored   = localStorage.getItem(THEME_KEY);
    const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(stored ?? preferred);
  }

  themeToggle.addEventListener("click", () => {
    const current = htmlEl.getAttribute("data-theme");
    applyTheme(current === "dark" ? "light" : "dark");
  });

  initTheme();

  /* ════════════════════════════════════════════════════════════════
     CHAT
     ════════════════════════════════════════════════════════════════ */

  // ── DOM refs ───────────────────────────────────────────────────
  const messagesEl = document.getElementById("chat-messages");
  const inputEl    = document.getElementById("user-input");
  const sendBtn    = document.getElementById("send-btn");
  const typingEl   = document.getElementById("typing-indicator");
  const clearBtn   = document.getElementById("btn-clear-chat");

  // ── State ──────────────────────────────────────────────────────
  let isLoading = false;
  let abortCtrl = null;

  // ── Marked.js config ───────────────────────────────────────────
  if (window.marked) {
    marked.setOptions({
      breaks:   true,
      gfm:      true,
      pedantic: false,
    });
  }

  /* ── Utilities ──────────────────────────────────────────────── */

  /** Format current time as HH:MM */
  function nowTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  /** Scroll messages area to the bottom smoothly */
  function scrollToBottom() {
    messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
  }

  /** Remove the welcome card (animated) */
  function hideWelcome() {
    const card = document.getElementById("welcome-card");
    if (!card) return;
    card.style.transition = "opacity 0.22s ease, transform 0.22s ease";
    card.style.opacity    = "0";
    card.style.transform  = "translateY(-10px) scale(0.98)";
    setTimeout(() => card.remove(), 240);
  }

  /** Auto-resize textarea (max 160 px) */
  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }

  /** Show / hide the typing indicator */
  function setTyping(visible) {
    typingEl.classList.toggle("hidden", !visible);
    typingEl.setAttribute("aria-hidden", String(!visible));
    if (visible) scrollToBottom();
  }

  /** Disable or enable input controls while loading */
  function setLoading(loading) {
    isLoading        = loading;
    inputEl.disabled = loading;
    sendBtn.disabled = loading;
    sendBtn.setAttribute("aria-busy", String(loading));
  }

  /** Render Markdown into a DOM element (falls back to textContent) */
  function renderMarkdown(el, text) {
    if (window.marked) {
      el.innerHTML = marked.parse(text);
    } else {
      el.textContent = text;
    }
  }

  /* ── Message creation ─────────────────────────────────────────── */

  /**
   * Create and append a message group.
   * Returns { group, bubble, contentEl } for streaming updates.
   */
  function appendMessage(role, text = "", isError = false) {
    const group = document.createElement("div");
    group.className = `message-group ${role}`;

    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    // Avatar
    const avatar = document.createElement("div");
    avatar.className = `avatar ${role}`;
    if (role === "bot") {
      avatar.innerHTML = `
        <svg viewBox="0 0 20 20" fill="none" stroke="#fff" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="2,14 7,8 11,11 18,4"/>
          <circle cx="18" cy="4" r="1.5" fill="#fff" stroke="none"/>
        </svg>`;
    } else {
      avatar.textContent = "🧑";
      avatar.setAttribute("aria-hidden", "true");
    }

    // Bubble
    const bubble = document.createElement("div");
    bubble.className = `bubble${isError ? " error" : ""}`;

    const contentEl = document.createElement("div");
    contentEl.className = "bubble-content";

    if (text) {
      if (role === "bot" && !isError) {
        renderMarkdown(contentEl, text);
      } else {
        contentEl.textContent = text;
      }
    }

    bubble.appendChild(contentEl);

    if (role === "user") {
      row.appendChild(bubble);
      row.appendChild(avatar);
    } else {
      row.appendChild(avatar);
      row.appendChild(bubble);
    }

    // Timestamp meta row
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.innerHTML = `<span class="msg-time">${nowTime()}</span>`;

    group.appendChild(row);
    group.appendChild(meta);
    messagesEl.appendChild(group);
    scrollToBottom();

    return { group, bubble, contentEl };
  }

  /* ── Core: send & stream ─────────────────────────────────────── */

  async function sendMessage(prefilled) {
    const message = (prefilled !== undefined ? prefilled : inputEl.value).trim();
    if (!message || isLoading) return;

    hideWelcome();

    inputEl.value = "";
    autoResize();

    appendMessage("user", message);
    setLoading(true);
    setTyping(true);

    // Create an empty bot bubble – filled by streaming tokens
    const { bubble: botBubble, contentEl: botContent } = appendMessage("bot", "");
    botBubble.classList.add("streaming-cursor");

    let accumulated = "";
    abortCtrl = new AbortController();

    try {
      const response = await fetch("/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message }),
        signal:  abortCtrl.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
        botContent.textContent = `⚠️  ${err.error || "Server error"}`;
        botBubble.classList.add("error");
        botBubble.classList.remove("streaming-cursor");
        return;
      }

      // SSE stream reader
      const reader     = response.body.getReader();
      const decoder    = new TextDecoder();
      let   buffer     = "";
      let   firstToken = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are double-newline separated
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // retain incomplete trailing fragment

        for (const part of parts) {
          const lines = part.trim().split("\n");
          let event = "message";
          let data  = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            if (line.startsWith("data: "))  data  = line.slice(6).trim();
          }

          if (event === "token") {
            if (firstToken) {
              setTyping(false);
              firstToken = false;
            }
            const token = JSON.parse(data).data;
            accumulated += token;
            renderMarkdown(botContent, accumulated);
            scrollToBottom();

          } else if (event === "error") {
            const errText = JSON.parse(data).data;
            botContent.textContent = `⚠️  ${errText}`;
            botBubble.classList.add("error");

          } else if (event === "done") {
            // Stream complete — final render already done above
          }
        }
      }

    } catch (err) {
      if (err.name !== "AbortError") {
        botContent.textContent =
          "⚠️  Could not reach the server. Make sure Flask is running on port 5000.";
        botBubble.classList.add("error");
      }
    } finally {
      botBubble.classList.remove("streaming-cursor");
      setTyping(false);
      setLoading(false);
      abortCtrl = null;
      inputEl.focus();
    }
  }

  /* ── Clear Chat / Reset ──────────────────────────────────────── */

  /** Build the welcome card HTML and re-attach chip listeners */
  function buildWelcomeCard() {
    const card = document.createElement("div");
    card.className = "welcome-card";
    card.id        = "welcome-card";
    card.innerHTML = `
      <div class="welcome-icon" aria-hidden="true">
        <svg viewBox="0 0 28 28" fill="none" stroke="#fff" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3,20 9,12 14,15 22,6"/>
          <circle cx="22" cy="6" r="2" fill="#fff" stroke="none"/>
        </svg>
      </div>
      <h2>Your private financial advisor</h2>
      <p>Ask me anything about investing, budgeting, taxes, or personal finance.<br>
         All processing happens <strong>locally on your machine</strong> — zero data leaves your device.</p>
      <div class="welcome-chips">
        <button class="chip" data-prompt="What are the best strategies to save money each month?"><span class="chip-icon">💡</span> Saving strategies</button>
        <button class="chip" data-prompt="Explain index funds for a beginner."><span class="chip-icon">📈</span> Index funds 101</button>
        <button class="chip" data-prompt="How do I build an emergency fund?"><span class="chip-icon">🏦</span> Emergency fund</button>
        <button class="chip" data-prompt="What is the difference between a Roth IRA and a traditional IRA?"><span class="chip-icon">🔄</span> Roth vs Traditional IRA</button>
        <button class="chip" data-prompt="How should I diversify my investment portfolio?"><span class="chip-icon">📊</span> Portfolio diversification</button>
        <button class="chip" data-prompt="What are the tax advantages of a 401(k)?"><span class="chip-icon">🧾</span> 401(k) tax benefits</button>
      </div>`;

    card.querySelectorAll(".chip").forEach(c => {
      c.addEventListener("click", () => sendMessage(c.dataset.prompt));
    });

    return card;
  }

  async function clearChat() {
    if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }

    // Tell the backend to reset conversation history
    try { await fetch("/reset", { method: "POST" }); } catch (_) { /* ignore */ }

    // Animate messages out
    messagesEl.style.transition = "opacity 0.25s ease, transform 0.25s ease";
    messagesEl.style.opacity    = "0";
    messagesEl.style.transform  = "translateY(6px)";

    setTimeout(() => {
      messagesEl.innerHTML = "";
      messagesEl.style.transition = "opacity 0.3s ease, transform 0.3s ease";
      messagesEl.style.opacity    = "1";
      messagesEl.style.transform  = "translateY(0)";

      messagesEl.appendChild(buildWelcomeCard());

      setLoading(false);
      setTyping(false);
      inputEl.value = "";
      autoResize();
      inputEl.focus();
    }, 280);
  }

  /* ── Events ──────────────────────────────────────────────────── */

  sendBtn.addEventListener("click", () => sendMessage());

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener("input", autoResize);

  clearBtn.addEventListener("click", () => clearChat());

  // Chip clicks on the initial welcome card
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => sendMessage(chip.dataset.prompt));
  });

  /* ── Init ────────────────────────────────────────────────────── */
  inputEl.focus();

})();
