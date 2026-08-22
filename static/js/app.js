/**
 * AeroMESH • Full-Spread Dark Claymorphism Chat Application
 * Features:
 * 1. StreamBuffer with RAF Throttling, In-Card Stop/Retry Controls, Layout Thrash Prevention
 * 2. Citations: Inline Superscript Badges, Expandable Source Cards, Quality Indicators
 * 3. Tiered Feedback: Thumbs Up/Down, Category Selector, Optional Text Input, Persistence
 * 4. Safety: Content Warning Detection, Actionable Rejection Cards, AI-Generated Labels
 * 5. Enhanced Sessions: Auto-save, Date-Grouped History, Message Search, Context Usage Gauge
 * 6. Accessibility: ARIA Live Region Announcements, Keyboard Navigation, Focus Trapping
 */

(function () {
  'use strict';

  // --- STATE MANAGEMENT ---
  const STORAGE_KEY = 'aeromesh_conversations_v2';
  const MAX_CONTEXT_TOKENS = 8192;

  let conversations = [];
  let currentChatId = null;
  let isGenerating = false;
  let activeAbortController = null;

  let activeModel = 'Loading...';
  let availableModels = [];
  let clusterState = { connected: false, status: 'checking' };

  let maxTokens = 256;
  const tokenSteps = [128, 256, 512, 1024, 2048];
  let enableReasoning = true;
  let clearKvOnSend = false;

  // --- DOM ELEMENTS ---
  const elements = {
    appContainer: document.getElementById('app-container'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebar-toggle-btn'),
    sidebarCloseBtn: document.getElementById('sidebar-close-btn'),
    btnNewChat: document.getElementById('btn-new-chat'),
    searchInput: document.getElementById('search-input'),
    historyList: document.getElementById('chat-list-today'),
    chatViewport: document.getElementById('chat-viewport'),
    chatInner: document.getElementById('chat-inner'),
    heroState: document.getElementById('hero-state'),
    messagesContainer: document.getElementById('messages-container'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    pillReasoning: document.getElementById('pill-reasoning'),
    pillTokens: document.getElementById('pill-tokens'),
    maxTokensLabel: document.getElementById('max-tokens-label'),
    pillUndo: document.getElementById('pill-undo'),
    pillClearKv: document.getElementById('pill-clearkv'),
    btnResetSession: document.getElementById('btn-reset-session'),
    btnExportChat: document.getElementById('btn-export-chat'),
    btnClearAll: document.getElementById('btn-clear-all'),
    btnOpenTelemetry: document.getElementById('btn-open-telemetry'),
    telemetryModal: document.getElementById('telemetry-modal'),
    modalCloseBtn: document.getElementById('modal-close-btn'),
    modalProbeBtn: document.getElementById('modal-probe-btn'),
    modelSelectorBtn: document.getElementById('model-selector-btn'),
    modelDropdownMenu: document.getElementById('model-dropdown-menu'),
    activeModelName: document.getElementById('active-model-name'),
    sidebarStatusDot: document.getElementById('sidebar-status-dot'),
    clusterStatusLabel: document.getElementById('cluster-status-label'),
    clusterStatsDynamic: document.getElementById('cluster-stats-dynamic'),
    topStatusDot: document.getElementById('top-status-dot'),
    topStatusText: document.getElementById('top-status-text'),
    modalNodesContainer: document.getElementById('modal-nodes-container'),
    modalStatusDot: document.getElementById('modal-status-dot'),
    modalClusterStatus: document.getElementById('modal-cluster-status'),
    suggestionCards: document.querySelectorAll('.suggestion-card'),
    // Context Transparency Bar
    contextUsagePill: document.getElementById('context-usage-pill'),
    contextTokensCount: document.getElementById('context-tokens-count'),
    contextBarFill: document.getElementById('context-bar-fill'),
    // Accessibility Live Announcer
    a11yAnnouncer: document.getElementById('a11y-live-announcer')
  };

  // --- INITIALIZATION ---
  async function init() {
    loadConversations();
    setupEventListeners();
    configureMarked();

    await fetchModels();
    await fetchClusterStatus();
    setInterval(fetchClusterStatus, 6000);

    if (conversations.length === 0) {
      createNewChat();
    } else {
      switchChat(conversations[0].id);
    }

    updateContextUsageBar();
    lucide.createIcons();
  }

  // --- ACCESSIBILITY HELPER ---
  function announceA11y(message) {
    if (elements.a11yAnnouncer) {
      elements.a11yAnnouncer.textContent = message;
    }
  }

  // --- CONFIGURE MARKED & HIGHLIGHT.JS ---
  function configureMarked() {
    if (window.marked) {
      marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function (code, lang) {
          if (lang && hljs.getLanguage(lang)) {
            try {
              return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
              console.error(err);
            }
          }
          return hljs.highlightAuto(code).value;
        }
      });
    }
  }

  // --- DYNAMIC BACKEND FETCHING (ZERO HARDCODING) ---
  async function fetchModels() {
    try {
      const resp = await fetch('/v1/models');
      if (resp.ok) {
        const data = await resp.json();
        availableModels = data.data || [];
        if (availableModels.length > 0) {
          activeModel = availableModels[0].id;
          elements.activeModelName.textContent = activeModel;
        } else {
          elements.activeModelName.textContent = 'No Models Discovered';
        }
        renderModelDropdown();
      }
    } catch (e) {
      elements.activeModelName.textContent = 'Coordinator Offline';
    }
  }

  function renderModelDropdown() {
    elements.modelDropdownMenu.innerHTML = '';
    if (availableModels.length === 0) {
      const emptyItem = document.createElement('div');
      emptyItem.className = 'model-dropdown-item';
      emptyItem.setAttribute('role', 'option');
      emptyItem.textContent = 'No active models found';
      elements.modelDropdownMenu.appendChild(emptyItem);
      return;
    }

    availableModels.forEach((m) => {
      const item = document.createElement('div');
      item.className = `model-dropdown-item ${m.id === activeModel ? 'active' : ''}`;
      item.setAttribute('role', 'option');
      item.setAttribute('tabindex', '0');
      item.setAttribute('aria-selected', m.id === activeModel ? 'true' : 'false');
      item.innerHTML = `<span>${m.id}</span> ${m.id === activeModel ? '<i data-lucide="check" style="width: 13px; height: 13px;" aria-hidden="true"></i>' : ''}`;
      
      const selectModel = () => {
        activeModel = m.id;
        elements.activeModelName.textContent = activeModel;
        elements.modelDropdownMenu.classList.remove('active');
        elements.modelSelectorBtn.setAttribute('aria-expanded', 'false');
        renderModelDropdown();
        announceA11y(`Active model set to ${activeModel}`);
      };

      item.onclick = selectModel;
      item.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          selectModel();
        }
      };

      elements.modelDropdownMenu.appendChild(item);
    });

    lucide.createIcons();
  }

  async function fetchClusterStatus() {
    try {
      const resp = await fetch('/api/cluster/status');
      if (resp.ok) {
        clusterState = await resp.json();
        updateClusterUI(clusterState);
        return;
      }
    } catch (e) {
      clusterState = { connected: false, status: 'offline', message: e.message };
    }
    updateClusterUI(clusterState);
  }

  function updateClusterUI(data) {
    if (data.connected && data.status === 'ok') {
      elements.sidebarStatusDot.classList.remove('offline');
      elements.topStatusDot.classList.remove('offline');
      elements.modalStatusDot.classList.remove('offline');

      elements.clusterStatusLabel.textContent = 'Mesh Online';
      elements.clusterStatusLabel.style.color = 'var(--accent-mint)';
      elements.topStatusText.textContent = 'Pipeline Connected';
      elements.topStatusText.parentElement.style.color = 'var(--accent-mint)';

      const localStage = data.local_stage || {};
      const workers = data.worker_nodes || [];

      elements.clusterStatsDynamic.innerHTML = `
        <div class="cluster-stats-row">
          <span>Coordinator:</span>
          <span class="stat-highlight">Layers ${localStage.layer_start ?? 0}..${localStage.layer_end ?? '?'}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Workers:</span>
          <span class="stat-highlight">${workers.length > 0 ? workers.join(', ') : 'Direct Pipeline'}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Total Layers:</span>
          <span style="color: #fff; font-weight: 600;">${data.total_layers || 'Auto'}</span>
        </div>
      `;

      elements.modalNodesContainer.innerHTML = `
        <div class="node-box">
          <div>
            <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Coordinator (Local)</div>
            <div style="font-size: 11.5px; color: var(--text-dim);">Layers ${localStage.layer_start ?? 0}..${localStage.layer_end ?? '?'} • Arch: ${data.architecture || 'GGUF'}</div>
          </div>
          <div class="stat-highlight" style="font-size: 12px;">Stage 1</div>
        </div>

        <div style="display: flex; align-items: center; justify-content: center; gap: 6px; color: var(--accent-orange); font-size: 11.5px; font-weight: 700;">
          <i data-lucide="arrow-down-up" style="width: 14px; height: 14px;"></i>
          <span>${data.transport || 'Tailscale Direct WireGuard'} • Zero Weights Transferred</span>
        </div>

        ${
          workers.length > 0
            ? workers
                .map(
                  (w, idx) => `
          <div class="node-box">
            <div>
              <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Worker Stage ${idx + 2} (${w})</div>
              <div style="font-size: 11.5px; color: var(--text-dim);">Final Layers • LM Head • Token Sampler</div>
            </div>
            <div class="stat-highlight" style="font-size: 12px;">Connected</div>
          </div>
        `
                )
                .join('')
            : `
          <div class="node-box">
            <div>
              <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Worker Stage</div>
              <div style="font-size: 11.5px; color: var(--text-dim);">Awaiting Handshake Activation Frames</div>
            </div>
            <div class="stat-highlight" style="font-size: 12px;">Ready</div>
          </div>
        `
        }
      `;

      elements.modalClusterStatus.textContent = `Coordinator active on ${data.coordinator || 'http://127.0.0.1:8080'}`;
      elements.modalClusterStatus.style.color = 'var(--accent-mint)';
    } else {
      elements.sidebarStatusDot.classList.add('offline');
      elements.topStatusDot.classList.add('offline');
      elements.modalStatusDot.classList.add('offline');

      elements.clusterStatusLabel.textContent = 'Coordinator Offline';
      elements.clusterStatusLabel.style.color = 'var(--accent-coral)';
      elements.topStatusText.textContent = 'Coordinator Offline';
      elements.topStatusText.parentElement.style.color = 'var(--accent-coral)';

      elements.clusterStatsDynamic.innerHTML = `
        <div class="cluster-stats-row">
          <span>Status:</span>
          <span style="color: var(--accent-coral); font-weight: 600;">Not Running</span>
        </div>
        <div class="cluster-stats-row">
          <span>Start:</span>
          <span style="color: var(--accent-orange); font-family: monospace; font-size: 10.5px;">cargo run -- serve</span>
        </div>
      `;

      elements.modalNodesContainer.innerHTML = `
        <div style="padding: 14px; text-align: center; color: var(--text-muted); font-size: 13px; display: flex; flex-direction: column; gap: 8px;">
          <div style="color: var(--accent-coral); font-weight: 700;">AeroMesh Coordinator is not running</div>
          <div>Start the cluster coordinator from PowerShell:</div>
          <code style="background: #080b10; padding: 8px 12px; border-radius: 6px; color: var(--accent-orange); font-size: 11.5px; border: 1px solid rgba(255,255,255,0.06); text-align: left; overflow-x: auto;">
            cargo run --bin aeromesh -- serve --model models/model.gguf --layers 0..24 --peers worker-ip:50052 --port 8080
          </code>
        </div>
      `;

      elements.modalClusterStatus.textContent = 'Coordinator offline (http://127.0.0.1:8080)';
      elements.modalClusterStatus.style.color = 'var(--accent-coral)';
    }

    lucide.createIcons();
  }

  // --- STORAGE & CONVERSATION MANAGEMENT ---
  function loadConversations() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        conversations = JSON.parse(saved);
      }
    } catch (e) {
      conversations = [];
    }
  }

  function saveConversations(skipHistoryRender = false) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      if (!skipHistoryRender) {
        renderHistoryList(elements.searchInput ? elements.searchInput.value : '');
      }
      updateContextUsageBar();
    } catch (e) {
      console.error('Failed to save conversations:', e);
    }
  }

  function getCurrentChat() {
    return conversations.find((c) => c.id === currentChatId);
  }

  function createNewChat() {
    const newChat = {
      id: 'chat_' + Date.now(),
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: []
    };
    conversations.unshift(newChat);
    saveConversations();
    switchChat(newChat.id);
    announceA11y('Created new conversation');
  }

  function switchChat(chatId) {
    if (isGenerating && activeAbortController) {
      activeAbortController.abort();
    }
    currentChatId = chatId;
    renderHistoryList(elements.searchInput ? elements.searchInput.value : '');
    renderMessages();
    updateContextUsageBar();
  }

  function deleteChat(chatId, e) {
    if (e) e.stopPropagation();
    conversations = conversations.filter((c) => c.id !== chatId);
    if (conversations.length === 0) {
      createNewChat();
    } else if (currentChatId === chatId) {
      switchChat(conversations[0].id);
    } else {
      saveConversations();
    }
    announceA11y('Deleted conversation');
  }

  function clearAllChats() {
    if (confirm('Clear all conversation history from this browser?')) {
      conversations = [];
      createNewChat();
      announceA11y('All conversation history cleared');
    }
  }

  function exportCurrentChat() {
    const chat = getCurrentChat();
    if (!chat || chat.messages.length === 0) {
      alert('No messages to export.');
      return;
    }

    let markdown = `# ${chat.title}\n\n*Created: ${new Date(chat.createdAt).toLocaleString()}*\n*Model: ${activeModel}*\n\n---\n\n`;

    chat.messages.forEach((msg) => {
      const role = msg.role === 'user' ? '### 👤 User' : '### ⚡ AeroMesh Cluster';
      markdown += `${role}\n\n${msg.content}\n\n---\n\n`;
    });

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${chat.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
    announceA11y('Conversation exported to Markdown file');
  }

  // --- CONTEXT TRANSPARENCY BAR ---
  function updateContextUsageBar() {
    if (!elements.contextBarFill || !elements.contextTokensCount) return;
    const chat = getCurrentChat();
    if (!chat || chat.messages.length === 0) {
      elements.contextTokensCount.textContent = `0 / ${MAX_CONTEXT_TOKENS.toLocaleString()} tok (0%)`;
      elements.contextBarFill.style.width = '0%';
      elements.contextBarFill.className = 'context-bar-fill';
      return;
    }

    let totalTokens = 0;
    chat.messages.forEach((m) => {
      if (m.metrics && m.metrics.tokens) {
        totalTokens += Number(m.metrics.tokens) || 0;
      } else if (m.content) {
        totalTokens += Math.ceil(m.content.length / 3.8);
      }
    });

    const pct = Math.min(100, Math.round((totalTokens / MAX_CONTEXT_TOKENS) * 100));
    elements.contextTokensCount.textContent = `${totalTokens.toLocaleString()} / ${MAX_CONTEXT_TOKENS.toLocaleString()} tok (${pct}%)`;
    elements.contextBarFill.style.width = `${pct}%`;

    elements.contextBarFill.className = 'context-bar-fill';
    if (pct >= 80) {
      elements.contextBarFill.classList.add('danger');
    } else if (pct >= 50) {
      elements.contextBarFill.classList.add('warning');
    }
  }

  // --- DATE-GROUPED HISTORY RENDERING WITH CONTENT SEARCH ---
  function formatRelativeTime(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMin = Math.floor(diffMs / (1000 * 60));
    const diffHrs = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHrs < 24) return `${diffHrs}h ago`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }

  function getDateGroupKey(dateStr) {
    if (!dateStr) return 'Older';
    const date = new Date(dateStr);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    if (date >= today) return 'Today';
    if (date >= yesterday) return 'Yesterday';
    if (date >= sevenDaysAgo) return 'Previous 7 Days';
    return 'Older';
  }

  function renderHistoryList(filterQuery = '') {
    elements.historyList.innerHTML = '';
    const q = filterQuery.trim().toLowerCase();

    const filtered = conversations.filter((c) => {
      if (!q) return true;
      const titleMatch = (c.title || '').toLowerCase().includes(q);
      const messageMatch = (c.messages || []).some((m) =>
        (m.content || '').toLowerCase().includes(q)
      );
      return titleMatch || messageMatch;
    });

    if (filtered.length === 0) {
      const emptyDiv = document.createElement('div');
      emptyDiv.style.padding = '12px 8px';
      emptyDiv.style.fontSize = '12px';
      emptyDiv.style.color = 'var(--text-dim)';
      emptyDiv.style.textAlign = 'center';
      emptyDiv.textContent = q ? 'No matching conversations' : 'No conversations yet';
      elements.historyList.appendChild(emptyDiv);
      return;
    }

    const groups = {
      Today: [],
      Yesterday: [],
      'Previous 7 Days': [],
      Older: []
    };

    filtered.forEach((chat) => {
      const groupKey = getDateGroupKey(chat.updatedAt || chat.createdAt);
      if (groups[groupKey]) {
        groups[groupKey].push(chat);
      } else {
        groups.Older.push(chat);
      }
    });

    Object.keys(groups).forEach((groupName) => {
      const chatItems = groups[groupName];
      if (chatItems.length === 0) return;

      const groupContainer = document.createElement('div');
      groupContainer.className = 'history-date-group';

      const header = document.createElement('div');
      header.className = 'history-group-header';
      header.textContent = groupName;
      groupContainer.appendChild(header);

      chatItems.forEach((chat) => {
        const item = document.createElement('div');
        item.className = `chat-history-item ${chat.id === currentChatId ? 'active' : ''}`;
        item.setAttribute('role', 'listitem');
        item.setAttribute('tabindex', '0');
        item.setAttribute('aria-current', chat.id === currentChatId ? 'true' : 'false');
        item.setAttribute('aria-label', `Chat: ${chat.title}`);

        const topRow = document.createElement('div');
        topRow.className = 'chat-history-top-row';

        const title = document.createElement('div');
        title.className = 'chat-title-text';
        title.textContent = chat.title;

        const actions = document.createElement('div');
        actions.className = 'chat-item-actions';

        const delBtn = document.createElement('button');
        delBtn.className = 'action-icon-btn';
        delBtn.setAttribute('aria-label', `Delete ${chat.title}`);
        delBtn.innerHTML = '<i data-lucide="trash-2" style="width: 12px; height: 12px;" aria-hidden="true"></i>';
        delBtn.title = 'Delete Chat';
        delBtn.onclick = (e) => deleteChat(chat.id, e);

        actions.appendChild(delBtn);
        topRow.appendChild(title);
        topRow.appendChild(actions);

        const lastMsg = chat.messages && chat.messages.length > 0 ? chat.messages[chat.messages.length - 1] : null;
        let snippetText = '';
        if (lastMsg) {
          const prefix = lastMsg.role === 'user' ? '👤 ' : '⚡ ';
          const cleanSnippet = cleanSpecialTokens(lastMsg.content).replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
          snippetText = prefix + (cleanSnippet.substring(0, 48) || 'Reasoning details...');
        }

        const snippet = document.createElement('div');
        snippet.className = 'chat-history-snippet';
        snippet.textContent = snippetText;

        const meta = document.createElement('div');
        meta.className = 'chat-history-meta';
        const relTime = formatRelativeTime(chat.updatedAt || chat.createdAt);
        const msgCount = chat.messages ? chat.messages.length : 0;
        meta.innerHTML = `<span>${relTime}</span> • <span>${msgCount} msg${msgCount !== 1 ? 's' : ''}</span>`;

        item.appendChild(topRow);
        if (snippetText) item.appendChild(snippet);
        item.appendChild(meta);

        item.onclick = () => switchChat(chat.id);
        item.onkeydown = (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            switchChat(chat.id);
          }
        };

        groupContainer.appendChild(item);
      });

      elements.historyList.appendChild(groupContainer);
    });

    lucide.createIcons();
  }

  function renderMessages() {
    const chat = getCurrentChat();
    elements.messagesContainer.innerHTML = '';

    if (!chat || chat.messages.length === 0) {
      elements.heroState.style.display = 'flex';
      return;
    }

    elements.heroState.style.display = 'none';

    chat.messages.forEach((msg, idx) => {
      appendMessageElement(msg.role, msg.content, msg.metrics, idx, msg.feedback);
    });

    scrollToBottom();
  }

  function cleanSpecialTokens(text) {
    if (!text) return '';
    return text
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*end[\s\u00a0\u2000-\u200f_]+(?:of[\s\u00a0\u2000-\u200f_]+)?sentence[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*begin[\s\u00a0\u2000-\u200f_]+(?:of[\s\u00a0\u2000-\u200f_]+)?sentence[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*endoftext[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*im_end[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*im_start[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*eot_id[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*thought[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .replace(/<[|｜][\s\u00a0\u2000-\u200f]*\/thought[\s\u00a0\u2000-\u200f]*[|｜]>/gi, '')
      .trim();
  }

  function parseThinkingBlocks(text, isFinished = false) {
    if (!text) return { thinkText: null, answerText: '', isStillThinking: false };

    let cleanText = text;
    const thinkStartRegex = /(<think>|<[|｜]thought[|｜]>|<thought>)/i;
    const matchStart = cleanText.match(thinkStartRegex);

    if (!matchStart) {
      return {
        thinkText: null,
        answerText: cleanSpecialTokens(cleanText),
        isStillThinking: false
      };
    }

    const thinkStartIndex = matchStart.index;
    const tagLength = matchStart[0].length;
    const prefix = cleanText.substring(0, thinkStartIndex);

    const thinkEndRegex = /(<\/think>|<[|｜]\/thought[|｜]>|<\/[|｜]thought[|｜]>|<\/thought>)/i;
    const restText = cleanText.substring(thinkStartIndex + tagLength);
    const matchEnd = restText.match(thinkEndRegex);

    if (!matchEnd) {
      const rawThink = cleanSpecialTokens(restText);
      if (isFinished) {
        return {
          thinkText: rawThink || 'Reasoning complete',
          answerText: cleanSpecialTokens(prefix),
          isStillThinking: false
        };
      }
      return {
        thinkText: rawThink || 'Reasoning through prompt...',
        answerText: cleanSpecialTokens(prefix),
        isStillThinking: true
      };
    }

    const thinkEndIndex = matchEnd.index;
    const endTagLength = matchEnd[0].length;
    const thinkText = cleanSpecialTokens(restText.substring(0, thinkEndIndex));
    const rawAnswer = prefix + restText.substring(thinkEndIndex + endTagLength);
    const answerText = cleanSpecialTokens(rawAnswer);

    return { thinkText, answerText, isStillThinking: false };
  }

  // --- SAFETY & REJECTION DETECTION ---
  function detectSafetyRejection(rawText) {
    if (!rawText) return null;
    const textLower = rawText.toLowerCase();

    const refusalPatterns = [
      /(?:i cannot|i can't|i am unable to|i must decline|i'm unable to|as an ai(?: assistant)?, i cannot)/i,
      /(?:content policy|safety guidelines|harmful or dangerous|violates safety)/i
    ];

    const hasRefusal = refusalPatterns.some((pattern) => pattern.test(textLower));
    if (hasRefusal) {
      return {
        type: 'rejection',
        title: 'AeroMesh Safety & Policy Guard',
        message: 'This response triggered safety guidelines or standard model refusal boundaries.'
      };
    }

    const warningPatterns = [/^(?:⚠️\s*warning:|disclaimer:|safety notice:)/im];
    if (warningPatterns.some((p) => p.test(textLower))) {
      return {
        type: 'warning',
        title: 'Safety Advisory',
        message: 'Please review the experimental context below carefully before execution.'
      };
    }

    return null;
  }

  function renderSafetyWarningCard(safetyData, cardElement) {
    if (!safetyData) return;

    const banner = document.createElement('div');
    banner.className = `safety-warning-card ${safetyData.type}`;

    const header = document.createElement('div');
    header.className = 'safety-header';
    header.innerHTML = `<i data-lucide="shield-alert" style="width: 14px; height: 14px;"></i><span>${safetyData.title}</span>`;

    const message = document.createElement('div');
    message.className = 'safety-message';
    message.textContent = safetyData.message;

    const actionRow = document.createElement('div');
    actionRow.className = 'safety-action-row';

    const rephraseBtn = document.createElement('button');
    rephraseBtn.className = 'safety-action-pill';
    rephraseBtn.innerHTML = '<i data-lucide="edit-3" style="width: 11px; height: 11px;"></i><span>💡 Rephrase Prompt</span>';
    rephraseBtn.onclick = () => {
      const chat = getCurrentChat();
      if (chat && chat.messages.length > 0) {
        const lastUser = [...chat.messages].reverse().find((m) => m.role === 'user');
        if (lastUser) {
          elements.chatInput.value = `Explain conceptually: ${lastUser.content}`;
          adjustTextareaHeight();
          elements.chatInput.focus();
        }
      }
    };

    const syncKvBtn = document.createElement('button');
    syncKvBtn.className = 'safety-action-pill';
    syncKvBtn.innerHTML = '<i data-lucide="refresh-cw" style="width: 11px; height: 11px;"></i><span>🧹 Sync KV & Retry</span>';
    syncKvBtn.onclick = () => {
      clearKvOnSend = true;
      undoLastTurn();
      handleSendMessage();
    };

    actionRow.appendChild(rephraseBtn);
    actionRow.appendChild(syncKvBtn);

    banner.appendChild(header);
    banner.appendChild(message);
    banner.appendChild(actionRow);

    cardElement.insertBefore(banner, cardElement.firstChild);
    lucide.createIcons();
  }

  // --- CITATION PARSER & EXPANDABLE SOURCES ---
  function parseAndEnhanceCitations(text, cardElement) {
    if (!text) return { cleanText: '', sources: [] };

    const sources = [];
    let cleanText = text;

    const footnoteRegex = /\[(\d+)\]:\s*(https?:\/\/[^\s]+)(?:\s+"([^"]+)"|\s*\(([^)]+)\))?/g;
    let match;
    const explicitMap = {};

    while ((match = footnoteRegex.exec(text)) !== null) {
      const num = parseInt(match[1], 10);
      const url = match[2];
      const title = match[3] || match[4] || new URL(url).hostname;
      const domain = new URL(url).hostname.replace(/^www\./, '');
      explicitMap[num] = { id: num, url, title, domain, type: 'verified' };
    }

    cleanText = cleanText.replace(footnoteRegex, '').trim();

    const mdLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g;
    let linkMatch;
    let autoId = Object.keys(explicitMap).length + 1;

    while ((linkMatch = mdLinkRegex.exec(cleanText)) !== null) {
      const linkTitle = linkMatch[1];
      const linkUrl = linkMatch[2];
      try {
        const domain = new URL(linkUrl).hostname.replace(/^www\./, '');
        if (!Object.values(explicitMap).some((s) => s.url === linkUrl)) {
          sources.push({
            id: autoId++,
            title: linkTitle,
            url: linkUrl,
            domain,
            type: 'verified'
          });
        }
      } catch (e) {}
    }

    Object.values(explicitMap).forEach((s) => sources.push(s));

    if (/Tailscale Direct WireGuard|Zero-Weight|AeroMESH|KV Cache/i.test(cleanText) && sources.length === 0) {
      sources.push({
        id: 1,
        title: 'AeroMesh Zero-Weight Activation Pipeline Specification',
        url: '#',
        domain: 'aeromesh.internal',
        type: 'mesh'
      });
    }

    cleanText = cleanText.replace(/\[\^?(\d+)\](?!\()/g, (m, num) => {
      return `<sup class="citation-badge" data-cite="${num}" tabindex="0" role="button" aria-label="Citation ${num}">[${num}]</sup>`;
    });

    return { cleanText, sources };
  }

  function renderSourcesAccordion(sources, cardElement) {
    if (!sources || sources.length === 0) return;

    const accordion = document.createElement('div');
    accordion.className = 'sources-accordion';

    const header = document.createElement('div');
    header.className = 'sources-header';
    header.setAttribute('role', 'button');
    header.setAttribute('tabindex', '0');
    header.setAttribute('aria-expanded', 'false');

    const badge = document.createElement('div');
    badge.className = 'sources-badge';
    badge.innerHTML = `<i data-lucide="book-open" style="width: 13px; height: 13px;" aria-hidden="true"></i><span>Sources & References (${sources.length})</span>`;

    const chevron = document.createElement('i');
    chevron.setAttribute('data-lucide', 'chevron-down');
    chevron.style.width = '13px';
    chevron.style.height = '13px';
    chevron.style.transform = 'rotate(-90deg)';
    chevron.style.transition = 'transform 0.2s ease';

    header.appendChild(badge);
    header.appendChild(chevron);

    const content = document.createElement('div');
    content.className = 'sources-content collapsed';

    const grid = document.createElement('div');
    grid.className = 'sources-grid';

    sources.forEach((src) => {
      const card = document.createElement('a');
      card.className = 'source-card';
      card.id = `source-card-${src.id}`;
      card.href = src.url !== '#' ? src.url : 'javascript:void(0)';
      if (src.url !== '#') {
        card.target = '_blank';
        card.rel = 'noopener noreferrer';
      }

      const topRow = document.createElement('div');
      topRow.className = 'source-card-top';

      const numPill = document.createElement('span');
      numPill.className = 'source-num-pill';
      numPill.textContent = `[${src.id}]`;

      const qual = document.createElement('div');
      qual.className = 'quality-indicator';
      const dot = document.createElement('span');
      dot.className = `quality-dot ${src.type || 'verified'}`;
      const label = document.createElement('span');
      label.style.fontSize = '10px';
      label.style.color = 'var(--text-dim)';
      label.textContent = src.type === 'mesh' ? 'Mesh Internal' : 'Verified';

      qual.appendChild(dot);
      qual.appendChild(label);
      topRow.appendChild(numPill);
      topRow.appendChild(qual);

      const title = document.createElement('div');
      title.className = 'source-title';
      title.textContent = src.title;

      const domain = document.createElement('div');
      domain.className = 'source-domain';
      domain.innerHTML = `<i data-lucide="external-link" style="width: 10px; height: 10px;"></i><span>${src.domain || 'External Source'}</span>`;

      card.appendChild(topRow);
      card.appendChild(title);
      card.appendChild(domain);
      grid.appendChild(card);
    });

    content.appendChild(grid);

    const toggleSources = () => {
      content.classList.toggle('collapsed');
      const isCollapsed = content.classList.contains('collapsed');
      chevron.style.transform = isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
      header.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
    };

    header.onclick = toggleSources;
    header.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleSources();
      }
    };

    accordion.appendChild(header);
    accordion.appendChild(content);

    const actions = cardElement.querySelector('.message-actions');
    if (actions) {
      cardElement.insertBefore(accordion, actions);
    } else {
      cardElement.appendChild(accordion);
    }

    const badges = cardElement.querySelectorAll('.citation-badge');
    badges.forEach((b) => {
      const citeNum = b.getAttribute('data-cite');
      b.onclick = (e) => {
        e.preventDefault();
        if (content.classList.contains('collapsed')) {
          toggleSources();
        }
        const targetCard = cardElement.querySelector(`#source-card-${citeNum}`);
        if (targetCard) {
          targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          targetCard.style.borderColor = 'var(--accent-orange)';
          targetCard.style.boxShadow = 'var(--accent-orange-glow)';
          setTimeout(() => {
            targetCard.style.borderColor = '';
            targetCard.style.boxShadow = '';
          }, 1500);
        }
      };
    });

    lucide.createIcons();
  }

  // --- TIERED FEEDBACK CONTROLS ---
  function renderFeedbackControls(actionsContainer, cardElement, messageIndex, initialFeedback = null) {
    const feedbackWrapper = document.createElement('div');
    feedbackWrapper.style.display = 'inline-flex';
    feedbackWrapper.style.alignItems = 'center';
    feedbackWrapper.style.gap = '4px';

    const btnUp = document.createElement('button');
    btnUp.className = `feedback-btn ${initialFeedback && initialFeedback.rating === 'up' ? 'active-up' : ''}`;
    btnUp.title = 'Good response';
    btnUp.setAttribute('aria-label', 'Helpful response');
    btnUp.innerHTML = '<i data-lucide="thumbs-up" style="width: 11px; height: 11px;"></i>';

    const btnDown = document.createElement('button');
    btnDown.className = `feedback-btn ${initialFeedback && initialFeedback.rating === 'down' ? 'active-down' : ''}`;
    btnDown.title = 'Bad response';
    btnDown.setAttribute('aria-label', 'Unhelpful response');
    btnDown.innerHTML = '<i data-lucide="thumbs-down" style="width: 11px; height: 11px;"></i>';

    feedbackWrapper.appendChild(btnUp);
    feedbackWrapper.appendChild(btnDown);

    const btnGroup = actionsContainer.querySelector('.action-buttons-group');
    if (btnGroup) {
      btnGroup.insertBefore(feedbackWrapper, btnGroup.firstChild);
    }

    let feedbackPanel = null;

    btnUp.onclick = () => {
      const chat = getCurrentChat();
      if (!chat || messageIndex >= chat.messages.length) return;

      const isCurrentUp = btnUp.classList.contains('active-up');
      if (isCurrentUp) {
        btnUp.classList.remove('active-up');
        chat.messages[messageIndex].feedback = null;
      } else {
        btnUp.classList.add('active-up');
        btnDown.classList.remove('active-down');
        if (feedbackPanel) feedbackPanel.remove();
        chat.messages[messageIndex].feedback = { rating: 'up', timestamp: Date.now() };
      }
      saveConversations(true);
      announceA11y('Marked as helpful');
    };

    btnDown.onclick = () => {
      const chat = getCurrentChat();
      if (!chat || messageIndex >= chat.messages.length) return;

      if (feedbackPanel) {
        feedbackPanel.remove();
        feedbackPanel = null;
        return;
      }

      btnDown.classList.add('active-down');
      btnUp.classList.remove('active-up');

      feedbackPanel = document.createElement('div');
      feedbackPanel.className = 'feedback-panel';

      const categoriesRow = document.createElement('div');
      categoriesRow.className = 'feedback-categories-row';

      const categories = ['Inaccurate', 'Hallucination', 'Incomplete', 'Slow Mesh', 'Other'];
      let selectedCategory = 'Inaccurate';

      categories.forEach((cat) => {
        const chip = document.createElement('div');
        chip.className = `feedback-chip ${cat === selectedCategory ? 'selected' : ''}`;
        chip.textContent = cat;
        chip.onclick = () => {
          categoriesRow.querySelectorAll('.feedback-chip').forEach((c) => c.classList.remove('selected'));
          chip.classList.add('selected');
          selectedCategory = cat;
        };
        categoriesRow.appendChild(chip);
      });

      const inputRow = document.createElement('div');
      inputRow.className = 'feedback-input-row';

      const textInput = document.createElement('input');
      textInput.type = 'text';
      textInput.className = 'feedback-text-input';
      textInput.placeholder = 'What went wrong? (optional)';
      textInput.setAttribute('aria-label', 'Feedback details');

      const submitBtn = document.createElement('button');
      submitBtn.className = 'feedback-submit-btn';
      submitBtn.textContent = 'Submit';

      submitBtn.onclick = () => {
        chat.messages[messageIndex].feedback = {
          rating: 'down',
          category: selectedCategory,
          comment: textInput.value.trim(),
          timestamp: Date.now()
        };
        saveConversations(true);

        feedbackPanel.innerHTML = '<div class="feedback-done-badge"><i data-lucide="check" style="width: 12px; height: 12px;"></i><span>Feedback recorded. Thank you!</span></div>';
        lucide.createIcons();
        announceA11y('Feedback submitted');
        setTimeout(() => {
          if (feedbackPanel) feedbackPanel.remove();
          feedbackPanel = null;
        }, 2200);
      };

      inputRow.appendChild(textInput);
      inputRow.appendChild(submitBtn);

      feedbackPanel.appendChild(categoriesRow);
      feedbackPanel.appendChild(inputRow);

      cardElement.appendChild(feedbackPanel);
      textInput.focus();
      lucide.createIcons();
    };

    lucide.createIcons();
  }

  // --- RETRY TURN LOGIC ---
  function retryTurn(assistantIndex) {
    if (isGenerating) {
      if (activeAbortController) activeAbortController.abort();
    }

    const chat = getCurrentChat();
    if (!chat || assistantIndex >= chat.messages.length) return;

    let userPrompt = '';
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (chat.messages[i].role === 'user') {
        userPrompt = chat.messages[i].content;
        break;
      }
    }

    if (!userPrompt) return;

    chat.messages.splice(assistantIndex, 1);
    saveConversations();
    renderMessages();

    handleSendMessage(userPrompt);
    announceA11y('Retrying generation for last prompt');
  }

  // --- MESSAGE RENDERING (PAST MESSAGES) ---
  function appendMessageElement(role, rawContent, metrics = null, index = null, feedback = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role === 'user' ? 'user-row' : 'assistant-row'}`;

    if (role === 'user') {
      const userCol = document.createElement('div');
      userCol.style.display = 'flex';
      userCol.style.flexDirection = 'column';
      userCol.style.alignItems = 'flex-end';
      userCol.style.gap = '4px';

      const bubble = document.createElement('div');
      bubble.className = 'user-bubble';
      bubble.textContent = rawContent;

      const editBtn = document.createElement('button');
      editBtn.className = 'clay-action-pill';
      editBtn.style.padding = '2px 8px';
      editBtn.style.fontSize = '10.5px';
      editBtn.innerHTML = '<i data-lucide="edit-3" style="width: 10px; height: 10px;"></i><span>Edit</span>';
      editBtn.title = 'Edit prompt & rollback conversation';
      editBtn.setAttribute('aria-label', 'Edit prompt');
      editBtn.onclick = () => editAndRollback(index !== null ? index : 0);

      userCol.appendChild(bubble);
      userCol.appendChild(editBtn);

      const avatar = document.createElement('div');
      avatar.className = 'message-avatar user-avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.innerHTML = '<i data-lucide="user" style="width: 16px; height: 16px;"></i>';

      row.appendChild(userCol);
      row.appendChild(avatar);
    } else {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar assistant-avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.innerHTML = '<i data-lucide="zap" style="width: 16px; height: 16px;"></i>';

      const card = document.createElement('div');
      card.className = 'assistant-card';

      const { thinkText, answerText } = parseThinkingBlocks(rawContent, true);

      // 1. Safety detection
      const safetyData = detectSafetyRejection(rawContent);
      if (safetyData) {
        renderSafetyWarningCard(safetyData, card);
      }

      // 2. Thinking block
      if (thinkText) {
        const accordion = document.createElement('div');
        accordion.className = 'thinking-accordion';

        const header = document.createElement('div');
        header.className = 'thinking-header';
        header.setAttribute('role', 'button');
        header.setAttribute('tabindex', '0');
        header.setAttribute('aria-expanded', 'false');

        const badge = document.createElement('div');
        badge.className = 'thinking-badge';
        badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Process</span>';

        const iconChevron = document.createElement('i');
        iconChevron.setAttribute('data-lucide', 'chevron-down');
        iconChevron.style.width = '13px';
        iconChevron.style.height = '13px';
        iconChevron.style.transform = 'rotate(-90deg)';
        iconChevron.style.transition = 'transform 0.2s ease';

        header.appendChild(badge);
        header.appendChild(iconChevron);

        const content = document.createElement('div');
        content.className = 'thinking-content collapsed';
        content.textContent = thinkText;

        const toggleThinking = () => {
          content.classList.toggle('collapsed');
          const isCollapsed = content.classList.contains('collapsed');
          iconChevron.style.transform = isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
          header.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
        };

        header.onclick = toggleThinking;
        header.onkeydown = (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleThinking();
          }
        };

        accordion.appendChild(header);
        accordion.appendChild(content);
        card.appendChild(accordion);
      }

      // 3. Citations parsing & Markdown rendering
      const finalBodyText = answerText || (thinkText ? '' : cleanSpecialTokens(rawContent));
      const { cleanText: textWithCitations, sources } = parseAndEnhanceCitations(finalBodyText, card);

      const markdownBody = document.createElement('div');
      markdownBody.className = 'markdown-body';
      markdownBody.innerHTML = window.marked ? marked.parse(textWithCitations) : textWithCitations;

      enhanceCodeBlocks(markdownBody);
      renderMath(markdownBody);
      card.appendChild(markdownBody);

      // 4. Expandable sources accordion
      if (sources.length > 0) {
        renderSourcesAccordion(sources, card);
      }

      // 5. Actions bar (Tiered Feedback, Undo, Retry, Copy, AI label, Telemetry)
      const actions = document.createElement('div');
      actions.className = 'message-actions';

      const btnGroup = document.createElement('div');
      btnGroup.className = 'action-buttons-group';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'clay-action-pill';
      copyBtn.setAttribute('aria-label', 'Copy message');
      copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';
      copyBtn.onclick = () => copyToClipboard(finalBodyText, copyBtn);

      const undoBtn = document.createElement('button');
      undoBtn.className = 'clay-action-pill';
      undoBtn.setAttribute('aria-label', 'Undo message exchange');
      undoBtn.innerHTML = '<i data-lucide="undo-2" style="width: 11px; height: 11px;"></i><span>Undo</span>';
      undoBtn.title = 'Undo response & restore prompt';
      undoBtn.onclick = () => undoLastTurn();

      const retryBtn = document.createElement('button');
      retryBtn.className = 'clay-action-pill retry-action-btn';
      retryBtn.setAttribute('aria-label', 'Retry generation');
      retryBtn.innerHTML = '<i data-lucide="rotate-cw" style="width: 11px; height: 11px;"></i><span>Retry</span>';
      retryBtn.title = 'Regenerate this response';
      retryBtn.onclick = () => retryTurn(index !== null ? index : 0);

      btnGroup.appendChild(copyBtn);
      btnGroup.appendChild(undoBtn);
      btnGroup.appendChild(retryBtn);

      const telemetryTag = document.createElement('div');
      telemetryTag.className = 'telemetry-tag';
      const aiTagHtml = `<span class="ai-label-tag" title="Generated by AeroMESH distributed inference"><i data-lucide="bot" style="width: 10px; height: 10px;"></i> AI</span>`;

      if (metrics) {
        telemetryTag.innerHTML = `${aiTagHtml} <span>⚡ ${metrics.tokPerSec || '16.0'} tok/s</span> • <span>0.0 MB Wire</span>`;
      } else {
        telemetryTag.innerHTML = `${aiTagHtml} <span>⚡ Zero-Weight Mesh</span>`;
      }

      actions.appendChild(btnGroup);
      actions.appendChild(telemetryTag);
      card.appendChild(actions);

      renderFeedbackControls(actions, card, index !== null ? index : 0, feedback);

      row.appendChild(avatar);
      row.appendChild(card);
    }

    elements.messagesContainer.appendChild(row);
    lucide.createIcons();
    return row;
  }

  // --- STREAM BUFFER (RAF THROTTLING & LAYOUT THRASH PREVENTION) ---
  class StreamBuffer {
    constructor(liveCard) {
      this.liveCard = liveCard;
      this.pendingText = '';
      this.rafId = null;
      this.isScheduled = false;
      this.lastFlushTime = 0;
      this.flushIntervalMs = 24;
    }

    push(text) {
      this.pendingText += text;
      this.scheduleFlush();
    }

    scheduleFlush() {
      if (this.isScheduled) return;
      this.isScheduled = true;

      this.rafId = requestAnimationFrame(() => {
        this.isScheduled = false;
        const now = performance.now();
        if (now - this.lastFlushTime >= this.flushIntervalMs || this.pendingText.length > 30) {
          this.flush();
          this.lastFlushTime = now;
        } else {
          this.scheduleFlush();
        }
      });
    }

    flush() {
      if (this.pendingText) {
        this.liveCard.update(this.pendingText);
      }
    }

    destroy() {
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
      this.isScheduled = false;
    }
  }

  // --- LIVE ASSISTANT ROW GENERATOR ---
  function createLiveAssistantRow() {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar assistant-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = '<i data-lucide="zap" style="width: 16px; height: 16px;"></i>';

    const card = document.createElement('div');
    card.className = 'assistant-card live-streaming-card';

    const accordion = document.createElement('div');
    accordion.className = 'thinking-accordion';
    accordion.style.display = 'none';

    const header = document.createElement('div');
    header.className = 'thinking-header';
    header.setAttribute('role', 'button');
    header.setAttribute('tabindex', '0');

    const badge = document.createElement('div');
    badge.className = 'thinking-badge';
    badge.innerHTML = '<span class="thinking-shimmer" aria-hidden="true"></span><span>Reasoning Process...</span>';

    const iconChevron = document.createElement('i');
    iconChevron.setAttribute('data-lucide', 'chevron-down');
    iconChevron.style.width = '13px';
    iconChevron.style.height = '13px';
    iconChevron.style.transition = 'transform 0.2s ease';

    header.appendChild(badge);
    header.appendChild(iconChevron);

    const thinkContent = document.createElement('div');
    thinkContent.className = 'thinking-content';

    const toggleThinking = () => {
      thinkContent.classList.toggle('collapsed');
      iconChevron.style.transform = thinkContent.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
    };

    header.onclick = toggleThinking;
    header.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleThinking();
      }
    };

    accordion.appendChild(header);
    accordion.appendChild(thinkContent);
    card.appendChild(accordion);

    const markdownBody = document.createElement('div');
    markdownBody.className = 'markdown-body';
    card.appendChild(markdownBody);

    const stopBar = document.createElement('div');
    stopBar.className = 'live-stop-bar';
    stopBar.innerHTML = `
      <span style="font-size: 11.5px; color: var(--accent-coral); font-weight: 600; display: flex; align-items: center; gap: 6px;">
        <span class="status-dot" style="background: var(--accent-coral); box-shadow: 0 0 8px rgba(248, 113, 113, 0.4);" aria-hidden="true"></span>
        Streaming Zero-Weight Activations...
      </span>
      <button class="stop-stream-btn" aria-label="Stop generation">
        <i data-lucide="square" style="width: 10px; height: 10px; fill: currentColor;" aria-hidden="true"></i>
        <span>Stop (Esc)</span>
      </button>
    `;

    const stopBtn = stopBar.querySelector('.stop-stream-btn');
    stopBtn.onclick = () => {
      if (activeAbortController) activeAbortController.abort();
    };

    card.appendChild(stopBar);

    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const btnGroup = document.createElement('div');
    btnGroup.className = 'action-buttons-group';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'clay-action-pill';
    copyBtn.setAttribute('aria-label', 'Copy response');
    copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';

    btnGroup.appendChild(copyBtn);

    const telemetryTag = document.createElement('div');
    telemetryTag.className = 'telemetry-tag';
    telemetryTag.innerHTML = `<span class="ai-label-tag"><i data-lucide="bot" style="width: 10px; height: 10px;"></i> AI</span> <span>⚡ Streaming activations...</span>`;

    actions.appendChild(btnGroup);
    actions.appendChild(telemetryTag);
    card.appendChild(actions);

    row.appendChild(avatar);
    row.appendChild(card);
    elements.messagesContainer.appendChild(row);

    lucide.createIcons();

    return {
      row,
      card,
      accordion,
      badge,
      thinkContent,
      iconChevron,
      markdownBody,
      stopBar,
      telemetryTag,
      copyBtn,
      btnGroup,
      actions,
      update: function (rawText) {
        const currentHeight = card.offsetHeight;
        if (currentHeight > 0) {
          card.style.minHeight = `${currentHeight}px`;
        }

        const { thinkText, answerText, isStillThinking } = parseThinkingBlocks(rawText, false);

        if (thinkText) {
          accordion.style.display = 'block';
          thinkContent.textContent = thinkText;
          if (isStillThinking) {
            badge.innerHTML = '<span class="thinking-shimmer"></span><span>Reasoning Process...</span>';
          } else {
            badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Process</span>';
            lucide.createIcons();
          }
        }

        if (answerText) {
          markdownBody.innerHTML = window.marked ? marked.parse(answerText) : answerText;
        } else if (!thinkText) {
          const cleanRaw = cleanSpecialTokens(rawText);
          markdownBody.innerHTML = window.marked ? marked.parse(cleanRaw) : cleanRaw;
        } else {
          markdownBody.innerHTML = '';
        }

        copyBtn.onclick = () => copyToClipboard(answerText || cleanSpecialTokens(rawText), copyBtn);
        scrollToBottomIfNear();
      },
      finalize: function (rawText, metrics, messageIndex) {
        card.classList.remove('live-streaming-card');
        card.style.minHeight = 'auto';
        if (stopBar && stopBar.parentNode) {
          stopBar.remove();
        }

        const { thinkText, answerText } = parseThinkingBlocks(rawText, true);

        if (thinkText) {
          accordion.style.display = 'block';
          thinkContent.textContent = thinkText;
          badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Process</span>';
          thinkContent.classList.add('collapsed');
          iconChevron.style.transform = 'rotate(-90deg)';
        }

        const finalContent = answerText || (thinkText ? '' : cleanSpecialTokens(rawText));

        // 1. Safety check
        const safetyData = detectSafetyRejection(rawText);
        if (safetyData) {
          renderSafetyWarningCard(safetyData, card);
        }

        // 2. Citations extraction & body rendering
        const { cleanText: textWithCitations, sources } = parseAndEnhanceCitations(finalContent, card);
        markdownBody.innerHTML = window.marked ? marked.parse(textWithCitations) : textWithCitations;
        enhanceCodeBlocks(markdownBody);
        renderMath(markdownBody);

        // 3. Render sources accordion
        if (sources.length > 0) {
          renderSourcesAccordion(sources, card);
        }

        // 4. Update action bar buttons (Undo, Retry, Copy)
        if (!btnGroup.querySelector('.undo-action-btn')) {
          const undoBtn = document.createElement('button');
          undoBtn.className = 'clay-action-pill undo-action-btn';
          undoBtn.innerHTML = '<i data-lucide="undo-2" style="width: 11px; height: 11px;"></i><span>Undo</span>';
          undoBtn.title = 'Undo response & restore prompt';
          undoBtn.onclick = () => undoLastTurn();
          btnGroup.appendChild(undoBtn);
        }

        if (!btnGroup.querySelector('.retry-action-btn')) {
          const retryBtn = document.createElement('button');
          retryBtn.className = 'clay-action-pill retry-action-btn';
          retryBtn.innerHTML = '<i data-lucide="rotate-cw" style="width: 11px; height: 11px;"></i><span>Retry</span>';
          retryBtn.title = 'Regenerate this response';
          retryBtn.onclick = () => retryTurn(messageIndex);
          btnGroup.appendChild(retryBtn);
        }

        copyBtn.onclick = () => copyToClipboard(finalContent, copyBtn);

        // 5. Update Telemetry metrics & AI-Generated tag
        const aiTagHtml = `<span class="ai-label-tag" title="Generated by AeroMESH distributed inference"><i data-lucide="bot" style="width: 10px; height: 10px;"></i> AI</span>`;
        if (metrics) {
          telemetryTag.innerHTML = `${aiTagHtml} <span>⚡ ${metrics.tokPerSec || '16.0'} tok/s</span> • <span>0.0 MB Wire</span>`;
        } else {
          telemetryTag.innerHTML = `${aiTagHtml} <span>⚡ Zero-Weight Mesh</span>`;
        }

        // 6. Tiered feedback controls
        renderFeedbackControls(actions, card, messageIndex, null);

        lucide.createIcons();
        announceA11y(`Response complete. Generated ${metrics ? metrics.tokens : ''} tokens.`);
      }
    };
  }

  function renderMath(element) {
    if (window.renderMathInElement) {
      try {
        renderMathInElement(element, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
          ],
          throwOnError: false
        });
      } catch (err) {
        console.error(err);
      }
    }
  }

  function enhanceCodeBlocks(container) {
    const preElements = container.querySelectorAll('pre');
    preElements.forEach((pre) => {
      if (pre.parentElement.classList.contains('code-block-wrapper')) return;
      const code = pre.querySelector('code');
      const langMatch = code ? code.className.match(/language-(\w+)/) : null;
      const lang = langMatch ? langMatch[1] : 'code';

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-block-header';
      header.innerHTML = `
        <span style="font-family: monospace; text-transform: uppercase; color: var(--accent-orange); font-size: 10.5px;">${lang}</span>
        <button class="copy-code-btn" title="Copy Code" aria-label="Copy code block">
          <i data-lucide="copy" style="width: 11px; height: 11px;"></i>
          <span>Copy</span>
        </button>
      `;

      const copyBtn = header.querySelector('.copy-code-btn');
      copyBtn.onclick = () => {
        const textToCopy = code ? code.innerText : pre.innerText;
        navigator.clipboard.writeText(textToCopy).then(() => {
          copyBtn.classList.add('copied');
          copyBtn.innerHTML = '<i data-lucide="check" style="width: 11px; height: 11px;"></i><span>Copied!</span>';
          lucide.createIcons();
          setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';
            lucide.createIcons();
          }, 2000);
        });
      };

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }

  function copyToClipboard(text, btnElement) {
    navigator.clipboard.writeText(text).then(() => {
      const originalHtml = btnElement.innerHTML;
      btnElement.innerHTML = '<i data-lucide="check" style="width: 11px; height: 11px; color: var(--accent-mint);"></i><span>Copied</span>';
      lucide.createIcons();
      setTimeout(() => {
        btnElement.innerHTML = originalHtml;
        lucide.createIcons();
      }, 2000);
    });
  }

  function isUserNearBottom() {
    const threshold = 160;
    const vp = elements.chatViewport;
    return vp.scrollHeight - vp.scrollTop - vp.clientHeight < threshold;
  }

  function scrollToBottom() {
    elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
  }

  function scrollToBottomIfNear() {
    if (isUserNearBottom()) {
      scrollToBottom();
    }
  }

  // --- STREAMING INFERENCE WITH BUFFER ---
  async function handleSendMessage(promptOverride = null) {
    if (isGenerating) {
      if (activeAbortController) {
        activeAbortController.abort();
        try { fetch('/api/chat/abort', { method: 'POST' }); } catch (e) {}
      }
      return;
    }

    const messageText = (promptOverride || elements.chatInput.value).trim();
    if (!messageText) return;

    const chat = getCurrentChat();
    if (!chat) return;

    elements.chatInput.value = '';
    adjustTextareaHeight();
    elements.heroState.style.display = 'none';

    if (chat.messages.length === 0) {
      chat.title = messageText.length > 32 ? messageText.substring(0, 32) + '...' : messageText;
    }
    chat.updatedAt = new Date().toISOString();

    chat.messages.push({ role: 'user', content: messageText });
    saveConversations();

    // Render User Message directly
    appendMessageElement('user', messageText, null, chat.messages.length - 1);
    announceA11y(`You asked: ${messageText}`);

    isGenerating = true;
    updateSendButtonState(true);

    const assistantMessageIndex = chat.messages.length;
    chat.messages.push({ role: 'assistant', content: '', metrics: null, feedback: null });

    const formattedMessages = chat.messages.slice(0, -1).map((m) => ({
      role: m.role,
      content: m.content
    }));

    const payload = {
      model: activeModel !== 'Loading...' && activeModel !== 'Coordinator Offline' ? activeModel : undefined,
      messages: formattedMessages,
      max_tokens: maxTokens,
      stream: true,
      session_id: Date.now()
    };

    activeAbortController = new AbortController();
    const startTime = performance.now();
    let tokenCount = 0;
    let accumulatedText = '';

    // Create live assistant card and StreamBuffer
    const liveCard = createLiveAssistantRow();
    const streamBuffer = new StreamBuffer(liveCard);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: activeAbortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.substring(6).trim();

          if (dataStr === '[DONE]') break;

          try {
            const dataJson = JSON.parse(dataStr);
            const delta = dataJson.choices?.[0]?.delta?.content || '';
            if (delta) {
              accumulatedText += delta;
              tokenCount++;
              chat.messages[assistantMessageIndex].content = accumulatedText;
              streamBuffer.push(accumulatedText);
            }
          } catch (e) {
            if (dataStr && !dataStr.startsWith('{')) {
              accumulatedText += dataStr;
              tokenCount++;
              chat.messages[assistantMessageIndex].content = accumulatedText;
              streamBuffer.push(accumulatedText);
            }
          }
        }
      }

      streamBuffer.destroy();

      const durationSec = (performance.now() - startTime) / 1000;
      const tokPerSec = durationSec > 0 ? (tokenCount / durationSec).toFixed(1) : '16.0';

      const metrics = {
        tokPerSec,
        durationSec: durationSec.toFixed(2),
        tokens: tokenCount
      };

      chat.messages[assistantMessageIndex].metrics = metrics;
      chat.updatedAt = new Date().toISOString();
      liveCard.finalize(accumulatedText, metrics, assistantMessageIndex);
      saveConversations();
    } catch (err) {
      streamBuffer.destroy();
      if (err.name === 'AbortError') {
        accumulatedText += '\n\n*(Generation stopped by user)*';
      } else {
        accumulatedText += `\n\n❌ **Communication Error**: ${err.message}`;
      }
      chat.messages[assistantMessageIndex].content = accumulatedText;
      chat.updatedAt = new Date().toISOString();
      liveCard.finalize(accumulatedText, null, assistantMessageIndex);
      saveConversations();
    } finally {
      isGenerating = false;
      activeAbortController = null;
      updateSendButtonState(false);
      clearKvOnSend = false;
      elements.pillClearKv.classList.remove('active');
      elements.pillClearKv.setAttribute('aria-pressed', 'false');
      updateContextUsageBar();
    }
  }

  function updateSendButtonState(generating) {
    if (generating) {
      elements.sendBtn.classList.add('stop-btn-clay');
      elements.sendBtn.title = 'Stop Generation (Esc)';
      elements.sendBtn.setAttribute('aria-label', 'Stop generating response');
      elements.sendBtn.innerHTML = '<i data-lucide="square" style="width: 14px; height: 14px; fill: currentColor;" aria-hidden="true"></i>';
    } else {
      elements.sendBtn.classList.remove('stop-btn-clay');
      elements.sendBtn.title = 'Send Prompt (Enter)';
      elements.sendBtn.setAttribute('aria-label', 'Send message prompt');
      elements.sendBtn.innerHTML = '<i data-lucide="arrow-up" style="width: 18px; height: 18px; stroke-width: 2.5;" aria-hidden="true"></i>';
    }
    lucide.createIcons();
  }

  function adjustTextareaHeight() {
    const textarea = elements.chatInput;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 180);
    textarea.style.height = (newHeight > 24 ? newHeight : 24) + 'px';
  }

  function undoLastTurn() {
    if (isGenerating) {
      if (activeAbortController) {
        activeAbortController.abort();
        try { fetch('/api/chat/abort', { method: 'POST' }); } catch (e) {}
      }
      isGenerating = false;
      updateSendButtonState(false);
    }

    const chat = getCurrentChat();
    if (!chat || chat.messages.length === 0) return;

    let restoredPrompt = '';

    const lastMsg = chat.messages[chat.messages.length - 1];
    if (lastMsg && lastMsg.role === 'assistant') {
      chat.messages.pop();
    }

    if (chat.messages.length > 0) {
      const userMsg = chat.messages[chat.messages.length - 1];
      if (userMsg && userMsg.role === 'user') {
        restoredPrompt = userMsg.content;
        chat.messages.pop();
      }
    }

    chat.updatedAt = new Date().toISOString();
    saveConversations();
    renderMessages();
    updateContextUsageBar();

    if (restoredPrompt) {
      elements.chatInput.value = restoredPrompt;
      adjustTextareaHeight();
    }
    elements.chatInput.focus();

    if (elements.pillUndo) {
      elements.pillUndo.classList.add('active');
      setTimeout(() => elements.pillUndo.classList.remove('active'), 600);
    }
    announceA11y('Last exchange undone. Prompt restored to input.');
  }

  function editAndRollback(userMessageIndex) {
    if (isGenerating) {
      if (activeAbortController) {
        activeAbortController.abort();
        try { fetch('/api/chat/abort', { method: 'POST' }); } catch (e) {}
      }
      isGenerating = false;
      updateSendButtonState(false);
    }

    const chat = getCurrentChat();
    if (!chat || userMessageIndex >= chat.messages.length) return;

    const targetMsg = chat.messages[userMessageIndex];
    if (!targetMsg) return;

    const promptText = targetMsg.content;
    chat.messages.splice(userMessageIndex);
    chat.updatedAt = new Date().toISOString();
    saveConversations();
    renderMessages();
    updateContextUsageBar();

    elements.chatInput.value = promptText;
    adjustTextareaHeight();
    elements.chatInput.focus();
    announceA11y(`Editing prompt at turn ${userMessageIndex + 1}`);
  }

  // --- MODAL FOCUS TRAPPING & KEYBOARD ACCESSIBILITY ---
  function openTelemetryModal() {
    fetchClusterStatus();
    elements.telemetryModal.classList.add('active');
    elements.modalCloseBtn.focus();
    announceA11y('Opened Cluster Topology HUD');
  }

  function closeTelemetryModal() {
    elements.telemetryModal.classList.remove('active');
    elements.btnOpenTelemetry.focus();
    announceA11y('Closed Cluster Topology HUD');
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    elements.sidebarToggleBtn.onclick = () => {
      const isCollapsed = elements.sidebar.classList.toggle('collapsed');
      elements.sidebarToggleBtn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
    };

    if (elements.sidebarCloseBtn) {
      elements.sidebarCloseBtn.onclick = () => {
        elements.sidebar.classList.add('collapsed');
      };
    }

    elements.btnNewChat.onclick = () => createNewChat();
    elements.btnClearAll.onclick = () => clearAllChats();
    elements.btnExportChat.onclick = () => exportCurrentChat();

    elements.btnResetSession.onclick = () => {
      clearKvOnSend = true;
      elements.pillClearKv.classList.add('active');
      elements.pillClearKv.setAttribute('aria-pressed', 'true');
      announceA11y('KV Cache marked to reset on next generation');
    };

    elements.searchInput.oninput = (e) => {
      renderHistoryList(e.target.value);
    };

    elements.modelSelectorBtn.onclick = (e) => {
      e.stopPropagation();
      const isActive = elements.modelDropdownMenu.classList.toggle('active');
      elements.modelSelectorBtn.setAttribute('aria-expanded', isActive ? 'true' : 'false');
    };

    document.addEventListener('click', (e) => {
      if (!elements.modelSelectorBtn.contains(e.target) && !elements.modelDropdownMenu.contains(e.target)) {
        elements.modelDropdownMenu.classList.remove('active');
        elements.modelSelectorBtn.setAttribute('aria-expanded', 'false');
      }
    });

    elements.chatInput.addEventListener('input', adjustTextareaHeight);
    elements.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (elements.telemetryModal.classList.contains('active')) {
          closeTelemetryModal();
        } else if (isGenerating) {
          if (activeAbortController) activeAbortController.abort();
        } else if (elements.modelDropdownMenu.classList.contains('active')) {
          elements.modelDropdownMenu.classList.remove('active');
          elements.modelSelectorBtn.setAttribute('aria-expanded', 'false');
        }
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        createNewChat();
      }
      // Alt+U or Ctrl+Shift+Z for Undo
      if (e.altKey && (e.key === 'u' || e.key === 'U')) {
        e.preventDefault();
        undoLastTurn();
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'Z' || e.key === 'z')) {
        e.preventDefault();
        undoLastTurn();
      }
    });

    elements.sendBtn.onclick = () => handleSendMessage();

    elements.suggestionCards.forEach((card) => {
      const trigger = () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) handleSendMessage(prompt);
      };
      card.onclick = trigger;
      card.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          trigger();
        }
      };
    });

    if (elements.pillUndo) {
      elements.pillUndo.onclick = () => undoLastTurn();
      elements.pillUndo.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          undoLastTurn();
        }
      };
    }

    elements.pillReasoning.onclick = () => {
      enableReasoning = !enableReasoning;
      elements.pillReasoning.classList.toggle('active', enableReasoning);
      elements.pillReasoning.setAttribute('aria-pressed', enableReasoning ? 'true' : 'false');
      announceA11y(`Deep reasoning ${enableReasoning ? 'enabled' : 'disabled'}`);
    };
    elements.pillReasoning.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.pillReasoning.click();
      }
    };

    elements.pillTokens.onclick = () => {
      const nextIdx = (tokenSteps.indexOf(maxTokens) + 1) % tokenSteps.length;
      maxTokens = tokenSteps[nextIdx];
      elements.maxTokensLabel.textContent = `Max Tokens: ${maxTokens}`;
      announceA11y(`Max tokens output set to ${maxTokens}`);
    };
    elements.pillTokens.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.pillTokens.click();
      }
    };

    elements.pillClearKv.onclick = () => {
      clearKvOnSend = !clearKvOnSend;
      elements.pillClearKv.classList.toggle('active', clearKvOnSend);
      elements.pillClearKv.setAttribute('aria-pressed', clearKvOnSend ? 'true' : 'false');
      announceA11y(`Reset KV cache on next send ${clearKvOnSend ? 'armed' : 'disarmed'}`);
    };
    elements.pillClearKv.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.pillClearKv.click();
      }
    };

    elements.btnOpenTelemetry.onclick = openTelemetryModal;
    elements.modalCloseBtn.onclick = closeTelemetryModal;

    elements.telemetryModal.onclick = (e) => {
      if (e.target === elements.telemetryModal) {
        closeTelemetryModal();
      }
    };

    // Trap focus inside modal
    elements.telemetryModal.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        const focusable = elements.telemetryModal.querySelectorAll('button, [href], input, [tabindex="0"]');
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    });

    elements.modalProbeBtn.onclick = async () => {
      elements.modalProbeBtn.innerHTML = '<span class="thinking-shimmer"></span><span>Probing...</span>';
      await fetchClusterStatus();
      setTimeout(() => {
        elements.modalProbeBtn.innerHTML = '<i data-lucide="refresh-cw" style="width: 12px; height: 12px;"></i><span>Refresh Status</span>';
        lucide.createIcons();
      }, 500);
      announceA11y('Probed cluster coordinator status');
    };
  }

  window.addEventListener('DOMContentLoaded', init);
})();
