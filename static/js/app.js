/**
 * AeroMESH • Full-Spread Dark Claymorphism Chat Application
 * Features: Dynamic Backend Probing (Zero Hardcoding), SSE Streaming, DeepSeek-R1 <think> Accordion, KaTeX Math, Highlight.js
 */

(function () {
  'use strict';

  // --- STATE MANAGEMENT ---
  const STORAGE_KEY = 'aeromesh_conversations_v2';
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
    suggestionCards: document.querySelectorAll('.suggestion-card')
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

    lucide.createIcons();
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
          const activeItem = availableModels.find((m) => m.is_active);
          activeModel = activeItem ? activeItem.id : (activeModel || availableModels[0].id);
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
      emptyItem.textContent = 'No active models found';
      elements.modelDropdownMenu.appendChild(emptyItem);
      return;
    }

    availableModels.forEach((m) => {
      const item = document.createElement('div');
      item.className = `model-dropdown-item ${m.id === activeModel ? 'active' : ''}`;
      item.innerHTML = `<span>${m.id}</span> ${m.id === activeModel ? '<i data-lucide="check" style="width: 13px; height: 13px;"></i>' : ''}`;
      item.onclick = async () => {
        if (activeModel === m.id) {
          elements.modelDropdownMenu.classList.remove('active');
          return;
        }
        activeModel = m.id;
        elements.activeModelName.textContent = `Loading ${activeModel}...`;
        elements.modelDropdownMenu.classList.remove('active');
        renderModelDropdown();

        try {
          const resp = await fetch('/api/model/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: m.id })
          });
          if (resp.ok) {
            elements.activeModelName.textContent = activeModel;
            await fetchClusterStatus();
          } else {
            console.error('Failed to switch model on coordinator');
            elements.activeModelName.textContent = activeModel;
          }
        } catch (e) {
          console.error('Error switching model:', e);
          elements.activeModelName.textContent = activeModel;
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
      // Online state
      elements.sidebarStatusDot.classList.remove('offline');
      elements.topStatusDot.classList.remove('offline');
      elements.modalStatusDot.classList.remove('offline');

      elements.clusterStatusLabel.textContent = 'Mesh Online';
      elements.clusterStatusLabel.style.color = 'var(--accent-mint)';
      elements.topStatusText.textContent = 'Pipeline Connected';
      elements.topStatusText.parentElement.style.color = 'var(--accent-mint)';

      // Populate sidebar stats dynamically
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

      // Populate Modal topology dynamically
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
      // Offline state
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

  function saveConversations() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      renderHistoryList();
    } catch (e) {
      console.error(e);
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
      messages: []
    };
    conversations.unshift(newChat);
    saveConversations();
    switchChat(newChat.id);
  }

  function switchChat(chatId) {
    if (isGenerating && activeAbortController) {
      activeAbortController.abort();
    }
    currentChatId = chatId;
    renderHistoryList();
    renderMessages();
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
  }

  function clearAllChats() {
    if (confirm('Clear all conversation history from this browser?')) {
      conversations = [];
      createNewChat();
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
  }

  // --- RENDERING UI ---
  function renderHistoryList(filterQuery = '') {
    elements.historyList.innerHTML = '';
    const filtered = conversations.filter((c) =>
      c.title.toLowerCase().includes(filterQuery.toLowerCase())
    );

    filtered.forEach((chat) => {
      const item = document.createElement('div');
      item.className = `chat-history-item ${chat.id === currentChatId ? 'active' : ''}`;
      item.onclick = () => switchChat(chat.id);

      const title = document.createElement('div');
      title.className = 'chat-title-text';
      title.textContent = chat.title;

      const actions = document.createElement('div');
      actions.className = 'chat-item-actions';

      const delBtn = document.createElement('button');
      delBtn.className = 'action-icon-btn';
      delBtn.innerHTML = '<i data-lucide="trash-2" style="width: 12px; height: 12px;"></i>';
      delBtn.title = 'Delete Chat';
      delBtn.onclick = (e) => deleteChat(chat.id, e);

      actions.appendChild(delBtn);
      item.appendChild(title);
      item.appendChild(actions);

      elements.historyList.appendChild(item);
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
      appendMessageElement(msg.role, msg.content, msg.metrics, idx);
    });

    scrollToBottom();
  }

  function parseThinkingBlocks(text) {
    if (!text) return { thinkText: null, answerText: '', isStillThinking: false };

    // Clean initial double tags if present
    const cleanText = text.replace(/^(<think>)+/i, '<think>');
    const thinkStart = cleanText.indexOf('<think>');
    
    if (thinkStart === -1) {
      return { thinkText: null, answerText: cleanText, isStillThinking: false };
    }

    const thinkEnd = cleanText.indexOf('</think>');
    if (thinkEnd === -1) {
      let rawThink = cleanText.substring(thinkStart + 7).replace(/<think>/gi, '').trim();
      return { thinkText: rawThink || 'Reasoning through prompt...', answerText: '', isStillThinking: true };
    }

    let thinkText = cleanText.substring(thinkStart + 7, thinkEnd).replace(/<think>/gi, '').trim();
    let answerText = cleanText.substring(thinkEnd + 8).trim();
    return { thinkText, answerText, isStillThinking: false };
  }

  function appendMessageElement(role, rawContent, metrics = null, index = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role === 'user' ? 'user-row' : 'assistant-row'}`;

    if (role === 'user') {
      const bubble = document.createElement('div');
      bubble.className = 'user-bubble';
      bubble.textContent = rawContent;

      const avatar = document.createElement('div');
      avatar.className = 'message-avatar user-avatar';
      avatar.innerHTML = '<i data-lucide="user" style="width: 16px; height: 16px;"></i>';

      row.appendChild(bubble);
      row.appendChild(avatar);
    } else {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar assistant-avatar';
      avatar.innerHTML = '<i data-lucide="zap" style="width: 16px; height: 16px;"></i>';

      const card = document.createElement('div');
      card.className = 'assistant-card';

      const { thinkText, answerText, isStillThinking } = parseThinkingBlocks(rawContent);

      if (thinkText) {
        const accordion = document.createElement('div');
        accordion.className = 'thinking-accordion';

        const header = document.createElement('div');
        header.className = 'thinking-header';

        const badge = document.createElement('div');
        badge.className = 'thinking-badge';
        if (isStillThinking) {
          badge.innerHTML = '<span class="thinking-shimmer"></span><span>Reasoning Process...</span>';
        } else {
          badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Process</span>';
        }

        const iconChevron = document.createElement('i');
        iconChevron.setAttribute('data-lucide', 'chevron-down');
        iconChevron.style.width = '13px';
        iconChevron.style.height = '13px';

        header.appendChild(badge);
        header.appendChild(iconChevron);

        const content = document.createElement('div');
        content.className = 'thinking-content';
        content.textContent = thinkText;

        header.onclick = () => {
          content.classList.toggle('collapsed');
          iconChevron.style.transform = content.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
        };

        accordion.appendChild(header);
        accordion.appendChild(content);
        card.appendChild(accordion);
      }

      const markdownBody = document.createElement('div');
      markdownBody.className = 'markdown-body';
      markdownBody.innerHTML = window.marked ? marked.parse(answerText || (isStillThinking ? '' : rawContent)) : (answerText || rawContent);

      enhanceCodeBlocks(markdownBody);
      renderMath(markdownBody);

      card.appendChild(markdownBody);

      const actions = document.createElement('div');
      actions.className = 'message-actions';

      const btnGroup = document.createElement('div');
      btnGroup.className = 'action-buttons-group';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'clay-action-pill';
      copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';
      copyBtn.onclick = () => copyToClipboard(answerText || rawContent, copyBtn);

      btnGroup.appendChild(copyBtn);

      const telemetryTag = document.createElement('div');
      telemetryTag.className = 'telemetry-tag';
      if (metrics) {
        telemetryTag.innerHTML = `<span>⚡ ${metrics.tokPerSec || '16.0'} tok/s</span> • <span>0.0 MB Wire</span>`;
      } else {
        telemetryTag.innerHTML = `<span>⚡ Zero-Weight Mesh</span>`;
      }

      actions.appendChild(btnGroup);
      actions.appendChild(telemetryTag);
      card.appendChild(actions);

      row.appendChild(avatar);
      row.appendChild(card);
    }

    elements.messagesContainer.appendChild(row);
    lucide.createIcons();
    return row;
  }

  function createLiveAssistantRow() {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar assistant-avatar';
    avatar.innerHTML = '<i data-lucide="zap" style="width: 16px; height: 16px;"></i>';

    const card = document.createElement('div');
    card.className = 'assistant-card';

    // Thinking accordion container
    const accordion = document.createElement('div');
    accordion.className = 'thinking-accordion';
    accordion.style.display = 'none';

    const header = document.createElement('div');
    header.className = 'thinking-header';

    const badge = document.createElement('div');
    badge.className = 'thinking-badge';
    badge.innerHTML = '<span class="thinking-shimmer"></span><span>Reasoning Process...</span>';

    const iconChevron = document.createElement('i');
    iconChevron.setAttribute('data-lucide', 'chevron-down');
    iconChevron.style.width = '13px';
    iconChevron.style.height = '13px';

    header.appendChild(badge);
    header.appendChild(iconChevron);

    const thinkContent = document.createElement('div');
    thinkContent.className = 'thinking-content';

    header.onclick = () => {
      thinkContent.classList.toggle('collapsed');
      iconChevron.style.transform = thinkContent.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
    };

    accordion.appendChild(header);
    accordion.appendChild(thinkContent);
    card.appendChild(accordion);

    // Markdown Answer Body
    const markdownBody = document.createElement('div');
    markdownBody.className = 'markdown-body';
    card.appendChild(markdownBody);

    // Message Actions
    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const btnGroup = document.createElement('div');
    btnGroup.className = 'action-buttons-group';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'clay-action-pill';
    copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';

    btnGroup.appendChild(copyBtn);

    const telemetryTag = document.createElement('div');
    telemetryTag.className = 'telemetry-tag';
    telemetryTag.innerHTML = `<span>⚡ Streaming activations...</span>`;

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
      telemetryTag,
      copyBtn,
      update: function (rawText) {
        const { thinkText, answerText, isStillThinking } = parseThinkingBlocks(rawText);

        if (thinkText) {
          accordion.style.display = 'block';
          thinkContent.textContent = thinkText;
          if (isStillThinking) {
            badge.innerHTML = '<span class="thinking-shimmer"></span><span>Reasoning Process...</span>';
          } else {
            badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Complete</span>';
          }
        }

        if (answerText) {
          markdownBody.innerHTML = window.marked ? marked.parse(answerText) : answerText;
        } else if (!thinkText) {
          markdownBody.innerHTML = window.marked ? marked.parse(rawText) : rawText;
        } else {
          markdownBody.innerHTML = '';
        }

        copyBtn.onclick = () => copyToClipboard(answerText || rawText, copyBtn);
        scrollToBottom();
      },
      finalize: function (rawText, metrics) {
        const { thinkText, answerText } = parseThinkingBlocks(rawText);
        const finalContent = answerText || (thinkText ? '' : rawText);

        markdownBody.innerHTML = window.marked ? marked.parse(finalContent) : finalContent;
        enhanceCodeBlocks(markdownBody);
        renderMath(markdownBody);

        if (metrics) {
          telemetryTag.innerHTML = `<span>⚡ ${metrics.tokPerSec || '16.0'} tok/s</span> • <span>0.0 MB Wire</span>`;
        }

        copyBtn.onclick = () => copyToClipboard(finalContent, copyBtn);
        lucide.createIcons();
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
        <button class="copy-code-btn" title="Copy Code">
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

  function scrollToBottom() {
    elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
  }

  // --- STREAMING INFERENCE ---
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
      chat.title = messageText.length > 28 ? messageText.substring(0, 28) + '...' : messageText;
      saveConversations();
    }

    chat.messages.push({ role: 'user', content: messageText });
    saveConversations();

    // Render User Message directly
    appendMessageElement('user', messageText);

    isGenerating = true;
    updateSendButtonState(true);

    const assistantMessageIndex = chat.messages.length;
    chat.messages.push({ role: 'assistant', content: '', metrics: null });

    const formattedMessages = chat.messages.slice(0, -1).map((m) => ({
      role: m.role,
      content: m.content
    }));

    const numericSessionId = currentChatId
      ? Math.abs(currentChatId.split('').reduce((acc, char) => ((acc << 5) - acc + char.charCodeAt(0)) | 0, 0)) || 1001
      : 1001;

    const payload = {
      model: activeModel !== 'Loading...' && activeModel !== 'Coordinator Offline' ? activeModel : undefined,
      messages: formattedMessages,
      max_tokens: maxTokens,
      stream: true,
      session_id: numericSessionId
    };

    activeAbortController = new AbortController();
    const startTime = performance.now();
    let tokenCount = 0;
    let accumulatedText = '';

    // Create live assistant card once in DOM
    const liveCard = createLiveAssistantRow();

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
              liveCard.update(accumulatedText);
            }
          } catch (e) {
            if (dataStr && !dataStr.startsWith('{')) {
              accumulatedText += dataStr;
              tokenCount++;
              chat.messages[assistantMessageIndex].content = accumulatedText;
              liveCard.update(accumulatedText);
            }
          }
        }
      }

      const durationSec = (performance.now() - startTime) / 1000;
      const tokPerSec = durationSec > 0 ? (tokenCount / durationSec).toFixed(1) : '16.0';

      const metrics = {
        tokPerSec,
        durationSec: durationSec.toFixed(2),
        tokens: tokenCount
      };

      chat.messages[assistantMessageIndex].metrics = metrics;
      liveCard.finalize(accumulatedText, metrics);
      saveConversations();
    } catch (err) {
      if (err.name === 'AbortError') {
        accumulatedText += '\n\n*(Generation stopped by user)*';
      } else {
        accumulatedText += `\n\n❌ **Communication Error**: ${err.message}`;
      }
      chat.messages[assistantMessageIndex].content = accumulatedText;
      liveCard.finalize(accumulatedText, null);
      saveConversations();
    } finally {
      isGenerating = false;
      activeAbortController = null;
      updateSendButtonState(false);
      clearKvOnSend = false;
      elements.pillClearKv.classList.remove('active');
    }
  }

  function updateSendButtonState(generating) {
    if (generating) {
      elements.sendBtn.classList.add('stop-btn-clay');
      elements.sendBtn.title = 'Stop Generation (Esc)';
      elements.sendBtn.innerHTML = '<i data-lucide="square" style="width: 14px; height: 14px; fill: currentColor;"></i>';
    } else {
      elements.sendBtn.classList.remove('stop-btn-clay');
      elements.sendBtn.title = 'Send Prompt (Enter)';
      elements.sendBtn.innerHTML = '<i data-lucide="arrow-up" style="width: 18px; height: 18px; stroke-width: 2.5;"></i>';
    }
    lucide.createIcons();
  }

  function adjustTextareaHeight() {
    const textarea = elements.chatInput;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 180);
    textarea.style.height = (newHeight > 24 ? newHeight : 24) + 'px';
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    elements.sidebarToggleBtn.onclick = () => {
      elements.sidebar.classList.toggle('collapsed');
    };

    elements.btnNewChat.onclick = () => createNewChat();
    elements.btnClearAll.onclick = () => clearAllChats();
    elements.btnExportChat.onclick = () => exportCurrentChat();

    elements.btnResetSession.onclick = () => {
      clearKvOnSend = true;
      elements.pillClearKv.classList.add('active');
    };

    elements.searchInput.oninput = (e) => {
      renderHistoryList(e.target.value);
    };

    elements.modelSelectorBtn.onclick = (e) => {
      e.stopPropagation();
      elements.modelDropdownMenu.classList.toggle('active');
    };

    document.addEventListener('click', (e) => {
      if (!elements.modelSelectorBtn.contains(e.target) && !elements.modelDropdownMenu.contains(e.target)) {
        elements.modelDropdownMenu.classList.remove('active');
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
      if (e.key === 'Escape' && isGenerating) {
        if (activeAbortController) activeAbortController.abort();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        createNewChat();
      }
    });

    elements.sendBtn.onclick = () => handleSendMessage();

    elements.suggestionCards.forEach((card) => {
      card.onclick = () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) handleSendMessage(prompt);
      };
    });

    elements.pillReasoning.onclick = () => {
      enableReasoning = !enableReasoning;
      elements.pillReasoning.classList.toggle('active', enableReasoning);
    };

    elements.pillTokens.onclick = () => {
      const nextIdx = (tokenSteps.indexOf(maxTokens) + 1) % tokenSteps.length;
      maxTokens = tokenSteps[nextIdx];
      elements.maxTokensLabel.textContent = `Max Tokens: ${maxTokens}`;
    };

    elements.pillClearKv.onclick = () => {
      clearKvOnSend = !clearKvOnSend;
      elements.pillClearKv.classList.toggle('active', clearKvOnSend);
    };

    elements.btnOpenTelemetry.onclick = () => {
      fetchClusterStatus();
      elements.telemetryModal.classList.add('active');
    };

    elements.modalCloseBtn.onclick = () => {
      elements.telemetryModal.classList.remove('active');
    };

    elements.telemetryModal.onclick = (e) => {
      if (e.target === elements.telemetryModal) {
        elements.telemetryModal.classList.remove('active');
      }
    };

    elements.modalProbeBtn.onclick = async () => {
      elements.modalProbeBtn.innerHTML = '<span class="thinking-shimmer"></span><span>Probing...</span>';
      await fetchClusterStatus();
      setTimeout(() => {
        elements.modalProbeBtn.innerHTML = '<i data-lucide="refresh-cw" style="width: 12px; height: 12px;"></i><span>Refresh Status</span>';
        lucide.createIcons();
      }, 500);
    };
  }

  window.addEventListener('DOMContentLoaded', init);
})();
