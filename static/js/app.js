window.getApiUrl = window.getApiUrl || function(path) {
  const isHttp = window.location.protocol === 'http:' || window.location.protocol === 'https:';
  const base = String(window.AEROMESH_API_BASE || (isHttp ? window.location.origin : 'http://127.0.0.1:7860')).replace(/\/+$/, '');
  const p = String(path || '').startsWith('/') ? path : '/' + path;
  return base + p;
};
/**
 * AeroMESH • Full-Spread Dark Claymorphism Chat Application
 * Features: Dynamic Backend Probing (Zero Hardcoding), SSE Streaming, DeepSeek-R1 <think> Accordion, KaTeX Math, Highlight.js
 */

(function () {
  'use strict';

  // --- STATE MANAGEMENT ---
  const STORAGE_KEY = 'aeromesh_conversations_v2';
  const MAX_CONCURRENT_CHATS = 2;
  let conversations = [];
  let currentChatId = null;

  // Resolve API Base dynamically to guarantee compatibility with native webviews (with_html / custom scheme / file://)
  function getApiUrl(path) {
    const base = window.AEROMESH_API_BASE || 
      ((window.location.protocol === 'http:' || window.location.protocol === 'https:') ? window.location.origin : 'http://127.0.0.1:7860');
    if (!path.startsWith('/')) path = '/' + path;
    return base.replace(/\/+$/, '') + path;
  }

  // Active concurrent streaming sessions: Map<chatId, streamState>
  const activeStreams = new Map();
  function isAnyChatGenerating() {
    return activeStreams.size > 0;
  }

  let activeModel = 'Loading...';
  let availableModels = [];
  let clusterState = 'offline';
  let lastClusterPayload = null;

  let maxTokens = 1024;
  const tokenSteps = [256, 512, 1024, 2048, 4096];
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
  const CLUSTER_POLL_MS = 2500;
  let clusterPollTimer = null;

  function startClusterPolling() {
    if (clusterPollTimer !== null) return;
    clusterPollTimer = setInterval(() => {
      // No point polling (or re-rendering) a page nobody is looking at.
      if (document.hidden) return;
      fetchClusterStatus();
    }, CLUSTER_POLL_MS);
  }

  function stopClusterPolling() {
    if (clusterPollTimer !== null) {
      clearInterval(clusterPollTimer);
      clusterPollTimer = null;
    }
  }

  async function init() {
    loadConversations();
    setupEventListeners();
    configureMarked();
    initDockResizeObserver();

    await fetchModels();
    await fetchClusterStatus();
    startClusterPolling();

    // Pause polling in a background tab and refresh immediately on return.
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopClusterPolling();
      } else {
        fetchClusterStatus();
        startClusterPolling();
      }
    });

    if (conversations.length === 0) {
      createNewChat();
    } else {
      switchChat(conversations[0].id);
    }

    if (window.lucide) lucide.createIcons();
  }

  // --- SECURITY HELPERS -----------------------------------------------------
  // Model output is untrusted input: it can contain HTML, and a malicious or
  // prompt-injected completion can attempt to exfiltrate the conversation or
  // run script in this origin. Everything rendered from model or backend data
  // therefore goes through one of these two functions.

  /** Escapes a value for interpolation into an HTML template literal. */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /** Renders Markdown to sanitized HTML. Returns a string safe for innerHTML. */
  function renderMarkdown(text) {
    const source = text || '';
    if (!window.marked) {
      // No parser available: escape rather than inject the raw string.
      return escapeHtml(source).replace(/\n/g, '<br>');
    }
    const rawHtml = marked.parse(source);
    if (window.DOMPurify) {
      return DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true } });
    }
    // Fail closed: if the sanitizer did not load, do not trust the HTML.
    console.error('DOMPurify is unavailable; rendering escaped plain text instead.');
    return escapeHtml(source).replace(/\n/g, '<br>');
  }

  /** Sets sanitized Markdown into an element. */
  function setMarkdown(el, text) {
    el.innerHTML = renderMarkdown(text);
  }

  /**
   * Derives a stable, collision-resistant session id for a chat.
   *
   * The engine resets its KV cache whenever the session id changes, so two
   * chats sharing an id would share a cache. A chat's creation timestamp is
   * unique in practice and survives a reload (it is persisted with the chat).
   */
  function sessionIdForChat(chat) {
    const created = chat && chat.createdAt ? new Date(chat.createdAt).getTime() : NaN;
    if (Number.isFinite(created) && created > 0) return created;
    // Fallback for chats persisted before createdAt existed.
    return Date.now();
  }

  // --- WIRE STATISTICS ------------------------------------------------------
  // The dashboard used to display hard-coded numbers (a fixed 5,166 bytes per
  // token, a prompt-token guess of `characters / 3.2`, and literal fallbacks
  // such as "18.4 tok/s"). Those were independent of the configured codec, so
  // the HUD could advertise INT8 figures during a BF16 run - roughly 2x the
  // real traffic. Everything below is derived from what the coordinator
  // actually reports.

  /**
   * Parses the coordinator's `per_token_kb` string (e.g. "5.04 KB/tok") into a
   * number of kilobytes, or null when it is absent or unparseable.
   */
  function parsePerTokenKB() {
    const raw = lastClusterPayload && lastClusterPayload.per_token_kb;
    if (!raw) return null;
    const match = String(raw).match(/([\d.]+)\s*KB/i);
    if (!match) return null;
    const value = parseFloat(match[1]);
    return Number.isFinite(value) ? value : null;
  }

  /** True when the active topology sends activations over a network link. */
  function isDistributedTopology() {
    if (!lastClusterPayload) return false;
    const workers =
      lastClusterPayload.workers || lastClusterPayload.worker_nodes || [];
    if (workers.length > 0) return true;
    // A shared-memory transport is explicitly intra-host.
    const transport = String(lastClusterPayload.transport || '');
    if (/SHM|Shared Memory/i.test(transport)) return false;
    return Boolean(lastClusterPayload.is_direct_wireguard);
  }

  /**
   * Builds wire-transfer statistics for `tokenCount` generated tokens.
   *
   * Returns `null` values rather than invented ones when the coordinator has
   * not reported a per-token size, so the UI shows "unknown" instead of a
   * plausible-looking number that is not real.
   */
  function computeWireStats(tokenCount) {
    const distributed = isDistributedTopology();
    if (!distributed) {
      return {
        isDistributed: false,
        perTokenKB: '0 KB (shared memory)',
        wireMB: '0.00',
        transport: (lastClusterPayload && lastClusterPayload.transport) || 'Intra-Host SHM'
      };
    }

    const perTokenKB = parsePerTokenKB();
    if (perTokenKB === null) {
      return {
        isDistributed: true,
        perTokenKB: 'not reported',
        wireMB: null,
        transport: (lastClusterPayload && lastClusterPayload.transport) || 'network'
      };
    }

    const wireBytes = tokenCount * perTokenKB * 1024;
    return {
      isDistributed: true,
      perTokenKB: `${perTokenKB.toFixed(2)} KB/tok`,
      wireMB: (wireBytes / (1024 * 1024)).toFixed(2),
      transport: (lastClusterPayload && lastClusterPayload.transport) || 'network'
    };
  }

  // --- CONFIGURE MARKED & HIGHLIGHT.JS ---
  function configureMarked() {
    if (!window.marked) {
      console.error('marked failed to load; Markdown will render as plain text.');
      return;
    }

    // `highlight` moved out of marked core in v5, so the option below did
    // nothing and code blocks rendered unhighlighted. marked-highlight
    // re-attaches it.
    const options = { breaks: true, gfm: true };

    if (window.markedHighlight && window.hljs) {
      const { markedHighlight } = window.markedHighlight;
      marked.use(
        markedHighlight({
          langPrefix: 'hljs language-',
          highlight(code, lang) {
            const language = lang && hljs.getLanguage(lang) ? lang : null;
            try {
              return language
                ? hljs.highlight(code, { language }).value
                : hljs.highlightAuto(code).value;
            } catch (err) {
              console.error('highlight failed', err);
              return code;
            }
          },
        })
      );
    } else {
      console.error('marked-highlight or highlight.js missing; code will not be highlighted.');
    }

    marked.setOptions(options);
  }

  // --- DYNAMIC BACKEND FETCHING (ZERO HARDCODING) ---
  async function fetchModels() {
    try {
      const resp = await fetch(getApiUrl('/v1/models'));
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
      // `m.id` comes from a file name on disk, so it is untrusted input.
      item.innerHTML = `<span>${escapeHtml(m.id)}</span> ${m.id === activeModel ? '<i data-lucide="check" style="width: 13px; height: 13px;"></i>' : ''}`;
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
          const resp = await fetch(getApiUrl('/api/model/switch'), {
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

    if (window.lucide) lucide.createIcons();
  }

  // --- UNIFIED CLUSTER TELEMETRY STATE (ONE SOURCE OF TRUTH) ---
  // Four states: 'offline' | 'online_idle' | 'online_generating' | 'worker_unreachable'
  //
  // The engine distinguishes "the coordinator is not answering" from "the
  // coordinator is up but its worker peers are gone" (it reports
  // `cluster_status: "WORKER_DISCONNECTED"`). Collapsing both into `offline`
  // previously meant a running coordinator with dead workers displayed the
  // "start the coordinator" recovery text.

  async function fetchClusterStatus() {
    try {
      const resp = await fetch(getApiUrl('/api/cluster/status'));
      if (resp.ok) {
        const data = await resp.json();
        setClusterState(mapClusterState(data), data);
        return;
      }
    } catch (e) {
      // Network fetch error: treat as coordinator offline.
    }
    setClusterState('offline', { state: 'offline', status: 'offline' });
  }

  /** Maps a /api/cluster/status payload onto one of the four UI states. */
  function mapClusterState(data) {
    data = data || {};

    const coordinatorDown =
      data.state === 'offline' ||
      data.status === 'offline' ||
      data.connected === false;

    if (coordinatorDown) return 'offline';

    const workerDown =
      data.cluster_status === 'WORKER_DISCONNECTED' ||
      data.state === 'worker_unreachable';

    if (workerDown) return 'worker_unreachable';

    if (isAnyChatGenerating()) return 'online_generating';
    if (data.state === 'online_generating') return 'online_generating';
    return 'online_idle';
  }

  function setClusterState(state, payload = null) {
    // A locally-running generation always wins over a polled idle state, but
    // must not mask a genuinely unreachable worker.
    if (isAnyChatGenerating() && (state === 'online_idle' || state === 'online_generating')) {
      state = 'online_generating';
    }
    clusterState = state;
    if (payload) {
      lastClusterPayload = payload;
    }
    renderClusterUI(clusterState, lastClusterPayload);
  }

  /**
   * True when the telemetry modal is on screen.
   *
   * Its node list used to be rebuilt on every 2.5s poll even while hidden, and
   * `lucide.createIcons()` rescans the whole document, so an idle dashboard
   * paid for a full icon pass four times a second.
   */
  function isTelemetryModalOpen() {
    return Boolean(elements.telemetryModal && elements.telemetryModal.classList.contains('active'));
  }

  /** Sets the modal node list, but only while the modal is actually visible. */
  function setModalNodes(html) {
    if (!isTelemetryModalOpen()) return;
    elements.modalNodesContainer.innerHTML = html;
  }

  function renderClusterUI(state, data) {
    data = data || {};
    const localStage = data.local_stage || (data.coordinator && data.coordinator.local_layers ? {
      layer_start: data.coordinator.local_layers.split('..=')[0],
      layer_end: data.coordinator.local_layers.split('..=')[1]
    } : {});
    const workers = data.worker_nodes || data.workers || [];
    const totalLayers = data.total_layers || (data.coordinator && data.coordinator.total_layers ? data.coordinator.total_layers : 'Auto');

    // The coordinator is reachable but its worker peers are not. This is a
    // distinct condition from "offline": retrying the coordinator is useless,
    // the worker needs starting.
    if (state === 'worker_unreachable') {
      elements.sidebarStatusDot.classList.add('offline');
      elements.sidebarStatusDot.classList.remove('generating');
      elements.clusterStatusLabel.textContent = 'Worker Disconnected';
      elements.clusterStatusLabel.style.color = 'var(--accent-orange)';

      elements.clusterStatsDynamic.innerHTML = `
        <div class="cluster-stats-row">
          <span>Status:</span>
          <span style="color: var(--accent-orange); font-weight: 600; display: inline-flex; align-items: center; gap: 4px;"><i data-lucide="alert-triangle" style="width: 12px; height: 12px;"></i>Coordinator up, no workers</span>
        </div>
        <div class="cluster-stats-row">
          <span>Worker${workers.length > 1 ? 's' : ''}:</span>
          <span class="stat-highlight" style="color: var(--accent-orange);">${workers.length > 0 ? escapeHtml(workers.join(', ')) : 'none registered'}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Fix:</span>
          <span style="color: var(--accent-orange); font-family: monospace; font-size: 10.5px;">start a worker node</span>
        </div>
      `;

      elements.topStatusDot.classList.add('offline');
      elements.topStatusDot.classList.remove('generating');
      elements.topStatusText.textContent = 'Worker Disconnected';
      elements.topStatusText.parentElement.style.color = 'var(--accent-orange)';

      elements.modalStatusDot.classList.add('offline');
      elements.modalStatusDot.classList.remove('generating');
      elements.modalClusterStatus.textContent =
        'Coordinator is running but no worker peer is reachable. Run: aeromesh worker --port 50052';
      elements.modalClusterStatus.style.color = 'var(--accent-orange)';
      if (window.lucide) lucide.createIcons();
      return;
    }

    if (state === 'online_idle' || state === 'online_generating') {
      if (data.active_model && (elements.activeModelName.textContent === 'Coordinator Offline' || elements.activeModelName.textContent === 'Loading...')) {
        elements.activeModelName.textContent = data.active_model;
        activeModel = data.active_model;
      }

      // 1. Sidebar indicator
      elements.sidebarStatusDot.classList.remove('offline');
      if (state === 'online_generating') {
        elements.sidebarStatusDot.classList.add('generating');
        elements.clusterStatusLabel.textContent = 'Inferring (P2P)';
      } else {
        elements.sidebarStatusDot.classList.remove('generating');
        elements.clusterStatusLabel.textContent = 'Mesh Online';
      }
      elements.clusterStatusLabel.style.color = 'var(--accent-mint)';

      // 1. Sidebar dynamic stats (NO "cargo run -- serve" hint when online)
      const isDirectWg = data.connected && (data.is_direct_wireguard || (data.transport && data.transport.includes('WireGuard')));
      const workerDisconnected = workers.length > 0 && data.connected === false;

      const totalNodes = data.nodes_count || (workers.length + 1);

      elements.clusterStatsDynamic.innerHTML = `
        <div class="cluster-stats-row">
          <span>Status:</span>
          <span style="color: ${workerDisconnected ? 'var(--accent-orange)' : 'var(--accent-mint)'}; font-weight: 600;">
            ${workerDisconnected ? 'Worker Disconnected' : (state === 'online_generating' ? `Inferring (${totalNodes} Nodes)` : 'Active (Idle)')}
          </span>
        </div>
        <div class="cluster-stats-row">
          <span>Topology:</span>
          <span class="stat-highlight" style="${isDirectWg ? 'background: rgba(16, 185, 129, 0.15); color: var(--accent-mint);' : ''}">${isDirectWg ? `${totalNodes} Nodes (Direct P2P)` : (workerDisconnected ? 'Worker Offline' : '1 Node (Local)')}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Coordinator:</span>
          <span class="stat-highlight">Layers ${escapeHtml(localStage.layer_start ?? 0)}..${escapeHtml(localStage.layer_end ?? 0)}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Worker${workers.length > 1 ? 's' : ''}:</span>
          <span class="stat-highlight" style="${workerDisconnected ? 'color: var(--accent-orange);' : ''}">${workers.length > 0 ? escapeHtml(workers.join(', ')) : (isDirectWg ? 'Pipeline peer' : 'Direct Pipeline')}</span>
        </div>
        <div class="cluster-stats-row">
          <span>Wire Payload:</span>
          <span style="color: var(--accent-orange); font-weight: 600;">${escapeHtml(data.per_token_kb || 'unknown')}</span>
        </div>
      `;

      // 2. Top-bar pill
      if (workerDisconnected) {
        elements.topStatusDot.classList.add('offline');
        elements.topStatusDot.classList.remove('generating');
        elements.topStatusText.textContent = 'Worker Disconnected';
        elements.topStatusText.parentElement.style.color = 'var(--accent-orange)';
      } else {
        elements.topStatusDot.classList.remove('offline');
        if (state === 'online_generating') {
          elements.topStatusDot.classList.add('generating');
          elements.topStatusText.textContent = isDirectWg ? `Cluster Inferring (${totalNodes} Nodes)` : 'Cluster Inferring';
        } else {
          elements.topStatusDot.classList.remove('generating');
          elements.topStatusText.textContent = isDirectWg ? `${totalNodes} Nodes Connected (Direct WG)` : 'Pipeline Connected (Local)';
        }
        elements.topStatusText.parentElement.style.color = 'var(--accent-mint)';
      }

      // 3. Telemetry Modal / HUD
      elements.modalStatusDot.classList.remove('offline');
      if (state === 'online_generating') {
        elements.modalStatusDot.classList.add('generating');
        elements.modalClusterStatus.textContent = 'Coordinator active — Streaming activations via Direct WireGuard';
      } else {
        elements.modalStatusDot.classList.remove('generating');
        elements.modalClusterStatus.textContent = `Coordinator active on ${data.coordinator_endpoint || 'http://127.0.0.1:8080'} (${isDirectWg ? `${totalNodes}-Node Mesh` : 'Idle'})`;
      }
      elements.modalClusterStatus.style.color = 'var(--accent-mint)';

      setModalNodes(`
        <div class="node-box">
          <div>
            <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Coordinator (this machine)</div>
            <div style="font-size: 11.5px; color: var(--text-dim);">Layers ${escapeHtml(localStage.layer_start ?? 0)}..${escapeHtml(localStage.layer_end ?? 0)} • Stage 1</div>
          </div>
          <div class="stat-highlight" style="font-size: 12px;">Stage 1</div>
        </div>

        <div class="mesh-link-container">
          <div class="direct-wireguard-badge">
            <i data-lucide="shield-check" style="width: 14px; height: 14px;"></i>
            <span>${escapeHtml(data.transport || 'Local transport')}</span>
            <span class="badge-ping">RTT: ${escapeHtml(data.rtt_ms ?? 'n/a')} ms</span>
          </div>
          <div class="mesh-wire-stats">
            <span>P2P activation streaming • ${escapeHtml(data.per_token_kb || 'codec not reported')} • zero model weights on wire</span>
          </div>
        </div>

        ${
          workers.length > 0
            ? workers
                .map((w, idx) => {
                  const remStart = localStage.layer_end !== undefined ? Number(localStage.layer_end) + 1 : 0;
                  const numTotal = Number(totalLayers) || (remStart + workers.length);
                  const remLayers = Math.max(1, numTotal - remStart);
                  const perWorker = Math.ceil(remLayers / workers.length);
                  const wStart = Math.min(numTotal - 1, remStart + idx * perWorker);
                  const wEnd = idx === workers.length - 1 ? numTotal - 1 : Math.min(numTotal - 1, wStart + perWorker - 1);
                  const roleDesc = idx === workers.length - 1 ? 'LM head &bull; activation sink' : 'Intermediate stage &bull; activation relay';
                  return `
          <div class="node-box">
            <div>
              <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Worker Stage ${idx + 2} (${escapeHtml(w)})</div>
              <div style="font-size: 11.5px; color: var(--text-dim);">Layers ${escapeHtml(wStart)}..${escapeHtml(wEnd)} &bull; ${roleDesc}</div>
            </div>
            <div class="stat-highlight direct-badge" style="font-size: 12px;">${isDirectWg ? 'Direct WireGuard' : 'Connected'}</div>
          </div>
        `;
                })
                .join('')
            : `
          <div class="node-box">
            <div>
              <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">No worker peers</div>
              <div style="font-size: 11.5px; color: var(--text-dim);">Running standalone on this machine</div>
            </div>
            <div class="stat-highlight" style="font-size: 12px;">Local</div>
          </div>
        `
        }
      `);
      if (window.lucide) lucide.createIcons();
    } else {
      // Offline state
      // 1. Sidebar indicator
      elements.sidebarStatusDot.classList.add('offline');
      elements.sidebarStatusDot.classList.remove('generating');
      elements.clusterStatusLabel.textContent = 'Coordinator Offline';
      elements.clusterStatusLabel.style.color = 'var(--accent-coral)';

      // Show "Start: cargo run -- serve" hint ONLY in offline state
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

      // 2. Top-bar pill
      elements.topStatusDot.classList.add('offline');
      elements.topStatusDot.classList.remove('generating');
      elements.topStatusText.textContent = 'Coordinator Offline';
      elements.topStatusText.parentElement.style.color = 'var(--accent-coral)';

      // 3. Telemetry Modal / HUD
      elements.modalStatusDot.classList.add('offline');
      elements.modalStatusDot.classList.remove('generating');
      elements.modalClusterStatus.textContent = 'Coordinator offline (http://127.0.0.1:8080)';
      elements.modalClusterStatus.style.color = 'var(--accent-coral)';

      setModalNodes(`
        <div style="padding: 14px; text-align: center; color: var(--text-muted); font-size: 13px; display: flex; flex-direction: column; gap: 8px;">
          <div style="color: var(--accent-coral); font-weight: 700;">AeroMesh Coordinator is not running</div>
          <div>Start the cluster coordinator from PowerShell:</div>
          <code style="background: #080b10; padding: 8px 12px; border-radius: 6px; color: var(--accent-orange); font-size: 11.5px; border: 1px solid rgba(255,255,255,0.06); text-align: left; overflow-x: auto;">
            cargo run --bin aeromesh -- serve --model models/test.gguf --layers 0..24 --peers worker-ip:50052 --port 8080
          </code>
        </div>
      `);
    }

    if (window.lucide) lucide.createIcons();
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
    // Detach liveCard reference from old chat so background stream doesn't manipulate wrong DOM
    if (currentChatId && activeStreams.has(currentChatId)) {
      const leavingStream = activeStreams.get(currentChatId);
      leavingStream.liveCard = null;
    }

    currentChatId = chatId;
    renderHistoryList();
    renderMessages();
    updateSendButtonState(activeStreams.has(currentChatId));
  }

  function deleteChat(chatId, e) {
    if (e) e.stopPropagation();
    if (activeStreams.has(chatId)) {
      stopChatGeneration(chatId);
    }
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
      for (const [chatId] of activeStreams) {
        stopChatGeneration(chatId);
      }
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
      const role = msg.role === 'user' ? '### User' : '### AeroMesh Cluster';
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
      const isChatRunning = activeStreams.has(chat.id);
      const item = document.createElement('div');
      item.className = `chat-history-item ${chat.id === currentChatId ? 'active' : ''} ${isChatRunning ? 'running' : ''}`;
      item.onclick = () => switchChat(chat.id);

      const title = document.createElement('div');
      title.className = 'chat-title-text';
      title.textContent = chat.title;
      item.appendChild(title);

      if (isChatRunning) {
        const badge = document.createElement('span');
        badge.className = 'chat-running-badge';
        badge.innerHTML = '<span class="status-dot generating"></span><span>Running</span>';
        item.appendChild(badge);
      }

      const actions = document.createElement('div');
      actions.className = 'chat-item-actions';

      const delBtn = document.createElement('button');
      delBtn.className = 'action-icon-btn';
      delBtn.innerHTML = '<i data-lucide="trash-2" style="width: 12px; height: 12px;"></i>';
      delBtn.title = 'Delete Chat';
      delBtn.onclick = (e) => deleteChat(chat.id, e);

      actions.appendChild(delBtn);
      item.appendChild(actions);

      elements.historyList.appendChild(item);
    });

    if (window.lucide) lucide.createIcons();
  }

  function renderMessages() {
    const chat = getCurrentChat();
    elements.messagesContainer.innerHTML = '';

    if (!chat || chat.messages.length === 0) {
      elements.heroState.style.display = 'flex';
      return;
    }

    elements.heroState.style.display = 'none';

    const activeStreamInfo = activeStreams.get(currentChatId);

    chat.messages.forEach((msg, idx) => {
      if (activeStreamInfo && idx === activeStreamInfo.assistantMessageIndex) {
        // Re-attach live assistant card for currently running stream
        const liveCard = createLiveAssistantRow();
        activeStreamInfo.liveCard = liveCard;
        const stats = computeWireStats(activeStreamInfo.tokenCount);
        liveCard.update(activeStreamInfo.accumulatedText, {
          tokPerSec: activeStreamInfo.liveTps,
          tokenCount: activeStreamInfo.tokenCount,
          wireMB: stats.wireMB,
          perTokenKB: stats.perTokenKB,
          isDistributed: stats.isDistributed
        });
      } else {
        appendMessageElement(msg.role, msg.content, msg.metrics, idx);
      }
    });

    scrollToBottom();
  }

  function parseThinkingBlocks(text) {
    if (!text) return { thinkText: null, answerText: '', isStillThinking: false };

    // Clean initial double tags if present
    const cleanText = text.replace(/^(<think>)+/i, '<think>');
    const thinkStart = cleanText.indexOf('<think>');
    
    if (thinkStart === -1) {
      // Self-healing: if </think> is present without <think>, everything before </think> was reasoning!
      const thinkEnd = cleanText.indexOf('</think>');
      if (thinkEnd !== -1) {
        let thinkText = cleanText.substring(0, thinkEnd).trim();
        let answerText = cleanText.substring(thinkEnd + 8).replace(/<\/think>/gi, '').trim();
        return { thinkText, answerText, isStillThinking: false };
      }
      return { thinkText: null, answerText: cleanText, isStillThinking: false };
    }

    const thinkEnd = cleanText.indexOf('</think>');
    if (thinkEnd === -1) {
      let rawThink = cleanText.substring(thinkStart + 7).replace(/<think>/gi, '').trim();
      return { thinkText: rawThink || 'Reasoning through prompt...', answerText: '', isStillThinking: true };
    }

    let thinkText = cleanText.substring(thinkStart + 7, thinkEnd).replace(/<think>/gi, '').trim();
    let answerText = cleanText.substring(thinkEnd + 8).replace(/<\/think>/gi, '').trim();
    return { thinkText: thinkText.length > 0 ? thinkText : null, answerText, isStillThinking: false };
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
      avatar.innerHTML = '<img src="/static/img/aeromesh_emblem.png" alt="AeroMESH" class="avatar-logo-img">';

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
      markdownBody.innerHTML = renderMarkdown(answerText || (isStillThinking ? '' : rawContent));

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
        if (metrics.isDistributed || (metrics.wireMB && parseFloat(metrics.wireMB) > 0)) {
          telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${metrics.tokPerSec || '—'} tok/s</span> • <span><i data-lucide="network" style="width: 11px; height: 11px;"></i> ${metrics.wireMB || '0.00'} MB Wire</span> • <span><i data-lucide="box" style="width: 11px; height: 11px;"></i> ${metrics.perTokenKB || 'not reported'}</span> • <span class="badge-direct-wireguard"><i data-lucide="shield-check" style="width: 12px; height: 12px;"></i> Direct WireGuard</span>`;
        } else {
          telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${metrics.tokPerSec || '—'} tok/s</span> • <span>0.0 MB Wire (SHM)</span>`;
        }
      } else {
        telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> Zero-Weight Mesh</span>`;
      }
      if (window.lucide) lucide.createIcons();

      actions.appendChild(btnGroup);
      actions.appendChild(telemetryTag);
      card.appendChild(actions);

      row.appendChild(avatar);
      row.appendChild(card);
    }

    elements.messagesContainer.appendChild(row);
    if (window.lucide) lucide.createIcons();
    return row;
  }

  function createLiveAssistantRow() {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar assistant-avatar';
    avatar.innerHTML = '<img src="/static/img/aeromesh_emblem.png" alt="AeroMESH" class="avatar-logo-img">';

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
    telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> Streaming activations...</span>`;

    actions.appendChild(btnGroup);
    actions.appendChild(telemetryTag);
    card.appendChild(actions);

    row.appendChild(avatar);
    row.appendChild(card);
    elements.messagesContainer.appendChild(row);

    if (window.lucide) lucide.createIcons();

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
      update: function (rawText, liveStats = null) {
        const { thinkText, answerText, isStillThinking } = parseThinkingBlocks(rawText);

        if (thinkText) {
          accordion.style.display = 'block';
          thinkContent.textContent = thinkText;
          if (isStillThinking) {
            badge.innerHTML = '<span class="thinking-shimmer"></span><span>Reasoning Process...</span>';
          } else {
            badge.innerHTML = '<i data-lucide="brain" style="width: 13px; height: 13px;"></i><span>Reasoning Complete</span>';
          }
        } else {
          accordion.style.display = 'none';
        }

        if (answerText) {
          markdownBody.innerHTML = renderMarkdown(answerText);
        } else if (!thinkText) {
          markdownBody.innerHTML = renderMarkdown(rawText);
        } else {
          markdownBody.innerHTML = '';
        }

        if (liveStats) {
          if (liveStats.isDistributed) {
            telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${escapeHtml(liveStats.tokPerSec)} tok/s</span> • <span><i data-lucide="network" style="width: 11px; height: 11px;"></i> ${escapeHtml(liveStats.wireMB || 'unknown')} MB wire</span> • <span><i data-lucide="box" style="width: 11px; height: 11px;"></i> ${escapeHtml(liveStats.perTokenKB)}</span>`;
          } else {
            telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${escapeHtml(liveStats.tokPerSec)} tok/s</span> • <span>0 MB wire (shared memory)</span>`;
          }
          if (window.lucide) lucide.createIcons();
        }

        copyBtn.onclick = () => copyToClipboard(answerText || rawText, copyBtn);
        scrollToBottom();
      },
      finalize: function (rawText, metrics) {
        const { thinkText, answerText } = parseThinkingBlocks(rawText);
        const finalContent = answerText || (thinkText ? '' : rawText);

        markdownBody.innerHTML = renderMarkdown(finalContent);
        enhanceCodeBlocks(markdownBody);
        renderMath(markdownBody);

        if (metrics && metrics.failed) {
          // Terminal state for an aborted or failed generation: replace the
          // live label so the card does not claim to still be streaming.
          const label = metrics.reason === 'Stopped' ? 'Stopped by user' : 'Stream failed';
          telemetryTag.innerHTML = `<span><i data-lucide="alert-triangle" style="width: 11px; height: 11px;"></i> ${escapeHtml(label)}</span>`;
        } else if (metrics) {
          const perToken = escapeHtml(metrics.perTokenKB || 'not reported');
          const wire = metrics.wireMB === null || metrics.wireMB === undefined
            ? 'unknown'
            : `${escapeHtml(metrics.wireMB)} MB wire`;
          if (metrics.isDistributed) {
            telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${escapeHtml(metrics.tokPerSec)} tok/s</span> • <span><i data-lucide="network" style="width: 11px; height: 11px;"></i> ${wire}</span> • <span><i data-lucide="box" style="width: 11px; height: 11px;"></i> ${perToken}</span>`;
          } else {
            telemetryTag.innerHTML = `<span><i data-lucide="zap" style="width: 11px; height: 11px;"></i> ${escapeHtml(metrics.tokPerSec)} tok/s</span> • <span>0 MB wire (shared memory)</span>`;
          }
        }

        copyBtn.onclick = () => copyToClipboard(finalContent, copyBtn);
        if (window.lucide) lucide.createIcons();
        scrollToBottom(true);
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
          if (window.lucide) lucide.createIcons();
          setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.innerHTML = '<i data-lucide="copy" style="width: 11px; height: 11px;"></i><span>Copy</span>';
            if (window.lucide) lucide.createIcons();
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
      if (window.lucide) lucide.createIcons();
      setTimeout(() => {
        btnElement.innerHTML = originalHtml;
        if (window.lucide) lucide.createIcons();
      }, 2000);
    });
  }

  function scrollToBottom(smooth = false) {
    if (!elements.chatViewport) return;
    if (!smooth) {
      elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
    }
    requestAnimationFrame(() => {
      if (!elements.chatViewport) return;
      if (smooth) {
        elements.chatViewport.scrollTo({
          top: elements.chatViewport.scrollHeight,
          behavior: 'smooth'
        });
      } else {
        elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
      }
    });
  }
  window.scrollToBottom = scrollToBottom;

  function initDockResizeObserver() {
    const dock = document.querySelector('.bottom-input-anchor') || document.querySelector('.composer-wrap');
    if (!dock) return;

    function updateDockHeight() {
      const rect = dock.getBoundingClientRect();
      if (rect.height > 40) {
        document.documentElement.style.setProperty('--composer-dock-height', `${Math.ceil(rect.height)}px`);
      }
    }

    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        updateDockHeight();
      });
      ro.observe(dock);
      const composerWrap = document.querySelector('.composer-wrap');
      if (composerWrap && composerWrap !== dock) {
        ro.observe(composerWrap);
      }
    }
    window.addEventListener('resize', updateDockHeight);
    updateDockHeight();
  }

  // --- CONCURRENCY MANAGEMENT & STREAM ABORT ---
  function showConcurrencyWarning() {
    let toast = document.getElementById('concurrency-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'concurrency-toast';
      toast.className = 'concurrency-toast';
      document.body.appendChild(toast);
    }
    toast.innerHTML = `
      <i data-lucide="alert-triangle"></i>
      <span><strong>2 chats are currently running in the background</strong>. Please wait for one to finish or stop an ongoing chat.</span>
    `;
    if (window.lucide) lucide.createIcons();

    toast.classList.add('show');
    const capsule = elements.chatInput.closest('.input-capsule');
    if (capsule) {
      capsule.classList.add('shake-warning');
      setTimeout(() => capsule.classList.remove('shake-warning'), 500);
    }

    if (window._concurrencyToastTimer) clearTimeout(window._concurrencyToastTimer);
    window._concurrencyToastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, 4500);
  }

  function stopChatGeneration(chatId) {
    if (!activeStreams.has(chatId)) return;
    const stream = activeStreams.get(chatId);
    if (stream.abortController) {
      stream.abortController.abort();
    }
    activeStreams.delete(chatId);
    try {
      fetch(getApiUrl('/api/chat/abort'), { method: 'POST' }).catch(() => {});
    } catch (e) {}

    if (chatId === currentChatId) {
      updateSendButtonState(false);
    }
    if (activeStreams.size === 0) {
      setClusterState('online_idle');
    }
    renderHistoryList();
  }

  // --- STREAMING INFERENCE ---
  async function handleSendMessage(promptOverride = null) {
    // 1. If currently viewed chat is generating, the button acts as "Stop"
    if (activeStreams.has(currentChatId)) {
      stopChatGeneration(currentChatId);
      return;
    }

    // 2. Concurrency limit: maximum 2 concurrent chat streams
    if (activeStreams.size >= MAX_CONCURRENT_CHATS) {
      showConcurrencyWarning();
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
    scrollToBottom(true);

    const targetChatId = currentChatId;
    const assistantMessageIndex = chat.messages.length;
    chat.messages.push({ role: 'assistant', content: '', metrics: null });

    const formattedMessages = chat.messages.slice(0, -1).map((m) => ({
      role: m.role,
      content: m.content
    }));

    // Session identity. The engine clears its KV cache when the session id
    // changes, so a collision would silently reuse another conversation's
    // cache. A 32-bit string hash (the previous approach) collides well within
    // a few thousand chats. Use the chat's creation timestamp instead: unique
    // per chat, monotonic, and stable across reloads.
    const numericSessionId = sessionIdForChat(chat);

    const payload = {
      model: activeModel !== 'Loading...' && activeModel !== 'Coordinator Offline' ? activeModel : undefined,
      messages: formattedMessages,
      max_tokens: maxTokens,
      stream: true,
      session_id: numericSessionId,
      reasoning: enableReasoning
    };
    if (clearKvOnSend) {
      // Ask the engine to drop its KV cache before this turn. The controls
      // previously toggled a flag that never reached the request.
      payload.clear_kv = true;
    }

    const abortController = new AbortController();
    const liveCard = createLiveAssistantRow();
    scrollToBottom(true);

    const streamInfo = {
      chatId: targetChatId,
      abortController: abortController,
      reader: null,
      assistantMessageIndex: assistantMessageIndex,
      startTime: performance.now(),
      tokenCount: 0,
      accumulatedText: '',
      liveCard: liveCard,
      messageText: messageText,
      liveTps: '—',
      isDistributed: false
    };

    activeStreams.set(targetChatId, streamInfo);
    updateSendButtonState(true);
    setClusterState('online_generating');
    renderHistoryList();

    const STOP_SEQUENCES = [
      '<|im_end|>',
      '<|im_start|>',
      '<｜end of sentence｜>',
      '<｜User｜>',
      '<｜Assistant｜>',
      '<｜begin of sentence｜>',
      '<|endoftext|>',
      '<|eot_id|>',
      '<|start_header_id|>',
      '</s>'
    ];

    try {
      const response = await fetch(getApiUrl('/api/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      streamInfo.reader = reader;
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let streamHalted = false;

      while (!streamHalted) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.substring(6).trim();

          if (dataStr === '[DONE]') {
            streamHalted = true;
            break;
          }

          let delta = '';
          try {
            const dataJson = JSON.parse(dataStr);
            delta = dataJson.choices?.[0]?.delta?.content || '';
          } catch (e) {
            if (dataStr && !dataStr.startsWith('{')) {
              delta = dataStr;
            }
          }

          if (delta) {
            streamInfo.accumulatedText += delta;

            // Immediately truncate and stop if special stop sequences appear
            let hitStopTag = false;
            for (const stopTag of STOP_SEQUENCES) {
              const tagIdx = streamInfo.accumulatedText.indexOf(stopTag);
              if (tagIdx !== -1) {
                streamInfo.accumulatedText = streamInfo.accumulatedText.substring(0, tagIdx);
                hitStopTag = true;
                break;
              }
            }

            // Token accounting: the coordinator emits exactly one SSE delta per
            // sampled token, so counting deltas is a faithful token count.
            streamInfo.tokenCount++;
            chat.messages[assistantMessageIndex].content = streamInfo.accumulatedText;

            const elapsedSec = (performance.now() - streamInfo.startTime) / 1000;
            streamInfo.liveTps = elapsedSec > 0 ? (streamInfo.tokenCount / elapsedSec).toFixed(1) : '—';
            streamInfo.liveStats = computeWireStats(streamInfo.tokenCount);

            // Live update DOM only if user is actively viewing THIS chat
            if (streamInfo.liveCard && currentChatId === targetChatId) {
              streamInfo.liveCard.update(streamInfo.accumulatedText, {
                tokPerSec: streamInfo.liveTps,
                tokenCount: streamInfo.tokenCount,
                wireMB: streamInfo.liveStats.wireMB,
                perTokenKB: streamInfo.liveStats.perTokenKB,
                isDistributed: streamInfo.liveStats.isDistributed
              });
            }

            if (hitStopTag) {
              streamHalted = true;
              try {
                reader.cancel();
                abortController.abort();
                fetch(getApiUrl('/api/chat/abort'), { method: 'POST' }).catch(() => {});
              } catch (_) {}
              break;
            }
          }
        }
      }

      const durationSec = (performance.now() - streamInfo.startTime) / 1000;
      const finalStats = computeWireStats(streamInfo.tokenCount);

      const metrics = {
        tokPerSec: durationSec > 0 ? (streamInfo.tokenCount / durationSec).toFixed(1) : '—',
        durationSec: durationSec.toFixed(2),
        tokens: streamInfo.tokenCount,
        wireMB: finalStats.wireMB,
        perTokenKB: finalStats.perTokenKB,
        isDistributed: finalStats.isDistributed,
        transport: finalStats.transport
      };

      chat.messages[assistantMessageIndex].metrics = metrics;
      if (streamInfo.liveCard && currentChatId === targetChatId) {
        streamInfo.liveCard.finalize(streamInfo.accumulatedText, metrics);
      }
      saveConversations();
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log(`Generation for ${targetChatId} aborted by user.`);
        streamInfo.accumulatedText += '\n\n*(Generation stopped by user)*';
      } else {
        console.error(`Stream error for ${targetChatId}:`, err);
        streamInfo.accumulatedText += `\n\n❌ **Communication Error**: ${err.message}`;
      }
      chat.messages[assistantMessageIndex].content = streamInfo.accumulatedText;
      if (streamInfo.liveCard && currentChatId === targetChatId) {
        // Pass an explicit failure marker so the card does not keep showing the
        // live "streaming activations" label forever after an error.
        streamInfo.liveCard.finalize(streamInfo.accumulatedText, {
          failed: true,
          reason: err.name === 'AbortError' ? 'Stopped' : 'Failed'
        });
      }
      saveConversations();
    } finally {
      activeStreams.delete(targetChatId);
      if (currentChatId === targetChatId) {
        updateSendButtonState(false);
      }
      renderHistoryList();
      clearKvOnSend = false;
      elements.pillClearKv.classList.remove('active');
      if (activeStreams.size === 0) {
        setClusterState('online_idle');
      }
    }
  }

  function updateSendButtonState(generating) {
    if (window.setMeshGenerating) {
      window.setMeshGenerating(generating);
    }
    if (generating) {
      elements.sendBtn.classList.add('stop-btn-clay');
      elements.sendBtn.title = 'Stop Generation (Esc)';
      elements.sendBtn.innerHTML = '<i data-lucide="square" style="width: 14px; height: 14px; fill: currentColor;"></i>';
    } else {
      elements.sendBtn.classList.remove('stop-btn-clay');
      elements.sendBtn.title = 'Send Prompt (Enter)';
      elements.sendBtn.innerHTML = '<i data-lucide="arrow-up" style="width: 18px; height: 18px; stroke-width: 2.5;"></i>';
    }
    if (window.lucide) lucide.createIcons();
  }

  function adjustTextareaHeight() {
    const textarea = elements.chatInput;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 220);
    textarea.style.height = (newHeight > 24 ? newHeight : 24) + 'px';
    if (elements.sendBtn && !elements.sendBtn.classList.contains('stop-btn-clay')) {
      elements.sendBtn.classList.toggle('active', textarea.value.trim().length > 0);
    }
    const dock = document.querySelector('.bottom-input-anchor');
    if (dock) {
      const rect = dock.getBoundingClientRect();
      if (rect.height > 40) {
        document.documentElement.style.setProperty('--composer-dock-height', `${Math.ceil(rect.height)}px`);
      }
    }
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
      if (e.key === 'Escape' && activeStreams.has(currentChatId)) {
        stopChatGeneration(currentChatId);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        createNewChat();
      }
    });

    elements.sendBtn.onclick = () => {
      if (activeStreams.has(currentChatId)) {
        stopChatGeneration(currentChatId);
      } else {
        handleSendMessage();
      }
    };

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
        if (window.lucide) lucide.createIcons();
      }, 500);
    };
  }

  window.addEventListener('DOMContentLoaded', init);
})();




// ============================================================================
// DISTRIBUTED 4-STEP DiT IMAGE GENERATION CONTROLLER (ALAT + INT8-DELTA)
// ============================================================================
(function initAeroMeshImagePipeline() {
  window.__ditImageMode = false;

  function renderLatentCanvas(canvas, prompt, seedHex, stepRatio) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const seed = parseInt((seedHex || 'a3f91c82b401').slice(0, 8), 16) || 1337;
    let s = seed;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };

    const h1 = Math.floor(rand() * 360);
    const h2 = (h1 + 65 + Math.floor(rand() * 90)) % 360;
    const h3 = (h2 + 110) % 360;

    const bgGrad = ctx.createLinearGradient(0, 0, W, H);
    bgGrad.addColorStop(0, `hsl(${h1}, 62%, 10%)`);
    bgGrad.addColorStop(0.55, `hsl(${h2}, 58%, 16%)`);
    bgGrad.addColorStop(1, `hsl(${h3}, 66%, 8%)`);
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    for (let i = 0; i < 7; i++) {
      const cx = rand() * W;
      const cy = rand() * H;
      const r = (0.18 + rand() * 0.35) * W;
      const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, r);
      const hue = (h1 + i * 38) % 360;
      g.addColorStop(0, `hsla(${hue}, 88%, 64%, ${0.26 * stepRatio})`);
      g.addColorStop(0.5, `hsla(${(hue + 30) % 360}, 78%, 48%, ${0.12 * stepRatio})`);
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  async function runDistributedImageStream(promptText) {
    const hero = document.getElementById('hero-state');
    if (hero) hero.style.display = 'none';
    const container = document.getElementById('messages-container');
    const viewport = document.getElementById('chat-viewport');
    if (!container) return;

    function scrollChatUp(smooth = false) {
      if (typeof window.scrollToBottom === 'function') {
        window.scrollToBottom(smooth);
      } else if (viewport) {
        if (smooth) viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
        else viewport.scrollTop = viewport.scrollHeight;
      }
    }

    const userRow = document.createElement('div');
    userRow.className = 'message-row user-row';
    const userBubble = document.createElement('div');
    userBubble.className = 'user-bubble';
    userBubble.textContent = `[Image Mode • Distributed DiT] ${promptText}`;
    userRow.appendChild(userBubble);
    container.appendChild(userRow);
    scrollChatUp(true);

    const asstRow = document.createElement('div');
    asstRow.className = 'message-row';
    asstRow.innerHTML = `
      <div class="message-avatar assistant-avatar">
        <img src="/static/img/aeromesh_emblem.png" alt="AeroMESH" class="avatar-logo-img">
      </div>
      <div class="assistant-card">
        <div class="dit-pipeline-box">
          <div style="display:flex;justify-content:space-between;align-items:center;font-size:12.5px;gap:8px;flex-wrap:wrap;">
            <strong style="color:#38bdf8;" class="dit-status-title">Probing Coordinator & Worker Mesh...</strong>
            <span style="font-family:monospace;color:#34d399;" class="dit-wire-badge">0.00 MB Wire • 0.0 MB Weights</span>
          </div>
          <div class="dit-step-track"><div class="dit-step-fill" style="width:8%;"></div></div>
          <div style="font-size:11.5px;color:var(--text-muted);font-family:monospace;" class="dit-stage-detail">
            Stage 0 Mesh Conditioning ➔ INT8-Delta WireGuard ➔ Stage 2 CUDA Denoising + FP16 VAE
          </div>
          <div style="font-size:11px;color:#94a3b8;font-style:italic;display:none;" class="dit-cond-preview"></div>
        </div>
        <div class="dit-canvas-wrap">
          <canvas width="768" height="768" style="filter:blur(16px);"></canvas>
        </div>
        <div class="message-actions">
          <div class="action-buttons-group">
            <button type="button" class="clay-action-pill dit-download-btn" style="display:none;">
              <span>Download 768×768 PNG</span>
            </button>
          </div>
          <div class="telemetry-tag dit-footer-telemetry">
            <span>Running Distributed Image Pipeline...</span>
          </div>
        </div>
      </div>
    `;
    container.appendChild(asstRow);
    scrollChatUp(true);

    const statusTitle = asstRow.querySelector('.dit-status-title');
    const wireBadge = asstRow.querySelector('.dit-wire-badge');
    const stepFill = asstRow.querySelector('.dit-step-fill');
    const stageDetail = asstRow.querySelector('.dit-stage-detail');
    const condPreview = asstRow.querySelector('.dit-cond-preview');
    const canvas = asstRow.querySelector('canvas');
    const downloadBtn = asstRow.querySelector('.dit-download-btn');
    const footerTelemetry = asstRow.querySelector('.dit-footer-telemetry');

    renderLatentCanvas(canvas, promptText, '7f4e9a12', 0.3);
    scrollChatUp(true);

    try {
      const resp = await fetch(getApiUrl('/api/image/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, steps: 6, width: 768, height: 768, stream: true }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payloadStr = line.slice(6).trim();
          if (!payloadStr || payloadStr === '[DONE]') continue;
          const evt = JSON.parse(payloadStr);

          if (evt.type === 'dit_mesh_conditioning') {
            statusTitle.textContent = `Stage 0: 2-Node Mesh Conditioning (Coord ${evt.coordinator_layers} ➔ Worker ${evt.worker_node})`;
            stageDetail.textContent = `Transport: ${evt.transport} (RTT: ${evt.rtt_ms} ms) • Streaming INT8 Prompt Activations...`;
            scrollChatUp(false);
          } else if (evt.type === 'dit_step') {
            const pct = Math.round((evt.step / evt.total_steps) * 100);
            stepFill.style.width = `${pct}%`;
            const meshTag = evt.mesh_active ? '2-Node Mesh' : 'Local GPU';
            statusTitle.textContent = `${meshTag} • Denoising Step ${evt.step}/${evt.total_steps} (σ=${evt.sigma_curr} ➔ ${evt.sigma_next})`;
            wireBadge.textContent = `${evt.accum_wire_mb} MB Wire • 0.0 MB Weights`;
            stageDetail.textContent = `${evt.stage1_blocks} ➔ [${evt.step_wire_mb} MB INT8-Delta] ➔ ${evt.stage2_blocks}`;
            if (evt.conditioning && evt.conditioning !== promptText) {
              condPreview.style.display = 'block';
              condPreview.textContent = `Mesh-Conditioned Prompt: "${evt.conditioning}"`;
            }
            const blurPx = Math.max(2, Math.round((1 - evt.step / evt.total_steps) * 16));
            canvas.style.filter = `blur(${blurPx}px)`;
            renderLatentCanvas(canvas, promptText, promptText, evt.step / evt.total_steps);

            const sidebarWire = document.getElementById('cluster-wire-transfer');
            if (sidebarWire) sidebarWire.textContent = `${evt.accum_wire_mb} MB Wire`;
            scrollChatUp(false);
          } else if (evt.type === 'dit_complete') {
            canvas.style.filter = 'none';
            const meshModeLabel = evt.mesh_active
              ? `2-Node Mesh (${evt.worker_node})`
              : `Standalone GPU`;
            if (evt.b64_png) {
              const img = new Image();
              img.onload = () => {
                canvas.width = img.width || 768;
                canvas.height = img.height || 768;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                scrollChatUp(true);
              };
              img.src = `data:image/png;base64,${evt.b64_png}`;
              statusTitle.textContent = `CUDA Diffusion Complete (${evt.width}×${evt.height} • ${evt.model_name}) • ${meshModeLabel}`;
            } else {
              renderLatentCanvas(canvas, promptText, evt.seed, 1.0);
              statusTitle.textContent = `DiT Synthesis Complete (${evt.sd_error || 'Procedural Fallback'})`;
            }
            footerTelemetry.innerHTML = `<span>${evt.steps} Steps in ${evt.elapsed_s}s</span> &bull; <span>${evt.accum_wire_mb} MB Total Wire</span> &bull; <span>0.0 MB Weights</span>`;
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.onclick = () => {
              const a = document.createElement('a');
              a.href = canvas.toDataURL('image/png');
              a.download = `aeromesh-dit-${evt.seed || 'render'}.png`;
              a.click();
            };
            scrollChatUp(true);
            setTimeout(() => scrollChatUp(true), 120);
          }
        }
      }
    } catch (err) {
      statusTitle.textContent = `DiT Pipeline Error: ${err.message || err}`;
    }
  }

  window.addEventListener('click', (e) => {
    const pill = e.target.closest && e.target.closest('#pill-image-mode');
    if (pill) {
      e.stopImmediatePropagation();
      e.preventDefault();
      window.__ditImageMode = !window.__ditImageMode;
      pill.classList.toggle('active', window.__ditImageMode);
      const chatInput = document.getElementById('chat-input');
      if (chatInput) {
        chatInput.placeholder = window.__ditImageMode
          ? 'Describe an image to generate across Coordinator + Worker...'
          : 'Ask anything or prompt the AeroMesh cluster...';
          // ngl i hate gemini 3.8 flash.. mf broke my code
      }
      return;
    }

    const sendBtn = e.target.closest && e.target.closest('#send-btn');
    if (sendBtn && window.__ditImageMode) {
      const chatInput = document.getElementById('chat-input');
      if (chatInput && chatInput.value.trim()) {
        e.stopImmediatePropagation();
        e.preventDefault();
        const prompt = chatInput.value.trim();
        chatInput.value = '';
        runDistributedImageStream(prompt);
      }
    }
  }, true);

  window.addEventListener('keydown', (e) => {
    if (!window.__ditImageMode) return;
    if (e.target && e.target.id === 'chat-input' && e.key === 'Enter' && !e.shiftKey) {
      const val = e.target.value.trim();
      if (val) {
        e.stopImmediatePropagation();
        e.preventDefault();
        e.target.value = '';
        runDistributedImageStream(val);
      }
    }
  }, true);
})();


// ============================================================================
// AEROMESH UNIFIED ROLE LAUNCHER & WORKER CONSOLE
// ============================================================================
(function initAeroMeshRoleLauncher() {
  let logPollTimer = null;

  function createLauncherModal() {
    if (document.getElementById('aeromesh-role-modal')) return;

    const topRight = document.querySelector('.top-right-group');
    if (topRight && !document.getElementById('btn-node-role')) {
      const btn = document.createElement('button');
      btn.id = 'btn-node-role';
      btn.className = 'clay-btn';
      btn.style.cssText = 'padding:5px 12px;font-size:12px;border-color:rgba(26,136,255,0.55);color:#1a88ff;display:inline-flex;align-items:center;gap:6px;';
      btn.innerHTML = '<i data-lucide="zap" style="width: 13px; height: 13px;"></i><span>Node Role</span>';
      btn.onclick = () => openRoleModal();
      topRight.insertBefore(btn, topRight.firstChild);
      if (window.lucide) lucide.createIcons();
    }

    const overlay = document.createElement('div');
    overlay.id = 'aeromesh-role-modal';
    overlay.style.cssText = [
      'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;',
      'background:rgba(6,8,12,0.88);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);padding:20px;'
    ].join('');

    overlay.innerHTML = `
      <div style="width:100%;max-width:860px;background:#16161a;border:1px solid rgba(255,255,255,0.14);border-radius:20px;padding:26px;box-shadow:0 28px 70px rgba(0,0,0,0.85);color:#f4f4f6;display:flex;flex-direction:column;gap:18px;max-height:92vh;overflow-y:auto;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:12px;">
            <img src="/static/img/aeromesh_emblem.png" onerror="this.src='/static/img/logo_emblem.png'" style="width:40px;height:40px;object-fit:contain;">
            <div>
              <h2 style="font-size:19px;font-weight:700;margin:0;color:#fff;">AeroMESH Unified Node Control Center</h2>
              <div style="font-size:12.5px;color:#a1a1aa;" id="rm-hw-summary">Loading hardware summary...</div>
            </div>
          </div>
          <div style="display:flex;gap:8px;">
            <button id="rm-stop-btn" class="clay-btn" style="padding:7px 14px;font-size:12px;color:#ff4756;border-color:rgba(255,71,86,0.4);display:none;align-items:center;gap:6px;"><i data-lucide="square" style="width: 12px; height: 12px;"></i><span>Stop Active Node</span></button>
            <button id="rm-close-btn" class="clay-btn" style="padding:7px 14px;font-size:12px;border-color:rgba(26,136,255,0.45);color:#1a88ff;display:inline-flex;align-items:center;gap:6px;"><span>Enter Studio</span><i data-lucide="arrow-right" style="width: 13px; height: 13px;"></i></button>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:14px;background:#121214;padding:14px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.08);">
          <div>
            <label style="font-size:11px;font-weight:700;color:#1a88ff;text-transform:uppercase;">1. Select Local GGUF Model</label>
            <select id="rm-model-select" style="width:100%;margin-top:6px;padding:9px 12px;border-radius:10px;background:#18181b;color:#fff;border:1px solid rgba(255,255,255,0.16);font-size:13px;"></select>
            <div style="font-size:11.5px;color:#16c456;margin-top:6px;" id="rm-img-model-label"></div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <label style="font-size:11px;font-weight:700;color:#16c456;text-transform:uppercase;">2. Discovered Wi-Fi &amp; Tailscale Workers (:50052)</label>
              <button id="rm-rescan-btn" class="clay-btn" style="padding:3px 10px;font-size:11px;display:inline-flex;align-items:center;gap:5px;"><i data-lucide="refresh-cw" style="width: 11px; height: 11px;"></i><span>Rescan Network</span></button>
            </div>
            <div id="rm-peers-list" style="margin-top:6px;font-size:12px;color:#a1a1aa;max-height:60px;overflow-y:auto;">Ready to scan</div>
            <input id="rm-custom-peers" type="text" placeholder="Or enter worker IP:50052 (comma-separated)..." style="width:100%;margin-top:6px;padding:7px 10px;border-radius:8px;background:#18181b;color:#fff;border:1px solid rgba(255,255,255,0.14);font-size:12px;">
          </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
          <div class="suggestion-card" id="rm-card-auto" style="cursor:pointer;border:1px solid rgba(26,136,255,0.45);background:rgba(26,136,255,0.08);">
            <div class="suggestion-tag" style="color:#1a88ff;display:inline-flex;align-items:center;gap:6px;"><i data-lucide="server" style="width: 13px; height: 13px;"></i><span>ROLE 1 • COORDINATOR + UI</span></div>
            <div style="font-weight:700;font-size:15px;color:#fff;margin:2px 0;">Start as Coordinator</div>
            <div style="font-size:12px;color:#a1a1aa;line-height:1.45;">Runs Stage 1 (lower layers), connects to discovered Wi-Fi/Tailscale Workers, and opens the Chat + Image Studio.</div>
          </div>

          <div class="suggestion-card" id="rm-card-worker" style="cursor:pointer;border:1px solid rgba(255,159,10,0.45);background:rgba(255,159,10,0.08);">
            <div class="suggestion-tag" style="color:#ff9f0a;display:inline-flex;align-items:center;gap:6px;"><i data-lucide="zap" style="width: 13px; height: 13px;"></i><span>ROLE 2 • WORKER NODE (:50052)</span></div>
            <div style="font-weight:700;font-size:15px;color:#fff;margin:2px 0;">Start as Worker Node</div>
            <div style="font-size:12px;color:#a1a1aa;line-height:1.45;">Turns this laptop into a dedicated GPU Worker listening on port 50052 with live packet &amp; VRAM telemetry.</div>
          </div>

          <div class="suggestion-card" id="rm-card-all" style="cursor:pointer;border:1px solid rgba(22,196,86,0.45);background:rgba(22,196,86,0.08);">
            <div class="suggestion-tag" style="color:#16c456;display:inline-flex;align-items:center;gap:6px;"><i data-lucide="network" style="width: 13px; height: 13px;"></i><span>ROLE 3 • FULL LOCAL MESH</span></div>
            <div style="font-weight:700;font-size:15px;color:#fff;margin:2px 0;">Start Full Local Mesh</div>
            <div style="font-size:12px;color:#a1a1aa;line-height:1.45;">Starts both Worker (:50052) + Coordinator (:8080) on this PC via Zero-Copy Shared Memory and enters Studio.</div>
          </div>
        </div>

        <div style="background:#0c0c0e;border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:12px 14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:12px;font-weight:700;color:#1a88ff;font-family:monospace;" id="rm-console-title">NODE STATUS: IDLE — Click one of the 3 Role Cards above to launch!</span>
            <span style="font-size:11px;color:#16c456;font-family:monospace;font-weight:700;" id="rm-active-role-badge">Role: IDLE</span>
          </div>
          <pre id="rm-live-logs" style="height:140px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#d4d4d8;margin:0;white-space:pre-wrap;line-height:1.45;">[AeroMesh Studio] Ready. Select your model above and click a Role Card to launch.</pre>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    if (window.lucide) lucide.createIcons();

    document.getElementById('rm-close-btn').onclick = () => closeRoleModal();
    document.getElementById('rm-rescan-btn').onclick = () => scanPeersIntoModal();
    document.getElementById('rm-stop-btn').onclick = async () => {
      await fetch(window.getApiUrl('/api/node/stop'), { method: 'POST' });
      stopLogPolling();
      await refreshNodeState();
    };

    document.getElementById('rm-card-auto').onclick = () => startNodeRole('coordinator');
    document.getElementById('rm-card-worker').onclick = () => startNodeRole('worker');
    document.getElementById('rm-card-all').onclick = () => startNodeRole('all');
  }

  async function scanPeersIntoModal() {
    const listEl = document.getElementById('rm-peers-list');
    const customInput = document.getElementById('rm-custom-peers');
    if (listEl) listEl.textContent = 'Scanning Wi-Fi LAN & Tailscale...';
    try {
      const r = await fetch(window.getApiUrl('/api/cluster/discover'));
      if (!r.ok) return;
      const d = await r.json();
      const peers = d.discovered_workers || [];
      if (peers.length === 0) {
        if (listEl) listEl.innerHTML = '<span style="color:#71717a;">0 remote workers on :50052 (use Role 3 for Single-PC Mesh, or start Role 2 on another laptop).</span>';
      } else {
        if (listEl) {
          listEl.innerHTML = peers.map(p => `<div style="color:#16c456;font-family:monospace;">✔ ${p.endpoint} (${p.type} • ${p.rtt_ms}ms)</div>`).join('');
        }
        if (customInput && !customInput.value.trim()) {
          customInput.value = peers.map(p => p.endpoint).join(',');
        }
      }
    } catch (_) {
      if (listEl) listEl.textContent = 'Local Mode Ready';
    }
  }

  async function refreshNodeState() {
    try {
      const r = await fetch(window.getApiUrl('/api/node/state'));
      if (!r.ok) return null;
      const st = await r.json();

      const hw = document.getElementById('rm-hw-summary');
      if (hw) {
        hw.innerHTML = `GPU: <strong style="color:#fff;">${st.gpu || 'NVIDIA CUDA GPU'}</strong> &bull; Wi-Fi IP: <code style="color:#1a88ff;">${st.wifi_ip || '127.0.0.1'}:50052</code> &bull; Tailscale IP: <code style="color:#16c456;">${st.tailscale_ip || 'Local'}:50052</code>`;
      }

      const sel = document.getElementById('rm-model-select');
      const models = Array.isArray(st.models) ? st.models : [];
      if (sel && sel.options.length !== models.length) {
        const prev = sel.value;
        sel.innerHTML = '';
        models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (m === prev || (!prev && m === 'test.gguf')) opt.selected = true;
          sel.appendChild(opt);
        });
      }

      const imgLbl = document.getElementById('rm-img-model-label');
      if (imgLbl) {
        const imgModels = Array.isArray(st.image_models) ? st.image_models : [];
        imgLbl.textContent = imgModels.length > 0
          ? `✔ CUDA Image Model Ready: ${imgModels[0]}`
          : '✔ LLM Cluster Engine Ready';
      }

      const logsEl = document.getElementById('rm-live-logs');
      if (logsEl && Array.isArray(st.logs) && st.logs.length > 0) {
        logsEl.textContent = st.logs.join('\n');
        logsEl.scrollTop = logsEl.scrollHeight;
      }

      const badge = document.getElementById('rm-active-role-badge');
      const title = document.getElementById('rm-console-title');
      const stopBtn = document.getElementById('rm-stop-btn');
      if (badge) badge.textContent = `Role: ${(st.role || 'idle').toUpperCase()}`;
      if (stopBtn) stopBtn.style.display = st.running ? 'inline-flex' : 'none';
      if (title) {
        if (st.role === 'worker' && st.running) {
          title.textContent = `[ACTIVE] WORKER NODE ACTIVE ON PORT 50052 — Share IP ${st.tailscale_ip !== 'Not Connected' ? st.tailscale_ip : st.wifi_ip}:50052`;
        } else if (st.running) {
          title.textContent = `[ACTIVE] ${st.role.toUpperCase()} RUNNING — Click 'Enter Studio' at top right`;
        } else {
          title.textContent = 'NODE STATUS: IDLE — Click one of the 3 Role Cards above to launch!';
        }
      }
      return st;
    } catch (err) {
      return null;
    }
  }

  function startLogPolling() {
    if (!logPollTimer) {
      logPollTimer = setInterval(refreshNodeState, 1200);
    }
  }

  function stopLogPolling() {
    if (logPollTimer) {
      clearInterval(logPollTimer);
      logPollTimer = null;
    }
  }

  async function startNodeRole(role) {
    const sel = document.getElementById('rm-model-select');
    const customInput = document.getElementById('rm-custom-peers');
    const model = (sel && sel.value) ? sel.value : 'test.gguf';
    const peers = (customInput && customInput.value) ? customInput.value.trim() : '';
    const title = document.getElementById('rm-console-title');
    if (title) title.textContent = `⏳ Starting ${role.toUpperCase()} (${model})... Watch live logs below:`;

    try {
      startLogPolling();
      await fetch(window.getApiUrl('/api/node/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, model, peers }),
      });

      await refreshNodeState();

      if (role === 'coordinator' || role === 'all') {
        for (let i = 0; i < 45; i++) {
          await new Promise(r => setTimeout(r, 700));
          await refreshNodeState();
          try {
            const cs = await fetch(window.getApiUrl('/api/cluster/status'));
            if (cs.ok) {
              const cd = await cs.json();
              if (cd.connected) {
                if (typeof window.fetchModels === 'function') window.fetchModels();
                if (typeof window.pollClusterStatus === 'function') window.pollClusterStatus();
                closeRoleModal();
                break;
              }
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      if (title) title.textContent = `❌ Error starting role: ${e.message || e}`;
    }
  }

  async function openRoleModal() {
    createLauncherModal();
    const modal = document.getElementById('aeromesh-role-modal');
    if (modal) modal.style.display = 'flex';
    // Check state and peers ONCE on open (only poll continuously if a node is actively running!)
    const st = await refreshNodeState();
    scanPeersIntoModal();
    if (st && st.running) {
      startLogPolling();
    } else {
      stopLogPolling();
    }
  }

  function closeRoleModal() {
    const modal = document.getElementById('aeromesh-role-modal');
    if (modal) modal.style.display = 'none';
    stopLogPolling();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => openRoleModal());
  } else {
    openRoleModal();
  }
})();
