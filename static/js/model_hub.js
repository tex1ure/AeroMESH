/**
 * AeroMESH Model Hub — LM Studio / Ollama-style Model Installer & Local Library
 *
 * Supports:
 * - Switching cleanly between Chat and Model Hub workspaces
 * - Curated 1-click GGUF model installs + live Hugging Face Hub GGUF search
 * - Inline quantization file browser with size & VRAM estimates
 * - Live download progress, speed (MB/s), ETA, and cancellation
 * - Manual file path or directory registration (zero copy by default)
 * - Native Windows File / Folder browser dialog
 * - Drag-and-drop or file-picker streaming upload into AeroMESH's managed folder
 */
(function () {
  'use strict';

  const getApiUrl = (path) => {
    if (typeof window.getApiUrl === 'function') return window.getApiUrl(path);
    const p = String(path || '').startsWith('/') ? path : '/' + path;
    return p;
  };

  const esc = (v) => {
    if (typeof window.escapeHtml === 'function') return window.escapeHtml(v);
    if (v === null || v === undefined) return '';
    return String(v)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  };

  const STAGE_COLORS = ['#2563eb', '#0d9488', '#7c3aed', '#d97706', '#059669', '#db2777', '#0284c7', '#65a30d'];

  const SIM_PRESETS = {
    'rtx3060-6g': { name: 'Sim RTX 3060', gpu_name: 'NVIDIA GeForce RTX 3060 Laptop GPU', vram_total_mb: 6144, ram_total_mb: 16384 },
    'rtx4060-8g': { name: 'Sim RTX 4060', gpu_name: 'NVIDIA GeForce RTX 4060', vram_total_mb: 8192, ram_total_mb: 16384 },
    'rtx4070-12g': { name: 'Sim RTX 4070', gpu_name: 'NVIDIA GeForce RTX 4070', vram_total_mb: 12288, ram_total_mb: 32768 },
    'rtx4080-16g': { name: 'Sim RTX 4080', gpu_name: 'NVIDIA GeForce RTX 4080', vram_total_mb: 16384, ram_total_mb: 32768 },
    'rtx4090-24g': { name: 'Sim RTX 4090', gpu_name: 'NVIDIA GeForce RTX 4090', vram_total_mb: 24576, ram_total_mb: 65536 },
    'cpu-16g': { name: 'Sim CPU Worker', gpu_name: 'CPU Only (System RAM)', vram_total_mb: 0, ram_total_mb: 16384 },
  };

  const state = {
    activeView: 'chat', // 'chat' | 'models'
    activeSubTab: 'discover', // 'discover' | 'local'
    library: null,
    downloads: [],
    uploads: [], // local XHR upload transfers
    downloadPollTimer: null,
    pendingDeleteName: null,
    expandedRepos: {}, // repo -> { loading, error, files }
    // Hardware-Aware VRAM Fit Meter & Auto Layer-Split Planner
    fitMode: 'cluster', // 'cluster' | 'local'
    fitContext: 4096,
    hardwareNodes: [],
    simulatedNodes: [],
    selectedFitModel: null, // { name, displayName, params, quant, size_bytes, size_hint }
    fitPlan: null,
    fitLoading: false,
    fitRequestSeq: 0,
  };

  const dom = {};

  function cacheDom() {
    dom.navChat = document.getElementById('nav-tab-chat');
    dom.navModels = document.getElementById('nav-tab-models');
    dom.navDlCount = document.getElementById('nav-models-dl-count');
    dom.btnNewChat = document.getElementById('btn-new-chat');
    dom.historyList = document.getElementById('chat-list-today');

    dom.chatViewport = document.getElementById('chat-viewport');
    dom.composerAnchor = document.querySelector('.bottom-input-anchor');
    dom.hubView = document.getElementById('models-hub-view');

    dom.storageSummary = document.getElementById('mh-storage-summary');
    dom.tabDiscover = document.getElementById('mh-tab-discover');
    dom.tabLocal = document.getElementById('mh-tab-local');
    dom.installedCount = document.getElementById('mh-installed-count');
    dom.refreshBtn = document.getElementById('mh-refresh-btn');
    dom.feedback = document.getElementById('mh-feedback');

    dom.downloadsPanel = document.getElementById('mh-downloads-panel');
    dom.downloadsList = document.getElementById('mh-downloads-list');

    dom.fitCard = document.getElementById('mh-fit-planner-card');
    dom.fitPoolSummary = document.getElementById('mh-fit-pool-summary');
    dom.fitModeCluster = document.getElementById('mh-fit-mode-cluster');
    dom.fitModeLocal = document.getElementById('mh-fit-mode-local');
    dom.fitCtxSelect = document.getElementById('mh-fit-ctx-select');
    dom.fitSimSelect = document.getElementById('mh-fit-sim-select');
    dom.fitSimAddBtn = document.getElementById('mh-fit-sim-add-btn');
    dom.fitNodeChips = document.getElementById('mh-fit-node-chips');
    dom.fitPlanBody = document.getElementById('mh-fit-plan-body');

    dom.paneDiscover = document.getElementById('mh-pane-discover');
    dom.paneLocal = document.getElementById('mh-pane-local');

    dom.hfQuery = document.getElementById('mh-hf-query');
    dom.hfSearchBtn = document.getElementById('mh-hf-search-btn');
    dom.hfTokenToggle = document.getElementById('mh-hf-token-toggle');
    dom.hfTokenRow = document.getElementById('mh-hf-token-row');
    dom.hfTokenInput = document.getElementById('mh-hf-token-input');
    dom.hfResultsSection = document.getElementById('mh-hf-results-section');
    dom.hfResultsTitle = document.getElementById('mh-hf-results-title');
    dom.hfResultsList = document.getElementById('mh-hf-results-list');
    dom.hfClearBtn = document.getElementById('mh-hf-clear-btn');
    dom.catalogGrid = document.getElementById('mh-catalog-grid');

    dom.manualPathInput = document.getElementById('mh-manual-path-input');
    dom.importPathBtn = document.getElementById('mh-import-path-btn');
    dom.browseFileBtn = document.getElementById('mh-browse-file-btn');
    dom.browseFolderBtn = document.getElementById('mh-browse-folder-btn');
    dom.copyOnImport = document.getElementById('mh-copy-on-import');

    dom.dropzone = document.getElementById('mh-dropzone');
    dom.fileInput = document.getElementById('mh-file-input');
    dom.pickUploadBtn = document.getElementById('mh-pick-upload-btn');

    dom.installedList = document.getElementById('mh-installed-list');
    dom.foldersList = document.getElementById('mh-folders-list');
  }

  let feedbackTimer = null;
  function showFeedback(message, kind = 'info') {
    if (!dom.feedback) return;
    dom.feedback.textContent = message;
    dom.feedback.className = 'mh-feedback' + (kind === 'ok' ? ' is-ok' : kind === 'error' ? ' is-error' : '');
    dom.feedback.style.display = 'block';
    if (feedbackTimer) clearTimeout(feedbackTimer);
    if (kind !== 'error') {
      feedbackTimer = setTimeout(() => {
        if (dom.feedback) dom.feedback.style.display = 'none';
      }, 6500);
    }
  }

  function formatBytes(bytes) {
    const n = Number(bytes || 0);
    if (n >= 1024 ** 3) return (n / (1024 ** 3)).toFixed(2) + ' GB';
    if (n >= 1024 ** 2) return (n / (1024 ** 2)).toFixed(1) + ' MB';
    if (n >= 1024) return Math.round(n / 1024) + ' KB';
    return n + ' B';
  }

  function formatEta(seconds) {
    const s = Math.max(0, Math.round(Number(seconds || 0)));
    if (!s) return '';
    if (s < 60) return `${s}s left`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}m ${rem}s left`;
  }

  function formatCount(n) {
    const num = Number(n || 0);
    if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
    if (num >= 1_000) return (num / 1_000).toFixed(1) + 'K';
    return String(num);
  }

  // ---------------------------------------------------------------------------
  // VIEW & SUB-TAB SWITCHING
  // ---------------------------------------------------------------------------
  function setWorkspaceView(view, subTab) {
    const normalized = view === 'models'
      ? 'models'
      : (view === 'cluster' ? 'cluster' : (view === 'api' ? 'api' : 'chat'));
    state.activeView = normalized;
    if (subTab && normalized === 'models') setSubTab(subTab);

    const isChat = normalized === 'chat';
    const isModels = normalized === 'models';
    const isCluster = normalized === 'cluster';
    const isApi = normalized === 'api';

    const clusterView = document.getElementById('cluster-pair-view');
    const navCluster = document.getElementById('nav-tab-cluster');
    const apiView = document.getElementById('dev-api-view');
    const navApi = document.getElementById('nav-tab-api');

    if (dom.chatViewport) dom.chatViewport.style.display = isChat ? '' : 'none';
    if (dom.composerAnchor) dom.composerAnchor.style.display = isChat ? '' : 'none';
    if (dom.hubView) dom.hubView.style.display = isModels ? 'block' : 'none';
    if (clusterView) clusterView.style.display = isCluster ? 'block' : 'none';
    if (apiView) apiView.style.display = isApi ? 'block' : 'none';

    if (dom.navChat) {
      dom.navChat.classList.toggle('active', isChat);
      dom.navChat.setAttribute('aria-selected', String(isChat));
    }
    if (dom.navModels) {
      dom.navModels.classList.toggle('active', isModels);
      dom.navModels.setAttribute('aria-selected', String(isModels));
    }
    if (navCluster) {
      navCluster.classList.toggle('active', isCluster);
      navCluster.setAttribute('aria-selected', String(isCluster));
    }
    if (navApi) {
      navApi.classList.toggle('active', isApi);
      navApi.setAttribute('aria-selected', String(isApi));
    }

    if (isModels) {
      refreshLibrary();
      const roleModal = document.getElementById('aeromesh-role-modal');
      if (roleModal) roleModal.style.display = 'none';
    } else if (isCluster) {
      const roleModal = document.getElementById('aeromesh-role-modal');
      if (roleModal) roleModal.style.display = 'none';
      if (typeof window.refreshClusterPairing === 'function') {
        window.refreshClusterPairing();
      }
    } else if (isApi) {
      const roleModal = document.getElementById('aeromesh-role-modal');
      if (roleModal) roleModal.style.display = 'none';
      if (typeof window.refreshDevServerPanel === 'function') {
        window.refreshDevServerPanel();
      }
    }
  }

  function setSubTab(tab) {
    state.activeSubTab = tab === 'local' ? 'local' : 'discover';
    const isLocal = state.activeSubTab === 'local';
    if (dom.paneDiscover) dom.paneDiscover.style.display = isLocal ? 'none' : 'flex';
    if (dom.paneLocal) dom.paneLocal.style.display = isLocal ? 'flex' : 'none';
    if (dom.tabDiscover) {
      dom.tabDiscover.classList.toggle('active', !isLocal);
      dom.tabDiscover.setAttribute('aria-selected', String(!isLocal));
    }
    if (dom.tabLocal) {
      dom.tabLocal.classList.toggle('active', isLocal);
      dom.tabLocal.setAttribute('aria-selected', String(isLocal));
    }
  }

  window.setWorkspaceView = setWorkspaceView;

  window.openModelHub = function (subTab) {
    setWorkspaceView('models', subTab || 'discover');
  };

  window.openChatWorkspace = function () {
    setWorkspaceView('chat');
  };

  window.openClusterPairing = function () {
    setWorkspaceView('cluster');
  };

  window.openDevApiServer = function () {
    setWorkspaceView('api');
  };

  // ---------------------------------------------------------------------------
  // HARDWARE-AWARE VRAM FIT METER & AUTO LAYER-SPLIT PLANNER
  // ---------------------------------------------------------------------------
  function mbToGbLabel(mb) {
    const n = Number(mb || 0);
    if (n >= 1024) return `${(n / 1024).toFixed(1)} GB`;
    return `${Math.round(n)} MB`;
  }

  function fitStatusClass(status) {
    if (status === 'fits_vram') return 'is-ok';
    if (status === 'tight_vram') return 'is-tight';
    if (status === 'ram_offload') return 'is-spill';
    return 'is-oom';
  }

  function getActiveHardwareNodes() {
    const detected = Array.isArray(state.hardwareNodes) ? state.hardwareNodes : [];
    const base = detected.length > 0
      ? detected
      : [{ id: 'local', name: 'This PC', role: 'coordinator', gpu_name: 'Local GPU / CPU', vram_total_mb: 0, ram_total_mb: 16384 }];

    if (state.fitMode === 'local') {
      return [base[0]];
    }
    return [...base, ...state.simulatedNodes].slice(0, 8);
  }

  function parseHintToMb(sizeBytes, sizeHint) {
    if (Number(sizeBytes) > 0) return Number(sizeBytes) / (1024 * 1024);
    const m = String(sizeHint || '').match(/(\d+(?:\.\d+)?)\s*(TB|GB|MB)/i);
    if (!m) return 0;
    const val = parseFloat(m[1]);
    const unit = m[2].toUpperCase();
    if (unit === 'TB') return val * 1024 * 1024;
    if (unit === 'GB') return val * 1024;
    return val;
  }

  function estimateParamsB(paramsHint, nameHint, weightMb) {
    const src = `${paramsHint || ''} ${nameHint || ''}`;
    const m = src.match(/(?:^|[^A-Za-z0-9])(\d+(?:\.\d+)?)\s*[bB](?:[^A-Za-z0-9]|$)/);
    if (m) return parseFloat(m[1]);
    if (weightMb > 0) return Math.max(0.5, (weightMb * 8) / (4.85 * 1000));
    return 7.0;
  }

  function quickFitSummary({ name, sizeBytes, sizeHint, paramsHint }) {
    // If this model matches the currently computed server-side plan, mirror its exact verdict
    if (
      state.fitPlan &&
      state.selectedFitModel &&
      (state.selectedFitModel.name || '').toLowerCase() === String(name || '').toLowerCase()
    ) {
      return {
        status: state.fitPlan.fit_status,
        cls: fitStatusClass(state.fitPlan.fit_status),
        label: state.fitPlan.fit_label,
      };
    }

    const nodes = getActiveHardwareNodes();
    const weightMb = parseHintToMb(sizeBytes, sizeHint);
    if (weightMb <= 0) return null;

    const paramsB = estimateParamsB(paramsHint, name, weightMb);
    // Approximate GQA FP16 KV cache at 4K ctx (matches vram_planner.py profiles closely)
    let kv4kMb = 224;
    if (paramsB <= 2.2) kv4kMb = 112;
    else if (paramsB <= 4.5) kv4kMb = 144;
    else if (paramsB <= 9.5) kv4kMb = 224;
    else if (paramsB <= 18) kv4kMb = 768;
    else if (paramsB <= 42) kv4kMb = 1024;
    else kv4kMb = 1280;

    const kvMb = kv4kMb * (Number(state.fitContext || 4096) / 4096.0);
    const totalModelMb = weightMb + kvMb;

    let totalVramMb = 0;
    let totalUsableVramMb = 0;
    let totalScratchMb = 0;
    let totalUsableRamMb = 0;
    let localUsableVramMb = 0;

    nodes.forEach((n, idx) => {
      const vram = Math.max(0, Number(n.vram_total_mb || 0));
      const ram = Math.max(2048, Number(n.ram_total_mb || 16384));
      totalVramMb += vram;
      if (vram > 0) {
        const osRes = Math.min(Math.max(512, vram * 0.08), 1536);
        const scratch = Math.min(Math.max(280, vram * 0.05), 900);
        const usable = Math.max(0, vram - osRes - scratch);
        totalScratchMb += scratch;
        totalUsableVramMb += usable;
        if (idx === 0) localUsableVramMb = usable;
      }
      totalUsableRamMb += Math.max(0, ram * 0.72 - 1536);
    });

    if (totalVramMb > 0 && totalModelMb <= totalUsableVramMb) {
      const pct = ((totalModelMb + totalScratchMb) / Math.max(1, totalVramMb)) * 100;
      if (pct <= 88) {
        if (nodes.length > 1 && localUsableVramMb > 0 && totalModelMb <= localUsableVramMb) {
          return { status: 'fits_vram', cls: 'is-ok', label: 'Fits Local GPU' };
        }
        return {
          status: 'fits_vram',
          cls: 'is-ok',
          label: nodes.length > 1 ? `${nodes.length}-Node VRAM Fit` : 'Full VRAM Fit',
        };
      }
      return { status: 'tight_vram', cls: 'is-tight', label: 'Tight VRAM Fit' };
    }
    if (totalModelMb <= totalUsableVramMb + totalUsableRamMb) {
      return {
        status: 'ram_offload',
        cls: 'is-spill',
        label: totalVramMb > 0 ? 'GPU + RAM Spill' : 'CPU RAM Only',
      };
    }
    return { status: 'wont_fit', cls: 'is-oom', label: 'Exceeds Memory' };
  }

  function renderFitHardwareHeader() {
    if (!dom.fitCard) return;
    const nodes = getActiveHardwareNodes();
    const totalVram = nodes.reduce((a, n) => a + Number(n.vram_total_mb || 0), 0);
    const totalRam = nodes.reduce((a, n) => a + Number(n.ram_total_mb || 0), 0);

    if (dom.fitModeCluster) dom.fitModeCluster.classList.toggle('active', state.fitMode === 'cluster');
    if (dom.fitModeLocal) dom.fitModeLocal.classList.toggle('active', state.fitMode === 'local');

    if (dom.fitPoolSummary) {
      const scopeLabel = state.fitMode === 'local' ? 'This PC only' : `${nodes.length} node${nodes.length === 1 ? '' : 's'} pooled`;
      const simCount = state.fitMode === 'cluster' ? state.simulatedNodes.length : 0;
      const simNote = simCount > 0 ? ` (${simCount} simulated)` : '';
      dom.fitPoolSummary.textContent = `${scopeLabel}${simNote} · ${mbToGbLabel(totalVram)} VRAM · ${mbToGbLabel(totalRam)} System RAM · ${(state.fitContext / 1024)}K context`;
    }

    if (dom.fitNodeChips) {
      dom.fitNodeChips.innerHTML = '';
      nodes.forEach((n, idx) => {
        const chip = document.createElement('div');
        const isSim = Boolean(n._simulated);
        chip.className = 'mh-fit-node-chip' + (isSim ? ' is-sim' : '');
        const color = STAGE_COLORS[idx % STAGE_COLORS.length];
        const roleTag = idx === 0 ? 'Coord' : (isSim ? 'Sim Worker' : 'Worker');
        const vramTxt = Number(n.vram_total_mb || 0) > 0 ? `${mbToGbLabel(n.vram_total_mb)} VRAM` : 'CPU Only';
        const ramTxt = `${mbToGbLabel(n.ram_total_mb || 16384)} RAM`;

        chip.innerHTML = `
          <span class="mh-fit-node-dot" style="background-color:${color};"></span>
          <span class="mh-fit-node-name">${esc(n.name || roleTag)}</span>
          <span class="mh-fit-node-spec">${esc(roleTag)} · ${esc(vramTxt)} · ${esc(ramTxt)}</span>
        `;

        if (isSim) {
          const rmBtn = document.createElement('button');
          rmBtn.type = 'button';
          rmBtn.className = 'mh-fit-node-remove';
          rmBtn.title = 'Remove simulated worker';
          rmBtn.setAttribute('aria-label', `Remove ${n.name || 'simulated worker'}`);
          rmBtn.textContent = '×';
          rmBtn.onclick = () => {
            state.simulatedNodes = state.simulatedNodes.filter((s) => s.id !== n.id);
            onHardwareConfigChanged();
          };
          chip.appendChild(rmBtn);
        }
        dom.fitNodeChips.appendChild(chip);
      });
    }
  }

  function renderFitPlanBody() {
    if (!dom.fitPlanBody) return;
    if (state.fitLoading && !state.fitPlan) {
      dom.fitPlanBody.innerHTML = '<div class="mh-meta-line">Computing exact VRAM budget and proportional layer split…</div>';
      return;
    }
    const plan = state.fitPlan;
    if (!plan || !plan.model) {
      dom.fitPlanBody.innerHTML = '<div class="mh-meta-line">Click <strong>Plan Split</strong> on any model below to calculate exact VRAM budgets and layer ranges.</div>';
      return;
    }

    const m = plan.model;
    const totals = plan.cluster_totals || {};
    const stages = Array.isArray(plan.stages) ? plan.stages : [];
    const recs = Array.isArray(plan.recommendations) ? plan.recommendations : [];
    const badgeCls = fitStatusClass(plan.fit_status);
    const displayName = (state.selectedFitModel && state.selectedFitModel.displayName) || m.name;
    const archTag = m.arch_source === 'gguf_header' ? 'Exact GGUF header' : 'Estimated arch';

    const splitSegsHtml = stages.map((st, idx) => {
      const pct = Math.max(4, Math.min(100, Number(st.share_percent || 0)));
      const bg = STAGE_COLORS[idx % STAGE_COLORS.length];
      const hasRamSpill = Number(st.ram_layers || 0) > 0;
      const spillSuffix = hasRamSpill ? ` · ${st.ram_layers}L RAM` : '';
      const fullLabel = `${st.node_name} · L${st.layer_range || '-'} (${st.layer_count}L${spillSuffix})`;
      const visibleLabel = pct < 14
        ? `L${st.layer_range || '-'}`
        : pct < 26
          ? `L${st.layer_range || '-'} (${st.layer_count}L)`
          : fullLabel;
      return `<div class="mh-split-seg" style="width:${pct}%;background-color:${bg};" title="${esc(fullLabel)}"><span class="mh-split-seg-label">${esc(visibleLabel)}</span></div>`;
    }).join('');

    const stageCardsHtml = stages.map((st, idx) => {
      const color = STAGE_COLORS[idx % STAGE_COLORS.length];
      const roleTitle = idx === 0 ? 'Stage 0 · Coordinator' : `Stage ${idx} · Worker`;
      const vramPct = Math.max(0, Math.min(100, Number(st.vram_percent || 0)));
      const vramFillCls = Number(st.ram_layers || 0) > 0 ? 'is-spill' : (vramPct > 88 ? 'is-tight' : 'is-ok');
      const ramPct = Math.max(0, Math.min(100, Number(st.ram_percent || 0)));
      const ramFillCls = Number(st.ram_used_mb || 0) > Number(st.ram_total_mb || 1) * 0.72 ? 'is-oom' : 'is-spill';
      const extras = [
        st.includes_embeddings ? 'Embeddings' : '',
        st.includes_lm_head ? 'LM Head' : '',
        `${mbToGbLabel(st.weights_mb)} weights`,
        `${mbToGbLabel(st.kv_cache_mb)} KV`,
      ].filter(Boolean).join(' · ');

      const vramRow = Number(st.vram_total_mb || 0) > 0
        ? `
          <div class="mh-meter-row">
            <div class="mh-meter-head">
              <span>GPU VRAM (${st.gpu_layers}/${st.layer_count} layers)</span>
              <span class="mh-meter-val">${esc(st.vram_used_label)} / ${esc(st.vram_total_label)} (${Math.round(vramPct)}%)</span>
            </div>
            <div class="mh-meter-track">
              <div class="mh-meter-fill ${vramFillCls}" style="width:${vramPct}%;"></div>
            </div>
          </div>
        `
        : `<div class="mh-meter-head"><span>GPU VRAM</span><span class="mh-meter-val">No discrete GPU (CPU RAM)</span></div>`;

      const ramRow = Number(st.ram_used_mb || 0) > 0
        ? `
          <div class="mh-meter-row">
            <div class="mh-meter-head">
              <span>System RAM Spill (${st.ram_layers} layers)</span>
              <span class="mh-meter-val">${esc(st.ram_used_label)} / ${esc(st.ram_total_label)} (${Math.round(ramPct)}%)</span>
            </div>
            <div class="mh-meter-track">
              <div class="mh-meter-fill ${ramFillCls}" style="width:${ramPct}%;"></div>
            </div>
          </div>
        `
        : `
          <div class="mh-meter-row">
            <div class="mh-meter-head">
              <span>System RAM Spill</span>
              <span class="mh-meter-val" style="color:#4ade80;">0 MB (100% GPU)</span>
            </div>
          </div>
        `;

      return `
        <div class="mh-stage-card">
          <div class="mh-stage-top">
            <div style="min-width:0;">
              <div class="mh-stage-node-title">
                <span class="mh-fit-node-dot" style="background-color:${color};"></span>
                <span>${esc(st.node_name)}</span>
              </div>
              <div class="mh-stage-gpu">${esc(roleTitle)} · ${esc(st.gpu_name)}</div>
            </div>
            ${st.cli_flag ? `<code class="mh-stage-flag" title="CLI layer slice flag">${esc(st.cli_flag)}</code>` : ''}
          </div>
          ${vramRow}
          ${ramRow}
          <div class="mh-stage-foot">${esc(extras)}</div>
        </div>
      `;
    }).join('');

    const recsHtml = recs.length > 0
      ? `
        <div class="mh-fit-recs">
          <div class="mh-fit-rec-title">Fit Recommendations</div>
          ${recs.map((r) => `<div class="mh-fit-rec-item">• ${esc(r)}</div>`).join('')}
        </div>
      `
      : '';

    dom.fitPlanBody.innerHTML = `
      <div class="mh-fit-verdict-row">
        <div class="mh-fit-model-heading">
          <div class="mh-fit-model-name">${esc(displayName)}</div>
          <div class="mh-fit-model-meta">
            ${esc(m.params_label || '')} · ${esc(m.quant || '')} · ${esc(m.size_label || '')} weights · ${esc(totals.kv_cache_label || '')} KV (${Math.round((plan.context_length || 4096) / 1024)}K ctx) · ${m.layers} layers · ${m.per_token_kb} KB/tok wire (${esc(archTag)})
          </div>
        </div>
        <div class="mh-fit-verdict-right">
          <span class="mh-fit-speed-tier">${esc(plan.speed_tier || '')}</span>
          <span id="mh-fit-verdict-badge" class="mh-fit-badge ${badgeCls}">${esc(plan.fit_label || '')}</span>
        </div>
      </div>
      <p class="mh-fit-summary-text">${esc(plan.summary || '')}</p>
      <div id="mh-split-bar" class="mh-split-bar" role="img" aria-label="Proportional multi-node layer split">${splitSegsHtml}</div>
      <div class="mh-stage-grid">${stageCardsHtml}</div>
      ${recsHtml}
    `;
  }

  async function requestFitPlan(modelInfo, scrollIntoView = false) {
    if (!modelInfo || !modelInfo.name) return;
    state.selectedFitModel = modelInfo;
    state.fitLoading = true;
    const reqId = ++state.fitRequestSeq;

    renderFitHardwareHeader();
    renderCatalog();
    renderInstalledModels();
    if (scrollIntoView && dom.fitCard) {
      dom.fitCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const activeNodes = getActiveHardwareNodes();
    const payload = {
      model: modelInfo.name,
      size_bytes: Number(modelInfo.size_bytes || 0),
      size_hint: modelInfo.size_hint || '',
      params: modelInfo.params || '',
      quant: modelInfo.quant || '',
      context_length: Number(state.fitContext || 4096),
      mode: state.fitMode,
    };
    if (state.fitMode === 'cluster' && state.simulatedNodes.length > 0) {
      payload.custom_nodes = activeNodes.map((n) => ({
        id: n.id,
        name: n.name,
        role: n.role,
        gpu_name: n.gpu_name,
        vram_total_mb: Number(n.vram_total_mb || 0),
        ram_total_mb: Number(n.ram_total_mb || 16384),
        endpoint: n.endpoint || '',
      }));
    }

    try {
      const resp = await fetch(getApiUrl('/api/cluster/fit/plan'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const plan = await resp.json();
      if (reqId !== state.fitRequestSeq) return;
      state.fitPlan = plan;
      state.fitLoading = false;
      renderFitPlanBody();
      renderCatalog();
      renderInstalledModels();
      if (lastHfResults.length > 0) renderHfResults(lastHfResults);
    } catch (err) {
      if (reqId !== state.fitRequestSeq) return;
      state.fitLoading = false;
      if (dom.fitPlanBody) {
        dom.fitPlanBody.innerHTML = `<div class="mh-meta-line" style="color:#fca5a5;">Could not compute VRAM split plan: ${esc(err.message || err)}</div>`;
      }
    }
  }

  function onHardwareConfigChanged() {
    renderFitHardwareHeader();
    renderCatalog();
    renderInstalledModels();
    if (lastHfResults.length > 0) renderHfResults(lastHfResults);
    if (state.selectedFitModel) {
      requestFitPlan(state.selectedFitModel, false);
    }
  }

  function pickDefaultFitModel(lib) {
    if (!lib) return null;
    const catalog = Array.isArray(lib.catalog) ? lib.catalog : [];
    const models = Array.isArray(lib.models) ? lib.models.filter((m) => m.kind === 'llm') : [];
    // Prefer a 14B catalog or installed model so users immediately see a clear multi-node split
    const installed14 = models.find((m) => /14b/i.test(m.name));
    if (installed14) {
      return {
        name: installed14.name,
        displayName: installed14.name,
        quant: installed14.quant || '',
        size_bytes: installed14.size_bytes || 0,
        size_hint: installed14.size_label || '',
      };
    }
    const cat14 = catalog.find((c) => /14b/i.test(c.id || c.params || c.name || ''));
    if (cat14) {
      return {
        name: cat14.recommended_file || cat14.name,
        displayName: `${cat14.name} (${cat14.quant})`,
        params: cat14.params || '',
        quant: cat14.quant || '',
        size_hint: cat14.size_hint || '',
      };
    }
    if (models.length > 0) {
      const largest = [...models].sort((a, b) => Number(b.size_bytes || 0) - Number(a.size_bytes || 0))[0];
      return {
        name: largest.name,
        displayName: largest.name,
        quant: largest.quant || '',
        size_bytes: largest.size_bytes || 0,
        size_hint: largest.size_label || '',
      };
    }
    if (catalog.length > 0) {
      const c = catalog[0];
      return {
        name: c.recommended_file || c.name,
        displayName: `${c.name} (${c.quant})`,
        params: c.params || '',
        quant: c.quant || '',
        size_hint: c.size_hint || '',
      };
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // API CALLS
  // ---------------------------------------------------------------------------
  async function refreshLibrary() {
    try {
      const resp = await fetch(getApiUrl('/api/models/library'));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.library = data;
      state.downloads = Array.isArray(data.downloads) ? data.downloads : [];
      if (data.cluster_hardware && Array.isArray(data.cluster_hardware.nodes)) {
        state.hardwareNodes = data.cluster_hardware.nodes;
      }
      renderLibrary();
      renderTransfers();
      updateDownloadPolling();
      if (!state.selectedFitModel) {
        const defModel = pickDefaultFitModel(data);
        if (defModel) requestFitPlan(defModel, false);
      } else {
        requestFitPlan(state.selectedFitModel, false);
      }
    } catch (err) {
      showFeedback(`Could not load model library: ${err.message || err}`, 'error');
    }
  }

  async function pollDownloads() {
    try {
      const resp = await fetch(getApiUrl('/api/models/downloads'));
      if (!resp.ok) return;
      const data = await resp.json();
      const prevActive = state.downloads.filter((d) => d.status === 'downloading' || d.status === 'starting').length;
      state.downloads = Array.isArray(data.downloads) ? data.downloads : [];
      const nowActive = state.downloads.filter((d) => d.status === 'downloading' || d.status === 'starting').length;
      renderTransfers();
      updateDownloadPolling();
      if (prevActive > nowActive) {
        await refreshLibrary();
        if (typeof window.fetchModels === 'function') window.fetchModels();
      }
    } catch (_) {}
  }

  function updateDownloadPolling() {
    const activeDownloads = state.downloads.filter(
      (d) => d.status === 'downloading' || d.status === 'starting'
    ).length;
    const activeUploads = state.uploads.filter((u) => u.status === 'uploading').length;
    const totalActive = activeDownloads + activeUploads;

    if (dom.navDlCount) {
      if (totalActive > 0) {
        dom.navDlCount.textContent = `(${totalActive})`;
        dom.navDlCount.style.display = 'inline';
      } else {
        dom.navDlCount.style.display = 'none';
      }
    }

    if (activeDownloads > 0 && !state.downloadPollTimer) {
      state.downloadPollTimer = setInterval(pollDownloads, 900);
    } else if (activeDownloads === 0 && state.downloadPollTimer) {
      clearInterval(state.downloadPollTimer);
      state.downloadPollTimer = null;
    }
  }

  async function startDownload({ url, repo, filename }) {
    const hfToken = dom.hfTokenInput ? dom.hfTokenInput.value.trim() : '';
    try {
      const resp = await fetch(getApiUrl('/api/models/download'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, repo, filename, hf_token: hfToken }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || `Download failed (HTTP ${resp.status})`);
      }
      if (data.download) {
        state.downloads.unshift(data.download);
        renderTransfers();
        updateDownloadPolling();
      }
      showFeedback(`Started downloading ${filename || repo}…`, 'ok');
    } catch (err) {
      showFeedback(err.message || String(err), 'error');
    }
  }

  async function cancelDownload(downloadId) {
    try {
      await fetch(getApiUrl('/api/models/download/cancel'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: downloadId }),
      });
      setTimeout(pollDownloads, 300);
    } catch (_) {}
  }

  async function inspectHfRepo(repo) {
    if (state.expandedRepos[repo] && state.expandedRepos[repo].files) {
      delete state.expandedRepos[repo];
      renderCatalog();
      renderHfResults(lastHfResults);
      return;
    }
    state.expandedRepos[repo] = { loading: true, error: null, files: null };
    renderCatalog();
    renderHfResults(lastHfResults);

    try {
      const resp = await fetch(getApiUrl(`/api/models/hf/files?repo=${encodeURIComponent(repo)}`));
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.expandedRepos[repo] = { loading: false, error: null, files: data.files || [] };
    } catch (err) {
      state.expandedRepos[repo] = { loading: false, error: err.message || String(err), files: [] };
    }
    renderCatalog();
    renderHfResults(lastHfResults);
  }

  let lastHfResults = [];

  async function runHfSearch() {
    const q = dom.hfQuery ? dom.hfQuery.value.trim() : '';
    if (!q) {
      showFeedback('Type a model name (e.g. Qwen 2.5, DeepSeek, Llama 3.2), owner/repo, or a Hugging Face .gguf URL.', 'info');
      return;
    }

    // 1. Direct Hugging Face .gguf URL pasted into search box
    if (/^https:\/\/(www\.)?(huggingface\.co|hf\.co)\/.*\.(gguf|safetensors)(\?.*)?$/i.test(q)) {
      const cleanUrl = q.split('?')[0];
      const fname = cleanUrl.split('/').pop();
      await startDownload({ url: cleanUrl, filename: fname, repo: cleanUrl });
      return;
    }

    // 2. Direct owner/repo pasted into search box
    if (/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(q)) {
      lastHfResults = [{ repo: q, author: q.split('/')[0], downloads: 0, likes: 0, gated: false }];
      if (dom.hfResultsSection) dom.hfResultsSection.style.display = 'flex';
      if (dom.hfResultsTitle) dom.hfResultsTitle.textContent = `Repository: ${q}`;
      await inspectHfRepo(q);
      return;
    }

    if (dom.hfSearchBtn) {
      dom.hfSearchBtn.disabled = true;
      dom.hfSearchBtn.textContent = 'Searching…';
    }
    try {
      const resp = await fetch(getApiUrl(`/api/models/hf/search?q=${encodeURIComponent(q)}&limit=12`));
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      lastHfResults = Array.isArray(data.results) ? data.results : [];
      if (dom.hfResultsSection) dom.hfResultsSection.style.display = 'flex';
      if (dom.hfResultsTitle) dom.hfResultsTitle.textContent = `Hugging Face GGUF Results for "${q}" (${lastHfResults.length})`;
      renderHfResults(lastHfResults);
    } catch (err) {
      showFeedback(`Search failed: ${err.message || err}`, 'error');
    } finally {
      if (dom.hfSearchBtn) {
        dom.hfSearchBtn.disabled = false;
        dom.hfSearchBtn.textContent = 'Search Hub';
      }
    }
  }

  async function importManualPath() {
    const rawPath = dom.manualPathInput ? dom.manualPathInput.value.trim() : '';
    if (!rawPath) {
      showFeedback('Enter or browse for a local .gguf file path or models folder first.', 'error');
      return;
    }
    const copy = Boolean(dom.copyOnImport && dom.copyOnImport.checked);
    if (dom.importPathBtn) {
      dom.importPathBtn.disabled = true;
      dom.importPathBtn.textContent = copy ? 'Copying…' : 'Linking…';
    }
    try {
      const resp = await fetch(getApiUrl('/api/models/import-path'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: rawPath, copy_to_library: copy }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (data.library) {
        state.library = data.library;
        renderLibrary();
      } else {
        await refreshLibrary();
      }
      if (typeof window.fetchModels === 'function') window.fetchModels();
      if (dom.manualPathInput) dom.manualPathInput.value = '';
      const msg = data.imported_type === 'directory'
        ? `Added model folder: ${data.path}`
        : `Imported model "${data.model}" (${copy ? 'copied to AeroMESH folder' : 'linked in-place'}).`;
      showFeedback(msg, 'ok');
    } catch (err) {
      showFeedback(err.message || String(err), 'error');
    } finally {
      if (dom.importPathBtn) {
        dom.importPathBtn.disabled = false;
        dom.importPathBtn.textContent = 'Add Path';
      }
    }
  }

  async function browseNativePath(mode) {
    try {
      const resp = await fetch(getApiUrl('/api/models/browse'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (!data.supported) {
        showFeedback(data.message || 'Paste a local path directly in the input box.', 'info');
        if (dom.manualPathInput) dom.manualPathInput.focus();
        return;
      }
      if (data.path && dom.manualPathInput) {
        dom.manualPathInput.value = data.path;
        await importManualPath();
      }
    } catch (err) {
      showFeedback(`Could not open native picker: ${err.message || err}`, 'error');
    }
  }

  async function removeCustomFolder(folderPath) {
    try {
      const resp = await fetch(getApiUrl('/api/models/paths'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'remove', path: folderPath }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (data.library) {
        state.library = data.library;
        renderLibrary();
      }
      if (typeof window.fetchModels === 'function') window.fetchModels();
      showFeedback(`Unlinked folder: ${folderPath}`, 'ok');
    } catch (err) {
      showFeedback(err.message || String(err), 'error');
    }
  }

  async function deleteManagedModel(name) {
    if (state.pendingDeleteName !== name) {
      state.pendingDeleteName = name;
      renderInstalledModels();
      return;
    }
    state.pendingDeleteName = null;
    try {
      const resp = await fetch(getApiUrl('/api/models/delete'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (data.library) {
        state.library = data.library;
        renderLibrary();
      }
      if (typeof window.fetchModels === 'function') window.fetchModels();
      showFeedback(`Deleted ${name} from managed folder.`, 'ok');
    } catch (err) {
      showFeedback(err.message || String(err), 'error');
    }
  }

  function uploadFile(file) {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.gguf') && !lower.endsWith('.safetensors')) {
      showFeedback(`Unsupported file "${file.name}". Drop a .gguf or .safetensors model file.`, 'error');
      return;
    }

    const uploadId = 'up-' + Date.now();
    const item = {
      id: uploadId,
      filename: file.name,
      status: 'uploading',
      downloaded_bytes: 0,
      total_bytes: file.size || 0,
      percent: 0,
      speed_mbps: 0,
      eta_seconds: 0,
      error: null,
      isUpload: true,
      xhr: null,
    };
    state.uploads.unshift(item);
    renderTransfers();
    updateDownloadPolling();

    const xhr = new XMLHttpRequest();
    item.xhr = xhr;
    const t0 = performance.now();

    xhr.open('POST', getApiUrl(`/api/models/upload?filename=${encodeURIComponent(file.name)}`));
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');

    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      item.downloaded_bytes = ev.loaded;
      item.total_bytes = ev.total;
      item.percent = Math.round((ev.loaded / ev.total) * 1000) / 10;
      const elapsed = Math.max(0.05, (performance.now() - t0) / 1000);
      item.speed_mbps = Math.round(((ev.loaded / (1024 * 1024)) / elapsed) * 100) / 100;
      item.eta_seconds = ev.total > ev.loaded ? Math.round((ev.total - ev.loaded) / (ev.loaded / elapsed)) : 0;
      renderTransfers();
    };

    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        item.status = 'completed';
        item.percent = 100;
        item.eta_seconds = 0;
        renderTransfers();
        updateDownloadPolling();
        await refreshLibrary();
        if (typeof window.fetchModels === 'function') window.fetchModels();
        showFeedback(`Imported "${file.name}" into your local library.`, 'ok');
      } else {
        let msg = `Upload failed (HTTP ${xhr.status})`;
        try {
          const j = JSON.parse(xhr.responseText);
          if (j.detail) msg = j.detail;
        } catch (_) {}
        item.status = 'error';
        item.error = msg;
        renderTransfers();
        updateDownloadPolling();
        showFeedback(msg, 'error');
      }
    };

    xhr.onerror = () => {
      item.status = 'error';
      item.error = 'Network error while streaming file to local server.';
      renderTransfers();
      updateDownloadPolling();
    };

    xhr.onabort = () => {
      item.status = 'cancelled';
      item.error = 'Upload cancelled.';
      renderTransfers();
      updateDownloadPolling();
    };

    xhr.send(file);
  }

  async function loadModelAndReturnToChat(modelName) {
    showFeedback(`Loading ${modelName} into AeroMESH…`, 'info');
    if (typeof window.switchActiveModel === 'function') {
      const ok = await window.switchActiveModel(modelName);
      if (ok) {
        showFeedback(`Loaded ${modelName}. Switching to Chat…`, 'ok');
        setWorkspaceView('chat');
      } else {
        showFeedback(`Could not load ${modelName}. Check that your node role is running.`, 'error');
      }
    }
  }

  // ---------------------------------------------------------------------------
  // RENDERERS
  // ---------------------------------------------------------------------------
  function renderLibrary() {
    if (!state.library) return;
    const lib = state.library;

    if (dom.storageSummary) {
      const freePart = lib.disk_free_label ? ` · ${esc(lib.disk_free_label)} free` : '';
      dom.storageSummary.innerHTML = `Managed folder: <code>${esc(lib.primary_dir || '')}</code>${freePart}`;
    }

    const totalModels = Array.isArray(lib.models) ? lib.models.length : 0;
    if (dom.installedCount) {
      dom.installedCount.textContent = `(${totalModels})`;
    }

    renderFitHardwareHeader();
    renderFitPlanBody();
    renderCatalog();
    renderInstalledModels();
    renderFolders();
    if (window.lucide) lucide.createIcons();
  }

  function renderTransfers() {
    if (!dom.downloadsPanel || !dom.downloadsList) return;
    const all = [...state.uploads, ...state.downloads];
    if (all.length === 0) {
      dom.downloadsPanel.style.display = 'none';
      return;
    }
    dom.downloadsPanel.style.display = 'flex';
    dom.downloadsList.innerHTML = '';

    all.forEach((t) => {
      const row = document.createElement('div');
      row.className = 'mh-dl-row';
      const active = t.status === 'downloading' || t.status === 'starting' || t.status === 'uploading';
      const done = t.status === 'completed';
      const err = t.status === 'error' || t.status === 'cancelled';

      const rawPct = Number(t.percent);
      const safePct = Number.isFinite(rawPct) ? Math.min(100, Math.max(0, rawPct)) : 0;
      const pct = safePct.toFixed(1);
      const bytesStr = t.total_bytes > 0
        ? `${formatBytes(t.downloaded_bytes)} / ${formatBytes(t.total_bytes)}`
        : formatBytes(t.downloaded_bytes);
      const speedStr = t.speed_mbps > 0 ? `${t.speed_mbps} MB/s` : '';
      const etaStr = formatEta(t.eta_seconds);
      const metaParts = active
        ? [`${pct}%`, bytesStr, speedStr, etaStr].filter(Boolean).join(' · ')
        : done
          ? `Completed · ${bytesStr}`
          : (t.error || t.status);

      row.innerHTML = `
        <div class="mh-dl-top">
          <div>
            <div class="mh-dl-name">${esc(t.filename)}</div>
            <div class="mh-dl-meta">${esc(metaParts)}</div>
          </div>
          <div class="mh-list-actions" data-transfer-actions></div>
        </div>
        <div class="mh-progress-track">
          <div class="mh-progress-fill ${done ? 'is-done' : err ? 'is-err' : ''}" style="width: ${safePct}%"></div>
        </div>
      `;

      const actions = row.querySelector('[data-transfer-actions]');
      if (active) {
        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button';
        cancelBtn.className = 'mh-btn-danger';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = () => {
          if (t.isUpload && t.xhr) t.xhr.abort();
          else cancelDownload(t.id);
        };
        actions.appendChild(cancelBtn);
      } else if (done && t.filename.toLowerCase().endsWith('.gguf')) {
        const loadBtn = document.createElement('button');
        loadBtn.type = 'button';
        loadBtn.className = 'mh-btn-primary mh-btn-sm';
        loadBtn.textContent = 'Load Model';
        loadBtn.onclick = () => loadModelAndReturnToChat(t.filename);
        actions.appendChild(loadBtn);
      }

      dom.downloadsList.appendChild(row);
    });
  }

  function renderFilesDrawer(repo) {
    const info = state.expandedRepos[repo];
    if (!info) return null;
    const drawer = document.createElement('div');
    drawer.className = 'mh-files-drawer';
    if (info.loading) {
      drawer.innerHTML = '<div class="mh-meta-line">Loading .gguf quantizations from Hugging Face…</div>';
      return drawer;
    }
    if (info.error) {
      drawer.innerHTML = `<div class="mh-meta-line" style="color:#fca5a5;">${esc(info.error)}</div>`;
      return drawer;
    }
    if (!info.files || info.files.length === 0) {
      drawer.innerHTML = '<div class="mh-meta-line">No .gguf files found in this repository.</div>';
      return drawer;
    }

    info.files.slice(0, 14).forEach((f) => {
      const opt = document.createElement('div');
      opt.className = 'mh-file-option';
      const vramTxt = f.est_vram_gb > 0 ? `Est. ${f.est_vram_gb} GB VRAM` : '';
      const recTxt = f.recommended ? 'Recommended' : '';
      const meta = [f.quant, f.size_label, vramTxt, recTxt].filter(Boolean).join(' · ');
      const fit = quickFitSummary({
        name: f.filename,
        sizeBytes: f.size_bytes,
        sizeHint: f.size_label,
      });
      const fitBadgeHtml = fit ? `<span class="mh-fit-badge ${fit.cls}">${esc(fit.label)}</span>` : '';

      opt.innerHTML = `
        <div style="min-width: 0; flex: 1;">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <div class="mh-file-option-name">${esc(f.filename)}</div>
            ${fitBadgeHtml}
          </div>
          <div class="mh-meta-line">${esc(meta)}</div>
        </div>
      `;

      const planBtn = document.createElement('button');
      planBtn.type = 'button';
      planBtn.className = 'clay-btn mh-btn-secondary';
      planBtn.textContent = 'Plan Split';
      planBtn.onclick = () => requestFitPlan({
        name: f.filename,
        displayName: f.filename,
        quant: f.quant || '',
        size_bytes: Number(f.size_bytes || 0),
        size_hint: f.size_label || '',
      }, true);
      opt.appendChild(planBtn);

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = f.recommended ? 'mh-btn-primary mh-btn-sm' : 'clay-btn mh-btn-secondary';
      btn.textContent = 'Download';
      btn.onclick = () => startDownload({ url: f.download_url, repo, filename: f.filename });
      opt.appendChild(btn);
      drawer.appendChild(opt);
    });
    return drawer;
  }

  function renderCatalog() {
    if (!dom.catalogGrid || !state.library) return;
    const catalog = Array.isArray(state.library.catalog) ? state.library.catalog : [];
    const installedNames = new Set((state.library.models || []).map((m) => m.name.toLowerCase()));
    dom.catalogGrid.innerHTML = '';

    catalog.forEach((item) => {
      const card = document.createElement('div');
      const modelKey = (item.recommended_file || item.name || '').toLowerCase();
      const isPlanned = Boolean(
        state.selectedFitModel &&
        (state.selectedFitModel.name || '').toLowerCase() === modelKey
      );
      card.className = 'mh-catalog-card' + (isPlanned ? ' is-planned' : '');
      const isInstalled = installedNames.has((item.recommended_file || '').toLowerCase());
      const fit = quickFitSummary({
        name: item.recommended_file || item.name,
        sizeHint: item.size_hint,
        paramsHint: item.params,
      });
      const fitBadgeHtml = fit ? `<span class="mh-fit-badge ${fit.cls}">${esc(fit.label)}</span>` : '';

      card.innerHTML = `
        <div class="mh-card-head-row">
          <div class="mh-card-title">${esc(item.name)}</div>
          ${fitBadgeHtml}
        </div>
        <div class="mh-meta-line">
          <span>${esc(item.params)}</span>
          <span class="mh-meta-sep" aria-hidden="true">·</span>
          <span>${esc(item.quant)}</span>
          <span class="mh-meta-sep" aria-hidden="true">·</span>
          <span>${esc(item.size_hint)}</span>
          <span class="mh-meta-sep" aria-hidden="true">·</span>
          <span>${esc(item.vram_hint)}</span>
        </div>
        <div class="mh-card-desc">${esc(item.description)}</div>
        <div class="mh-card-actions" data-card-actions></div>
      `;

      const actions = card.querySelector('[data-card-actions]');
      if (isInstalled) {
        const loadBtn = document.createElement('button');
        loadBtn.type = 'button';
        loadBtn.className = 'mh-btn-primary mh-btn-sm';
        loadBtn.textContent = 'Load Installed';
        loadBtn.onclick = () => loadModelAndReturnToChat(item.recommended_file);
        actions.appendChild(loadBtn);
      } else {
        const dlBtn = document.createElement('button');
        dlBtn.type = 'button';
        dlBtn.className = 'mh-btn-primary mh-btn-sm';
        dlBtn.textContent = `Install ${item.quant}`;
        dlBtn.onclick = () => startDownload({ repo: item.repo, filename: item.recommended_file });
        actions.appendChild(dlBtn);
      }

      const planBtn = document.createElement('button');
      planBtn.type = 'button';
      planBtn.className = 'clay-btn mh-btn-secondary';
      planBtn.setAttribute('data-plan-model', item.id || item.recommended_file || item.name);
      planBtn.textContent = isPlanned ? 'Planned ✓' : 'Plan Split';
      planBtn.onclick = () => requestFitPlan({
        name: item.recommended_file || item.name,
        displayName: `${item.name} (${item.quant})`,
        params: item.params || '',
        quant: item.quant || '',
        size_hint: item.size_hint || '',
      }, true);
      actions.appendChild(planBtn);

      const quantBtn = document.createElement('button');
      quantBtn.type = 'button';
      quantBtn.className = 'clay-btn mh-btn-secondary';
      quantBtn.textContent = state.expandedRepos[item.repo] ? 'Hide Quants' : 'All Quantizations';
      quantBtn.onclick = () => inspectHfRepo(item.repo);
      actions.appendChild(quantBtn);

      const drawer = renderFilesDrawer(item.repo);
      if (drawer) card.appendChild(drawer);

      dom.catalogGrid.appendChild(card);
    });
  }

  function renderHfResults(results) {
    if (!dom.hfResultsList) return;
    dom.hfResultsList.innerHTML = '';
    if (!results || results.length === 0) {
      dom.hfResultsList.innerHTML = '<div class="mh-empty-state">No GGUF repositories matched your search.</div>';
      return;
    }

    results.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'mh-repo-row';
      const meta = [
        r.author,
        r.downloads ? `${formatCount(r.downloads)} downloads` : '',
        r.likes ? `${formatCount(r.likes)} likes` : '',
        r.gated ? 'Gated (HF Token required)' : '',
      ].filter(Boolean).join(' · ');

      row.innerHTML = `
        <div class="mh-repo-row-head">
          <div>
            <div class="mh-list-name">${esc(r.repo)}</div>
            <div class="mh-meta-line">${esc(meta)}</div>
          </div>
          <div class="mh-list-actions" data-repo-actions></div>
        </div>
      `;

      const actions = row.querySelector('[data-repo-actions]');
      const inspectBtn = document.createElement('button');
      inspectBtn.type = 'button';
      inspectBtn.className = 'clay-btn mh-btn-secondary';
      inspectBtn.textContent = state.expandedRepos[r.repo] ? 'Hide Files' : 'Choose Quantization';
      inspectBtn.onclick = () => inspectHfRepo(r.repo);
      actions.appendChild(inspectBtn);

      const drawer = renderFilesDrawer(r.repo);
      if (drawer) row.appendChild(drawer);

      dom.hfResultsList.appendChild(row);
    });
  }

  function renderInstalledModels() {
    if (!dom.installedList || !state.library) return;
    const models = Array.isArray(state.library.models) ? state.library.models : [];
    dom.installedList.innerHTML = '';

    if (models.length === 0) {
      dom.installedList.innerHTML = '<div class="mh-empty-state">No .gguf or .safetensors models found yet. Download one from Discover, drop a file above, or add your existing LM Studio / Ollama models folder.</div>';
      return;
    }

    models.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'mh-list-row';
      const kindLabel = m.kind === 'diffusion' ? 'Diffusion' : 'LLM';
      const locLabel = m.managed ? 'Managed' : 'Linked Path';
      const meta = [kindLabel, m.quant, m.size_label, locLabel].filter(Boolean).join(' · ');
      const fit = m.kind === 'llm'
        ? quickFitSummary({ name: m.name, sizeBytes: m.size_bytes, sizeHint: m.size_label })
        : null;
      const fitBadgeHtml = fit ? `<span class="mh-fit-badge ${fit.cls}">${esc(fit.label)}</span>` : '';

      row.innerHTML = `
        <div class="mh-list-main">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <div class="mh-list-name">${esc(m.name)}</div>
            ${fitBadgeHtml}
          </div>
          <div class="mh-meta-line">${esc(meta)}</div>
          <div class="mh-list-path">${esc(m.path)}</div>
        </div>
        <div class="mh-list-actions" data-model-actions></div>
      `;

      const actions = row.querySelector('[data-model-actions]');
      if (m.kind === 'llm') {
        const planBtn = document.createElement('button');
        planBtn.type = 'button';
        planBtn.className = 'clay-btn mh-btn-secondary';
        planBtn.textContent = 'Plan Split';
        planBtn.onclick = () => requestFitPlan({
          name: m.name,
          displayName: m.name,
          quant: m.quant || '',
          size_bytes: Number(m.size_bytes || 0),
          size_hint: m.size_label || '',
        }, true);
        actions.appendChild(planBtn);

        const loadBtn = document.createElement('button');
        loadBtn.type = 'button';
        loadBtn.className = 'mh-btn-primary mh-btn-sm';
        loadBtn.textContent = 'Load in Cluster';
        loadBtn.onclick = () => loadModelAndReturnToChat(m.name);
        actions.appendChild(loadBtn);
      }

      if (m.managed) {
        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'mh-btn-danger';
        delBtn.textContent = state.pendingDeleteName === m.name ? 'Confirm Delete?' : 'Delete';
        delBtn.onclick = () => deleteManagedModel(m.name);
        actions.appendChild(delBtn);
      }

      dom.installedList.appendChild(row);
    });
  }

  function renderFolders() {
    if (!dom.foldersList || !state.library) return;
    const dirs = Array.isArray(state.library.directories) ? state.library.directories : [];
    dom.foldersList.innerHTML = '';

    if (dirs.length === 0) {
      dom.foldersList.innerHTML = '<div class="mh-empty-state">No model folders configured.</div>';
      return;
    }

    dirs.forEach((d) => {
      const row = document.createElement('div');
      row.className = 'mh-list-row';
      const kindLabel = d.kind === 'primary' ? 'Primary (Downloads & Drops)' : d.kind === 'custom' ? 'Custom Folder' : 'Default Fallback';
      const statusLabel = d.exists ? `${d.model_count} model${d.model_count === 1 ? '' : 's'}` : 'Not created yet';

      row.innerHTML = `
        <div class="mh-list-main">
          <div class="mh-list-name" style="font-family: 'JetBrains Mono', monospace; font-size: 13px;">${esc(d.path)}</div>
          <div class="mh-meta-line">${esc(kindLabel)} · ${esc(statusLabel)}</div>
        </div>
        <div class="mh-list-actions" data-folder-actions></div>
      `;

      const actions = row.querySelector('[data-folder-actions]');
      if (d.kind === 'custom') {
        const rmBtn = document.createElement('button');
        rmBtn.type = 'button';
        rmBtn.className = 'mh-btn-danger';
        rmBtn.textContent = 'Unlink';
        rmBtn.onclick = () => removeCustomFolder(d.path);
        actions.appendChild(rmBtn);
      }

      dom.foldersList.appendChild(row);
    });
  }

  // ---------------------------------------------------------------------------
  // EVENT WIRING
  // ---------------------------------------------------------------------------
  function wireEvents() {
    if (dom.navChat) dom.navChat.addEventListener('click', () => setWorkspaceView('chat'));
    if (dom.navModels) dom.navModels.addEventListener('click', () => setWorkspaceView('models'));

    // Returning to chat when user clicks New Chat or an existing conversation
    if (dom.btnNewChat) {
      dom.btnNewChat.addEventListener('click', () => {
        if (state.activeView !== 'chat') setWorkspaceView('chat');
      });
    }
    if (dom.historyList) {
      dom.historyList.addEventListener('click', () => {
        if (state.activeView !== 'chat') setWorkspaceView('chat');
      });
    }

    if (dom.tabDiscover) dom.tabDiscover.addEventListener('click', () => setSubTab('discover'));
    if (dom.tabLocal) dom.tabLocal.addEventListener('click', () => setSubTab('local'));
    if (dom.refreshBtn) dom.refreshBtn.addEventListener('click', refreshLibrary);

    // VRAM Fit Meter & Auto Layer-Split Planner controls
    if (dom.fitModeCluster) {
      dom.fitModeCluster.addEventListener('click', () => {
        state.fitMode = 'cluster';
        onHardwareConfigChanged();
      });
    }
    if (dom.fitModeLocal) {
      dom.fitModeLocal.addEventListener('click', () => {
        state.fitMode = 'local';
        onHardwareConfigChanged();
      });
    }
    if (dom.fitCtxSelect) {
      dom.fitCtxSelect.addEventListener('change', () => {
        state.fitContext = Number(dom.fitCtxSelect.value || 4096);
        onHardwareConfigChanged();
      });
    }
    if (dom.fitSimAddBtn && dom.fitSimSelect) {
      dom.fitSimAddBtn.addEventListener('click', () => {
        const presetKey = dom.fitSimSelect.value;
        const preset = SIM_PRESETS[presetKey];
        if (!preset) return;
        const activeCount = (state.hardwareNodes.length || 1) + state.simulatedNodes.length;
        if (activeCount >= 8) {
          showFeedback('Maximum of 8 cluster nodes supported in the split planner.', 'info');
          return;
        }
        state.fitMode = 'cluster';
        const simIdx = state.simulatedNodes.length + 1;
        state.simulatedNodes.push({
          id: `sim-${Date.now()}-${simIdx}`,
          name: `${preset.name} #${simIdx}`,
          role: 'worker',
          gpu_name: preset.gpu_name,
          vram_total_mb: preset.vram_total_mb,
          ram_total_mb: preset.ram_total_mb,
          _simulated: true,
        });
        onHardwareConfigChanged();
      });
    }

    if (dom.hfSearchBtn) dom.hfSearchBtn.addEventListener('click', runHfSearch);
    if (dom.hfQuery) {
      dom.hfQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          runHfSearch();
        }
      });
    }
    if (dom.hfTokenToggle && dom.hfTokenRow) {
      dom.hfTokenToggle.addEventListener('click', () => {
        const visible = dom.hfTokenRow.style.display !== 'none';
        dom.hfTokenRow.style.display = visible ? 'none' : 'flex';
      });
    }
    if (dom.hfClearBtn && dom.hfResultsSection) {
      dom.hfClearBtn.addEventListener('click', () => {
        lastHfResults = [];
        dom.hfResultsSection.style.display = 'none';
        if (dom.hfQuery) dom.hfQuery.value = '';
      });
    }

    if (dom.importPathBtn) dom.importPathBtn.addEventListener('click', importManualPath);
    if (dom.manualPathInput) {
      dom.manualPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          importManualPath();
        }
      });
    }
    if (dom.browseFileBtn) dom.browseFileBtn.addEventListener('click', () => browseNativePath('file'));
    if (dom.browseFolderBtn) dom.browseFolderBtn.addEventListener('click', () => browseNativePath('folder'));

    // Drag-and-drop & file picker upload
    if (dom.pickUploadBtn && dom.fileInput) {
      dom.pickUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        dom.fileInput.click();
      });
    }
    if (dom.dropzone && dom.fileInput) {
      dom.dropzone.addEventListener('click', (e) => {
        if (e.target === dom.pickUploadBtn) return;
        dom.fileInput.click();
      });
      dom.dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          dom.fileInput.click();
        }
      });
      ['dragenter', 'dragover'].forEach((evt) => {
        dom.dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dom.dropzone.classList.add('is-dragover');
        });
      });
      ['dragleave', 'drop'].forEach((evt) => {
        dom.dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dom.dropzone.classList.remove('is-dragover');
        });
      });
      dom.dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length > 0) {
          Array.from(files).forEach(uploadFile);
        }
      });
      dom.fileInput.addEventListener('change', () => {
        if (dom.fileInput.files && dom.fileInput.files.length > 0) {
          Array.from(dom.fileInput.files).forEach(uploadFile);
          dom.fileInput.value = '';
        }
      });
    }
  }

  function init() {
    cacheDom();
    if (!dom.hubView) return;
    wireEvents();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
