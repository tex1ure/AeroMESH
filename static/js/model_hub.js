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

  const state = {
    activeView: 'chat', // 'chat' | 'models'
    activeSubTab: 'discover', // 'discover' | 'local'
    library: null,
    downloads: [],
    uploads: [], // local XHR upload transfers
    downloadPollTimer: null,
    pendingDeleteName: null,
    expandedRepos: {}, // repo -> { loading, error, files }
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
    const normalized = view === 'models' ? 'models' : (view === 'cluster' ? 'cluster' : 'chat');
    state.activeView = normalized;
    if (subTab && normalized === 'models') setSubTab(subTab);

    const isChat = normalized === 'chat';
    const isModels = normalized === 'models';
    const isCluster = normalized === 'cluster';

    const clusterView = document.getElementById('cluster-pair-view');
    const navCluster = document.getElementById('nav-tab-cluster');

    if (dom.chatViewport) dom.chatViewport.style.display = isChat ? '' : 'none';
    if (dom.composerAnchor) dom.composerAnchor.style.display = isChat ? '' : 'none';
    if (dom.hubView) dom.hubView.style.display = isModels ? 'block' : 'none';
    if (clusterView) clusterView.style.display = isCluster ? 'block' : 'none';

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
      renderLibrary();
      renderTransfers();
      updateDownloadPolling();
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
      const data = await resp.json();
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
      const data = await resp.json();
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
      const data = await resp.json();
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
      const data = await resp.json();
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
      const data = await resp.json();
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
      const data = await resp.json();
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
      const data = await resp.json();
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

      const pct = Number(t.percent || 0).toFixed(1);
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
          <div class="mh-progress-fill ${done ? 'is-done' : err ? 'is-err' : ''}" style="width: ${Math.min(100, Math.max(0, Number(t.percent || 0)))}%"></div>
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

      opt.innerHTML = `
        <div style="min-width: 0; flex: 1;">
          <div class="mh-file-option-name">${esc(f.filename)}</div>
          <div class="mh-meta-line">${esc(meta)}</div>
        </div>
      `;
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
      card.className = 'mh-catalog-card';
      const isInstalled = installedNames.has((item.recommended_file || '').toLowerCase());

      card.innerHTML = `
        <div class="mh-card-title">${esc(item.name)}</div>
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

      row.innerHTML = `
        <div class="mh-list-main">
          <div class="mh-list-name">${esc(m.name)}</div>
          <div class="mh-meta-line">${esc(meta)}</div>
          <div class="mh-list-path">${esc(m.path)}</div>
        </div>
        <div class="mh-list-actions" data-model-actions></div>
      `;

      const actions = row.querySelector('[data-model-actions]');
      if (m.kind === 'llm') {
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
