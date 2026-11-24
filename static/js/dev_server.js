/**
 * AeroMESH Developer / OpenAI-Compatible Local API Server Panel Controller
 *
 * Manages:
 * - Live Base URL, Gateway Proxy URL, LAN & Tailscale URL display + one-click copy
 * - Masked API Key display, reveal/hide toggle, one-click copy, and key rotation
 * - LAN / Tailscale exposure toggle (127.0.0.1 vs 0.0.0.0 with mandatory Bearer auth)
 * - Copy-paste IDE & SDK snippets (cURL, OpenAI Python SDK, Tools+JSON, Continue.dev, Cline/Roo, Cursor/Open WebUI)
 * - Live protocol verification probes (GET /v1/models, JSON Mode, Tool Calling)
 */
(function initDevApiServerPanel() {
  'use strict';

  const state = {
    config: null,
    keyRevealed: false,
    activeSnippet: 'curl',
    includeLiveKey: false,
    busy: false,
  };

  function getApiUrl(path) {
    if (typeof window.getApiUrl === 'function') return window.getApiUrl(path);
    return path;
  }

  function esc(v) {
    if (typeof window.escapeHtml === 'function') return window.escapeHtml(v);
    return String(v ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[c]));
  }

  const dom = {};

  function cacheDom() {
    dom.navApi = document.getElementById('nav-tab-api');
    dom.view = document.getElementById('dev-api-view');
    dom.statusDot = document.getElementById('dev-status-dot');
    dom.statusSummary = document.getElementById('dev-status-summary');
    dom.testEndpointBtn = document.getElementById('dev-test-endpoint-btn');
    dom.refreshBtn = document.getElementById('dev-refresh-btn');
    dom.feedback = document.getElementById('dev-feedback');

    dom.bindBadge = document.getElementById('dev-bind-badge');
    dom.baseUrlInput = document.getElementById('dev-base-url-input');
    dom.copyUrlBtn = document.getElementById('dev-copy-url-btn');
    dom.altUrls = document.getElementById('dev-alt-urls');

    dom.apiKeyInput = document.getElementById('dev-api-key-input');
    dom.keyRevealBtn = document.getElementById('dev-key-reveal-btn');
    dom.copyKeyBtn = document.getElementById('dev-copy-key-btn');
    dom.rotateKeyBtn = document.getElementById('dev-rotate-key-btn');

    dom.lanToggle = document.getElementById('dev-lan-toggle');
    dom.lanDesc = document.getElementById('dev-lan-desc');
    dom.statCtx = document.getElementById('dev-stat-ctx');
    dom.statNgl = document.getElementById('dev-stat-ngl');
    dom.statModel = document.getElementById('dev-stat-model');
    dom.openInfDrawerBtn = document.getElementById('dev-open-inf-drawer-btn');
    dom.applyRestartBtn = document.getElementById('dev-apply-restart-btn');

    dom.includeKeyToggle = document.getElementById('dev-include-key-toggle');
    dom.copySnippetBtn = document.getElementById('dev-copy-snippet-btn');
    dom.snippetTabs = document.querySelectorAll('.dev-snippet-tab');
    dom.snippetCode = document.getElementById('dev-snippet-code');

    dom.probeModelsBtn = document.getElementById('dev-probe-models-btn');
    dom.testJsonBtn = document.getElementById('dev-test-json-btn');
    dom.testToolsBtn = document.getElementById('dev-test-tools-btn');
    dom.testMeta = document.getElementById('dev-test-meta');
    dom.testOutput = document.getElementById('dev-test-output');
  }

  let feedbackTimer = null;
  function showFeedback(message, kind) {
    if (!dom.feedback) return;
    dom.feedback.textContent = message;
    dom.feedback.className = 'mh-feedback' + (kind === 'ok' ? ' is-ok' : kind === 'error' ? ' is-error' : '');
    dom.feedback.style.display = 'block';
    if (feedbackTimer) clearTimeout(feedbackTimer);
    if (kind !== 'error') {
      feedbackTimer = setTimeout(() => {
        if (dom.feedback) dom.feedback.style.display = 'none';
      }, 6000);
    }
  }

  async function copyText(text, btn, doneLabel) {
    if (!text) return;
    let copied = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (_) {
      copied = false;
    }
    if (!copied) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        copied = true;
      } catch (_) {}
      document.body.removeChild(ta);
    }
    if (btn && copied) {
      if (!btn.dataset.origHtml) {
        btn.dataset.origHtml = btn.innerHTML;
      }
      btn.innerHTML = `<i data-lucide="check" style="width: 14px; height: 14px;"></i><span>${esc(doneLabel || 'Copied')}</span>`;
      if (window.lucide) lucide.createIcons();
      if (btn._copyTimer) clearTimeout(btn._copyTimer);
      btn._copyTimer = setTimeout(() => {
        btn.innerHTML = btn.dataset.origHtml;
        if (window.lucide) lucide.createIcons();
      }, 1800);
    }
  }

  function renderSnippet() {
    if (!dom.snippetCode || !state.config) return;
    const snippets = state.config.snippets || {};
    let code = snippets[state.activeSnippet] || snippets.curl || '';
    if (state.includeLiveKey && state.config.api_key) {
      code = code.split('YOUR_AEROMESH_API_KEY').join(state.config.api_key);
    }
    dom.snippetCode.textContent = code;
  }

  function renderConfig(cfg) {
    if (!cfg) return;
    state.config = cfg;

    // Header status
    const online = Boolean(cfg.coordinator_online || cfg.node_running);
    if (dom.statusDot) {
      dom.statusDot.classList.toggle('is-online', online);
      dom.statusDot.classList.toggle('is-offline', !online);
    }
    if (dom.statusSummary) {
      const stateText = cfg.coordinator_online
        ? 'Coordinator Live'
        : (cfg.node_running ? `Node Running (${cfg.node_role})` : 'Ready (Start Coordinator or Local Mesh to serve)');
      dom.statusSummary.textContent = `${stateText} · ${cfg.active_endpoint_url || cfg.base_url} · Tools & JSON Mode Enabled`;
    }

    // Bind badge & URL
    const lanOn = Boolean(cfg.lan_exposed);
    if (dom.bindBadge) {
      dom.bindBadge.textContent = lanOn ? '0.0.0.0 (LAN / Tailscale)' : '127.0.0.1 Only';
      dom.bindBadge.classList.toggle('is-lan', lanOn);
    }
    if (dom.baseUrlInput) {
      dom.baseUrlInput.value = cfg.base_url || 'http://127.0.0.1:8080/v1';
    }

    // Secondary endpoint chips (Gateway, LAN, Tailscale)
    if (dom.altUrls) {
      const altRows = [];
      if (cfg.gateway_base_url) {
        altRows.push({ label: 'Studio Gateway (Tools + JSON)', url: cfg.gateway_base_url });
      }
      if (cfg.lan_url) {
        altRows.push({
          label: lanOn ? 'LAN Endpoint (Active)' : 'LAN Endpoint (Enable toggle to expose)',
          url: cfg.lan_url,
          muted: !lanOn,
        });
      }
      if (cfg.tailscale_url) {
        altRows.push({
          label: lanOn ? 'Tailscale Direct (Active)' : 'Tailscale Direct (Enable toggle to expose)',
          url: cfg.tailscale_url,
          muted: !lanOn,
        });
      }
      dom.altUrls.innerHTML = altRows
        .map(
          (row, idx) => `
          <div class="dev-alt-url-row ${row.muted ? 'is-muted' : ''}">
            <span class="dev-alt-label">${esc(row.label)}:</span>
            <code class="dev-alt-code">${esc(row.url)}</code>
            <button class="cp-inline-link dev-alt-copy-btn" type="button" data-url-idx="${idx}">Copy</button>
          </div>`
        )
        .join('');
      dom.altUrls.querySelectorAll('.dev-alt-copy-btn').forEach((btn) => {
        btn.onclick = () => {
          const idx = Number(btn.getAttribute('data-url-idx'));
          const item = altRows[idx];
          if (item) copyText(item.url, btn, 'Copied');
        };
      });
    }

    // Masked / Revealed API Key
    if (dom.apiKeyInput) {
      dom.apiKeyInput.value = state.keyRevealed
        ? (cfg.api_key || '')
        : (cfg.api_key_masked || 'sk-aerom…');
    }
    if (dom.keyRevealBtn) {
      dom.keyRevealBtn.innerHTML = state.keyRevealed
        ? '<i data-lucide="eye-off" style="width: 14px; height: 14px;"></i><span>Hide</span>'
        : '<i data-lucide="eye" style="width: 14px; height: 14px;"></i><span>Reveal</span>';
    }

    // LAN Toggle & Description
    if (dom.lanToggle) {
      dom.lanToggle.checked = lanOn;
    }
    if (dom.lanDesc) {
      if (lanOn) {
        const lanTarget = cfg.lan_url || 'http://<this-pc-ip>:8080/v1';
        dom.lanDesc.innerHTML = `Bound to <code>0.0.0.0:8080</code>. Other devices can connect via <code>${esc(lanTarget)}</code> using the Bearer API Key.`;
      } else {
        dom.lanDesc.innerHTML = 'Currently bound to <code>127.0.0.1</code> (this PC only). Toggle ON to bind <code>0.0.0.0:8080</code> with mandatory Bearer API Key auth.';
      }
    }

    // Runtime Defaults summary
    if (dom.statCtx) {
      dom.statCtx.textContent = `${Number(cfg.n_ctx || 4096).toLocaleString()} tok`;
    }
    if (dom.statNgl) {
      const ngl = Number(cfg.ngl ?? -1);
      dom.statNgl.textContent = ngl < 0 ? 'All GPU (-1)' : (ngl === 0 ? '0 (CPU Only)' : `${ngl} layers`);
    }
    if (dom.statModel) {
      dom.statModel.textContent = cfg.active_model || '—';
      dom.statModel.title = cfg.active_model || '';
    }

    renderSnippet();
    if (window.lucide) lucide.createIcons();

    // Notify Inference Settings Drawer so n_ctx and ngl stay synchronized
    if (typeof window.syncInferenceRuntimeFromServer === 'function') {
      window.syncInferenceRuntimeFromServer(cfg);
    }
  }

  async function refreshDevServerPanel() {
    try {
      const resp = await fetch(getApiUrl('/api/server/config'));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const cfg = await resp.json();
      renderConfig(cfg);
      return cfg;
    } catch (err) {
      showFeedback(`Could not load API server configuration: ${err.message}`, 'error');
      return null;
    }
  }
  window.refreshDevServerPanel = refreshDevServerPanel;

  async function saveLanExposure(lanExposed, applyRestart) {
    if (state.busy) return;
    state.busy = true;
    try {
      const resp = await fetch(getApiUrl('/api/server/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lan_exposed: Boolean(lanExposed),
          apply_restart: Boolean(applyRestart),
        }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`);
      }
      renderConfig(data);
      const modeMsg = data.lan_exposed
        ? 'LAN / Tailscale exposure enabled (0.0.0.0 with Bearer API Key auth).'
        : 'Server restricted to localhost only (127.0.0.1).';
      const restartMsg = data.restarted ? ' Active node restarted with new settings.' : '';
      showFeedback(modeMsg + restartMsg, 'ok');
    } catch (err) {
      showFeedback(`Failed to update network exposure: ${err.message}`, 'error');
      await refreshDevServerPanel();
    } finally {
      state.busy = false;
    }
  }

  async function rotateApiKey() {
    if (state.busy) return;
    state.busy = true;
    try {
      const resp = await fetch(getApiUrl('/api/server/api-key/rotate'), {
        method: 'POST',
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.keyRevealed = true;
      renderConfig(data);
      const restartNote = data.restarted ? ' Active coordinator restarted with the new key.' : '';
      showFeedback(`Rotated AEROMESH_API_KEY and saved to .env.${restartNote} Update IDE clients with the new key.`, 'ok');
    } catch (err) {
      showFeedback(`Failed to rotate API key: ${err.message}`, 'error');
    } finally {
      state.busy = false;
    }
  }

  function showTestOutput(label, status, ms, payload) {
    if (dom.testMeta) {
      dom.testMeta.style.display = 'flex';
      const ok = status >= 200 && status < 300;
      dom.testMeta.innerHTML = `
        <span class="dev-test-badge ${ok ? 'is-ok' : 'is-err'}">HTTP ${esc(status)}</span>
        <span>${esc(label)}</span>
        <span class="dev-test-ms">${esc(ms)} ms</span>
      `;
    }
    if (dom.testOutput) {
      dom.testOutput.textContent = typeof payload === 'string'
        ? payload
        : JSON.stringify(payload, null, 2);
    }
  }

  async function runModelsProbe() {
    const t0 = performance.now();
    if (dom.testOutput) dom.testOutput.textContent = 'Probing GET /v1/models…';
    try {
      const headers = {};
      if (state.config && state.config.api_key) {
        headers['Authorization'] = `Bearer ${state.config.api_key}`;
      }
      const resp = await fetch(getApiUrl('/v1/models'), { headers });
      const ms = Math.round(performance.now() - t0);
      const rawText = await resp.text();
      let data;
      try { data = JSON.parse(rawText); } catch (_) { data = rawText || { error: `HTTP ${resp.status}` }; }
      showTestOutput('GET /v1/models', resp.status, ms, data);
      if (resp.ok) {
        showFeedback(`OpenAI endpoint healthy (HTTP 200 in ${ms} ms).`, 'ok');
      } else {
        showFeedback(`Endpoint returned HTTP ${resp.status}.`, 'error');
      }
    } catch (err) {
      const ms = Math.round(performance.now() - t0);
      showTestOutput('GET /v1/models', 0, ms, { error: err.message });
      showFeedback(`Endpoint probe failed: ${err.message}`, 'error');
    }
  }

  async function runStructuredTest(mode) {
    const t0 = performance.now();
    const model = (state.config && state.config.active_model) || 'aeromesh-model';
    const headers = { 'Content-Type': 'application/json' };
    if (state.config && state.config.api_key) {
      headers['Authorization'] = `Bearer ${state.config.api_key}`;
    }

    let body;
    let label;
    if (mode === 'json') {
      label = 'POST /v1/chat/completions (response_format: json_object)';
      body = {
        model,
        stream: false,
        temperature: 0.1,
        max_tokens: 180,
        response_format: { type: 'json_object' },
        messages: [
          {
            role: 'user',
            content: 'Return a JSON object with keys "engine" ("AeroMESH"), "protocol" ("OpenAI /v1"), and "status" ("ready").',
          },
        ],
      };
    } else {
      label = 'POST /v1/chat/completions (tools + tool_choice: auto)';
      body = {
        model,
        stream: false,
        temperature: 0.1,
        max_tokens: 200,
        tool_choice: 'auto',
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_cluster_telemetry',
              description: 'Read live GPU VRAM utilization and WireGuard RTT for a cluster node.',
              parameters: {
                type: 'object',
                properties: {
                  node_id: { type: 'string', description: 'Node identifier, e.g. coordinator or worker-1' },
                },
                required: ['node_id'],
              },
            },
          },
        ],
        messages: [
          { role: 'user', content: 'Check live GPU telemetry on node coordinator.' },
        ],
      };
    }

    if (dom.testOutput) dom.testOutput.textContent = `Sending ${label}…`;
    try {
      const resp = await fetch(getApiUrl('/v1/chat/completions'), {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      });
      const ms = Math.round(performance.now() - t0);
      const rawText = await resp.text();
      let data;
      try { data = JSON.parse(rawText); } catch (_) { data = rawText || { error: `HTTP ${resp.status}` }; }
      showTestOutput(label, resp.status, ms, data);
      if (resp.ok) {
        const choice = (data.choices && data.choices[0]) || {};
        if (mode === 'tools' && choice.finish_reason === 'tool_calls') {
          showFeedback(`Tool calling verified! Model emitted ${(choice.message?.tool_calls || []).length} OpenAI tool_call(s) in ${ms} ms.`, 'ok');
        } else if (mode === 'json') {
          showFeedback(`JSON Mode verified in ${ms} ms.`, 'ok');
        } else {
          showFeedback(`Completed request in ${ms} ms (finish_reason: ${choice.finish_reason || 'stop'}).`, 'ok');
        }
      } else {
        showFeedback(`Server returned HTTP ${resp.status}. Start a model/coordinator to run live inference probes.`, 'error');
      }
    } catch (err) {
      const ms = Math.round(performance.now() - t0);
      showTestOutput(label, 0, ms, { error: err.message });
      showFeedback(`Request failed: ${err.message}`, 'error');
    }
  }

  function bindEvents() {
    if (dom.navApi) {
      dom.navApi.addEventListener('click', () => {
        if (typeof window.setWorkspaceView === 'function') {
          window.setWorkspaceView('api');
        }
      });
    }

    if (dom.refreshBtn) {
      dom.refreshBtn.addEventListener('click', () => refreshDevServerPanel());
    }

    if (dom.testEndpointBtn) {
      dom.testEndpointBtn.addEventListener('click', () => runModelsProbe());
    }

    if (dom.copyUrlBtn) {
      dom.copyUrlBtn.addEventListener('click', () => {
        const url = (dom.baseUrlInput && dom.baseUrlInput.value) || 'http://127.0.0.1:8080/v1';
        copyText(url, dom.copyUrlBtn, 'Copied');
      });
    }

    if (dom.keyRevealBtn) {
      dom.keyRevealBtn.addEventListener('click', () => {
        state.keyRevealed = !state.keyRevealed;
        if (state.config) renderConfig(state.config);
      });
    }

    if (dom.copyKeyBtn) {
      dom.copyKeyBtn.addEventListener('click', () => {
        const key = state.config && state.config.api_key;
        if (key) {
          copyText(key, dom.copyKeyBtn, 'Copied');
        }
      });
    }

    if (dom.rotateKeyBtn) {
      dom.rotateKeyBtn.addEventListener('click', () => rotateApiKey());
    }

    if (dom.lanToggle) {
      dom.lanToggle.addEventListener('change', () => {
        saveLanExposure(dom.lanToggle.checked, false);
      });
    }

    if (dom.applyRestartBtn) {
      dom.applyRestartBtn.addEventListener('click', () => {
        const lanOn = dom.lanToggle ? dom.lanToggle.checked : Boolean(state.config && state.config.lan_exposed);
        saveLanExposure(lanOn, true);
      });
    }

    if (dom.openInfDrawerBtn) {
      dom.openInfDrawerBtn.addEventListener('click', () => {
        if (typeof window.openInferenceDrawer === 'function') {
          window.openInferenceDrawer();
        }
      });
    }

    dom.snippetTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const key = tab.getAttribute('data-snippet') || 'curl';
        state.activeSnippet = key;
        dom.snippetTabs.forEach((t) => {
          const isActive = t === tab;
          t.classList.toggle('active', isActive);
          t.setAttribute('aria-selected', String(isActive));
        });
        renderSnippet();
      });
    });

    if (dom.includeKeyToggle) {
      dom.includeKeyToggle.addEventListener('change', () => {
        state.includeLiveKey = Boolean(dom.includeKeyToggle.checked);
        renderSnippet();
      });
    }

    if (dom.copySnippetBtn) {
      dom.copySnippetBtn.addEventListener('click', () => {
        const text = dom.snippetCode ? dom.snippetCode.textContent : '';
        copyText(text, dom.copySnippetBtn, 'Copied');
      });
    }

    if (dom.probeModelsBtn) {
      dom.probeModelsBtn.addEventListener('click', () => runModelsProbe());
    }
    if (dom.testJsonBtn) {
      dom.testJsonBtn.addEventListener('click', () => runStructuredTest('json'));
    }
    if (dom.testToolsBtn) {
      dom.testToolsBtn.addEventListener('click', () => runStructuredTest('tools'));
    }
  }

  function init() {
    cacheDom();
    bindEvents();
    // Pre-fetch server config quietly so runtime n_ctx / ngl syncs with the Inference Drawer on load
    refreshDevServerPanel();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
