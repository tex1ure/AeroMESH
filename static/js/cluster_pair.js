/**
 * AeroMESH Cluster Pairing & Secret Sync Controller
 *
 * Manages one-click LAN / Tailscale cluster pairing (`aeromesh-pair-v1`),
 * live 6-digit PIN + invite token hosting, nearby pairing host discovery,
 * encrypted secret sync, cluster key rotation, and paired peer management.
 */
(function initClusterPairing() {
  'use strict';

  const state = {
    status: null,
    nearbyHosts: [],
    pollTimer: null,
    countdownTimer: null,
    localExpiresIn: 0,
    confirmRotate: false,
    confirmUnpairId: null,
    joining: false,
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
    dom.navCluster = document.getElementById('nav-tab-cluster');
    dom.navPairCount = document.getElementById('nav-cluster-pair-count');
    dom.view = document.getElementById('cluster-pair-view');

    dom.keyFingerprint = document.getElementById('cp-key-fingerprint');
    dom.localIps = document.getElementById('cp-local-ips');
    dom.rotateBtn = document.getElementById('cp-rotate-btn');
    dom.refreshBtn = document.getElementById('cp-refresh-btn');
    dom.feedback = document.getElementById('cp-feedback');

    dom.hostStatusLabel = document.getElementById('cp-host-status-label');
    dom.hostIdle = document.getElementById('cp-host-idle');
    dom.hostActive = document.getElementById('cp-host-active');
    dom.startHostBtn = document.getElementById('cp-start-host-btn');
    dom.regenHostBtn = document.getElementById('cp-regen-host-btn');
    dom.stopHostBtn = document.getElementById('cp-stop-host-btn');
    dom.pinDisplay = document.getElementById('cp-pin-display');
    dom.countdown = document.getElementById('cp-countdown');
    dom.pairedCounter = document.getElementById('cp-paired-counter');
    dom.attemptsCounter = document.getElementById('cp-attempts-counter');
    dom.inviteReadonly = document.getElementById('cp-invite-readonly');
    dom.copyInviteBtn = document.getElementById('cp-copy-invite-btn');
    dom.inviteLanRow = document.getElementById('cp-invite-lan-row');
    dom.inviteLanText = document.getElementById('cp-invite-lan-text');
    dom.copyLanBtn = document.getElementById('cp-copy-lan-btn');

    dom.scanHostsBtn = document.getElementById('cp-scan-hosts-btn');
    dom.nearbyList = document.getElementById('cp-nearby-list');
    dom.joinInviteInput = document.getElementById('cp-join-invite-input');
    dom.joinHostInput = document.getElementById('cp-join-host-input');
    dom.joinPinInput = document.getElementById('cp-join-pin-input');
    dom.joinSubmitBtn = document.getElementById('cp-join-submit-btn');

    dom.peersCount = document.getElementById('cp-peers-count');
    dom.peersList = document.getElementById('cp-peers-list');
  }

  let feedbackTimer = null;
  function showFeedback(message, kind) {
    if (!dom.feedback) return;
    dom.feedback.textContent = message;
    dom.feedback.className = `mh-feedback ${kind || 'info'}`;
    dom.feedback.style.display = 'block';
    if (feedbackTimer) clearTimeout(feedbackTimer);
    if (kind !== 'error') {
      feedbackTimer = setTimeout(() => {
        if (dom.feedback) dom.feedback.style.display = 'none';
      }, 6500);
    }
  }

  function formatCountdown(secs) {
    const s = Math.max(0, Math.floor(secs || 0));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')} remaining`;
  }

  async function copyText(text, btn, labelAfter) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (_) {}
      document.body.removeChild(ta);
    }
    if (btn) {
      const orig = btn.innerHTML;
      btn.textContent = labelAfter || 'Copied!';
      setTimeout(() => {
        btn.innerHTML = orig;
        if (window.lucide) lucide.createIcons();
      }, 1500);
    }
  }

  function parseInviteString(raw) {
    const trimmed = String(raw || '').trim();
    if (!trimmed) return null;
    const stripped = trimmed.replace(/^AERO-/i, '');
    const dashIdx = stripped.lastIndexOf('-');
    if (dashIdx <= 0) return null;
    const hostPart = stripped.slice(0, dashIdx).trim();
    const pinDigits = stripped.slice(dashIdx + 1).replace(/\D/g, '');
    if (!hostPart || pinDigits.length !== 6) return null;
    return {
      host: hostPart,
      pin: `${pinDigits.slice(0, 3)}-${pinDigits.slice(3)}`,
    };
  }

  function formatPinInput(val) {
    const digits = String(val || '').replace(/\D/g, '').slice(0, 6);
    if (digits.length > 3) return `${digits.slice(0, 3)}-${digits.slice(3)}`;
    return digits;
  }

  // ---------------------------------------------------------------------------
  // RENDERING
  // ---------------------------------------------------------------------------
  function renderStatus() {
    const st = state.status;
    if (!st) return;

    if (dom.keyFingerprint) {
      dom.keyFingerprint.textContent = st.cluster_fingerprint || 'UNSET';
    }
    if (dom.localIps) {
      const parts = [];
      if (st.wifi_ip && st.wifi_ip !== '127.0.0.1') parts.push(`Wi-Fi ${st.wifi_ip}`);
      if (st.tailscale_ip && st.tailscale_ip !== 'Not Connected' && st.tailscale_ip !== 'Local') {
        parts.push(`Tailscale ${st.tailscale_ip}`);
      }
      dom.localIps.textContent = parts.length > 0 ? parts.join(' · ') : 'Localhost only';
    }

    const sess = st.session;
    const isActive = !!(sess && sess.active);

    if (dom.hostStatusLabel) {
      dom.hostStatusLabel.textContent = isActive ? `Listening on :${sess.port}` : 'Idle';
      dom.hostStatusLabel.className = isActive ? 'cp-status-text active' : 'cp-status-text';
    }
    if (dom.hostIdle) dom.hostIdle.style.display = isActive ? 'none' : 'flex';
    if (dom.hostActive) dom.hostActive.style.display = isActive ? 'flex' : 'none';

    if (isActive) {
      state.localExpiresIn = Number(sess.expires_in || 0);
      if (dom.pinDisplay) dom.pinDisplay.textContent = sess.pin_formatted || sess.pin || '--- ---';
      if (dom.countdown) dom.countdown.textContent = formatCountdown(state.localExpiresIn);
      if (dom.pairedCounter) {
        const cnt = Number(sess.paired_count || 0);
        dom.pairedCounter.textContent = `${cnt} ${cnt === 1 ? 'node' : 'nodes'} paired`;
      }
      if (dom.attemptsCounter) {
        dom.attemptsCounter.textContent = `${sess.failed_attempts || 0}/${sess.max_attempts || 5} failed`;
      }
      if (dom.inviteReadonly) dom.inviteReadonly.value = sess.invite_code || '';

      if (dom.inviteLanRow && dom.inviteLanText) {
        if (sess.invite_lan && sess.invite_lan !== sess.invite_code) {
          dom.inviteLanText.textContent = sess.invite_lan;
          dom.inviteLanRow.style.display = 'flex';
        } else {
          dom.inviteLanRow.style.display = 'none';
        }
      }
    }

    renderPairedPeers(Array.isArray(st.paired_peers) ? st.paired_peers : [], Array.isArray(st.local_models) ? st.local_models : []);
    syncTimers(isActive);
  }

  function renderPairedPeers(peers, localModels) {
    if (dom.peersCount) dom.peersCount.textContent = String(peers.length);
    if (dom.navPairCount) {
      if (peers.length > 0) {
        dom.navPairCount.textContent = `(${peers.length})`;
        dom.navPairCount.style.display = 'inline';
      } else {
        dom.navPairCount.style.display = 'none';
      }
    }
    if (!dom.peersList) return;

    if (peers.length === 0) {
      dom.peersList.innerHTML = `
        <div class="mh-empty-state">
          No paired nodes yet. Click <strong>Start Pairing Session</strong> on your Coordinator PC, then join from your second PC with the 6-digit code.
        </div>
      `;
      return;
    }

    dom.peersList.innerHTML = peers.map((p) => {
      const synced = p.secret_synced !== false;
      const peerModels = Array.isArray(p.models) ? p.models : [];
      const shared = Array.isArray(p.shared_models)
        ? p.shared_models
        : peerModels.filter((m) => localModels.includes(m));
      const modelNote = shared.length > 0
        ? `${shared.length} matching GGUF ${shared.length === 1 ? 'model' : 'models'} ready`
        : (peerModels.length > 0
          ? `0 matching models (peer has ${peerModels.length})`
          : 'No GGUF models reported yet');
      const syncClass = synced ? 'cp-sync-ok' : 'cp-sync-warn';
      const syncText = synced ? 'Secret Synced' : 'Key Changed — Re-Pair';
      const isConfirming = state.confirmUnpairId === p.id;

      return `
        <div class="mh-row">
          <div class="mh-row-main">
            <div class="mh-row-title">${esc(p.node_name || 'Cluster Node')}</div>
            <div class="mh-row-meta">
              <span class="${syncClass}">${esc(syncText)}</span>
              <span>&middot;</span>
              <span>${esc(p.gpu || 'GPU / CPU')}</span>
              <span>&middot;</span>
              <span>${esc(modelNote)}</span>
            </div>
            <div class="mh-row-path">Worker: ${esc(p.worker_endpoint || '')} &middot; Key: ${esc(p.cluster_fingerprint || 'UNSET')}</div>
          </div>
          <div class="mh-row-actions">
            <button class="mh-btn mh-btn-primary mh-btn-sm cp-use-worker-btn" type="button" data-endpoint="${esc(p.worker_endpoint || '')}">Use as Worker</button>
            <button class="mh-btn mh-btn-danger-ghost mh-btn-sm cp-unpair-btn" type="button" data-peer-id="${esc(p.id || '')}" data-peer-name="${esc(p.node_name || '')}">${isConfirming ? 'Confirm Unpair?' : 'Unpair'}</button>
          </div>
        </div>
      `;
    }).join('');
  }

  function renderNearbyHosts() {
    if (!dom.nearbyList) return;
    if (!state.nearbyHosts || state.nearbyHosts.length === 0) {
      dom.nearbyList.innerHTML = '<div class="cp-nearby-empty">No active pairing hosts found on Wi-Fi/Tailscale (:50053). Start a session on the host PC or paste an Invite Token below.</div>';
      return;
    }

    dom.nearbyList.innerHTML = state.nearbyHosts.map((h) => `
      <button type="button" class="cp-nearby-item" data-host="${esc(h.host)}:${esc(h.port || 50053)}">
        <div class="cp-nearby-item-main">
          <span class="cp-nearby-name">${esc(h.node_name || h.host)}</span>
          <span class="cp-nearby-meta">${esc(h.host)}:${esc(h.port || 50053)} &middot; ${esc(h.network || 'LAN')} &middot; ${esc(h.gpu || 'GPU')}</span>
        </div>
        <span class="cp-nearby-select">Select</span>
      </button>
    `).join('');
  }

  // ---------------------------------------------------------------------------
  // TIMERS & API CALLS
  // ---------------------------------------------------------------------------
  function syncTimers(sessionActive) {
    if (sessionActive) {
      if (!state.countdownTimer) {
        state.countdownTimer = setInterval(() => {
          if (state.localExpiresIn > 0) {
            state.localExpiresIn -= 1;
            if (dom.countdown) dom.countdown.textContent = formatCountdown(state.localExpiresIn);
          } else {
            refreshStatus();
          }
        }, 1000);
      }
      if (!state.pollTimer) {
        state.pollTimer = setInterval(() => {
          if (dom.view && dom.view.style.display !== 'none') {
            refreshStatus(true);
          }
        }, 2500);
      }
    } else {
      if (state.countdownTimer) {
        clearInterval(state.countdownTimer);
        state.countdownTimer = null;
      }
      if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    }
  }

  async function refreshStatus(silent) {
    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/status'));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const prevPairedCount = state.status && state.status.session ? Number(state.status.session.paired_count || 0) : 0;
      const data = await resp.json();
      state.status = data;
      renderStatus();
      if (data.session && Number(data.session.paired_count || 0) > prevPairedCount) {
        showFeedback('New cluster node paired and synced its secret!', 'success');
      }
    } catch (err) {
      if (!silent) showFeedback(`Could not load pairing status: ${err.message || err}`, 'error');
    }
  }

  async function startPairingSession() {
    try {
      if (dom.startHostBtn) dom.startHostBtn.disabled = true;
      const resp = await fetch(getApiUrl('/api/cluster/pair/start'), { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.status = data.pair_status || state.status;
      renderStatus();
      showFeedback('Pairing session live on TCP 50053 for 5 minutes. Share the 6-digit code or invite token with your second PC.', 'success');
    } catch (err) {
      showFeedback(`Could not start pairing session: ${err.message || err}`, 'error');
    } finally {
      if (dom.startHostBtn) dom.startHostBtn.disabled = false;
    }
  }

  async function stopPairingSession() {
    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/stop'), { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.status = data.pair_status || state.status;
      renderStatus();
      showFeedback('Pairing session closed.', 'info');
    } catch (err) {
      showFeedback(`Could not stop pairing session: ${err.message || err}`, 'error');
    }
  }

  async function scanNearbyHosts() {
    if (dom.scanHostsBtn) dom.scanHostsBtn.disabled = true;
    if (dom.nearbyList) {
      dom.nearbyList.innerHTML = '<div class="cp-nearby-empty">Scanning Wi-Fi LAN &amp; Tailscale peers on :50053...</div>';
    }
    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/discover'));
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.nearbyHosts = Array.isArray(data.pairing_hosts) ? data.pairing_hosts : [];
      renderNearbyHosts();
    } catch (err) {
      if (dom.nearbyList) {
        dom.nearbyList.innerHTML = `<div class="cp-nearby-empty">Scan error: ${esc(err.message || err)}</div>`;
      }
    } finally {
      if (dom.scanHostsBtn) dom.scanHostsBtn.disabled = false;
    }
  }

  async function joinCluster() {
    if (state.joining) return;
    const inviteCode = dom.joinInviteInput ? dom.joinInviteInput.value.trim() : '';
    const host = dom.joinHostInput ? dom.joinHostInput.value.trim() : '';
    const pin = dom.joinPinInput ? dom.joinPinInput.value.trim() : '';

    if (!inviteCode && (!host || pin.replace(/\D/g, '').length !== 6)) {
      showFeedback('Paste an AERO-... invite token, or enter both the Host IP and the 6-digit pairing code.', 'error');
      return;
    }

    state.joining = true;
    if (dom.joinSubmitBtn) {
      dom.joinSubmitBtn.disabled = true;
      dom.joinSubmitBtn.textContent = 'Verifying & Syncing Secret...';
    }

    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/join'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ invite_code: inviteCode, host, pin }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

      state.status = data.pair_status || state.status;
      if (dom.joinInviteInput) dom.joinInviteInput.value = '';
      if (dom.joinPinInput) dom.joinPinInput.value = '';
      renderStatus();
      showFeedback(
        `Paired with ${data.paired_with || host}! Cluster secret synced (${data.cluster_fingerprint || ''}) and saved to .env. If a worker/coordinator is already running, restart it from Node Role to apply.`,
        'success'
      );
    } catch (err) {
      showFeedback(err.message || 'Pairing failed.', 'error');
    } finally {
      state.joining = false;
      if (dom.joinSubmitBtn) {
        dom.joinSubmitBtn.disabled = false;
        dom.joinSubmitBtn.innerHTML = '<i data-lucide="link-2" style="width: 15px; height: 15px;"></i><span>Pair &amp; Sync Cluster Secret</span>';
        if (window.lucide) lucide.createIcons();
      }
    }
  }

  async function rotateClusterKey() {
    if (!state.confirmRotate) {
      state.confirmRotate = true;
      if (dom.rotateBtn) dom.rotateBtn.textContent = 'Confirm Rotate Key?';
      setTimeout(() => {
        if (state.confirmRotate) {
          state.confirmRotate = false;
          if (dom.rotateBtn) {
            dom.rotateBtn.innerHTML = '<i data-lucide="key-round" style="width: 14px; height: 14px;"></i><span>Rotate Cluster Key</span>';
            if (window.lucide) lucide.createIcons();
          }
        }
      }, 4500);
      return;
    }

    state.confirmRotate = false;
    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/rotate'), { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.status = data.pair_status || state.status;
      renderStatus();
      showFeedback(`Generated new cluster secret (${data.cluster_fingerprint || ''}). Re-pair other cluster PCs to sync them.`, 'info');
    } catch (err) {
      showFeedback(`Key rotation failed: ${err.message || err}`, 'error');
    } finally {
      if (dom.rotateBtn) {
        dom.rotateBtn.innerHTML = '<i data-lucide="key-round" style="width: 14px; height: 14px;"></i><span>Rotate Cluster Key</span>';
        if (window.lucide) lucide.createIcons();
      }
    }
  }

  async function unpairPeer(peerId, peerName) {
    if (state.confirmUnpairId !== peerId) {
      state.confirmUnpairId = peerId;
      renderStatus();
      setTimeout(() => {
        if (state.confirmUnpairId === peerId) {
          state.confirmUnpairId = null;
          renderStatus();
        }
      }, 4500);
      return;
    }
    state.confirmUnpairId = null;
    try {
      const resp = await fetch(getApiUrl('/api/cluster/pair/remove'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: peerId }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      state.status = data.pair_status || state.status;
      renderStatus();
      showFeedback(`Removed ${peerName || 'peer'} from paired cluster list. Rotate Cluster Key if you want to revoke its access.`, 'info');
    } catch (err) {
      showFeedback(`Unpair failed: ${err.message || err}`, 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // EVENT BINDINGS
  // ---------------------------------------------------------------------------
  function bindEvents() {
    if (dom.navCluster) {
      dom.navCluster.addEventListener('click', () => {
        if (typeof window.openClusterPairing === 'function') {
          window.openClusterPairing();
        }
      });
    }
    if (dom.refreshBtn) dom.refreshBtn.addEventListener('click', () => refreshStatus());
    if (dom.rotateBtn) dom.rotateBtn.addEventListener('click', rotateClusterKey);
    if (dom.startHostBtn) dom.startHostBtn.addEventListener('click', startPairingSession);
    if (dom.regenHostBtn) dom.regenHostBtn.addEventListener('click', startPairingSession);
    if (dom.stopHostBtn) dom.stopHostBtn.addEventListener('click', stopPairingSession);

    if (dom.copyInviteBtn) {
      dom.copyInviteBtn.addEventListener('click', () => {
        copyText(dom.inviteReadonly ? dom.inviteReadonly.value : '', dom.copyInviteBtn, 'Copied!');
      });
    }
    if (dom.copyLanBtn) {
      dom.copyLanBtn.addEventListener('click', () => {
        copyText(dom.inviteLanText ? dom.inviteLanText.textContent : '', dom.copyLanBtn, 'Copied!');
      });
    }

    if (dom.scanHostsBtn) dom.scanHostsBtn.addEventListener('click', scanNearbyHosts);

    if (dom.joinInviteInput) {
      dom.joinInviteInput.addEventListener('input', () => {
        const parsed = parseInviteString(dom.joinInviteInput.value);
        if (parsed) {
          if (dom.joinHostInput) dom.joinHostInput.value = parsed.host;
          if (dom.joinPinInput) dom.joinPinInput.value = parsed.pin;
        }
      });
      dom.joinInviteInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') joinCluster();
      });
    }

    if (dom.joinHostInput) {
      dom.joinHostInput.addEventListener('input', () => {
        const parsed = parseInviteString(dom.joinHostInput.value);
        if (parsed) {
          if (dom.joinInviteInput) dom.joinInviteInput.value = dom.joinHostInput.value.trim();
          dom.joinHostInput.value = parsed.host;
          if (dom.joinPinInput) dom.joinPinInput.value = parsed.pin;
        }
      });
    }

    if (dom.joinPinInput) {
      dom.joinPinInput.addEventListener('input', () => {
        dom.joinPinInput.value = formatPinInput(dom.joinPinInput.value);
      });
      dom.joinPinInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') joinCluster();
      });
    }

    if (dom.joinSubmitBtn) dom.joinSubmitBtn.addEventListener('click', joinCluster);

    if (dom.nearbyList) {
      dom.nearbyList.addEventListener('click', (e) => {
        const btn = e.target.closest('.cp-nearby-item');
        if (!btn) return;
        const host = btn.getAttribute('data-host') || '';
        if (dom.joinHostInput) dom.joinHostInput.value = host;
        if (dom.joinInviteInput) dom.joinInviteInput.value = '';
        if (dom.joinPinInput) dom.joinPinInput.focus();
      });
    }

    if (dom.peersList) {
      dom.peersList.addEventListener('click', (e) => {
        const useBtn = e.target.closest('.cp-use-worker-btn');
        if (useBtn) {
          const ep = useBtn.getAttribute('data-endpoint') || '';
          if (typeof window.openRoleModalWithPeer === 'function') {
            window.openRoleModalWithPeer(ep);
          } else {
            showFeedback(`Copy worker endpoint ${ep} into Node Role -> Worker Peers.`, 'info');
          }
          return;
        }
        const unpairBtn = e.target.closest('.cp-unpair-btn');
        if (unpairBtn) {
          unpairPeer(unpairBtn.getAttribute('data-peer-id'), unpairBtn.getAttribute('data-peer-name'));
        }
      });
    }
  }

  window.refreshClusterPairing = function () {
    refreshStatus();
  };

  function boot() {
    cacheDom();
    bindEvents();
    refreshStatus(true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
