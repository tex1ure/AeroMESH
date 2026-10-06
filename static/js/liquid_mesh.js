/**
 * AeroMESH • Liquid Topological Mesh Canvas
 * Replicates the organic fluid looping network aesthetic of the official AeroMESH logo.
 * Features:
 * - Fluid orbiting nodes with smooth cubic bezier connections
 * - Interactive subtle mouse deflection & luminescence
 * - Energy packet pulses streaming along curves during generation
 */

(function () {
  'use strict';

  const canvas = document.getElementById('liquid-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  let mouse = { x: width * 0.5, y: height * 0.45, active: false };
  let isGenerating = false;

  // Window resize handler
  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    initNodes();
  });

  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  });

  window.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  // Track generation state from app
  window.setMeshGenerating = function (generating) {
    isGenerating = generating;
  };

  // Node class
  const NODE_COUNT = 14;
  let nodes = [];
  let energyPulses = [];

  class MeshNode {
    constructor(id) {
      this.id = id;
      this.reset();
    }

    reset() {
      // Clustered slightly around center-right for a natural composition
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * (Math.min(width, height) * 0.38) + 40;
      const cx = width * 0.52;
      const cy = height * 0.48;

      this.x = cx + Math.cos(angle) * dist;
      this.y = cy + Math.sin(angle) * dist;
      this.baseX = this.x;
      this.baseY = this.y;

      this.vx = (Math.random() - 0.5) * 0.38;
      this.vy = (Math.random() - 0.5) * 0.38;
      this.radius = Math.random() * 2 + 4.0; // Bold pearl node terminals matching new logo
      this.pulsePhase = Math.random() * Math.PI * 2;
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;

      // Soft boundary bounce around center
      const dx = this.x - width * 0.52;
      const dy = this.y - height * 0.48;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const maxDist = Math.min(width, height) * 0.45;

      if (dist > maxDist) {
        this.vx -= (dx / dist) * 0.02;
        this.vy -= (dy / dist) * 0.02;
      }

      // Mouse gentle interaction
      if (mouse.active) {
        const mdx = this.x - mouse.x;
        const mdy = this.y - mouse.y;
        const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
        if (mdist < 180 && mdist > 1) {
          const force = (180 - mdist) / 180 * 0.8;
          this.x += (mdx / mdist) * force;
          this.y += (mdy / mdist) * force;
        }
      }

      this.pulsePhase += 0.035;
    }

    draw() {
      const pulse = Math.sin(this.pulsePhase) * 0.2 + 0.8;
      const r = this.radius * pulse;

      // Dark shadow cutout under node
      ctx.beginPath();
      ctx.arc(this.x, this.y, r + 2.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(8, 9, 13, 0.8)';
      ctx.fill();

      // Glowing pearl node terminal
      ctx.beginPath();
      ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isGenerating ? '#7dd3fc' : '#ffffff';
      ctx.shadowColor = isGenerating ? 'rgba(56, 189, 248, 0.95)' : 'rgba(255, 255, 255, 0.7)';
      ctx.shadowBlur = isGenerating ? 16 : 10;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  function initNodes() {
    nodes = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push(new MeshNode(i));
    }
  }

  // Energy packet pulsing along links
  class EnergyPacket {
    constructor(nodeA, nodeB) {
      this.a = nodeA;
      this.b = nodeB;
      this.t = 0;
      this.speed = Math.random() * 0.025 + 0.015;
    }

    update() {
      this.t += this.speed;
      return this.t <= 1;
    }

    draw() {
      // Calculate cubic bezier point between a and b
      const cx = (this.a.x + this.b.x) * 0.5 + (this.b.y - this.a.y) * 0.25;
      const cy = (this.a.y + this.b.y) * 0.5 - (this.b.x - this.a.x) * 0.25;

      const omt = 1 - this.t;
      const px = omt * omt * this.a.x + 2 * omt * this.t * cx + this.t * this.t * this.b.x;
      const py = omt * omt * this.a.y + 2 * omt * this.t * cy + this.t * this.t * this.b.y;

      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.shadowColor = 'rgba(56, 189, 248, 1)';
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  initNodes();

  // Animation Loop
  let frame = 0;
  function animate() {
    ctx.clearRect(0, 0, width, height);

    // Update nodes
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].update();
    }

    // Connect nodes with smooth organic curves (matching the logo's continuous splines)
    const MAX_CONN_DIST = Math.min(width, height) * 0.38;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const na = nodes[i];
        const nb = nodes[j];
        const dx = nb.x - na.x;
        const dy = nb.y - na.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < MAX_CONN_DIST) {
          const alpha = (1 - dist / MAX_CONN_DIST) * (isGenerating ? 0.38 : 0.18);

          // Control point creates natural loop
          const cx = (na.x + nb.x) * 0.5 + (nb.y - na.y) * 0.22;
          const cy = (na.y + nb.y) * 0.5 - (nb.x - na.x) * 0.22;

          // 1. Dark under-shadow stroke for 3D ribbon depth
          ctx.beginPath();
          ctx.moveTo(na.x, na.y);
          ctx.quadraticCurveTo(cx, cy, nb.x, nb.y);
          ctx.strokeStyle = `rgba(6, 8, 12, ${alpha * 0.9})`;
          ctx.lineWidth = isGenerating ? 4.8 : 3.8;
          ctx.stroke();

          // 2. Luminous pearl ribbon stroke
          ctx.beginPath();
          ctx.moveTo(na.x, na.y);
          ctx.quadraticCurveTo(cx, cy, nb.x, nb.y);
          ctx.strokeStyle = isGenerating
            ? `rgba(56, 189, 248, ${alpha})`
            : `rgba(255, 255, 255, ${alpha * 1.15})`;
          ctx.lineWidth = isGenerating ? 2.4 : 1.8;
          ctx.stroke();

          // Spawn energy pulses during generation
          if (isGenerating && Math.random() < 0.015 && energyPulses.length < 8) {
            energyPulses.push(new EnergyPacket(na, nb));
          }
        }
      }
    }

    // Draw energy pulses
    for (let i = energyPulses.length - 1; i >= 0; i--) {
      const p = energyPulses[i];
      if (p.update()) {
        p.draw();
      } else {
        energyPulses.splice(i, 1);
      }
    }

    // Draw nodes
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].draw();
    }

    frame++;
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
