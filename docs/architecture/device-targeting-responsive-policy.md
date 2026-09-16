# Device Targeting & Responsive Layout Policy (FIX-26)

> **Status**: Intended Architecture & Responsive Guardrail Standard  
> **Component**: `static/index.html`, `static/css/responsive.css`, `static/js/app.js`  
> **Classification**: Architectural Decision Record (ADR)  
> **Related Documents**: `README.md`, `PROJECT_STATUS_AND_REMAINING_FIXES.md`, `docs/architecture/fastapi-proxy-gateway.md`

---

## 1. Executive Summary & Policy Statement

AeroMESH is engineered primarily as a distributed edge inference control system for consumer laptops and desktop workstations.

> **AeroMESH is desktop-first by design.** The web interface is tailored for operators orchestrating heterogeneous multi-node pipelines across physical machines. Mobile responsiveness is not a primary product target. Small-screen compatibility is implemented strictly as a non-destructive fallback to guarantee basic readability, fluid chat interaction, and graceful layout degradation without compromising the desktop claymorphic presentation.

A full mobile-first redesign, native mobile application, or touch-optimized cluster management UI are explicit non-goals for the project.

---

## 2. Primary Device Targets & Supported Resolutions

AeroMESH is designed and verified against standard desktop and laptop browser display viewports:

| Resolution Category | Typical Hardware | Supported Viewport Dimensions | Priority |
|---|---|---|---|
| **Primary Full HD** | Modern 15"/16" laptops & external monitors | `1920x1080` | **Primary (P0)** |
| **Wide Laptop** | 14"/16" creator laptops | `1600x900`, `1440x900` | **Primary (P0)** |
| **Standard Laptop** | 13"/14" consumer laptops | `1366x768`, `1280x800` | **Primary (P0)** |
| **Tablet Landscape** | Tablets, convertibles | `1024x768`, `1180x820` | Secondary (Graceful Fallback) |
| **Tablet Portrait** | Tablets, split-screen desktop windows | `768x1024`, `800x1280` | Secondary (Drawer Fallback) |
| **Mobile Phone** | Operator phone smoke tests & quick monitoring | `390x844` (iPhone), `412x915` (Android) | Tertiary (Readability & Chat Fallback) |

---

## 3. Rationale: Why Mobile-First Redesign is an Anti-Pattern

1. **Physical Operational Reality**:
   - AeroMESH coordinates local GPU weights and pipeline activation transfers across consumer gaming laptops running Windows 11 with NVIDIA CUDA.
   - Operators launch coordinators and workers from PowerShell terminals (`.\start.ps1`, `cargo run --bin aeromesh -- serve`). The operator is seated in front of a laptop or workstation.
2. **High-Density Telemetry Requirements**:
   - The Cluster HUD modal and sidebar telemetry panels present dense real-time diagnostics: stage allocations (Layers 0..24 vs 25..48), VRAM occupancy, Tailscale WireGuard direct RTT pings (1–2 ms), and per-token wire metrics (5.16 KB/tok). Compacting this into a thumb-first mobile layout sacrifices technical observability.
3. **Engineering Opportunity Cost**:
   - Rebuilding the custom dark claymorphic design system for mobile would divert resources away from distributed tensor evaluation, CUDA acceleration, and pipeline latency optimization.

---

## 4. Responsive Guardrail Architecture

Small-screen usability is implemented via **additive, non-destructive guardrails**:

```
Desktop (> 1024px) ────────► 100% Unchanged Luxury Claymorphic Layout
Tablet (<= 1024px) ────────► Fluid Max-Width Containers (Chat & Input Anchor)
Mobile (<= 768px)  ────────► Fixed Drawer Sidebar, Compact Nav, Vertical Modals
Ultra-Compact (<= 420px) ──► Word-Break Code Blocks, Hidden Desktop Shortcuts
```

### 4.1 HTML Viewport Meta Tag
`static/index.html` enforces proper mobile scaling:
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```
This ensures mobile browsers do not simulate a 980px desktop viewport with miniature, illegible text.

### 4.2 Additive Stylesheet (`static/css/responsive.css`)
To protect the desktop design from accidental regressions, all responsive overrides are isolated in `static/css/responsive.css`, imported directly after `claymorphic.css`:
- **Sidebar Drawer Transformation**: On viewports $\le 768\text{px}$, `#sidebar` transforms from an in-flow flex container into a fixed overlay drawer (`z-index: 100`) with a dark drop shadow.
- **Top Navigation Compaction**: Model selector pill truncates long model names with `text-overflow: ellipsis`, and secondary buttons (`Sync KV`) collapse to icon-only buttons.
- **Hero Suggestions Stacking**: The 2-column hero prompt suggestion grid collapses to a single column for comfortable tapping.
- **Floating Input Dock & Virtual Keyboards**: Desktop keyboard hints (`Enter ↵ to send`) are hidden on touchscreens; input capsule border radii and paddings adjust for smaller screens.
- **Modal Overflow Protection**: The Cluster Topology HUD modal (`.clay-modal`) is capped at `max-height: 88vh; overflow-y: auto;`, preventing modal clipping or screen lockup on small displays.

### 4.3 Mobile Interaction Enhancements (`static/js/app.js`)
- **Drawer Close Button**: `#sidebar-close-btn` is displayed when the drawer is active on mobile screens, allowing explicit closure.
- **Auto-Collapse on Navigation**: Selecting a conversation or creating a new chat on viewports $\le 768\text{px}$ automatically collapses the sidebar drawer, returning the operator to the active chat viewport.
- **Tap Outside to Dismiss**: Tapping the `#chat-viewport` while the drawer is open on mobile collapses the drawer automatically.

---

## 5. Verification Checklist

| Scenario | Target Resolution | Expected Behavior | Status |
|---|---|---|---|
| **Primary Full HD** | `1920x1080` | Full-spread claymorphic layout, fixed sidebar, 2-column suggestions, full top navbar | Pass (Unchanged) |
| **Standard Laptop** | `1366x768` | Zero horizontal scrollbars, input dock centered, complete HUD visibility | Pass (Unchanged) |
| **Tablet Portrait** | `768x1024` | Sidebar accessible via panel-left toggle button, suggestions 1-col, modals fit screen | Pass |
| **Mobile Smoke Test** | `390x844` / `412x915` | Text readable without pinch-to-zoom, chat input works, sidebar auto-collapses | Pass |
| **Code Block Overflow** | Mobile Viewport | Code blocks scroll horizontally within cards; zero body horizontal overflow | Pass |
