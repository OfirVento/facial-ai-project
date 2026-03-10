/**
 * AdminApp — Multi-store management dashboard.
 *
 * Features:
 *   - JWT-authenticated login (admin role required)
 *   - Multi-store overview with stats
 *   - Staff management (list, create, deactivate)
 *   - Consultation overview & trends
 *   - Audit log viewer
 *   - System health monitoring
 *
 * Self-contained SPA: all DOM + CSS created programmatically.
 * Communicates with Fastify API at /api/v1/*.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE = 'http://localhost:3335/api/v1';
const TOKEN_KEY = 'facialai_admin_token';
const USER_KEY = 'facialai_admin_user';

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function getUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY)); }
  catch { return null; }
}

function saveSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

async function api(path, opts = {}) {
  const token = getToken();
  const headers = { ...opts.headers };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (opts.body && typeof opts.body === 'object') {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
  if (res.status === 401) {
    clearSession();
    location.reload();
    throw new Error('Session expired');
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `API error ${res.status}`);
  return data;
}

// ---------------------------------------------------------------------------
// CSS
// ---------------------------------------------------------------------------

const ADMIN_CSS = `
  :root {
    --admin-bg: #0f1117;
    --admin-surface: #1a1d27;
    --admin-surface-hover: #22263a;
    --admin-border: #2a2e3d;
    --admin-text: #e8eaed;
    --admin-text-dim: #8b8fa3;
    --admin-accent: #4f8cff;
    --admin-accent-hover: #6ea0ff;
    --admin-success: #34d399;
    --admin-warning: #fbbf24;
    --admin-danger: #f87171;
    --admin-radius: 10px;
    --admin-font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: var(--admin-font);
    background: var(--admin-bg);
    color: var(--admin-text);
    min-height: 100vh;
  }

  #admin-app {
    display: flex;
    min-height: 100vh;
  }

  /* ---- Login ---- */
  .admin-login {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 100vh;
    background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%);
  }
  .admin-login-card {
    background: var(--admin-surface);
    border: 1px solid var(--admin-border);
    border-radius: 16px;
    padding: 48px 40px;
    width: 100%;
    max-width: 420px;
  }
  .admin-login-card h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .admin-login-card p {
    color: var(--admin-text-dim);
    margin-bottom: 32px;
    font-size: 0.9rem;
  }
  .admin-login-card label {
    display: block;
    font-size: 0.82rem;
    color: var(--admin-text-dim);
    margin-bottom: 6px;
    font-weight: 500;
  }
  .admin-login-card input {
    width: 100%;
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid var(--admin-border);
    background: var(--admin-bg);
    color: var(--admin-text);
    font-size: 0.95rem;
    margin-bottom: 18px;
    outline: none;
    transition: border-color 0.2s;
  }
  .admin-login-card input:focus {
    border-color: var(--admin-accent);
  }
  .admin-login-card .login-error {
    color: var(--admin-danger);
    font-size: 0.85rem;
    margin-bottom: 12px;
    min-height: 20px;
  }
  .admin-login-card button {
    width: 100%;
    padding: 12px;
    border-radius: 8px;
    border: none;
    background: var(--admin-accent);
    color: #fff;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  .admin-login-card button:hover { background: var(--admin-accent-hover); }
  .admin-login-card button:disabled { opacity: 0.5; cursor: wait; }

  /* ---- Sidebar ---- */
  .admin-sidebar {
    width: 240px;
    background: var(--admin-surface);
    border-right: 1px solid var(--admin-border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }
  .admin-sidebar-logo {
    padding: 24px 20px 16px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    border-bottom: 1px solid var(--admin-border);
  }
  .admin-sidebar-logo span { color: var(--admin-accent); }
  .admin-sidebar-nav {
    flex: 1;
    padding: 12px 8px;
  }
  .admin-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    color: var(--admin-text-dim);
    font-size: 0.88rem;
    font-weight: 500;
    transition: all 0.15s;
    user-select: none;
  }
  .admin-nav-item:hover { background: var(--admin-surface-hover); color: var(--admin-text); }
  .admin-nav-item.active { background: rgba(79,140,255,0.12); color: var(--admin-accent); }
  .admin-nav-item .nav-icon { font-size: 1.1rem; width: 22px; text-align: center; }

  .admin-sidebar-footer {
    padding: 16px 14px;
    border-top: 1px solid var(--admin-border);
  }
  .admin-user-info {
    font-size: 0.82rem;
    color: var(--admin-text-dim);
    margin-bottom: 8px;
  }
  .admin-user-info .user-name { color: var(--admin-text); font-weight: 600; }
  .admin-logout-btn {
    background: none;
    border: 1px solid var(--admin-border);
    color: var(--admin-text-dim);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
    width: 100%;
    transition: all 0.15s;
  }
  .admin-logout-btn:hover { border-color: var(--admin-danger); color: var(--admin-danger); }

  /* ---- Main Content ---- */
  .admin-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .admin-header {
    padding: 24px 32px 16px;
    border-bottom: 1px solid var(--admin-border);
  }
  .admin-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .admin-header .header-sub {
    font-size: 0.85rem;
    color: var(--admin-text-dim);
  }
  .admin-content {
    padding: 24px 32px;
    flex: 1;
  }

  /* ---- Cards Grid ---- */
  .stat-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .stat-card {
    background: var(--admin-surface);
    border: 1px solid var(--admin-border);
    border-radius: var(--admin-radius);
    padding: 20px;
  }
  .stat-card .stat-label {
    font-size: 0.78rem;
    color: var(--admin-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
  }
  .stat-card .stat-sub {
    font-size: 0.8rem;
    color: var(--admin-text-dim);
    margin-top: 4px;
  }

  /* ---- Tables ---- */
  .admin-table-wrap {
    background: var(--admin-surface);
    border: 1px solid var(--admin-border);
    border-radius: var(--admin-radius);
    overflow: hidden;
    margin-bottom: 24px;
  }
  .admin-table-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--admin-border);
  }
  .admin-table-header h3 {
    font-size: 1rem;
    font-weight: 600;
  }
  .admin-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .admin-table th {
    text-align: left;
    padding: 12px 16px;
    color: var(--admin-text-dim);
    font-weight: 500;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--admin-border);
  }
  .admin-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(42,46,61,0.5);
  }
  .admin-table tr:last-child td { border-bottom: none; }
  .admin-table tr:hover td { background: var(--admin-surface-hover); }

  /* ---- Badges ---- */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-active { background: rgba(52,211,153,0.15); color: var(--admin-success); }
  .badge-completed { background: rgba(79,140,255,0.12); color: var(--admin-accent); }
  .badge-cancelled { background: rgba(248,113,113,0.12); color: var(--admin-danger); }
  .badge-admin { background: rgba(251,191,36,0.15); color: var(--admin-warning); }
  .badge-manager { background: rgba(79,140,255,0.12); color: var(--admin-accent); }
  .badge-practitioner { background: rgba(52,211,153,0.15); color: var(--admin-success); }
  .badge-receptionist { background: rgba(139,143,163,0.15); color: var(--admin-text-dim); }

  /* ---- Buttons ---- */
  .btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: none;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-primary { background: var(--admin-accent); color: #fff; }
  .btn-primary:hover { background: var(--admin-accent-hover); }
  .btn-outline {
    background: none;
    border: 1px solid var(--admin-border);
    color: var(--admin-text-dim);
  }
  .btn-outline:hover { border-color: var(--admin-accent); color: var(--admin-accent); }
  .btn-danger { background: rgba(248,113,113,0.12); color: var(--admin-danger); border: none; }
  .btn-danger:hover { background: rgba(248,113,113,0.25); }
  .btn-sm { padding: 5px 10px; font-size: 0.78rem; }

  /* ---- Horizontal bar chart ---- */
  .h-bar-chart {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 16px 20px;
  }
  .h-bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .h-bar-label {
    width: 140px;
    font-size: 0.82rem;
    color: var(--admin-text-dim);
    text-align: right;
    flex-shrink: 0;
    text-transform: capitalize;
  }
  .h-bar-track {
    flex: 1;
    height: 22px;
    background: var(--admin-bg);
    border-radius: 4px;
    overflow: hidden;
  }
  .h-bar-fill {
    height: 100%;
    background: var(--admin-accent);
    border-radius: 4px;
    transition: width 0.6s ease;
    min-width: 2px;
  }
  .h-bar-value {
    width: 36px;
    font-size: 0.82rem;
    font-weight: 600;
    text-align: right;
  }

  /* ---- Modal ---- */
  .admin-modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9000;
    backdrop-filter: blur(4px);
  }
  .admin-modal {
    background: var(--admin-surface);
    border: 1px solid var(--admin-border);
    border-radius: 16px;
    padding: 32px;
    width: 100%;
    max-width: 480px;
    max-height: 80vh;
    overflow-y: auto;
  }
  .admin-modal h3 {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 20px;
  }
  .admin-modal label {
    display: block;
    font-size: 0.82rem;
    color: var(--admin-text-dim);
    margin-bottom: 6px;
    font-weight: 500;
  }
  .admin-modal input, .admin-modal select {
    width: 100%;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--admin-border);
    background: var(--admin-bg);
    color: var(--admin-text);
    font-size: 0.9rem;
    margin-bottom: 16px;
    outline: none;
  }
  .admin-modal input:focus, .admin-modal select:focus {
    border-color: var(--admin-accent);
  }
  .admin-modal .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 8px;
  }
  .admin-modal .modal-error {
    color: var(--admin-danger);
    font-size: 0.82rem;
    margin-bottom: 10px;
    min-height: 18px;
  }

  /* ---- Loading & Empty ---- */
  .admin-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 60px;
    color: var(--admin-text-dim);
    font-size: 0.9rem;
  }
  .admin-empty {
    text-align: center;
    padding: 40px;
    color: var(--admin-text-dim);
  }
  .admin-empty .empty-icon { font-size: 2.4rem; margin-bottom: 12px; }

  /* ---- Health indicators ---- */
  .health-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
  }
  .health-item {
    background: var(--admin-surface);
    border: 1px solid var(--admin-border);
    border-radius: var(--admin-radius);
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .health-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .health-dot.ok { background: var(--admin-success); box-shadow: 0 0 8px rgba(52,211,153,0.4); }
  .health-dot.warn { background: var(--admin-warning); box-shadow: 0 0 8px rgba(251,191,36,0.4); }
  .health-dot.err { background: var(--admin-danger); box-shadow: 0 0 8px rgba(248,113,113,0.4); }
  .health-info h4 { font-size: 0.9rem; font-weight: 600; margin-bottom: 2px; }
  .health-info p { font-size: 0.78rem; color: var(--admin-text-dim); }

  /* ---- Audit log ---- */
  .audit-filters {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .audit-filters select, .audit-filters input {
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--admin-border);
    background: var(--admin-surface);
    color: var(--admin-text);
    font-size: 0.85rem;
    outline: none;
  }

  /* ---- Responsive ---- */
  @media (max-width: 768px) {
    .admin-sidebar { width: 60px; }
    .admin-sidebar-logo span, .admin-nav-item span:not(.nav-icon),
    .admin-user-info, .admin-sidebar-logo { font-size: 0; }
    .admin-sidebar-logo { padding: 16px 0; text-align: center; }
    .admin-nav-item { justify-content: center; padding: 12px; }
    .admin-nav-item .nav-icon { font-size: 1.3rem; }
    .admin-content { padding: 16px; }
    .admin-header { padding: 16px; }
  }
`;

// ---------------------------------------------------------------------------
// Navigation items
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { id: 'overview',      icon: '📊', label: 'Overview' },
  { id: 'stores',        icon: '🏥', label: 'Stores' },
  { id: 'staff',         icon: '👤', label: 'Staff' },
  { id: 'consultations', icon: '📋', label: 'Consultations' },
  { id: 'audit',         icon: '🔒', label: 'Audit Log' },
  { id: 'health',        icon: '💚', label: 'System Health' },
];

// ---------------------------------------------------------------------------
// AdminApp
// ---------------------------------------------------------------------------

class AdminApp {
  constructor(rootEl) {
    this._root = rootEl;
    this._currentPage = 'overview';
    this._stores = [];
    this._selectedStore = null;
    this._storeStats = null;

    // Inject CSS
    const style = document.createElement('style');
    style.textContent = ADMIN_CSS;
    document.head.appendChild(style);

    this._init();
  }

  // =========================================================================
  // Init
  // =========================================================================

  _init() {
    const user = getUser();
    if (!user || !getToken()) {
      this._renderLogin();
    } else {
      this._user = user;
      this._renderDashboard();
      this._loadStores();
    }
  }

  // =========================================================================
  // Login
  // =========================================================================

  _renderLogin() {
    this._root.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'admin-login';
    wrap.innerHTML = `
      <div class="admin-login-card">
        <h1>FacialAI Admin</h1>
        <p>Sign in with your admin credentials</p>
        <label>Email</label>
        <input type="email" id="login-email" placeholder="admin@clinic.com" autocomplete="email" />
        <label>Password</label>
        <input type="password" id="login-password" placeholder="••••••••" autocomplete="current-password" />
        <div class="login-error" id="login-error"></div>
        <button id="login-btn">Sign In</button>
      </div>
    `;
    this._root.appendChild(wrap);

    const emailEl = wrap.querySelector('#login-email');
    const passEl = wrap.querySelector('#login-password');
    const btnEl = wrap.querySelector('#login-btn');
    const errEl = wrap.querySelector('#login-error');

    const doLogin = async () => {
      errEl.textContent = '';
      btnEl.disabled = true;
      btnEl.textContent = 'Signing in...';
      try {
        const data = await api('/auth/login', {
          method: 'POST',
          body: { email: emailEl.value.trim(), password: passEl.value },
        });
        if (data.staff.role !== 'admin' && data.staff.role !== 'manager') {
          errEl.textContent = 'Admin or manager role required';
          btnEl.disabled = false;
          btnEl.textContent = 'Sign In';
          return;
        }
        saveSession(data.token, data.staff);
        this._user = data.staff;
        this._root.innerHTML = '';
        this._renderDashboard();
        this._loadStores();
      } catch (err) {
        errEl.textContent = err.message;
        btnEl.disabled = false;
        btnEl.textContent = 'Sign In';
      }
    };

    btnEl.addEventListener('click', doLogin);
    passEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') doLogin(); });
    emailEl.focus();
  }

  // =========================================================================
  // Dashboard Shell
  // =========================================================================

  _renderDashboard() {
    this._root.innerHTML = '';

    // Sidebar
    const sidebar = document.createElement('aside');
    sidebar.className = 'admin-sidebar';

    const logo = document.createElement('div');
    logo.className = 'admin-sidebar-logo';
    logo.innerHTML = 'Facial<span>AI</span>';
    sidebar.appendChild(logo);

    const nav = document.createElement('nav');
    nav.className = 'admin-sidebar-nav';
    NAV_ITEMS.forEach((item) => {
      const el = document.createElement('div');
      el.className = 'admin-nav-item' + (item.id === this._currentPage ? ' active' : '');
      el.dataset.page = item.id;
      el.innerHTML = `<span class="nav-icon">${item.icon}</span> <span>${item.label}</span>`;
      el.addEventListener('click', () => this._navigate(item.id));
      nav.appendChild(el);
    });
    sidebar.appendChild(nav);

    const footer = document.createElement('div');
    footer.className = 'admin-sidebar-footer';
    footer.innerHTML = `
      <div class="admin-user-info">
        <div class="user-name">${this._escHtml(this._user?.name || 'Admin')}</div>
        <div>${this._escHtml(this._user?.role || '')}</div>
      </div>
      <button class="admin-logout-btn" id="admin-logout">Sign Out</button>
    `;
    sidebar.appendChild(footer);

    this._root.appendChild(sidebar);

    footer.querySelector('#admin-logout').addEventListener('click', () => {
      clearSession();
      location.reload();
    });

    // Main area
    const main = document.createElement('main');
    main.className = 'admin-main';
    main.innerHTML = `
      <div class="admin-header">
        <h2 id="page-title"></h2>
        <div class="header-sub" id="page-subtitle"></div>
      </div>
      <div class="admin-content" id="admin-content"></div>
    `;
    this._root.appendChild(main);

    this._mainEl = main;
    this._contentEl = main.querySelector('#admin-content');
    this._titleEl = main.querySelector('#page-title');
    this._subtitleEl = main.querySelector('#page-subtitle');

    this._navigate(this._currentPage);
  }

  // =========================================================================
  // Navigation
  // =========================================================================

  _navigate(pageId) {
    this._currentPage = pageId;

    // Update nav active state
    this._root.querySelectorAll('.admin-nav-item').forEach((el) => {
      el.classList.toggle('active', el.dataset.page === pageId);
    });

    const titles = {
      overview:      ['Overview',       'Platform-wide statistics'],
      stores:        ['Stores',         'Manage clinic locations'],
      staff:         ['Staff',          'Manage team members'],
      consultations: ['Consultations',  'View consultation activity'],
      audit:         ['Audit Log',      'Security and access history'],
      health:        ['System Health',  'Service status monitoring'],
    };
    const [title, sub] = titles[pageId] || ['', ''];
    this._titleEl.textContent = title;
    this._subtitleEl.textContent = sub;

    // Render page
    const pages = {
      overview:      () => this._renderOverview(),
      stores:        () => this._renderStores(),
      staff:         () => this._renderStaff(),
      consultations: () => this._renderConsultations(),
      audit:         () => this._renderAudit(),
      health:        () => this._renderHealth(),
    };
    this._contentEl.innerHTML = '<div class="admin-loading">Loading...</div>';
    (pages[pageId] || pages.overview)();
  }

  // =========================================================================
  // Load data
  // =========================================================================

  async _loadStores() {
    try {
      const data = await api('/stores');
      this._stores = data.stores || [];
      if (this._stores.length > 0 && !this._selectedStore) {
        this._selectedStore = this._stores[0];
      }
    } catch {
      this._stores = [];
    }
  }

  async _loadStoreStats(storeId) {
    try {
      return await api(`/stores/${storeId}/stats`);
    } catch {
      return null;
    }
  }

  // =========================================================================
  // Overview Page
  // =========================================================================

  async _renderOverview() {
    if (this._stores.length === 0) await this._loadStores();

    const c = this._contentEl;

    if (this._stores.length === 0) {
      c.innerHTML = `
        <div class="admin-empty">
          <div class="empty-icon">🏥</div>
          <p>No stores found. Create your first store to get started.</p>
          <br>
          <button class="btn btn-primary" id="ov-create-store">Create Store</button>
        </div>
      `;
      c.querySelector('#ov-create-store')?.addEventListener('click', () => this._showCreateStoreModal());
      return;
    }

    // Store selector
    c.innerHTML = `
      <div style="margin-bottom:20px; display:flex; align-items:center; gap:12px;">
        <label style="font-size:0.85rem; color:var(--admin-text-dim);">Store:</label>
        <select id="ov-store-select" style="padding:8px 12px; border-radius:8px; border:1px solid var(--admin-border); background:var(--admin-surface); color:var(--admin-text); font-size:0.88rem;"></select>
      </div>
      <div id="ov-stats"></div>
    `;

    const sel = c.querySelector('#ov-store-select');
    this._stores.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name;
      if (this._selectedStore?.id === s.id) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => {
      this._selectedStore = this._stores.find((s) => s.id === sel.value);
      this._fillOverviewStats();
    });

    await this._fillOverviewStats();
  }

  async _fillOverviewStats() {
    const statsEl = this._contentEl.querySelector('#ov-stats');
    if (!statsEl) return;
    statsEl.innerHTML = '<div class="admin-loading">Loading stats...</div>';

    if (!this._selectedStore) {
      statsEl.innerHTML = '<div class="admin-empty">Select a store</div>';
      return;
    }

    const stats = await this._loadStoreStats(this._selectedStore.id);
    if (!stats) {
      statsEl.innerHTML = '<div class="admin-empty">Could not load stats. Is the API server running?</div>';
      return;
    }

    const totalConsultations =
      (stats.consultations.active || 0) +
      (stats.consultations.completed || 0) +
      (stats.consultations.cancelled || 0);

    statsEl.innerHTML = `
      <div class="stat-cards">
        <div class="stat-card">
          <div class="stat-label">Total Patients</div>
          <div class="stat-value">${stats.totalPatients}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Active Consultations</div>
          <div class="stat-value" style="color:var(--admin-success)">${stats.consultations.active || 0}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Completed</div>
          <div class="stat-value" style="color:var(--admin-accent)">${stats.consultations.completed || 0}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Total Consultations</div>
          <div class="stat-value">${totalConsultations}</div>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
        <div class="admin-table-wrap">
          <div class="admin-table-header"><h3>Popular Treatments</h3></div>
          <div id="ov-treatments-chart"></div>
        </div>
        <div class="admin-table-wrap">
          <div class="admin-table-header"><h3>Recent Consultations</h3></div>
          <div id="ov-recent-consults"></div>
        </div>
      </div>
    `;

    // Treatments bar chart
    const chartEl = statsEl.querySelector('#ov-treatments-chart');
    if (stats.popularTreatments.length === 0) {
      chartEl.innerHTML = '<div class="admin-empty" style="padding:20px">No treatments yet</div>';
    } else {
      const maxCount = Math.max(...stats.popularTreatments.map((t) => parseInt(t.count)));
      const chartHtml = stats.popularTreatments.map((t) => {
        const pct = Math.round((parseInt(t.count) / maxCount) * 100);
        const label = t.treatment_type.replace(/_/g, ' ');
        return `
          <div class="h-bar-row">
            <div class="h-bar-label">${this._escHtml(label)}</div>
            <div class="h-bar-track"><div class="h-bar-fill" style="width:${pct}%"></div></div>
            <div class="h-bar-value">${t.count}</div>
          </div>
        `;
      }).join('');
      chartEl.innerHTML = `<div class="h-bar-chart">${chartHtml}</div>`;
    }

    // Recent consultations
    const recentEl = statsEl.querySelector('#ov-recent-consults');
    if (stats.recentConsultations.length === 0) {
      recentEl.innerHTML = '<div class="admin-empty" style="padding:20px">No consultations yet</div>';
    } else {
      const rows = stats.recentConsultations.map((c) => `
        <tr>
          <td>${this._escHtml(c.staff_name || '—')}</td>
          <td><span class="badge badge-${c.status}">${c.status}</span></td>
          <td>${this._formatDate(c.started_at)}</td>
        </tr>
      `).join('');
      recentEl.innerHTML = `
        <table class="admin-table">
          <thead><tr><th>Staff</th><th>Status</th><th>Date</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }
  }

  // =========================================================================
  // Stores Page
  // =========================================================================

  async _renderStores() {
    if (this._stores.length === 0) await this._loadStores();

    const c = this._contentEl;

    if (this._stores.length === 0) {
      c.innerHTML = `
        <div class="admin-empty">
          <div class="empty-icon">🏥</div>
          <p>No stores configured yet.</p>
          <br>
          <button class="btn btn-primary" id="st-create">Create First Store</button>
        </div>
      `;
      c.querySelector('#st-create')?.addEventListener('click', () => this._showCreateStoreModal());
      return;
    }

    const rows = this._stores.map((s) => `
      <tr>
        <td style="font-weight:600">${this._escHtml(s.name)}</td>
        <td>${this._escHtml(s.address || '—')}</td>
        <td>${this._escHtml(s.phone || '—')}</td>
        <td>${this._formatDate(s.created_at)}</td>
        <td>
          <button class="btn btn-outline btn-sm store-view-btn" data-id="${s.id}">View Stats</button>
        </td>
      </tr>
    `).join('');

    c.innerHTML = `
      <div style="display:flex; justify-content:flex-end; margin-bottom:16px;">
        <button class="btn btn-primary" id="st-create">+ New Store</button>
      </div>
      <div class="admin-table-wrap">
        <table class="admin-table">
          <thead><tr><th>Name</th><th>Address</th><th>Phone</th><th>Created</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;

    c.querySelector('#st-create')?.addEventListener('click', () => this._showCreateStoreModal());
    c.querySelectorAll('.store-view-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        this._selectedStore = this._stores.find((s) => s.id === btn.dataset.id);
        this._navigate('overview');
      });
    });
  }

  _showCreateStoreModal() {
    const overlay = document.createElement('div');
    overlay.className = 'admin-modal-overlay';
    overlay.innerHTML = `
      <div class="admin-modal">
        <h3>Create New Store</h3>
        <label>Store Name *</label>
        <input type="text" id="modal-store-name" placeholder="Main Street Clinic" />
        <label>Address</label>
        <input type="text" id="modal-store-address" placeholder="123 Main St, City" />
        <label>Phone</label>
        <input type="text" id="modal-store-phone" placeholder="+1 (555) 123-4567" />
        <div class="modal-error" id="modal-store-error"></div>
        <div class="modal-actions">
          <button class="btn btn-outline" id="modal-cancel">Cancel</button>
          <button class="btn btn-primary" id="modal-submit">Create Store</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.querySelector('#modal-cancel').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    overlay.querySelector('#modal-submit').addEventListener('click', async () => {
      const name = overlay.querySelector('#modal-store-name').value.trim();
      const address = overlay.querySelector('#modal-store-address').value.trim();
      const phone = overlay.querySelector('#modal-store-phone').value.trim();
      const errEl = overlay.querySelector('#modal-store-error');

      if (!name) { errEl.textContent = 'Store name is required'; return; }

      try {
        await api('/stores', {
          method: 'POST',
          body: { name, address: address || null, phone: phone || null },
        });
        close();
        await this._loadStores();
        this._navigate(this._currentPage);
      } catch (err) {
        errEl.textContent = err.message;
      }
    });
  }

  // =========================================================================
  // Staff Page
  // =========================================================================

  async _renderStaff() {
    const c = this._contentEl;

    try {
      if (this._stores.length === 0) await this._loadStores();

      const data = await api('/staff');
      const staffList = data.staff || [];

      const rows = staffList.map((s) => `
        <tr>
          <td style="font-weight:500">${this._escHtml(s.name)}</td>
          <td>${this._escHtml(s.email)}</td>
          <td><span class="badge badge-${s.role}">${s.role}</span></td>
          <td>${this._escHtml(s.store_name || '—')}</td>
          <td>
            <span class="badge ${s.active ? 'badge-active' : 'badge-cancelled'}">
              ${s.active ? 'Active' : 'Inactive'}
            </span>
          </td>
          <td>${this._formatDate(s.created_at)}</td>
          <td>
            ${s.id !== this._user?.id
              ? s.active
                ? `<button class="btn btn-danger btn-sm staff-toggle" data-id="${s.id}" data-action="deactivate">Deactivate</button>`
                : `<button class="btn btn-outline btn-sm staff-toggle" data-id="${s.id}" data-action="activate">Activate</button>`
              : '<span style="color:var(--admin-text-dim); font-size:0.78rem">You</span>'
            }
          </td>
        </tr>
      `).join('');

      c.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
          <span style="font-size:0.85rem; color:var(--admin-text-dim)">${staffList.length} member${staffList.length !== 1 ? 's' : ''}</span>
          <button class="btn btn-primary" id="staff-create">+ Add Staff Member</button>
        </div>
        <div class="admin-table-wrap">
          <table class="admin-table">
            <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Store</th><th>Status</th><th>Joined</th><th></th></tr></thead>
            <tbody>${rows || '<tr><td colspan="7" class="admin-empty">No staff found</td></tr>'}</tbody>
          </table>
        </div>
      `;

      c.querySelector('#staff-create')?.addEventListener('click', () => this._showCreateStaffModal());

      c.querySelectorAll('.staff-toggle').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const { id, action } = btn.dataset;
          try {
            await api(`/staff/${id}/${action}`, { method: 'PUT' });
            this._renderStaff(); // Refresh
          } catch (err) {
            alert('Error: ' + err.message);
          }
        });
      });

    } catch (err) {
      c.innerHTML = `<div class="admin-empty"><p>Error loading staff: ${this._escHtml(err.message)}</p></div>`;
    }
  }

  _showCreateStaffModal() {
    const overlay = document.createElement('div');
    overlay.className = 'admin-modal-overlay';

    const storeOptions = this._stores
      .map((s) => `<option value="${s.id}">${this._escHtml(s.name)}</option>`)
      .join('');

    overlay.innerHTML = `
      <div class="admin-modal">
        <h3>Add Staff Member</h3>
        <label>Full Name *</label>
        <input type="text" id="modal-staff-name" placeholder="Dr. Jane Smith" />
        <label>Email *</label>
        <input type="email" id="modal-staff-email" placeholder="jane@clinic.com" />
        <label>Password *</label>
        <input type="password" id="modal-staff-password" placeholder="Min 8 characters" />
        <label>Store *</label>
        <select id="modal-staff-store">${storeOptions}</select>
        <label>Role</label>
        <select id="modal-staff-role">
          <option value="practitioner">Practitioner</option>
          <option value="manager">Manager</option>
          <option value="receptionist">Receptionist</option>
          <option value="admin">Admin</option>
        </select>
        <div class="modal-error" id="modal-staff-error"></div>
        <div class="modal-actions">
          <button class="btn btn-outline" id="modal-cancel">Cancel</button>
          <button class="btn btn-primary" id="modal-submit">Create Account</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.querySelector('#modal-cancel').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    overlay.querySelector('#modal-submit').addEventListener('click', async () => {
      const name = overlay.querySelector('#modal-staff-name').value.trim();
      const email = overlay.querySelector('#modal-staff-email').value.trim();
      const password = overlay.querySelector('#modal-staff-password').value;
      const store_id = overlay.querySelector('#modal-staff-store').value;
      const role = overlay.querySelector('#modal-staff-role').value;
      const errEl = overlay.querySelector('#modal-staff-error');

      if (!name || !email || !password) {
        errEl.textContent = 'Name, email, and password are required';
        return;
      }
      if (password.length < 8) {
        errEl.textContent = 'Password must be at least 8 characters';
        return;
      }

      try {
        await api('/auth/register', {
          method: 'POST',
          body: { name, email, password, store_id, role },
        });
        close();
        this._navigate('staff');
      } catch (err) {
        errEl.textContent = err.message;
      }
    });
  }

  // =========================================================================
  // Consultations Page
  // =========================================================================

  async _renderConsultations() {
    const c = this._contentEl;

    try {
      const data = await api('/consultations');
      const consultations = data.consultations || [];

      if (consultations.length === 0) {
        c.innerHTML = `
          <div class="admin-empty">
            <div class="empty-icon">📋</div>
            <p>No consultations found.</p>
          </div>
        `;
        return;
      }

      // Summary cards
      const counts = { active: 0, completed: 0, cancelled: 0 };
      consultations.forEach((con) => { counts[con.status] = (counts[con.status] || 0) + 1; });

      const rows = consultations.slice(0, 50).map((con) => `
        <tr>
          <td><code style="font-size:0.75rem">${con.id.slice(0, 8)}...</code></td>
          <td><span class="badge badge-${con.status}">${con.status}</span></td>
          <td>${this._formatDate(con.started_at)}</td>
          <td>${con.completed_at ? this._formatDate(con.completed_at) : '—'}</td>
          <td>${this._escHtml(con.notes?.slice(0, 40) || '—')}</td>
        </tr>
      `).join('');

      c.innerHTML = `
        <div class="stat-cards">
          <div class="stat-card">
            <div class="stat-label">Active</div>
            <div class="stat-value" style="color:var(--admin-success)">${counts.active}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Completed</div>
            <div class="stat-value" style="color:var(--admin-accent)">${counts.completed}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Cancelled</div>
            <div class="stat-value" style="color:var(--admin-danger)">${counts.cancelled}</div>
          </div>
        </div>
        <div class="admin-table-wrap">
          <div class="admin-table-header">
            <h3>All Consultations</h3>
            <span style="font-size:0.8rem; color:var(--admin-text-dim)">Showing last 50</span>
          </div>
          <table class="admin-table">
            <thead><tr><th>ID</th><th>Status</th><th>Started</th><th>Completed</th><th>Notes</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    } catch (err) {
      c.innerHTML = `<div class="admin-empty"><p>Error: ${this._escHtml(err.message)}</p></div>`;
    }
  }

  // =========================================================================
  // Audit Log Page
  // =========================================================================

  async _renderAudit() {
    const c = this._contentEl;

    try {
      const [logData, summaryData] = await Promise.all([
        api('/audit?limit=50'),
        api('/audit/summary'),
      ]);

      const entries = logData.entries || [];
      const summary = summaryData;

      // Summary cards
      c.innerHTML = `
        <div class="stat-cards">
          <div class="stat-card">
            <div class="stat-label">Actions (24h)</div>
            <div class="stat-value">${summary.last24Hours?.totalActions || 0}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Active Staff (24h)</div>
            <div class="stat-value">${summary.last24Hours?.activeStaff || 0}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Privacy Model</div>
            <div style="font-size:0.82rem; margin-top:6px; line-height:1.5">
              AES-256-GCM · Per-patient IV<br>No raw photos · GDPR delete
            </div>
          </div>
        </div>

        <div class="admin-table-wrap">
          <div class="admin-table-header">
            <h3>Recent Activity</h3>
            <span style="font-size:0.8rem; color:var(--admin-text-dim)">Last 50 entries</span>
          </div>
          ${entries.length === 0
            ? '<div class="admin-empty" style="padding:24px">No audit entries yet</div>'
            : `<table class="admin-table">
                <thead><tr><th>Time</th><th>Staff</th><th>Action</th><th>Resource</th><th>IP</th></tr></thead>
                <tbody>${entries.map((e) => `
                  <tr>
                    <td style="font-size:0.8rem">${this._formatDate(e.created_at)}</td>
                    <td>${this._escHtml(e.staff_name || e.staff_id?.slice(0, 8) || '—')}</td>
                    <td><code style="font-size:0.78rem">${this._escHtml(e.action)}</code></td>
                    <td style="font-size:0.8rem">${this._escHtml(e.resource_type || '')} ${e.resource_id ? e.resource_id.slice(0, 8) + '...' : ''}</td>
                    <td style="font-size:0.78rem; color:var(--admin-text-dim)">${this._escHtml(e.ip_address || '—')}</td>
                  </tr>
                `).join('')}</tbody>
              </table>`
          }
        </div>

        ${summary.last30Days?.length > 0 ? `
          <div class="admin-table-wrap" style="margin-top:16px">
            <div class="admin-table-header"><h3>Actions (Last 30 Days)</h3></div>
            <div class="h-bar-chart">
              ${(() => {
                const maxCount = Math.max(...summary.last30Days.map((a) => parseInt(a.count)));
                return summary.last30Days.map((a) => {
                  const pct = Math.round((parseInt(a.count) / maxCount) * 100);
                  return `
                    <div class="h-bar-row">
                      <div class="h-bar-label">${this._escHtml(a.action)}</div>
                      <div class="h-bar-track"><div class="h-bar-fill" style="width:${pct}%"></div></div>
                      <div class="h-bar-value">${a.count}</div>
                    </div>
                  `;
                }).join('');
              })()}
            </div>
          </div>
        ` : ''}
      `;
    } catch (err) {
      // Fallback if API is not available
      c.innerHTML = `
        <div class="stat-cards">
          <div class="stat-card">
            <div class="stat-label">Privacy Model</div>
            <div style="font-size:0.9rem; margin-top:8px">
              <div style="margin-bottom:4px">AES-256-GCM encryption</div>
              <div style="margin-bottom:4px">Per-patient initialization vectors</div>
              <div style="margin-bottom:4px">No raw photos stored</div>
              <div style="margin-bottom:4px">Full deletion on GDPR request</div>
              <div>Every data access logged</div>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Tracked Actions</div>
            <div style="font-size:0.85rem; margin-top:8px; color:var(--admin-text-dim); line-height:1.6">
              login · register_staff · create_store ·
              view_patient · create_patient · update_patient · delete_patient ·
              view_facedna · create_facedna · delete_facedna ·
              create_consultation · complete_consultation ·
              deactivate_staff · activate_staff
            </div>
          </div>
        </div>
        <div class="admin-empty" style="margin-top:16px">
          <p>Could not load audit log: ${this._escHtml(err.message)}</p>
          <p style="margin-top:4px; font-size:0.82rem; color:var(--admin-text-dim)">Ensure the API server is running on port 3335.</p>
        </div>
      `;
    }
  }

  // =========================================================================
  // System Health Page
  // =========================================================================

  async _renderHealth() {
    const c = this._contentEl;
    c.innerHTML = '<div class="admin-loading">Checking services...</div>';

    const checks = [];

    // API health check
    try {
      const start = performance.now();
      const data = await api('/health');
      const latency = Math.round(performance.now() - start);
      checks.push({
        name: 'API Server',
        status: 'ok',
        detail: `v${data.version} — ${latency}ms latency`,
      });
    } catch {
      checks.push({
        name: 'API Server',
        status: 'err',
        detail: 'Cannot reach API at ' + API_BASE,
      });
    }

    // Frontend check
    try {
      const start = performance.now();
      const res = await fetch('http://localhost:3334/', { method: 'HEAD' });
      const latency = Math.round(performance.now() - start);
      checks.push({
        name: 'Frontend (Vite)',
        status: res.ok ? 'ok' : 'warn',
        detail: res.ok ? `Port 3334 — ${latency}ms` : 'Unexpected status: ' + res.status,
      });
    } catch {
      checks.push({
        name: 'Frontend (Vite)',
        status: 'err',
        detail: 'Not reachable on port 3334',
      });
    }

    // Database (inferred from API)
    const apiOk = checks[0]?.status === 'ok';
    checks.push({
      name: 'PostgreSQL Database',
      status: apiOk ? 'ok' : 'warn',
      detail: apiOk ? 'Connected (via API)' : 'Status unknown — API not available',
    });

    // Kiosk endpoint
    try {
      const res = await fetch('http://localhost:3334/kiosk.html', { method: 'HEAD' });
      checks.push({
        name: 'Kiosk App',
        status: res.ok ? 'ok' : 'warn',
        detail: res.ok ? 'Available at /kiosk.html' : 'Not found',
      });
    } catch {
      checks.push({
        name: 'Kiosk App',
        status: 'warn',
        detail: 'Not reachable',
      });
    }

    // Lab endpoint
    try {
      const res = await fetch('http://localhost:3334/lab.html', { method: 'HEAD' });
      checks.push({
        name: 'Lab / GLB Viewer',
        status: res.ok ? 'ok' : 'warn',
        detail: res.ok ? 'Available at /lab.html' : 'Not found',
      });
    } catch {
      checks.push({
        name: 'Lab / GLB Viewer',
        status: 'warn',
        detail: 'Not reachable',
      });
    }

    const itemsHtml = checks.map((ch) => `
      <div class="health-item">
        <div class="health-dot ${ch.status}"></div>
        <div class="health-info">
          <h4>${this._escHtml(ch.name)}</h4>
          <p>${this._escHtml(ch.detail)}</p>
        </div>
      </div>
    `).join('');

    c.innerHTML = `
      <div class="health-grid">${itemsHtml}</div>
      <div style="margin-top:24px; display:flex; justify-content:flex-end;">
        <button class="btn btn-outline" id="health-refresh">Refresh</button>
      </div>
    `;

    c.querySelector('#health-refresh')?.addEventListener('click', () => this._renderHealth());
  }

  // =========================================================================
  // Utilities
  // =========================================================================

  _escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str ?? '';
    return div.innerHTML;
  }

  _formatDate(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return iso;
    }
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('admin-app');
  if (root) new AdminApp(root);
});
