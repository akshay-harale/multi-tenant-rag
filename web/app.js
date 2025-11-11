/**
 * Multi-Tenant RAG Application with Source-Based Isolation
 * Main JavaScript Module
 */

// Configuration
const API_BASE = "";

// State Management
let currentTenant = null;
let currentSource = null;
let allSources = [];
let sessionId = null;

// ============================================================================
// Navigation Functions
// ============================================================================

function showPage(pageName) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + pageName).classList.add('active');
  updateBreadcrumb(pageName);
}

function updateBreadcrumb(pageName) {
  const breadcrumb = document.getElementById('breadcrumb');
  let html = '';
  
  if (pageName === 'tenants') {
    html = '<span class="breadcrumb-item active">Tenants</span>';
  } else if (pageName === 'sources') {
    html = `
      <span class="breadcrumb-item" onclick="goToTenants()">Tenants</span>
      <span class="breadcrumb-separator">›</span>
      <span class="breadcrumb-item active">Sources</span>
    `;
  } else if (pageName === 'chat') {
    html = `
      <span class="breadcrumb-item" onclick="goToTenants()">Tenants</span>
      <span class="breadcrumb-separator">›</span>
      <span class="breadcrumb-item" onclick="goToSources()">Sources</span>
      <span class="breadcrumb-separator">›</span>
      <span class="breadcrumb-item active">Chat</span>
    `;
  }
  
  breadcrumb.innerHTML = html;
}

function goToTenants() {
  showPage('tenants');
  loadTenants();
}

function goToSources() {
  if (!currentTenant) {
    goToTenants();
    return;
  }
  showPage('sources');
  loadSources();
}

function goToChat() {
  if (!currentTenant) {
    goToTenants();
    return;
  }
  showPage('chat');
  document.getElementById('chatTenantDisplay').textContent = currentTenant;
  document.getElementById('chatSourceDisplay').textContent = currentSource ? currentSource.name : 'All Sources';
  loadSourceFilters();
}

// ============================================================================
// API Helper Functions
// ============================================================================

async function apiCall(url, options = {}) {
  const res = await fetch(API_BASE + url, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

function log(elementId, msg) {
  const el = document.getElementById(elementId);
  const timestamp = new Date().toLocaleTimeString();
  el.textContent = `[${timestamp}] ${msg}\n` + el.textContent.slice(0, 2000);
}

// ============================================================================
// Tenant Management
// ============================================================================

async function loadTenants() {
  try {
    const data = await apiCall('/tenants');
    const list = document.getElementById('tenantList');
    const empty = document.getElementById('tenantEmpty');
    
    if (data.tenants.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    
    empty.style.display = 'none';
    list.innerHTML = data.tenants.map(t => `
      <div class="card-item" onclick="selectTenant('${t}')">
        <div class="card-item-title">${t}</div>
        <div class="card-item-meta">Click to select</div>
      </div>
    `).join('');
  } catch (e) {
    alert('Error loading tenants: ' + e.message);
  }
}

async function createTenant() {
  const input = document.getElementById('newTenantId');
  const id = input.value.trim();
  if (!id) return;
  
  try {
    await apiCall('/tenants', {
      method: 'POST',
      body: JSON.stringify({ tenant_id: id })
    });
    input.value = '';
    await loadTenants();
    selectTenant(id);
  } catch (e) {
    alert('Error creating tenant: ' + e.message);
  }
}

function selectTenant(tenantId) {
  currentTenant = tenantId;
  currentSource = null;
  sessionId = null;
  document.getElementById('currentTenantDisplay').textContent = tenantId;
  goToSources();
}

// ============================================================================
// Source Management
// ============================================================================

async function loadSources() {
  try {
    const data = await apiCall(`/tenants/${encodeURIComponent(currentTenant)}/sources`);
    allSources = data.sources;
    const list = document.getElementById('sourceList');
    const empty = document.getElementById('sourceEmpty');
    
    if (data.sources.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    
    empty.style.display = 'none';
    list.innerHTML = data.sources.map(s => `
      <div class="card-item" onclick="selectSource('${s.source_id}', '${escapeHtml(s.source_name)}')">
        <div class="card-item-title">${escapeHtml(s.source_name)}</div>
        <div class="card-item-meta">Created: ${new Date(s.created_at).toLocaleDateString()}</div>
        <div class="card-item-actions">
          <button class="small secondary" onclick="event.stopPropagation(); viewSourceDocs('${s.source_id}')">View Docs</button>
          <button class="small danger" onclick="event.stopPropagation(); deleteSource('${s.source_id}', '${escapeHtml(s.source_name)}')">Delete</button>
        </div>
      </div>
    `).join('');
  } catch (e) {
    alert('Error loading sources: ' + e.message);
  }
}

async function createSource() {
  const input = document.getElementById('newSourceName');
  const name = input.value.trim();
  if (!name) return;
  
  try {
    await apiCall(`/tenants/${encodeURIComponent(currentTenant)}/sources`, {
      method: 'POST',
      body: JSON.stringify({ source_name: name })
    });
    input.value = '';
    await loadSources();
  } catch (e) {
    alert('Error creating source: ' + e.message);
  }
}

function selectSource(sourceId, sourceName) {
  currentSource = { id: sourceId, name: sourceName };
  goToChat();
}

function proceedWithoutSource() {
  currentSource = null;
  goToChat();
}

async function deleteSource(sourceId, sourceName) {
  if (!confirm(`Delete source "${sourceName}"? This will also delete all associated documents and vectors.`)) {
    return;
  }
  
  try {
    await apiCall(`/tenants/${encodeURIComponent(currentTenant)}/sources/${sourceId}`, {
      method: 'DELETE'
    });
    await loadSources();
  } catch (e) {
    alert('Error deleting source: ' + e.message);
  }
}

async function viewSourceDocs(sourceId) {
  try {
    const data = await apiCall(`/tenants/${encodeURIComponent(currentTenant)}/sources/${sourceId}/documents`);
    const docs = data.documents.map(d => `• ${d.filename} (${new Date(d.uploaded_at).toLocaleString()})`).join('\n');
    alert(docs || 'No documents in this source yet.');
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

// ============================================================================
// Document Upload
// ============================================================================

async function uploadDocument() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files || !fileInput.files.length) {
    alert('Please select a file');
    return;
  }
  
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  log('uploadLog', `Uploading ${file.name}...`);
  
  try {
    const url = currentSource 
      ? `/tenants/${encodeURIComponent(currentTenant)}/sources/${currentSource.id}/upload`
      : `/tenants/${encodeURIComponent(currentTenant)}/upload`;
    
    const res = await fetch(API_BASE + url, {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) throw new Error(await res.text());
    
    const data = await res.json();
    log('uploadLog', `✓ Embedded: ${data.new_chunks} new chunks, ${data.skipped_duplicates} skipped`);
    fileInput.value = '';
  } catch (e) {
    log('uploadLog', `✗ Error: ${e.message}`);
  }
}

// ============================================================================
// Source Filtering
// ============================================================================

function loadSourceFilters() {
  const list = document.getElementById('sourceFilterList');
  if (currentSource) {
    list.innerHTML = `
      <div class="source-checkbox">
        <input type="checkbox" id="source-${currentSource.id}" checked disabled>
        <label for="source-${currentSource.id}">${escapeHtml(currentSource.name)} (current)</label>
      </div>
    `;
  } else if (allSources.length > 0) {
    list.innerHTML = allSources.map(s => `
      <div class="source-checkbox">
        <input type="checkbox" id="source-${s.source_id}" value="${s.source_id}" checked>
        <label for="source-${s.source_id}">${escapeHtml(s.source_name)}</label>
      </div>
    `).join('');
  } else {
    list.innerHTML = '<p style="color:#8893a4;font-size:.85rem;">No sources available. All documents will be searched.</p>';
  }
}

function getSelectedSourceIds() {
  if (currentSource) {
    return [currentSource.id];
  }
  
  const checkboxes = document.querySelectorAll('#sourceFilterList input[type=checkbox]:checked');
  const ids = Array.from(checkboxes).map(cb => cb.value);
  return ids.length > 0 ? ids : null;
}

// ============================================================================
// Chat Functions
// ============================================================================

function appendMessage(role, content, citations = []) {
  const box = document.getElementById('chatBox');
  if (box.children.length === 1 && box.children[0].textContent.includes('Start a conversation')) {
    box.innerHTML = '';
  }
  
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;
  
  const roleDiv = document.createElement('div');
  roleDiv.className = 'msg-role';
  roleDiv.textContent = role === 'user' ? '👤 You' : '🤖 Assistant';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'msg-content';
  contentDiv.textContent = content;
  
  msg.appendChild(roleDiv);
  msg.appendChild(contentDiv);
  
  if (citations.length > 0) {
    const citeDiv = document.createElement('div');
    citeDiv.className = 'msg-citations';
    citeDiv.textContent = '📎 ' + citations.join(', ');
    msg.appendChild(citeDiv);
  }
  
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;
  
  const topK = parseInt(document.getElementById('chatTopK').value, 10);
  const sourceIds = getSelectedSourceIds();
  
  appendMessage('user', message);
  input.value = '';
  
  try {
    const data = await apiCall(`/tenants/${encodeURIComponent(currentTenant)}/chat`, {
      method: 'POST',
      body: JSON.stringify({
        message,
        session_id: sessionId,
        top_k: topK,
        include_history: true,
        source_ids: sourceIds
      })
    });
    
    if (!sessionId) {
      sessionId = data.session_id;
    }
    
    appendMessage('assistant', data.answer, data.citations || []);
  } catch (e) {
    appendMessage('assistant', '❌ Error: ' + e.message);
  }
}

function newSession() {
  sessionId = null;
  document.getElementById('chatBox').innerHTML = '<div style="text-align:center;color:#8893a4;padding:40px 20px;">New session started</div>';
}

function clearChat() {
  document.getElementById('chatBox').innerHTML = '<div style="text-align:center;color:#8893a4;padding:40px 20px;">Chat cleared</div>';
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================================
// Event Listeners
// ============================================================================

function initializeEventListeners() {
  document.getElementById('btnCreateTenant').onclick = createTenant;
  document.getElementById('btnCreateSource').onclick = createSource;
  document.getElementById('btnProceedWithoutSource').onclick = proceedWithoutSource;
  document.getElementById('btnUpload').onclick = uploadDocument;
  document.getElementById('btnSend').onclick = sendMessage;
  document.getElementById('btnNewSession').onclick = newSession;
  document.getElementById('btnClearChat').onclick = clearChat;

  document.getElementById('chatInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ============================================================================
// Initialization
// ============================================================================

function initialize() {
  initializeEventListeners();
  loadTenants();
  updateBreadcrumb('tenants');
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
