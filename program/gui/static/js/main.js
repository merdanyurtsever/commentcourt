/**
 * CommentCourt - Main JavaScript
 * Interactive features for the influencer analysis dashboard
 */

// ============================================
// Global Configuration
// ============================================
const API_BASE = '/api';
const REFRESH_INTERVAL = 30000; // 30 seconds

// ============================================
// Utility Functions
// ============================================

/**
 * Format a number with locale-specific formatting
 * @param {number} num - Number to format
 * @returns {string} Formatted number string
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString('tr-TR');
}

/**
 * Format a date to locale string
 * @param {string} dateStr - ISO date string
 * @returns {string} Formatted date string
 */
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('tr-TR', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Get score color class based on score value
 * @param {number} score - Score value (0-10)
 * @returns {string} CSS class name
 */
function getScoreClass(score) {
    if (score >= 8) return 'excellent';
    if (score >= 6) return 'good';
    if (score >= 4) return 'average';
    return 'poor';
}

/**
 * Get sentiment icon based on sentiment value
 * @param {string} sentiment - Sentiment type (positive, negative, neutral)
 * @returns {string} FontAwesome icon class
 */
function getSentimentIcon(sentiment) {
    switch (sentiment.toLowerCase()) {
        case 'positive':
            return 'fas fa-smile text-success';
        case 'negative':
            return 'fas fa-frown text-danger';
        default:
            return 'fas fa-meh text-secondary';
    }
}

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Toast type (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

/**
 * Create toast container if it doesn't exist
 * @returns {HTMLElement} Toast container element
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

/**
 * Show loading spinner in an element
 * @param {HTMLElement} element - Element to show spinner in
 * @param {boolean} show - Whether to show or hide the spinner
 */
function toggleLoading(element, show = true) {
    if (show) {
        element.dataset.originalContent = element.innerHTML;
        element.innerHTML = '<div class="loading-spinner mx-auto"></div>';
        element.classList.add('loading');
    } else {
        element.innerHTML = element.dataset.originalContent || '';
        element.classList.remove('loading');
    }
}

// ============================================
// API Functions
// ============================================

/**
 * Make an API request
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Response data
 */
async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

/**
 * Get dashboard statistics
 * @returns {Promise<Object>} Dashboard stats
 */
async function getDashboardStats() {
    return apiRequest('/stats');
}

/**
 * Get influencer list with optional filters
 * @param {Object} filters - Filter parameters
 * @returns {Promise<Array>} List of influencers
 */
async function getInfluencers(filters = {}) {
    const params = new URLSearchParams(filters);
    return apiRequest(`/influencers?${params}`);
}

/**
 * Get single influencer details
 * @param {number} id - Influencer ID
 * @returns {Promise<Object>} Influencer details
 */
async function getInfluencer(id) {
    return apiRequest(`/influencers/${id}`);
}

/**
 * Trigger analysis for an influencer
 * @param {number} influencerId - Influencer ID (optional, null for all)
 * @returns {Promise<Object>} Analysis result
 */
async function triggerAnalysis(influencerId = null) {
    const endpoint = influencerId ? `/analyze/${influencerId}` : '/analyze';
    return apiRequest(endpoint, { method: 'POST' });
}

/**
 * Get model comparison data
 * @returns {Promise<Array>} Model comparison data
 */
async function getModelComparison() {
    return apiRequest('/models');
}

// ============================================
// UI Components
// ============================================

/**
 * Create score badge HTML
 * @param {number} score - Score value
 * @param {boolean} large - Whether to use large size
 * @returns {string} HTML string
 */
function createScoreBadge(score, large = false) {
    const scoreClass = getScoreClass(score);
    const sizeClass = large ? 'score-large' : '';
    return `<span class="score-badge ${scoreClass} ${sizeClass}">${score.toFixed(1)}</span>`;
}

/**
 * Create rank badge HTML
 * @param {number} rank - Rank position
 * @returns {string} HTML string
 */
function createRankBadge(rank) {
    let badgeClass = 'default';
    if (rank === 1) badgeClass = 'gold';
    else if (rank === 2) badgeClass = 'silver';
    else if (rank === 3) badgeClass = 'bronze';
    
    return `<span class="rank-badge ${badgeClass}">#${rank}</span>`;
}

/**
 * Create sentiment chart HTML
 * @param {Object} distribution - Sentiment distribution {positive, neutral, negative}
 * @returns {string} HTML string
 */
function createSentimentChart(distribution) {
    const total = distribution.positive + distribution.neutral + distribution.negative;
    if (total === 0) return '<div class="text-muted">Veri yok</div>';
    
    const posPercent = (distribution.positive / total * 100).toFixed(1);
    const neuPercent = (distribution.neutral / total * 100).toFixed(1);
    const negPercent = (distribution.negative / total * 100).toFixed(1);
    
    return `
        <div class="sentiment-chart">
            <div class="positive" style="width: ${posPercent}%" title="Pozitif: ${posPercent}%"></div>
            <div class="neutral" style="width: ${neuPercent}%" title="Nötr: ${neuPercent}%"></div>
            <div class="negative" style="width: ${negPercent}%" title="Negatif: ${negPercent}%"></div>
        </div>
        <div class="d-flex justify-content-between mt-1 small text-muted">
            <span><i class="fas fa-smile text-success me-1"></i>${posPercent}%</span>
            <span><i class="fas fa-meh text-secondary me-1"></i>${neuPercent}%</span>
            <span><i class="fas fa-frown text-danger me-1"></i>${negPercent}%</span>
        </div>
    `;
}

/**
 * Create influencer card HTML
 * @param {Object} influencer - Influencer data
 * @returns {string} HTML string
 */
function createInfluencerCard(influencer) {
    return `
        <div class="col-md-6 col-lg-4 mb-4">
            <div class="card influencer-card h-100" onclick="window.location='/influencer/${influencer.id}'">
                <div class="card-body text-center">
                    <div class="position-relative d-inline-block mb-3">
                        <img src="${influencer.avatar_url || '/static/images/default-avatar.png'}" 
                             alt="${influencer.name}" 
                             class="influencer-avatar">
                        ${createRankBadge(influencer.rank)}
                    </div>
                    <h5 class="influencer-name">${influencer.name}</h5>
                    <p class="influencer-platform mb-3">
                        <i class="fab fa-${influencer.platform.toLowerCase()} platform-${influencer.platform.toLowerCase()}"></i>
                        @${influencer.username}
                    </p>
                    <div class="mb-3">
                        ${createScoreBadge(influencer.score, true)}
                    </div>
                    <div class="small text-muted">
                        <span class="me-3"><i class="fas fa-comments me-1"></i>${formatNumber(influencer.comment_count)}</span>
                        <span><i class="fas fa-clock me-1"></i>${formatDate(influencer.last_analyzed)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================
// Page-Specific Functions
// ============================================

/**
 * Initialize dashboard page
 */
function initDashboard() {
    // Auto-refresh stats
    loadDashboardStats();
    setInterval(loadDashboardStats, REFRESH_INTERVAL);
}

/**
 * Load dashboard statistics
 */
async function loadDashboardStats() {
    try {
        const stats = await getDashboardStats();
        updateDashboardUI(stats);
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

/**
 * Update dashboard UI with new stats
 * @param {Object} stats - Dashboard statistics
 */
function updateDashboardUI(stats) {
    // Update stat cards
    const elements = {
        'stat-influencers': stats.total_influencers,
        'stat-comments': formatNumber(stats.total_comments),
        'stat-avg-score': stats.average_score?.toFixed(1) || '-',
        'stat-best-model': stats.best_model || '-'
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });
}

/**
 * Initialize influencers list page
 */
function initInfluencersPage() {
    const filterForm = document.getElementById('filter-form');
    if (filterForm) {
        filterForm.addEventListener('submit', handleFilterSubmit);
    }
    
    // Initialize sort buttons
    document.querySelectorAll('[data-sort]').forEach(btn => {
        btn.addEventListener('click', handleSortClick);
    });
}

/**
 * Handle filter form submission
 * @param {Event} e - Form submit event
 */
function handleFilterSubmit(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const params = new URLSearchParams(formData);
    window.location.search = params.toString();
}

/**
 * Handle sort button click
 * @param {Event} e - Click event
 */
function handleSortClick(e) {
    const sortField = e.target.dataset.sort;
    const currentSort = new URLSearchParams(window.location.search).get('sort');
    const currentOrder = new URLSearchParams(window.location.search).get('order') || 'desc';
    
    let newOrder = 'desc';
    if (currentSort === sortField && currentOrder === 'desc') {
        newOrder = 'asc';
    }
    
    const params = new URLSearchParams(window.location.search);
    params.set('sort', sortField);
    params.set('order', newOrder);
    window.location.search = params.toString();
}

/**
 * Initialize analysis page
 */
function initAnalysisPage() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', handleAnalyzeClick);
    }
    
    // Load model comparison
    loadModelComparison();
}

/**
 * Handle analyze button click
 * @param {Event} e - Click event
 */
async function handleAnalyzeClick(e) {
    const btn = e.target;
    const influencerId = btn.dataset.influencerId;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analiz ediliyor...';
    
    try {
        const result = await triggerAnalysis(influencerId);
        showToast('Analiz başarıyla tamamlandı!', 'success');
        
        // Reload page to show new results
        setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
        showToast('Analiz sırasında hata oluştu.', 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play me-2"></i>Analizi Başlat';
    }
}

/**
 * Load and display model comparison
 */
async function loadModelComparison() {
    const container = document.getElementById('model-comparison');
    if (!container) return;
    
    try {
        const models = await getModelComparison();
        renderModelComparison(container, models);
    } catch (error) {
        container.innerHTML = '<div class="alert alert-danger">Model verileri yüklenemedi.</div>';
    }
}

/**
 * Render model comparison cards
 * @param {HTMLElement} container - Container element
 * @param {Array} models - Model data array
 */
function renderModelComparison(container, models) {
    const html = models.map((model, index) => `
        <div class="col-md-6 col-lg-3 mb-4">
            <div class="card model-card ${model.is_best ? 'selected' : ''} h-100">
                ${model.is_best ? '<span class="best-model-badge">En İyi</span>' : ''}
                <div class="card-body text-center">
                    <h6 class="model-name mb-3">${model.name}</h6>
                    <div class="model-accuracy mb-2">${(model.accuracy * 100).toFixed(1)}%</div>
                    <div class="text-muted small mb-3">Doğruluk</div>
                    <div class="row small">
                        <div class="col-4">
                            <div class="text-muted">Precision</div>
                            <div class="fw-bold">${(model.precision * 100).toFixed(1)}%</div>
                        </div>
                        <div class="col-4">
                            <div class="text-muted">Recall</div>
                            <div class="fw-bold">${(model.recall * 100).toFixed(1)}%</div>
                        </div>
                        <div class="col-4">
                            <div class="text-muted">F1</div>
                            <div class="fw-bold">${(model.f1_score * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

// ============================================
// Comments Table Functions
// ============================================

/**
 * Initialize comments page
 */
function initCommentsPage() {
    // Initialize DataTable-like functionality
    const searchInput = document.getElementById('comment-search');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(handleCommentSearch, 300));
    }
}

/**
 * Handle comment search
 * @param {Event} e - Input event
 */
function handleCommentSearch(e) {
    const query = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#comments-table tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================
// Charts (using Chart.js if available)
// ============================================

/**
 * Create sentiment trend chart
 * @param {string} canvasId - Canvas element ID
 * @param {Array} data - Chart data
 */
function createSentimentTrendChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: data.map(d => formatDate(d.date)),
            datasets: [
                {
                    label: 'Pozitif',
                    data: data.map(d => d.positive),
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true
                },
                {
                    label: 'Negatif',
                    data: data.map(d => d.negative),
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

/**
 * Create model accuracy bar chart
 * @param {string} canvasId - Canvas element ID
 * @param {Array} models - Model data
 */
function createModelAccuracyChart(canvasId, models) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: models.map(m => m.name),
            datasets: [{
                label: 'Doğruluk',
                data: models.map(m => m.accuracy * 100),
                backgroundColor: models.map(m => m.is_best ? '#4a90d9' : '#6c757d')
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => value + '%'
                    }
                }
            }
        }
    });
}

// ============================================
// Document Ready
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
    
    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    popoverTriggerList.forEach(el => new bootstrap.Popover(el));
    
    // Page-specific initialization
    const page = document.body.dataset.page;
    switch (page) {
        case 'dashboard':
            initDashboard();
            break;
        case 'influencers':
            initInfluencersPage();
            break;
        case 'analysis':
            initAnalysisPage();
            break;
        case 'comments':
            initCommentsPage();
            break;
    }
    
    console.log('CommentCourt initialized');
});
