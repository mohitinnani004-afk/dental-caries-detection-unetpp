/**
 * Dental Caries UNet++ Web Application Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    currentSampleId: 'sample_proximal',
    customImageBase64: null,
    currentThreshold: 0.50,
    currentView: 'composite',
    lastAnalysis: null,
    originalImageB64: null,
    debounceTimer: null
  };

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const sampleItems = document.querySelectorAll('.sample-item');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdBadge = document.getElementById('threshold-val-badge');
  const pillButtons = document.querySelectorAll('.pill-btn');
  const tabButtons = document.querySelectorAll('.tab-btn');
  const spinner = document.getElementById('loading-spinner');
  
  const mainRadiographView = document.getElementById('main-radiograph-view');
  const dualViewContainer = document.getElementById('dual-view-container');
  const dualImgOrig = document.getElementById('dual-img-orig');
  const dualImgPred = document.getElementById('dual-img-pred');

  const statusBanner = document.getElementById('status-banner');
  const statusTitle = document.getElementById('status-title');
  const statusSubtitle = document.getElementById('status-subtitle');
  const statusRiskBadge = document.getElementById('status-risk-badge');

  const metricLesionCount = document.getElementById('metric-lesion-count');
  const metricHighestSeverity = document.getElementById('metric-highest-severity');
  const metricAffectedArea = document.getElementById('metric-affected-area');
  const metricConfidence = document.getElementById('metric-max-confidence');

  const lesionTableBody = document.getElementById('lesion-table-body');
  const lesionFindingsCount = document.getElementById('lesion-findings-count');

  const btnExportReport = document.getElementById('btn-export-report');
  const btnDownloadOverlay = document.getElementById('btn-download-overlay');
  const btnPrint = document.getElementById('btn-print');

  const archModal = document.getElementById('architecture-modal');
  const btnArchModal = document.getElementById('btn-architecture-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const btnCloseModalBottom = document.getElementById('btn-close-modal-bottom');

  // Initialize
  init();

  function init() {
    setupEventListeners();
    fetchOriginalSampleImage(state.currentSampleId);
    runAnalysis();
  }

  function setupEventListeners() {
    // Dropzone upload interactions
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Sample selection
    sampleItems.forEach(item => {
      item.addEventListener('click', () => {
        sampleItems.forEach(i => {
          i.classList.remove('active');
          i.setAttribute('aria-selected', 'false');
        });
        item.classList.add('active');
        item.setAttribute('aria-selected', 'true');

        state.currentSampleId = item.getAttribute('data-sample-id');
        state.customImageBase64 = null;
        fetchOriginalSampleImage(state.currentSampleId);
        runAnalysis();
      });

      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          item.click();
        }
      });
    });

    // Threshold slider (debounced)
    thresholdSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.currentThreshold = val;
      thresholdBadge.textContent = val.toFixed(2);
      updatePresetPillActiveState(val);

      clearTimeout(state.debounceTimer);
      state.debounceTimer = setTimeout(() => {
        runAnalysis();
      }, 120);
    });

    // Preset sensitivity buttons
    pillButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const val = parseFloat(btn.getAttribute('data-thresh'));
        state.currentThreshold = val;
        thresholdSlider.value = val;
        thresholdBadge.textContent = val.toFixed(2);
        updatePresetPillActiveState(val);
        runAnalysis();
      });
    });

    // View tabs
    tabButtons.forEach(tab => {
      tab.addEventListener('click', () => {
        tabButtons.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');

        state.currentView = tab.getAttribute('data-view');
        renderCurrentView();
      });
    });

    // Modal triggers
    btnArchModal.addEventListener('click', () => {
      archModal.style.display = 'flex';
    });

    const closeModal = () => {
      archModal.style.display = 'none';
    };
    btnCloseModal.addEventListener('click', closeModal);
    btnCloseModalBottom.addEventListener('click', closeModal);
    archModal.addEventListener('click', (e) => {
      if (e.target === archModal) closeModal();
    });

    // Actions
    btnExportReport.addEventListener('click', exportDiagnosticReport);
    btnDownloadOverlay.addEventListener('click', downloadAnnotatedImage);
    btnPrint.addEventListener('click', () => window.print());
  }

  function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid dental image file (PNG, JPG, TIFF).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      state.customImageBase64 = event.target.result;
      state.originalImageB64 = event.target.result;
      state.currentSampleId = null;

      sampleItems.forEach(i => {
        i.classList.remove('active');
        i.setAttribute('aria-selected', 'false');
      });

      runAnalysis();
    };
    reader.readAsDataURL(file);
  }

  function fetchOriginalSampleImage(sampleId) {
    fetch(`/api/sample/${sampleId}`)
      .then(res => res.json())
      .then(data => {
        state.originalImageB64 = `data:image/png;base64,${data.image_b64}`;
        if (state.currentView === 'dual') {
          dualImgOrig.src = state.originalImageB64;
        }
      })
      .catch(err => console.error('Error fetching sample image:', err));
  }

  function updatePresetPillActiveState(val) {
    pillButtons.forEach(btn => {
      const pVal = parseFloat(btn.getAttribute('data-thresh'));
      if (Math.abs(pVal - val) < 0.01) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  function showLoading(isLoading) {
    spinner.style.display = isLoading ? 'block' : 'none';
    mainRadiographView.style.opacity = isLoading ? '0.3' : '1.0';
  }

  function runAnalysis() {
    showLoading(true);

    const payload = {
      threshold: state.currentThreshold
    };

    if (state.customImageBase64) {
      payload.image_b64 = state.customImageBase64;
    } else {
      payload.sample_id = state.currentSampleId;
    }

    fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) throw new Error('Analysis request failed');
        return res.json();
      })
      .then(data => {
        state.lastAnalysis = data;
        updateUI(data);
      })
      .catch(err => {
        console.error('Analysis error:', err);
      })
      .finally(() => {
        showLoading(false);
      });
  }

  function updateUI(data) {
    const summary = data.summary;
    const lesions = data.lesions;

    // Status Banner
    if (summary.caries_detected) {
      statusBanner.className = 'status-banner';
      statusTitle.textContent = `Dental Caries Identified (${lesions.length} ${lesions.length === 1 ? 'Cavity' : 'Cavities'})`;
      statusSubtitle.textContent = `Radiographic breakdown observed in tooth enamel/dentin. Excavation and restoration recommended.`;
      statusRiskBadge.textContent = `${summary.highest_severity.split(' ')[0]} Risk`;
      statusRiskBadge.style.background = 'rgba(255, 71, 87, 0.2)';
      statusRiskBadge.style.borderColor = '#ff4757';
      statusRiskBadge.style.color = '#ff6b81';
    } else {
      statusBanner.className = 'status-banner healthy';
      statusTitle.textContent = 'Healthy Dentition — No Caries Detected';
      statusSubtitle.textContent = 'Radiograph shows intact enamel margins and standard coronal mineral density.';
      statusRiskBadge.textContent = 'Intact (Clear)';
      statusRiskBadge.style.background = 'rgba(16, 185, 129, 0.2)';
      statusRiskBadge.style.borderColor = '#10b981';
      statusRiskBadge.style.color = '#34d399';
    }

    // Metric Cards
    metricLesionCount.textContent = summary.lesion_count;
    metricHighestSeverity.textContent = summary.highest_severity.split(' ')[0] || 'Healthy';
    metricAffectedArea.textContent = `${summary.estimated_caries_area_mm2} mm²`;
    metricConfidence.textContent = `${Math.round(summary.max_confidence * 100)}%`;

    // Render Viewport
    renderCurrentView();

    // Populate Findings Table
    populateLesionTable(lesions);
  }

  function renderCurrentView() {
    if (!state.lastAnalysis) return;

    const data = state.lastAnalysis;

    if (state.currentView === 'dual') {
      mainRadiographView.style.display = 'none';
      dualViewContainer.style.display = 'grid';
      dualImgOrig.src = state.originalImageB64 || `data:image/png;base64,${data.clahe_image}`;
      dualImgPred.src = `data:image/png;base64,${data.composite_image}`;
      return;
    }

    dualViewContainer.style.display = 'none';
    mainRadiographView.style.display = 'block';

    if (state.currentView === 'composite') {
      mainRadiographView.src = `data:image/png;base64,${data.composite_image}`;
    } else if (state.currentView === 'heatmap') {
      mainRadiographView.src = `data:image/png;base64,${data.heatmap_image}`;
    } else if (state.currentView === 'clahe') {
      mainRadiographView.src = `data:image/png;base64,${data.clahe_image}`;
    }
  }

  function populateLesionTable(lesions) {
    lesionTableBody.innerHTML = '';
    lesionFindingsCount.textContent = `Showing ${lesions.length} lesion${lesions.length === 1 ? '' : 's'}`;

    if (lesions.length === 0) {
      const emptyRow = document.createElement('tr');
      emptyRow.innerHTML = `
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">
          ✓ No radiolucent caries lesions detected above probability threshold ${state.currentThreshold.toFixed(2)}.
        </td>
      `;
      lesionTableBody.appendChild(emptyRow);
      return;
    }

    lesions.forEach(lesion => {
      const row = document.createElement('tr');
      
      let tagClass = 'e1';
      const sev = lesion.severity_grade.toLowerCase();
      if (sev.includes('e2')) tagClass = 'e2';
      else if (sev.includes('d1')) tagClass = 'd1';
      else if (sev.includes('d2')) tagClass = 'd2';
      else if (sev.includes('d3')) tagClass = 'd3';

      row.innerHTML = `
        <td><strong>#${lesion.id}</strong></td>
        <td>${lesion.site}</td>
        <td><span class="tag-severity ${tagClass}">${lesion.severity_grade}</span></td>
        <td><strong>${Math.round(lesion.confidence * 100)}%</strong></td>
        <td>${lesion.estimated_area_mm2} mm²</td>
        <td style="color: var(--text-secondary);">${lesion.clinical_recommendation}</td>
      `;
      lesionTableBody.appendChild(row);
    });
  }

  function exportDiagnosticReport() {
    if (!state.lastAnalysis) return;
    const reportData = {
      timestamp: new Date().toISOString(),
      model: "UNet++ (Nested U-Net with Deep Supervision L1-L4)",
      parameters: {
        applied_threshold: state.currentThreshold,
        source: state.currentSampleId || "User Radiograph Upload"
      },
      summary: state.lastAnalysis.summary,
      detailed_lesions: state.lastAnalysis.lesions
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dental_caries_diagnostic_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadAnnotatedImage() {
    if (!state.lastAnalysis || !state.lastAnalysis.composite_image) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${state.lastAnalysis.composite_image}`;
    a.download = `annotated_caries_radiograph_${Date.now()}.png`;
    a.click();
  }
});
