(function () {
  'use strict';

  // --- State ---
  let uploadedFiles = []; // Array of File objects

  // --- DOM refs ---
  const form = document.getElementById('verify-form');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const previewContainer = document.getElementById(
    'preview-container',
  );
  const verifyBtn = document.getElementById('verify-btn');
  const verifyText = document.getElementById('verify-text');
  const verifySpinner = document.getElementById('verify-spinner');
  const resultsSection = document.getElementById('results-section');
  const resultsContainer = document.getElementById(
    'results-container',
  );
  const debugToggle = document.getElementById('debug-toggle');
  const engineGroup = document.getElementById('engine-group');
  const engineSelect = document.getElementById('engine-select');

  // --- Debug toggle: show/hide engine selector ---
  function syncEngineVisibility() {
    engineGroup.classList.toggle('d-none', !debugToggle.checked);
  }
  debugToggle.addEventListener('change', syncEngineVisibility);
  syncEngineVisibility();

  // --- File handling ---

  function addFiles(fileList) {
    const allowed = ['image/jpeg', 'image/png', 'image/webp'];
    for (const file of fileList) {
      if (allowed.includes(file.type)) {
        uploadedFiles.push(file);
      }
    }
    renderPreviews();
  }

  function removeFile(index) {
    uploadedFiles.splice(index, 1);
    renderPreviews();
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function renderPreviews() {
    previewContainer.innerHTML = '';
    uploadedFiles.forEach(function (file, i) {
      var item = document.createElement('div');
      item.className = 'preview-item';

      var img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = file.name;
      img.onload = function () {
        URL.revokeObjectURL(img.src);
      };

      var removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'remove-btn';
      removeBtn.innerHTML = '&#10005;';
      removeBtn.title = 'Remove';
      removeBtn.setAttribute('data-index', i);
      removeBtn.addEventListener('click', function () {
        removeFile(parseInt(this.getAttribute('data-index'), 10));
      });

      var info = document.createElement('div');
      info.className = 'file-info';
      info.textContent =
        file.name + ' (' + formatSize(file.size) + ')';

      item.appendChild(img);
      item.appendChild(removeBtn);
      item.appendChild(info);
      previewContainer.appendChild(item);
    });
  }

  // --- Drag & drop ---

  dropZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', function () {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      addFiles(e.dataTransfer.files);
    }
  });

  browseBtn.addEventListener('click', function () {
    fileInput.click();
  });

  fileInput.addEventListener('change', function () {
    if (fileInput.files.length) {
      addFiles(fileInput.files);
    }
    fileInput.value = '';
  });

  // --- Form submission ---

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Validate
    form.classList.add('was-validated');
    if (!form.checkValidity()) {
      return;
    }
    if (uploadedFiles.length === 0) {
      alert('Please upload at least one label image.');
      return;
    }

    // Build multipart payload
    var fd = new FormData();
    fd.append(
      'brand_name',
      document.getElementById('brand_name').value.trim(),
    );
    fd.append(
      'alcohol_content',
      document.getElementById('alcohol_content').value,
    );
    fd.append(
      'net_contents',
      document.getElementById('net_contents').value.trim(),
    );
    fd.append(
      'gov_warning_expected',
      document.getElementById('gov_warning').checked,
    );
    fd.append(
      'debug',
      document.getElementById('debug-toggle').checked,
    );
    fd.append('engine', engineSelect.value);
    uploadedFiles.forEach(function (file) {
      fd.append('files', file);
    });

    // UI: loading state
    setLoading(true);
    resultsSection.classList.add('d-none');
    resultsContainer.innerHTML = '';

    try {
      var response = await fetch('/api/verify', {
        method: 'POST',
        body: fd,
      });

      if (!response.ok) {
        throw new Error('Server returned ' + response.status);
      }

      var data = await response.json();
      renderResults(data.results);
    } catch (err) {
      resultsSection.classList.remove('d-none');
      resultsContainer.innerHTML =
        '<div class="alert alert-danger">Verification failed: ' +
        escapeHtml(err.message) +
        '</div>';
    } finally {
      setLoading(false);
    }
  });

  function setLoading(on) {
    verifyBtn.disabled = on;
    verifyText.textContent = on ? 'Verifying…' : 'Verify Labels';
    verifySpinner.classList.toggle('d-none', !on);
  }

  // --- Results rendering ---

  var FIELD_LABELS = {
    brand_name: 'Brand Name',
    alcohol_content: 'Alcohol Content',
    net_contents: 'Net Contents',
    gov_warning: 'Government Warning',
  };

  var STATUS_BADGE = {
    match: '<span class="badge badge-match">Match</span>',
    partial_match:
      '<span class="badge badge-partial">Partial Match</span>',
    mismatch: '<span class="badge badge-mismatch">Mismatch</span>',
    not_found: '<span class="badge badge-not-found">Not Found</span>',
  };

  var FIELD_COLORS = {
    brand_name: 'rgba(13,110,253,0.45)',
    alcohol_content: 'rgba(25,135,84,0.45)',
    net_contents: 'rgba(111,66,193,0.45)',
    gov_warning: 'rgba(255,193,7,0.45)',
  };

  var FIELD_BORDERS = {
    brand_name: '#0d6efd',
    alcohol_content: '#198754',
    net_contents: '#6f42c1',
    gov_warning: '#ffc107',
  };

  function formatValue(val) {
    if (typeof val === 'boolean') return val ? 'Yes' : 'No';
    return val == null ? '—' : String(val);
  }

  function confidenceBar(conf) {
    var pct = Math.round((conf || 0) * 100);
    return (
      '<span class="confidence-bar"><span class="confidence-bar-fill" style="width:' +
      pct +
      '%"></span></span> <small>' +
      pct +
      '%</small>'
    );
  }

  function overallClass(result) {
    var statuses = Object.values(result.fields).map(function (f) {
      return f.match_status;
    });
    if (
      statuses.every(function (s) {
        return s === 'match';
      })
    )
      return 'overall-match';
    if (
      statuses.some(function (s) {
        return s === 'mismatch';
      })
    )
      return 'overall-mismatch';
    return 'overall-partial';
  }

  function renderResults(results) {
    resultsSection.classList.remove('d-none');
    resultsContainer.innerHTML = '';

    if (!results || results.length === 0) {
      resultsContainer.innerHTML =
        '<div class="alert alert-warning">No results returned.</div>';
      return;
    }

    results.forEach(function (result, ri) {
      // Try to find matching uploaded file for thumbnail
      var thumbSrc = '';
      var matchedFile = uploadedFiles.find(function (f) {
        return f.name === result.filename;
      });
      if (matchedFile) {
        thumbSrc = URL.createObjectURL(matchedFile);
      }

      var card = document.createElement('div');
      card.className =
        'card result-card mb-3 ' + overallClass(result);

      var body = document.createElement('div');
      body.className = 'card-body';

      // Header row: thumbnail + overall score
      var header = document.createElement('div');
      header.className = 'd-flex align-items-start gap-3 mb-3';
      header.innerHTML =
        (thumbSrc
          ? '<img src="' + thumbSrc + '" alt="" class="result-thumb">'
          : '') +
        '<div>' +
        "<h6 class='mb-1'>" +
        escapeHtml(result.filename) +
        '</h6>' +
        '<div>Overall Confidence: ' +
        confidenceBar(result.overall_confidence) +
        '</div>' +
        '</div>';

      // Fields table
      var table = document.createElement('table');
      table.className = 'table table-sm table-bordered mb-0 mt-2';
      var thead =
        "<thead class='table-light'><tr>" +
        '<th>Field</th><th>Expected</th><th>Extracted</th><th>Status</th><th>Confidence</th>' +
        '</tr></thead>';

      var tbody = '<tbody>';
      var fieldOrder = [
        'brand_name',
        'alcohol_content',
        'net_contents',
        'gov_warning',
      ];
      fieldOrder.forEach(function (key) {
        var f = result.fields[key];
        if (!f) return;
        tbody +=
          '<tr>' +
          '<td>' +
          (FIELD_LABELS[key] || key) +
          '</td>' +
          '<td>' +
          escapeHtml(formatValue(f.expected)) +
          '</td>' +
          '<td>' +
          escapeHtml(formatValue(f.extracted)) +
          '</td>' +
          '<td>' +
          (STATUS_BADGE[f.match_status] || f.match_status) +
          '</td>' +
          '<td>' +
          confidenceBar(f.confidence) +
          '</td>' +
          '</tr>';
      });
      tbody += '</tbody>';
      table.innerHTML = thead + tbody;

      body.appendChild(header);
      body.appendChild(table);

      // Annotated image with bounding box overlays
      if (thumbSrc && result.field_boxes) {
        var annotWrap = document.createElement('div');
        annotWrap.className = 'annotated-image-wrap mt-3';

        var annotInner = document.createElement('div');
        annotInner.className = 'annotated-image-inner';

        var annotImg = document.createElement('img');
        annotImg.src = thumbSrc;
        annotImg.className = 'annotated-image';
        annotInner.appendChild(annotImg);

        // Draw boxes once image dimensions are known
        annotImg.onload = function () {
          var natW = annotImg.naturalWidth;
          var natH = annotImg.naturalHeight;
          if (!natW || !natH) return;

          fieldOrder.forEach(function (key) {
            var boxes = result.field_boxes[key];
            if (!boxes || !boxes.length) return;

            boxes.forEach(function (box) {
              var b = box.bbox;
              var overlay = document.createElement('div');
              overlay.className = 'bbox-overlay';
              overlay.style.left = (b[0] / natW) * 100 + '%';
              overlay.style.top = (b[1] / natH) * 100 + '%';
              overlay.style.width =
                ((b[2] - b[0]) / natW) * 100 + '%';
              overlay.style.height =
                ((b[3] - b[1]) / natH) * 100 + '%';
              overlay.style.borderColor =
                FIELD_BORDERS[key] || '#0d6efd';
              overlay.style.backgroundColor =
                FIELD_COLORS[key] || 'rgba(13,110,253,0.2)';
              overlay.title =
                (FIELD_LABELS[key] || key) +
                ': ' +
                escapeHtml(box.text || '');
              annotInner.appendChild(overlay);
            });
          });
        };

        // Legend
        var legend = document.createElement('div');
        legend.className = 'bbox-legend mt-1';
        fieldOrder.forEach(function (key) {
          var boxes = result.field_boxes[key];
          if (!boxes || !boxes.length) return;
          legend.innerHTML +=
            '<span class="bbox-legend-item">' +
            '<span class="bbox-legend-swatch" style="background:' +
            (FIELD_BORDERS[key] || '#0d6efd') +
            '"></span>' +
            (FIELD_LABELS[key] || key) +
            '</span>';
        });

        annotWrap.appendChild(annotInner);
        annotWrap.appendChild(legend);
        body.appendChild(annotWrap);
      }

      // Warnings
      if (result.warnings && result.warnings.length > 0) {
        var warnDiv = document.createElement('div');
        warnDiv.className = 'mt-2';
        warnDiv.innerHTML = result.warnings
          .map(function (w) {
            return (
              '<span class="badge bg-warning text-dark me-1">' +
              escapeHtml(w) +
              '</span>'
            );
          })
          .join('');
        body.appendChild(warnDiv);
      }

      // Debug section
      if (result.debug) {
        var dbg = result.debug;
        var debugDiv = document.createElement('div');
        debugDiv.className =
          'mt-3 p-3 bg-light border rounded debug-section';
        debugDiv.style.fontSize = '0.85rem';

        var debugHtml = '<h6 class="mb-2">Debug Info</h6>';

        // Raw OCR text
        debugHtml +=
          '<details class="mb-2"><summary class="fw-bold">Raw OCR Text</summary>';
        debugHtml +=
          '<pre class="mt-1 mb-0" style="white-space:pre-wrap;max-height:200px;overflow-y:auto">' +
          escapeHtml(dbg.raw_ocr_text || '(empty)') +
          '</pre></details>';

        // Normalized OCR text
        debugHtml +=
          '<details class="mb-2"><summary class="fw-bold">Normalized OCR Text</summary>';
        debugHtml +=
          '<pre class="mt-1 mb-0" style="white-space:pre-wrap;max-height:200px;overflow-y:auto">' +
          escapeHtml(dbg.normalized_ocr_text || '(empty)') +
          '</pre></details>';

        // OCR by angle
        if (
          dbg.ocr_by_angle &&
          Object.keys(dbg.ocr_by_angle).length > 0
        ) {
          debugHtml +=
            '<details class="mb-2"><summary class="fw-bold">OCR by Angle</summary><div class="mt-1">';
          Object.keys(dbg.ocr_by_angle).forEach(function (angle) {
            debugHtml +=
              '<div class="mb-1"><strong>' +
              escapeHtml(angle) +
              '°:</strong> ' +
              escapeHtml(dbg.ocr_by_angle[angle] || '(empty)') +
              '</div>';
          });
          debugHtml += '</div></details>';
        }

        // OCR attempts with timings
        if (dbg.ocr_attempts && dbg.ocr_attempts.length > 0) {
          debugHtml +=
            '<details class="mb-2"><summary class="fw-bold">OCR Attempts (' +
            dbg.ocr_attempts.length +
            ')</summary>';
          debugHtml +=
            '<table class="table table-sm table-bordered mt-1 mb-0" style="font-size:0.8rem">';
          debugHtml +=
            '<thead><tr><th>#</th><th>Region</th><th>Angle</th><th>Preprocess</th><th>PSM</th><th>Time (ms)</th><th>Text (preview)</th></tr></thead><tbody>';
          dbg.ocr_attempts.forEach(function (a, idx) {
            var preview = (a.text || '').substring(0, 80);
            if ((a.text || '').length > 80) preview += '…';
            debugHtml +=
              '<tr>' +
              '<td>' +
              (idx + 1) +
              '</td>' +
              '<td>' +
              escapeHtml(a.region || '') +
              '</td>' +
              '<td>' +
              (a.angle || 0) +
              '°</td>' +
              '<td>' +
              escapeHtml(a.preprocess_mode || '') +
              '</td>' +
              '<td>' +
              (a.psm || '') +
              '</td>' +
              '<td>' +
              (a.elapsed_ms || 0).toFixed(1) +
              '</td>' +
              '<td>' +
              escapeHtml(preview || '(empty)') +
              '</td>' +
              '</tr>';
          });
          debugHtml += '</tbody></table></details>';
        }

        // Gov warning detail
        if (dbg.government_warning_detail) {
          var gwd = dbg.government_warning_detail;
          debugHtml +=
            '<details class="mb-2"><summary class="fw-bold">Government Warning Detail</summary><div class="mt-1">';
          debugHtml +=
            '<div><strong>Status:</strong> ' +
            escapeHtml(gwd.status || '') +
            ' &nbsp; <strong>Confidence:</strong> ' +
            ((gwd.confidence || 0) * 100).toFixed(0) +
            '%</div>';
          if (gwd.matched_groups) {
            debugHtml +=
              '<div class="mt-1"><strong>Matched Groups:</strong></div><ul class="mb-1">';
            Object.keys(gwd.matched_groups).forEach(function (g) {
              var mg = gwd.matched_groups[g];
              var label = mg.hits
                ? mg.hits.join(', ')
                : mg.best_phrase || '';
              debugHtml +=
                '<li>' +
                escapeHtml(g) +
                ': "' +
                escapeHtml(label) +
                '" (score: ' +
                (mg.score || 0) +
                ')</li>';
            });
            debugHtml += '</ul>';
          }
          if (gwd.matched_anchors && gwd.matched_anchors.length) {
            debugHtml +=
              '<div><strong>Matched Anchors:</strong> ' +
              gwd.matched_anchors
                .map(function (a) {
                  return escapeHtml(a);
                })
                .join(', ') +
              '</div>';
          }
          debugHtml += '</div></details>';
        }

        // Detailed field info
        if (dbg.fields_detail) {
          debugHtml +=
            '<details class="mb-0"><summary class="fw-bold">Field Extraction Detail</summary>';
          debugHtml +=
            '<pre class="mt-1 mb-0" style="white-space:pre-wrap;max-height:200px;overflow-y:auto">' +
            escapeHtml(JSON.stringify(dbg.fields_detail, null, 2)) +
            '</pre></details>';
        }

        debugDiv.innerHTML = debugHtml;
        body.appendChild(debugDiv);
      }

      card.appendChild(body);
      resultsContainer.appendChild(card);
    });
  }

  // --- Util ---
  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }
})();
