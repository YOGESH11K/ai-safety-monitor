(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var THRESHOLD = 0.80;

  var API_BASE = (window.FIGHT_API_BASE || "").replace(/\/+$/, "");
  function api(url) { return API_BASE + url; }
  function wsUrl(path) {
    if (!API_BASE) {
      var proto = location.protocol === "https:" ? "wss://" : "ws://";
      return proto + location.host + path;
    }
    return API_BASE.replace(/^http/, "ws") + path;
  }

  var els = {
    connDot: $("connDot"),
    modelBadge: $("modelBadge"),
    stateBadge: $("stateBadge"),
    riskBadge: $("riskBadge"),
    cameraId: $("cameraId"),
    detectionList: $("detectionList"),
    violenceBadge: $("violenceBadge"),
    eventsBody: $("eventsBody"),
    refreshEvents: $("refreshEvents"),
    eventSelect: $("eventSelect"),
    evidenceImg: $("evidenceImg"),
    evidenceEmpty: $("evidenceEmpty"),
    markReviewed: $("markReviewed"),
    markIgnored: $("markIgnored"),
    feedFrame: $("feedFrame"),
    feedSource: $("feedSource"),
  };

  var latestDetections = [];
  var latestState = "NORMAL";

  function setState(state) {
    latestState = state;
    els.stateBadge.className = "state " + state;
    els.stateBadge.textContent = formatState(state);
  }

  function formatState(s) {
    return { NORMAL: "NORMAL", WATCHING: "WATCHING", DETECTION: "DETECTION", CONFIRMED_EVENT: "CONFIRMED EVENT" }[s] || s;
  }

  function setRisk(level) {
    els.riskBadge.className = "risk" + (level && level !== "none" ? " " + level : "");
    els.riskBadge.textContent = "risk: " + (level || "none");
  }

  function renderDetections(list) {
    latestDetections = list || [];
    if (!latestDetections.length) {
      els.detectionList.innerHTML = '<li class="empty">No detections.</li>';
      return;
    }
    var html = "";
    latestDetections.forEach(function (d) {
      var conf = Math.round(d.confidence * 100);
      var over = d.confidence >= THRESHOLD;
      html +=
        '<li>' +
        '<span class="det-class">' + escapeHtml(d["class"]) + "</span>" +
        '<span class="det-conf' + (over ? " over" : "") + '"><span class="fill" style="width:' + conf + '%"></span></span>' +
        '<span class="det-val">' + conf + "%</span>" +
        "</li>";
    });
    els.detectionList.innerHTML = html;
  }

  function renderEvents(events) {
    var tbody = els.eventsBody;
    if (!events.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No events yet.</td></tr>';
      els.eventSelect.innerHTML = '<option value="">(none)</option>';
      return;
    }
    var rows = "";
    var opts = "";
    events.forEach(function (e) {
      var time = formatTime(e.timestamp);
      var conf = Math.round(e.confidence * 100) + "%";
      rows +=
        '<tr>' +
        '<td>' + time + "</td>" +
        '<td><span class="type-tag ' + (e.event_type || "weapon") + '">' + escapeHtml(e.event_type || "weapon") + "</span></td>" +
        "<td>" + escapeHtml(e.detected_class || "-") + "</td>" +
        "<td>" + conf + "</td>" +
        '<td><button class="ghost evidence-open" data-id="' + escapeHtml(e.id) + '">View</button></td>' +
        "</tr>";
      opts +=
        '<option value="' + escapeHtml(e.id) + '">' +
        time + " — " + escapeHtml(e.event_type || "weapon") + " (" + escapeHtml(e.detected_class || "-") + ")" +
        "</option>";
    });
    tbody.innerHTML = rows;
    els.eventSelect.innerHTML = opts;

    Array.prototype.forEach.call(document.querySelectorAll(".evidence-open"), function (btn) {
      btn.addEventListener("click", function () { loadEvidence(btn.getAttribute("data-id")); });
    });
  }

  function loadEvidence(id) {
    if (!id) return;
    els.evidenceImg.src = api("/api/events/" + encodeURIComponent(id) + "/image?" + Date.now());
    els.evidenceImg.hidden = false;
    els.evidenceEmpty.style.display = "none";
  }

  function formatTime(iso) {
    if (!iso) return "-";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) +
      " " + d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fetchStatus() {
    fetch(api("/api/status"))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        setState(data.state || "NORMAL");
        setRisk(data.risk_level);
        renderDetections(data.detections);
        els.feedFrame.textContent = "frame " + (data.frame || 0);
        els.feedSource.textContent = "source: " + (data.source || "-");
        els.cameraId.textContent = data.camera_id || "-";
        els.connDot.className = "dot online pulse";

        var mode = data.stub_mode ? "stub (demo)" : data.model;
        els.modelBadge.textContent = mode;
        els.modelBadge.className = "mode" + (data.stub_mode ? " demo" : "");

        els.violenceBadge.className = "badge" + (data.violence ? " active" : " quiet");
        els.violenceBadge.textContent = data.violence ? "flagged" : "cleared";
      })
      .catch(function () {
        els.connDot.className = "dot";
      });
  }

  function fetchEvents() {
    fetch(api("/api/events?limit=20"))
      .then(function (r) { return r.json(); })
      .then(function (data) { renderEvents(data.events || []); })
      .catch(function () {});
  }

  function connectWS() {
    var ws = new WebSocket(wsUrl("/api/ws"));
    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === "snapshot" || msg.type === "detections") {
        var d = msg.data || {};
        if (d.state) setState(d.state);
        if (d.risk_level) setRisk(d.risk_level);
        if (d.detections) renderDetections(d.detections);
        if (d.violence !== undefined) {
          els.violenceBadge.className = "badge" + (d.violence ? " active" : " quiet");
          els.violenceBadge.textContent = d.violence ? "flagged" : "cleared";
        }
        if (d.model_name) {
          els.modelBadge.textContent = d.stub_mode ? "stub (demo)" : d.model_name;
          els.modelBadge.className = "mode" + (d.stub_mode ? " demo" : "");
        }
      } else if (msg.type === "event") {
        fetchEvents();
        fetchStatus();
      }
    };
    ws.onclose = function () { setTimeout(connectWS, 3000); };
  }

  els.refreshEvents.addEventListener("click", fetchEvents);
  els.eventSelect.addEventListener("change", function () { loadEvidence(els.eventSelect.value); });
  els.markReviewed.addEventListener("click", function () { mark(els.eventSelect.value, "reviewed"); });
  els.markIgnored.addEventListener("click", function () { mark(els.eventSelect.value, "ignored"); });

  function mark(id, status) {
    if (!id) return;
    fetch(api("/api/events/" + encodeURIComponent(id) + "/review?status=" + status), { method: "POST" })
      .then(function () { fetchEvents(); })
      .catch(function () {});
  }

  els.liveFeed.src = api("/api/feed");

  setInterval(fetchStatus, 2000);
  fetchStatus();
  fetchEvents();
  connectWS();
})();