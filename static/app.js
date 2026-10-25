/* ReviewLens front end – plain JavaScript, no build step. */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  /** Tiny DOM builder. Text is always inserted as text, never as HTML. */
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === false || value == null) continue;
      if (key === "class") el.className = value;
      else if (key === "style") el.style.cssText = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : value);
    }
    children.flat(Infinity).forEach((child) => {
      if (child == null || child === false) return;
      el.append(child.nodeType ? child : document.createTextNode(String(child)));
    });
    return el;
  }

  const pct = (x) => `${Math.round(x * 1000) / 10}%`;
  const num = (x) => Number(x).toLocaleString();
  const fill = (el, ...nodes) => { el.replaceChildren(...nodes.flat()); return el; };

  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed. Please try again.");
    return data;
  }

  /* ------------------------------------------------------------ main tabs */
  const views = ["analyze", "batch", "model"];
  let modelLoaded = false;

  function showView(name) {
    if (!views.includes(name)) name = "analyze";
    views.forEach((v) => {
      $(`#view-${v}`).hidden = v !== name;
    });
    $$(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.view === name)));
    if (name === "model" && !modelLoaded) loadModel();
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  }
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));

  /* ----------------------------------------------------------- composer */
  const input = $("#review-input");
  const counter = $("#char-count");
  const analyzeBtn = $("#analyze-btn");
  const errorBox = $("#analyze-error");
  const maxChars = Number(input.getAttribute("maxlength"));

  const updateCount = () => { counter.textContent = `${num(input.value.length)} / ${num(maxChars)}`; };
  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
  });

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  }

  $("#clear-btn").addEventListener("click", () => {
    input.value = "";
    updateCount();
    showError("");
    input.focus();
  });

  $("#sample-btn").addEventListener("click", async () => {
    showError("");
    try {
      const response = await fetch("/api/sample");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      input.value = data.text;
      updateCount();
      analyze();
    } catch (err) {
      showError(err.message || "Could not load a sample review.");
    }
  });

  analyzeBtn.addEventListener("click", analyze);

  async function analyze() {
    const text = input.value.trim();
    if (text.length < 10) {
      showError("Write at least one full sentence so there is something to analyze.");
      input.focus();
      return;
    }
    showError("");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing…";
    try {
      const data = await postJSON("/api/analyze", { text });
      renderVerdict(data.sentiment);
      renderAnnotated(data.sentiment);
      renderToolkit(data.nlp);
      if (window.matchMedia("(max-width: 900px)").matches) {
        $("#verdict").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (err) {
      showError(err.message);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze review";
    }
  }

  /* ------------------------------------------------------------ verdict */
  function driverList(title, items, cls) {
    return h("div", { class: cls },
      h("h3", null, title),
      items.length
        ? h("ul", null, items.map((d) =>
            h("li", { class: "driver" },
              h("span", { class: "term", title: d.term }, d.term),
              h("span", { class: "bar" }, h("i", { style: `width:${Math.max(6, d.strength * 100)}%` })))))
        : h("p", { class: "none" }, "Nothing strong in this direction."));
  }

  function renderVerdict(s) {
    const tone = s.summary === "Mixed" ? "mixed" : s.label;
    const sure = s.confidence >= 0.9 ? "The model is very sure."
      : s.confidence >= 0.75 ? "The model is fairly sure."
      : s.confidence >= 0.6 ? "The model is only moderately sure." : "The model could go either way.";

    const neg = h("span", { class: "n", style: "width:50%" });
    const pos = h("span", { class: "p", style: "width:50%" });

    fill($("#verdict"),
      h("div", { class: `verdict-word ${tone}` }, s.summary),
      h("p", { class: "verdict-line" }, `${pct(s.label === "positive" ? s.positive : s.negative)} ${s.label}. ${sure}`),
      h("div", { class: "split", role: "img", "aria-label": `${pct(s.negative)} negative, ${pct(s.positive)} positive` }, neg, pos),
      h("div", { class: "split-labels" },
        h("span", { class: "n" }, `Negative ${pct(s.negative)}`),
        h("span", { class: "p" }, `Positive ${pct(s.positive)}`)),
      h("div", { class: "drivers" },
        driverList("Words and phrases pushing positive", s.drivers.positive, "pos"),
        driverList("Words and phrases pushing negative", s.drivers.negative, "neg")));

    // animate the split bar from the middle to its final size
    requestAnimationFrame(() => requestAnimationFrame(() => {
      neg.style.width = `${s.negative * 100}%`;
      pos.style.width = `${s.positive * 100}%`;
    }));
  }

  /* ---------------------------------------------------- annotated review */
  function renderAnnotated(s) {
    const nodes = s.highlight.map((seg) => {
      if (!seg.w) return document.createTextNode(seg.t);
      const rgb = seg.w > 0 ? "11,138,111" : "196,39,80";
      const strength = Math.abs(seg.w);
      const meaning = seg.w > 0 ? "raises positive" : "raises negative";
      return h("mark", {
        title: `“${seg.t}” ${meaning} (${seg.w > 0 ? "+" : ""}${seg.w.toFixed(2)})`,
        style: `background:rgba(${rgb},${(0.1 + strength * 0.5).toFixed(2)});` +
               `box-shadow:inset 0 -2px 0 rgba(${rgb},${(0.35 + strength * 0.65).toFixed(2)})`,
      }, seg.t);
    });
    fill($("#annotated-text"), nodes);
    $("#annotated").hidden = false;
  }

  /* ------------------------------------------------------ language toolkit */
  const subtabs = $$(".subtab");
  subtabs.forEach((tab) => tab.addEventListener("click", () => {
    subtabs.forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $$(".subview").forEach((v) => { v.hidden = v.id !== `sub-${tab.dataset.sub}`; });
  }));

  function barRows(items, labelKey, countKey) {
    const max = Math.max(...items.map((i) => i[countKey]), 1);
    return h("div", { class: "bars" }, items.map((item) =>
      h("div", { class: "hbar" },
        h("span", null, item[labelKey]),
        h("span", { class: "track" }, h("i", { style: `width:${(item[countKey] / max) * 100}%` })),
        h("span", { class: "count" }, item[countKey]))));
  }

  function renderToolkit(nlp) {
    const st = nlp.stats;
    const lang = nlp.language;

    // overview
    const fact = (label, value) => h("div", null, h("dt", null, label), h("dd", null, value));
    fill($("#sub-overview"),
      h("dl", { class: "facts" },
        fact("Language", lang.confidence ? `${lang.name} (${pct(lang.confidence)})` : lang.name),
        fact("Words", num(st.words)),
        fact("Sentences", num(st.sentences)),
        fact("Unique words", num(st.unique_words)),
        fact("Average word length", `${st.avg_word_length} letters`),
        fact("Average sentence length", `${st.avg_sentence_length} words`),
        fact("Vocabulary variety", pct(st.lexical_diversity)),
        fact("Filler words (the, and, of…)", pct(st.stopword_share)),
        fact("Reading time", st.reading_minutes < 0.1 ? "under 6 sec" : `${st.reading_minutes} min`)));

    // tokens and roots
    const changed = nlp.roots.filter((r) => r.stem !== r.word.toLowerCase() || r.lemma !== r.word.toLowerCase()).length;
    fill($("#sub-roots"),
      h("p", { class: "explain" }, "Tokenization splits the review into words and punctuation. Stemming chops word endings with fixed rules, so it can leave fragments (“incept”). Lemmatization looks the word up in a dictionary and returns a real word, using the part of speech to pick the right form."),
      h("h3", { class: "sub-heading" }, `First ${nlp.tokens.length} tokens`),
      h("ul", { class: "token-list" }, nlp.tokens.map((t) => h("li", { class: "token" }, t))),
      h("h3", { class: "sub-heading" }, "Stem vs lemma"),
      nlp.roots.length
        ? h("div", { class: "table-wrap" }, h("table", null,
            h("thead", null, h("tr", null, h("th", null, "Word"), h("th", null, "Stem"), h("th", null, "Lemma"))),
            h("tbody", null, nlp.roots.map((r) => h("tr", null,
              h("td", null, r.word),
              h("td", { class: r.stem !== r.word.toLowerCase() ? "changed" : "" }, r.stem),
              h("td", { class: r.lemma !== r.word.toLowerCase() ? "changed" : "" }, r.lemma))))))
        : h("p", { class: "muted" }, "No content words to show."),
      changed ? h("p", { class: "note" }, "Bold entries differ from the original word.") : null);

    // parts of speech
    fill($("#sub-pos"),
      h("p", { class: "explain" }, "Each word is labelled with its grammatical role. Adjectives and adverbs tend to carry the opinion in a review."),
      nlp.pos_distribution.length ? barRows(nlp.pos_distribution, "label", "count") : null,
      h("h3", { class: "sub-heading" }, `First ${nlp.pos.length} words tagged`),
      h("div", { class: "table-wrap" }, h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Word"), h("th", null, "Tag"), h("th", null, "Role"))),
        h("tbody", null, nlp.pos.map((p) => h("tr", null,
          h("td", null, p.word), h("td", null, h("span", { class: "tag" }, p.tag)), h("td", null, p.meaning)))))));

    // entities
    const labelNames = { PERSON: "Person", ORGANIZATION: "Organization", GPE: "Place", LOCATION: "Location", FACILITY: "Facility", GSP: "Country or region" };
    fill($("#sub-entities"),
      h("p", { class: "explain" }, "Names of people, organizations and places found by NLTK’s entity chunker. It works best on properly capitalised text."),
      nlp.entities.length
        ? h("ul", { class: "entity-list" }, nlp.entities.map((e) =>
            h("li", null, h("span", null, e.text), h("span", { class: "entity-label" }, `${labelNames[e.label] || e.label}${e.count > 1 ? ` ×${e.count}` : ""}`))))
        : h("p", { class: "muted" }, "No names found in this review. Try one that mentions actors, directors or cities."));

    // phrases
    const column = (title, items) => h("div", null, h("h3", null, title),
      items.length ? h("ol", null, items.map((i) => h("li", null, h("span", null, i.gram), h("span", { class: "c" }, i.count))))
                   : h("p", { class: "muted" }, "None"));
    fill($("#sub-phrases"),
      h("p", { class: "explain" }, "The most frequent words and word sequences. In a short review most appear once, so longer reviews give more useful results."),
      h("div", { class: "phrase-cols" },
        column("Single words", nlp.ngrams.unigrams),
        column("Two-word phrases", nlp.ngrams.bigrams),
        column("Three-word phrases", nlp.ngrams.trigrams)));

    $("#toolkit").hidden = false;
  }

  /* ---------------------------------------------------------------- batch */
  const dropzone = $("#dropzone");
  const csvInput = $("#csv-input");
  const batchError = $("#batch-error");

  ["dragenter", "dragover"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) { csvInput.files = e.dataTransfer.files; runBatch(); }
  });
  csvInput.addEventListener("change", runBatch);

  async function runBatch() {
    const file = csvInput.files[0];
    if (!file) return;
    batchError.hidden = true;
    $("#batch-result").hidden = true;
    $("#drop-title").textContent = `Scoring ${file.name}…`;
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/batch", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Could not read that file.");
      renderBatch(data);
    } catch (err) {
      batchError.textContent = err.message;
      batchError.hidden = false;
    } finally {
      $("#drop-title").textContent = "Choose another CSV file";
      csvInput.value = "";
    }
  }

  function renderBatch(d) {
    const posShare = d.positive / d.total;
    const stats = [
      [num(d.total), `reviews scored${d.truncated ? " (first 500)" : ""}`],
      [`${num(d.positive)}`, "positive"],
      [`${num(d.negative)}`, "negative"],
    ];
    if (d.accuracy != null) stats.push([pct(d.accuracy), "agreement with your labels"]);

    fill($("#batch-result"),
      h("p", { class: "summary-line" }, `Read the “${d.column}” column.`),
      h("div", { class: "batch-stats" }, stats.map(([big, small]) => h("div", null, h("b", null, big), h("span", null, small)))),
      h("div", { class: "split", role: "img", "aria-label": `${pct(1 - posShare)} negative, ${pct(posShare)} positive` },
        h("span", { class: "n", style: `width:${(1 - posShare) * 100}%` }),
        h("span", { class: "p", style: `width:${posShare * 100}%` })),
      h("div", { class: "split-labels" },
        h("span", { class: "n" }, `Negative ${pct(1 - posShare)}`),
        h("span", { class: "p" }, `Positive ${pct(posShare)}`)),
      h("h3", { class: "sub-heading" }, `First ${d.rows.length} reviews`),
      h("div", { class: "table-wrap" }, h("table", null,
        h("thead", null, h("tr", null,
          h("th", null, "Review"), h("th", null, "Verdict"), h("th", { class: "num" }, "Positive score"),
          d.accuracy != null ? h("th", null, "Your label") : null)),
        h("tbody", null, d.rows.map((r) => h("tr", null,
          h("td", { class: "review-cell" }, r.text),
          h("td", null, h("span", { class: `pill ${r.label}` }, r.label)),
          h("td", { class: "num" }, pct(r.positive)),
          d.accuracy != null ? h("td", null, r.actual || "–") : null))))));
    $("#batch-result").hidden = false;
  }

  /* ---------------------------------------------------------------- model */
  async function loadModel() {
    const box = $("#model-content");
    try {
      const response = await fetch("/api/model");
      const m = await response.json();
      if (!m.accuracy) throw new Error("No metrics found. Run python train_model.py first.");
      renderModel(box, m);
      modelLoaded = true;
    } catch (err) {
      fill(box, h("p", { class: "error" }, err.message));
    }
  }

  function scatterSVG(points) {
    const W = 520, H = 330, pad = 36;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - pad - 10);
    const sy = (y) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - pad - 10);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "scatter");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Scatter plot of 500 reviews in two LSA dimensions, coloured by sentiment");
    const add = (name, attrs, text) => {
      const e = document.createElementNS(ns, name);
      Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
      if (text) e.textContent = text;
      svg.append(e);
    };
    add("line", { x1: pad, y1: H - pad, x2: W - 10, y2: H - pad, stroke: "#c4ccd8" });
    add("line", { x1: pad, y1: 10, x2: pad, y2: H - pad, stroke: "#c4ccd8" });
    add("text", { x: W / 2, y: H - 8, "text-anchor": "middle" }, "LSA component 1");
    add("text", { x: 12, y: H / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${H / 2})` }, "LSA component 2");
    points.forEach((p) => add("circle", {
      cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r: 3,
      fill: p.label ? "#0b8a6f" : "#c42750", "fill-opacity": 0.5,
    }));
    return svg;
  }

  function renderModel(box, m) {
    const d = m.dataset;
    const [[tn, fp], [fn, tp]] = m.confusion_matrix;
    const total = tn + fp + fn + tp;
    const cell = (value, good, label) => h("div", {
      class: "cell",
      style: `background:rgba(${good ? "11,138,111" : "196,39,80"},${(0.08 + (value / total) * (good ? 0.9 : 3)).toFixed(2)})`,
    }, num(value), h("small", null, label));

    const chipList = (items, cls) => h("ul", { class: "chips" }, items.slice(0, 24).map((i) => h("li", { class: `chip ${cls}` }, i.term)));

    fill(box,
      h("div", { class: "model-hero" },
        h("div", { class: "big" }, pct(m.accuracy)),
        h("p", null, `of ${num(d.test)} reviews the model never saw during training were classified correctly. ${m.model_name}.`)),
      h("div", { class: "model-grid" },

        h("div", { class: "block" },
          h("h2", null, "Where it gets things wrong"),
          h("p", null, "Rows are the true sentiment, columns are what the model predicted."),
          h("div", { class: "matrix" },
            h("div"), h("div", { class: "h" }, "Predicted negative"), h("div", { class: "h" }, "Predicted positive"),
            h("div", { class: "h" }, "Actually negative"), cell(tn, true, "correct"), cell(fp, false, "wrongly called positive"),
            h("div", { class: "h" }, "Actually positive"), cell(fn, false, "wrongly called negative"), cell(tp, true, "correct"))),

        h("div", { class: "block" },
          h("h2", null, "Scores by class"),
          h("p", null, "Precision is how often a verdict is right. Recall is how many real cases were found."),
          h("div", { class: "table-wrap" }, h("table", null,
            h("thead", null, h("tr", null, h("th", null, "Class"), h("th", { class: "num" }, "Precision"), h("th", { class: "num" }, "Recall"), h("th", { class: "num" }, "F1"))),
            h("tbody", null, ["negative", "positive"].map((c) => h("tr", null,
              h("td", null, c),
              h("td", { class: "num" }, pct(m.per_class[c].precision)),
              h("td", { class: "num" }, pct(m.per_class[c]["recall"])),
              h("td", { class: "num" }, pct(m.per_class[c]["f1-score"])))))))),

        h("div", { class: "block wide" },
          h("h2", null, "Models compared"),
          h("p", null, "All models were trained on the same 80% of the data and scored on the same held-out 20%. Bars run from 50% (a coin flip) to 100%."),
          h("div", { class: "table-wrap" }, h("table", null,
            h("thead", null, h("tr", null, h("th", null, "Model"), h("th", null, "Accuracy"), h("th", null, ""), h("th", null, "Note"))),
            h("tbody", null, m.comparison.map((c) => h("tr", { class: c.note.startsWith("Deployed") ? "deployed" : c.note.includes("notebook") ? "notebook" : "" },
              h("td", null, c.name),
              h("td", null, pct(c.accuracy)),
              h("td", null, h("div", { class: "cmp-bar" }, h("i", { style: `width:${Math.max(2, (c.accuracy - 0.5) / 0.5 * 100)}%` }))),
              h("td", null, c.note))))))),

        h("div", { class: "block" },
          h("h2", null, "Words that signal positive"),
          h("p", null, "Highest weights learned from the training reviews."),
          chipList(m.top_positive, "pos")),
        h("div", { class: "block" },
          h("h2", null, "Words that signal negative"),
          h("p", null, "Lowest weights learned from the training reviews."),
          chipList(m.top_negative, "neg")),

        h("div", { class: "block" },
          h("h2", null, "Reviews in two dimensions"),
          h("p", null, `Latent Semantic Analysis squeezes each review into a point. Green is positive, red is negative. These two axes hold only ${pct(m.lsa_variance)} of the variation, which is why the groups overlap.`),
          scatterSVG(m.lsa_scatter)),

        h("div", { class: "block" },
          h("h2", null, "Training data"),
          h("p", null, "IMDB movie reviews, split 80/20 with the same share of positive and negative in each part."),
          h("dl", { class: "facts", style: "grid-template-columns:1fr" },
            [["Reviews used", num(d.total)], ["Training set", num(d.train)], ["Test set", num(d.test)],
             ["Positive / negative", `${num(d.positive)} / ${num(d.negative)}`], ["Features", num(d.features)], ["Trained", m.trained_at]]
              .map(([k, v]) => h("div", null, h("dt", null, k), h("dd", null, v)))))));
  }

  /* ----------------------------------------------------------------- boot */
  updateCount();
  showView(location.hash.slice(1) || "analyze");
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
})();
