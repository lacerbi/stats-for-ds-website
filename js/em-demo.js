// AI Summary: Step-by-step EM demonstration for a one-dimensional two-component
// Gaussian mixture. A fixed dataset (n = 35, drawn once with a seeded PRNG) is
// shown as a jittered strip of points coloured by the responsibilities of the
// current iteration, under the two weighted component densities and the mixture
// density; a second panel traces the log-likelihood across iterations. The whole
// trajectory is precomputed from the two initial means, and an iteration slider
// picks the iteration on display. Exports the pure maths (PRNG, normal density,
// E-step, M-step, log-likelihood, trajectory, dataset) separately from
// emDemo(d3), which builds the SVG.

// ---------------------------------------------------------------------------
// Seeded pseudo-random numbers, so the dataset is the same on every page load
// ---------------------------------------------------------------------------

// mulberry32: a small 32-bit generator with a period of 2^32.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform; the second normal of the pair is discarded.
export function randomNormal(rng) {
  let u = rng();
  while (u <= 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// The model: a two-component Gaussian mixture on the line
// ---------------------------------------------------------------------------

export const mixtureConfig = {
  n: 35,
  seed: 750,
  weights: [0.5, 0.5],
  means: [0, 1],
  sds: [0.75, 0.15],
  // The initialisation the sliders start from: a deliberately poor one, with
  // both means between the two clusters and both components far too wide.
  initWeight: 0.5,
  initSds: [0.5, 0.5],
  maxIterations: 30,
  sigmaFloor: 0.02,
  xDomain: [-1.8, 1.8],
  yMax: 1.6
};

export function normalPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Parameters travel as {weight, means: [mu1, mu2], sds: [sigma1, sigma2]},
// where weight is the mixing weight pi of component 1.
export function mixtureDensity(x, params) {
  const { weight, means, sds } = params;
  return (
    weight * normalPdf(x, means[0], sds[0]) +
    (1 - weight) * normalPdf(x, means[1], sds[1])
  );
}

export function logLikelihood(data, params) {
  let total = 0;
  for (let i = 0; i < data.length; i++) total += Math.log(mixtureDensity(data[i], params));
  return total;
}

// E-step: the responsibility of component 2 for each observation. The
// responsibility of component 1 is one minus this.
export function eStep(data, params) {
  const { weight, means, sds } = params;
  const r = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const p1 = weight * normalPdf(data[i], means[0], sds[0]);
    const p2 = (1 - weight) * normalPdf(data[i], means[1], sds[1]);
    r[i] = p2 / (p1 + p2);
  }
  return r;
}

// M-step: weighted mean and weighted variance, with the responsibilities as
// weights.
export function mStep(data, r) {
  const n = data.length;
  const weights = [0, 0];
  const means = [0, 0];
  const sds = [0, 0];
  for (let k = 0; k < 2; k++) {
    let sumR = 0;
    let sumRy = 0;
    for (let i = 0; i < n; i++) {
      const rik = k === 0 ? 1 - r[i] : r[i];
      sumR += rik;
      sumRy += rik * data[i];
    }
    const mu = sumRy / sumR;
    let sumRd = 0;
    for (let i = 0; i < n; i++) {
      const rik = k === 0 ? 1 - r[i] : r[i];
      sumRd += rik * (data[i] - mu) * (data[i] - mu);
    }
    weights[k] = sumR / n;
    means[k] = mu;
    // A component that shrinks onto a single point sends the likelihood to
    // infinity, so the standard deviations are held above a floor.
    sds[k] = Math.max(Math.sqrt(sumRd / sumR), mixtureConfig.sigmaFloor);
  }
  return { weight: weights[0], means, sds };
}

// Iteration t carries the parameters theta^(t), the responsibilities the E-step
// computes from them, and the log-likelihood at theta^(t). Iteration 0 is the
// initialisation, and each further entry is one full EM iteration.
export function emTrajectory(data, init, nIterations) {
  const steps = [];
  let params = init;
  for (let t = 0; t <= nIterations; t++) {
    const responsibilities = eStep(data, params);
    steps.push({ params, responsibilities, logLik: logLikelihood(data, params) });
    params = mStep(data, responsibilities);
  }
  return steps;
}

let datasetCache = null;

// The one dataset the demo is built around, together with the vertical jitter
// that spreads the points out for display.
export function dataset() {
  if (datasetCache === null) {
    const cfg = mixtureConfig;
    const rng = mulberry32(cfg.seed);
    const values = [];
    const labels = [];
    for (let i = 0; i < cfg.n; i++) {
      const k = rng() < cfg.weights[0] ? 0 : 1;
      labels.push(k);
      values.push(cfg.means[k] + cfg.sds[k] * randomNormal(rng));
    }
    const jitter = values.map(() => 0.04 + 0.2 * rng());
    datasetCache = { values, labels, jitter };
  }
  return datasetCache;
}

export function initialParams(mu1, mu2) {
  return {
    weight: mixtureConfig.initWeight,
    means: [mu1, mu2],
    sds: [mixtureConfig.initSds[0], mixtureConfig.initSds[1]]
  };
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

export function emDemo(d3) {
  const width = 700;
  const height = 480;
  const margin = { top: 40, right: 20, bottom: 45, left: 55 };
  const panelGap = 55;
  const readoutHeight = 78;
  const panelWidth = (width - margin.left - margin.right - panelGap) / 2;
  const panelHeight = height - margin.top - margin.bottom - readoutHeight;

  // Okabe-Ito blue and vermillion: distinguishable for the common forms of
  // colour blindness.
  const color1 = "#0072b2";
  const color2 = "#d55e00";
  const mixtureColor = "#888";
  const currentColor = "crimson";

  const fmt3 = d3.format(".3g");

  function axisTitles(g, title, xLabel, yLabel) {
    g.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -18)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text(title);
    g.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", panelHeight + 36)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text(xLabel);
    g.append("text")
      .attr("transform", `translate(-42,${panelHeight / 2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text(yLabel);
  }

  function curvePath(xScale, yScale, f) {
    const [lo, hi] = mixtureConfig.xDomain;
    // A very narrow component runs far off the top of the panel; the curve is
    // capped well above it and then clipped.
    const cap = 10 * mixtureConfig.yMax;
    const points = [];
    for (let i = 0; i <= 400; i++) {
      const x = lo + ((hi - lo) * i) / 400;
      const y = f(x);
      points.push({ x, y: isFinite(y) ? Math.min(y, cap) : cap });
    }
    return d3
      .line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))(points);
  }

  const c1 = d3.rgb(color1);
  const c2 = d3.rgb(color2);

  // The colour of a point with responsibility r for component 2.
  function blend(r) {
    return d3.rgb(
      (1 - r) * c1.r + r * c2.r,
      (1 - r) * c1.g + r * c2.g,
      (1 - r) * c1.b + r * c2.b
    );
  }

  function createVisualization(options) {
    const { mu1 = 0.3, mu2 = 0.7, iteration = 0 } = options;

    const cfg = mixtureConfig;
    const { values, jitter } = dataset();
    const steps = emTrajectory(values, initialParams(mu1, mu2), cfg.maxIterations);
    const t = Math.max(0, Math.min(cfg.maxIterations, Math.round(iteration)));
    const step = steps[t];
    const { weight, means, sds } = step.params;

    const svg = d3.create("svg").attr("width", width).attr("height", height);

    // Unique ids so several copies of the demo can coexist on one page.
    const uid = "em-" + Math.random().toString(36).slice(2, 9);

    // The trace panel is clipped with a small margin, so that a marker sitting on
    // the first or the last iteration is not cut in half.
    svg
      .append("defs")
      .selectAll("clipPath")
      .data([{ id: "data", pad: 0 }, { id: "trace", pad: 8 }])
      .join("clipPath")
      .attr("id", d => `${uid}-${d.id}`)
      .append("rect")
      .attr("x", d => -d.pad)
      .attr("y", d => -d.pad)
      .attr("width", d => panelWidth + 2 * d.pad)
      .attr("height", d => panelHeight + 2 * d.pad);

    // --- Left panel: the data and the current components -------------------
    const g1 = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain(cfg.xDomain).range([0, panelWidth]);
    const yScale = d3.scaleLinear().domain([0, cfg.yMax]).range([panelHeight, 0]);

    const inner1 = g1.append("g").attr("clip-path", `url(#${uid}-data)`);

    inner1
      .append("path")
      .attr("d", curvePath(xScale, yScale, x => mixtureDensity(x, step.params)))
      .attr("fill", "none")
      .attr("stroke", mixtureColor)
      .attr("stroke-width", 1.2);

    inner1
      .append("path")
      .attr("d", curvePath(xScale, yScale, x => weight * normalPdf(x, means[0], sds[0])))
      .attr("fill", "none")
      .attr("stroke", color1)
      .attr("stroke-width", 2);

    inner1
      .append("path")
      .attr("d", curvePath(xScale, yScale, x => (1 - weight) * normalPdf(x, means[1], sds[1])))
      .attr("fill", "none")
      .attr("stroke", color2)
      .attr("stroke-width", 2);

    inner1
      .append("g")
      .selectAll("circle")
      .data(values)
      .join("circle")
      .attr("cx", d => xScale(d))
      .attr("cy", (d, i) => yScale(jitter[i]))
      .attr("r", 4.5)
      .attr("fill", (d, i) => blend(step.responsibilities[i]))
      .attr("stroke", "white")
      .attr("stroke-width", 0.7);

    g1.append("g").attr("transform", `translate(0,${panelHeight})`).call(d3.axisBottom(xScale));
    g1.append("g").call(d3.axisLeft(yScale).ticks(5));
    axisTitles(g1, "Data and current components", "y", "Density");

    const legend = [
      { label: "component 1", color: color1 },
      { label: "component 2", color: color2 },
      { label: "mixture", color: mixtureColor }
    ];
    legend.forEach((item, i) => {
      g1.append("text")
        .attr("x", 4)
        .attr("y", 14 + i * 16)
        .style("font-size", "12px")
        .style("fill", item.color)
        .style("font-weight", "bold")
        .text(item.label);
    });

    // --- Right panel: the log-likelihood trajectory ------------------------
    const g2 = svg
      .append("g")
      .attr("transform", `translate(${margin.left + panelWidth + panelGap},${margin.top})`);

    const trace = steps.map((s, i) => ({ t: i, ll: s.logLik }));
    const llMin = d3.min(trace, d => d.ll);
    const llMax = d3.max(trace, d => d.ll);
    const pad = Math.max((llMax - llMin) * 0.08, 0.5);

    const tScale = d3.scaleLinear().domain([0, cfg.maxIterations]).range([0, panelWidth]);
    const llScale = d3
      .scaleLinear()
      .domain([llMin - pad, llMax + pad])
      .range([panelHeight, 0]);

    const inner2 = g2.append("g").attr("clip-path", `url(#${uid}-trace)`);

    inner2
      .append("path")
      .datum(trace)
      .attr("fill", "none")
      .attr("stroke", "#444")
      .attr("stroke-width", 1.8)
      .attr(
        "d",
        d3
          .line()
          .x(d => tScale(d.t))
          .y(d => llScale(d.ll))
      );

    inner2
      .selectAll("circle")
      .data(trace)
      .join("circle")
      .attr("cx", d => tScale(d.t))
      .attr("cy", d => llScale(d.ll))
      .attr("r", 2.5)
      .attr("fill", "#444");

    inner2
      .append("circle")
      .attr("cx", tScale(t))
      .attr("cy", llScale(step.logLik))
      .attr("r", 6)
      .attr("fill", currentColor)
      .attr("stroke", "white")
      .attr("stroke-width", 1.5);

    g2.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(tScale).ticks(6));
    g2.append("g").call(d3.axisLeft(llScale).ticks(5));
    axisTitles(g2, "Log-likelihood by iteration", "Iteration", "ℓ(θ)");

    // --- Readout -----------------------------------------------------------
    const readout = svg
      .append("g")
      .attr(
        "transform",
        `translate(${margin.left},${margin.top + panelHeight + margin.bottom + 16})`
      );

    const lines = [
      `iteration ${t} of ${cfg.maxIterations}      log-likelihood = ${fmt3(step.logLik)}`,
      `component 1:   π = ${fmt3(weight)}    μ₁ = ${fmt3(means[0])}    σ₁ = ${fmt3(sds[0])}`,
      `component 2:   1 − π = ${fmt3(1 - weight)}    μ₂ = ${fmt3(means[1])}    σ₂ = ${fmt3(sds[1])}`
    ];

    readout
      .selectAll("text")
      .data(lines)
      .join("text")
      .attr("x", 0)
      .attr("y", (d, i) => i * 19)
      .attr("xml:space", "preserve")
      .style("font-family", "monospace")
      .style("font-size", "12.5px")
      .style("fill", (d, i) => (i === 1 ? color1 : i === 2 ? color2 : "#333"))
      .text(d => d);

    return svg.node();
  }

  return {
    createVisualization
  };
}
