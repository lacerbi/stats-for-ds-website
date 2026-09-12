// AI Summary: Interactive "moment matching game" for the Method of Moments.
// A fixed dataset (Gamma or Beta, drawn once with a seeded PRNG) is shown as a
// density histogram beside a "moment space" panel whose axes are the mean and the
// variance: the reader moves two parameter sliders until the model's moments land
// on the sample moments, and a checkbox reveals the closed-form MoM solution.
// Exports the pure maths (log-gamma, densities, moment formulas, MoM solver,
// dataset generation) separately from momDemo(d3), which builds the SVG.

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

// Marsaglia and Tsang (2000), with the usual boost for shape < 1.
export function randomGamma(rng, shape, scale) {
  if (shape < 1) {
    const u = rng();
    return randomGamma(rng, shape + 1, scale) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = randomNormal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}

// If X ~ Gamma(a, 1) and Y ~ Gamma(b, 1) then X / (X + Y) ~ Beta(a, b).
export function randomBeta(rng, a, b) {
  const x = randomGamma(rng, a, 1);
  const y = randomGamma(rng, b, 1);
  return x / (x + y);
}

// ---------------------------------------------------------------------------
// Densities (computed in log space)
// ---------------------------------------------------------------------------

const LANCZOS_G = 7;
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
];

// Natural logarithm of the gamma function (Lanczos approximation).
export function logGamma(x) {
  if (x < 0.5) {
    // Reflection formula: Gamma(x) Gamma(1 - x) = pi / sin(pi x).
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_C[0];
  const t = z + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_C.length; i++) a += LANCZOS_C[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

// Gamma density in the shape-scale parameterisation used in the chapter.
export function gammaPdf(x, shape, scale) {
  if (!(x > 0)) return 0;
  const logp =
    (shape - 1) * Math.log(x) - x / scale - logGamma(shape) - shape * Math.log(scale);
  return Math.exp(logp);
}

export function betaPdf(x, a, b) {
  if (!(x > 0) || !(x < 1)) return 0;
  const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

// ---------------------------------------------------------------------------
// Distributions, moments and the Method of Moments solution
// ---------------------------------------------------------------------------

// The plotting ranges below assume the sliders of the chapter: shape in
// [0.5, 8] and scale in [0.5, 5] for the Gamma, a and b in [0.5, 10] for the
// Beta. Every (mean, variance) pair those sliders can reach lies inside
// momentDomain.
export const distributions = {
  Gamma: {
    paramNames: ["α", "β"],
    paramLabels: ["shape α", "scale β"],
    trueParams: [3, 2],
    n: 100,
    seed: 20260065,
    xDomain: [0, 25],
    yMax: 0.3,
    momentScale: "log",
    momentDomain: { mean: [0.2, 45], variance: [0.1, 220] },
    momentTicks: { mean: [0.2, 0.5, 1, 2, 5, 10, 20, 40], variance: [0.1, 0.3, 1, 3, 10, 30, 100] },
    histogramBins: 25,
    pdf: (x, p1, p2) => gammaPdf(x, p1, p2),
    sample: (rng, p1, p2) => randomGamma(rng, p1, p2),
    moments: (p1, p2) => ({ mean: p1 * p2, variance: p1 * p2 * p2 }),
    // beta = variance / mean, alpha = mean / beta.
    mom: (m, v) => {
      const scale = v / m;
      return [m / scale, scale];
    }
  },
  Beta: {
    paramNames: ["a", "b"],
    paramLabels: ["a", "b"],
    trueParams: [2, 5],
    n: 100,
    seed: 20260016,
    xDomain: [0, 1],
    yMax: 4,
    momentScale: "linear",
    momentDomain: { mean: [0, 1], variance: [0, 0.26] },
    histogramBins: 20,
    pdf: (x, p1, p2) => betaPdf(x, p1, p2),
    sample: (rng, p1, p2) => randomBeta(rng, p1, p2),
    moments: (p1, p2) => ({
      mean: p1 / (p1 + p2),
      variance: (p1 * p2) / ((p1 + p2) * (p1 + p2) * (p1 + p2 + 1))
    }),
    // c = mean (1 - mean) / variance - 1, a = mean c, b = (1 - mean) c.
    mom: (m, v) => {
      const c = (m * (1 - m)) / v - 1;
      return [m * c, (1 - m) * c];
    }
  }
};

export function modelMoments(distName, p1, p2) {
  return distributions[distName].moments(p1, p2);
}

// Sample mean and the 1/n sample variance, which is what the moment equations
// of the chapter produce.
export function sampleMoments(data) {
  const n = data.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += data[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (data[i] - mean) * (data[i] - mean);
  variance /= n;
  return { mean, variance };
}

export function momEstimate(distName, mean, variance) {
  return distributions[distName].mom(mean, variance);
}

const datasetCache = new Map();

// The one dataset per distribution that the demo is built around.
export function dataset(distName) {
  if (!datasetCache.has(distName)) {
    const cfg = distributions[distName];
    const rng = mulberry32(cfg.seed);
    const values = [];
    for (let i = 0; i < cfg.n; i++) {
      values.push(cfg.sample(rng, cfg.trueParams[0], cfg.trueParams[1]));
    }
    datasetCache.set(distName, values);
  }
  return datasetCache.get(distName);
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

export function momDemo(d3) {
  const width = 700;
  const height = 465;
  const margin = { top: 40, right: 20, bottom: 45, left: 55 };
  const panelGap = 55;
  const readoutHeight = 62;
  const panelWidth = (width - margin.left - margin.right - panelGap) / 2;
  const panelHeight = height - margin.top - margin.bottom - readoutHeight;

  const dataColor = "#1f6f9c";
  const modelColor = "crimson";
  const momColor = "#2ca02c";
  const matchTolerance = 0.03; // relative, on both the mean and the variance

  const fmt3 = d3.format(".3g");
  const fmt2 = d3.format(".2f");

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

  function densityPath(cfg, xScale, yScale, p1, p2) {
    const [lo, hi] = cfg.xDomain;
    const eps = (hi - lo) / 1000;
    const points = [];
    for (let i = 0; i <= 400; i++) {
      const x = lo + eps + ((hi - lo - 2 * eps) * i) / 400;
      let y = cfg.pdf(x, p1, p2);
      if (!isFinite(y)) y = 10 * cfg.yMax;
      points.push({ x, y: Math.min(y, 10 * cfg.yMax) });
    }
    return d3
      .line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))(points);
  }

  function createVisualization(options) {
    const {
      distName = "Gamma",
      param1 = 1.5,
      param2 = 1,
      showSolution = false
    } = options;

    const cfg = distributions[distName];
    const data = dataset(distName);
    const sample = sampleMoments(data);
    const model = cfg.moments(param1, param2);
    const momParams = cfg.mom(sample.mean, sample.variance);
    const momValid = momParams.every(p => isFinite(p) && p > 0);

    const matched =
      Math.abs(model.mean - sample.mean) / sample.mean < matchTolerance &&
      Math.abs(model.variance - sample.variance) / sample.variance < matchTolerance;

    const svg = d3.create("svg").attr("width", width).attr("height", height);

    // Unique ids so several copies of the demo can coexist on one page.
    const uid = "mom-" + Math.random().toString(36).slice(2, 9);

    svg
      .append("defs")
      .selectAll("clipPath")
      .data(["density", "moment"])
      .join("clipPath")
      .attr("id", d => `${uid}-${d}`)
      .append("rect")
      .attr("width", panelWidth)
      .attr("height", panelHeight);

    // --- Left panel: the data and the densities ---------------------------
    const g1 = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain(cfg.xDomain).range([0, panelWidth]);
    const yScale = d3.scaleLinear().domain([0, cfg.yMax]).range([panelHeight, 0]);

    // The last threshold coincides with the end of the domain, which leaves an
    // empty bin of zero width; drop it so the density is always well defined.
    const bins = d3
      .histogram()
      .domain(cfg.xDomain)
      .thresholds(xScale.ticks(cfg.histogramBins))(data)
      .filter(b => b.x1 > b.x0);
    const binDensity = b => b.length / data.length / (b.x1 - b.x0);

    g1.append("g")
      .attr("clip-path", `url(#${uid}-density)`)
      .selectAll("rect")
      .data(bins)
      .join("rect")
      .attr("x", d => xScale(d.x0) + 1)
      .attr("y", d => yScale(binDensity(d)))
      .attr("width", d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
      .attr("height", d => panelHeight - yScale(binDensity(d)))
      .style("fill", "skyblue");

    const curves = g1.append("g").attr("clip-path", `url(#${uid}-density)`);

    curves
      .append("path")
      .attr("d", densityPath(cfg, xScale, yScale, param1, param2))
      .attr("fill", "none")
      .attr("stroke", modelColor)
      .attr("stroke-width", 2);

    if (showSolution && momValid) {
      curves
        .append("path")
        .attr("d", densityPath(cfg, xScale, yScale, momParams[0], momParams[1]))
        .attr("fill", "none")
        .attr("stroke", momColor)
        .attr("stroke-width", 2.5)
        .attr("stroke-dasharray", "6,4");
    }

    g1.append("g").attr("transform", `translate(0,${panelHeight})`).call(d3.axisBottom(xScale));
    g1.append("g").call(d3.axisLeft(yScale).ticks(5));
    axisTitles(g1, `Data (n = ${data.length}) and model`, "x", "Density");

    const legend = [
      { label: "data", color: "skyblue" },
      { label: "model", color: modelColor }
    ];
    if (showSolution && momValid) legend.push({ label: "MoM fit", color: momColor });
    legend.forEach((item, i) => {
      g1.append("text")
        .attr("x", panelWidth - 4)
        .attr("y", 14 + i * 16)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", item.color)
        .style("font-weight", "bold")
        .text(item.label);
    });

    // --- Right panel: moment space ----------------------------------------
    const g2 = svg
      .append("g")
      .attr("transform", `translate(${margin.left + panelWidth + panelGap},${margin.top})`);

    const isLog = cfg.momentScale === "log";
    const mScale = (isLog ? d3.scaleLog() : d3.scaleLinear())
      .domain(cfg.momentDomain.mean)
      .range([0, panelWidth]);
    const vScale = (isLog ? d3.scaleLog() : d3.scaleLinear())
      .domain(cfg.momentDomain.variance)
      .range([panelHeight, 0]);

    const inner = g2.append("g").attr("clip-path", `url(#${uid}-moment)`);

    if (distName === "Beta") {
      // Every Beta distribution has variance below mean (1 - mean).
      const curve = [];
      for (let i = 0; i <= 100; i++) {
        const m = i / 100;
        curve.push({ m, v: m * (1 - m) });
      }
      inner
        .append("path")
        .datum(curve)
        .attr("fill", "#dde7f0")
        .attr(
          "d",
          d3
            .area()
            .x(d => mScale(d.m))
            .y0(vScale(0))
            .y1(d => vScale(d.v))
        );
      inner
        .append("path")
        .datum(curve)
        .attr("fill", "none")
        .attr("stroke", "#9db4c8")
        .attr(
          "d",
          d3
            .line()
            .x(d => mScale(d.m))
            .y(d => vScale(d.v))
        );
      inner
        .append("text")
        .attr("x", mScale(0.5))
        .attr("y", vScale(0.25) + 16)
        .attr("text-anchor", "middle")
        .style("font-size", "11px")
        .style("fill", "#62798e")
        .text("every Beta lies below this curve");
    }

    inner
      .append("line")
      .attr("x1", mScale(sample.mean))
      .attr("y1", vScale(sample.variance))
      .attr("x2", mScale(model.mean))
      .attr("y2", vScale(model.variance))
      .attr("stroke", "#999")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3,3");

    inner
      .append("circle")
      .attr("cx", mScale(sample.mean))
      .attr("cy", vScale(sample.variance))
      .attr("r", 6)
      .attr("fill", dataColor);
    inner
      .append("text")
      .attr("x", mScale(sample.mean) + 10)
      .attr("y", vScale(sample.variance) + 4)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .style("fill", dataColor)
      .text("data");

    if (matched) {
      inner
        .append("circle")
        .attr("cx", mScale(model.mean))
        .attr("cy", vScale(model.variance))
        .attr("r", 12)
        .attr("fill", "none")
        .attr("stroke", momColor)
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);
    }
    inner
      .append("circle")
      .attr("cx", mScale(model.mean))
      .attr("cy", vScale(model.variance))
      .attr("r", 6)
      .attr("fill", matched ? momColor : modelColor)
      .attr("stroke", "white")
      .attr("stroke-width", 1.5);
    inner
      .append("text")
      .attr("x", mScale(model.mean) + 10)
      .attr("y", vScale(model.variance) - 8)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .style("fill", matched ? momColor : modelColor)
      .text("model");

    const mAxis = d3.axisBottom(mScale);
    const vAxis = d3.axisLeft(vScale);
    if (cfg.momentTicks) {
      mAxis.tickValues(cfg.momentTicks.mean).tickFormat(d3.format("~g"));
      vAxis.tickValues(cfg.momentTicks.variance).tickFormat(d3.format("~g"));
    } else {
      mAxis.ticks(5);
      vAxis.ticks(5);
    }
    g2.append("g").attr("transform", `translate(0,${panelHeight})`).call(mAxis);
    g2.append("g").call(vAxis);
    axisTitles(
      g2,
      "Moment space",
      isLog ? "Mean (log scale)" : "Mean",
      isLog ? "Variance (log scale)" : "Variance"
    );

    if (matched) {
      g2.append("text")
        .attr("x", panelWidth / 2)
        .attr("y", -3)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .style("fill", momColor)
        .text("moments matched");
    }

    // --- Readout -----------------------------------------------------------
    const readout = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top + panelHeight + margin.bottom + 20})`);

    const lines = [
      `data:   mean = ${fmt3(sample.mean)}   variance = ${fmt3(sample.variance)}`,
      `model:  mean = ${fmt3(model.mean)}   variance = ${fmt3(model.variance)}` +
        `   (${cfg.paramNames[0]} = ${fmt2(param1)}, ${cfg.paramNames[1]} = ${fmt2(param2)})`
    ];
    if (showSolution && momValid) {
      lines.push(
        `MoM solution: ${cfg.paramNames[0]} = ${fmt3(momParams[0])}, ` +
          `${cfg.paramNames[1]} = ${fmt3(momParams[1])}` +
          `   |   data generated with ${cfg.paramNames[0]} = ${cfg.trueParams[0]}, ` +
          `${cfg.paramNames[1]} = ${cfg.trueParams[1]}`
      );
    }

    readout
      .selectAll("text")
      .data(lines)
      .join("text")
      .attr("x", 0)
      .attr("y", (d, i) => i * 19)
      .attr("xml:space", "preserve")
      .style("font-family", "monospace")
      .style("font-size", "12.5px")
      .style("fill", (d, i) => (i === 2 ? momColor : "#333"))
      .text(d => d);

    return svg.node();
  }

  return {
    createVisualization
  };
}
