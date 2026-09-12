// AI Summary: Interactive Central Limit Theorem (CLT) demonstration.
// Simulates sample means from various distributions and visualizes their convergence
// to a normal distribution via histograms (PDF) and empirical CDFs.

export function cltDemo(d3) {
  // --- DISTRIBUTION DEFINITIONS ---
  // Helper for Normal(0,1) PDF and CDF
  const normal = {
    pdf: x => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x),
    // Abramowitz and Stegun approximation for erf(x)
    erf: x => {
        const sign = (x >= 0) ? 1 : -1;
        x = Math.abs(x);
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    },
    cdf: function(x) {
        return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
    }
  };

  // A custom skewed discrete distribution to show the CLT works for non-symmetric cases.
  const skewed_p1 = 0.7, skewed_p2 = 0.2; // P(X=10) is 0.1
  const skewed_mean = 0 * skewed_p1 + 1 * skewed_p2 + 10 * (1 - skewed_p1 - skewed_p2); // 1.2
  const skewed_mean_sq = 0 * skewed_p1 + 1 * skewed_p2 + 100 * (1 - skewed_p1 - skewed_p2); // 10.2
  const skewed_var = skewed_mean_sq - skewed_mean**2; // 8.76

  const distributions = {
    'Uniform': {
      mean: 0.5,
      std: 1 / Math.sqrt(12),
      random: () => Math.random()
    },
    'Exponential': {
      mean: 1,
      std: 1,
      random: () => -Math.log(1 - Math.random()) // Inverse transform for Exp(1)
    },
    'Bernoulli': {
      mean: 0.5,
      std: 0.5,
      random: () => Math.random() < 0.5 ? 1 : 0
    },
    'Skewed Discrete': {
      mean: skewed_mean,
      std: Math.sqrt(skewed_var),
      random: () => {
          const val = Math.random();
          if (val < skewed_p1) return 0;
          if (val < skewed_p1 + skewed_p2) return 1;
          return 10;
      }
    }
  };

  // --- VISUALIZATION FUNCTION ---
  function createVisualization(options) {
    const {
      distName = 'Uniform',
      sampleSize = 10,
      numSimulations = 2000
    } = options;

    const dist = distributions[distName];

    // 1. Run simulation
    const standardizedMeans = [];
    if (sampleSize > 0) {
        for (let i = 0; i < numSimulations; i++) {
            let sum = 0;
            for (let j = 0; j < sampleSize; j++) {
                sum += dist.random();
            }
            const sampleMean = sum / sampleSize;
            
            if (dist.std > 0) {
                const z = (sampleMean - dist.mean) / (dist.std / Math.sqrt(sampleSize));
                standardizedMeans.push(z);
            }
        }
    }

    // 2. Setup SVG and scales for side-by-side layout
    const width = 700;
    const height = 400;
    const margin = {top: 40, right: 30, bottom: 40, left: 50};
    const panelGap = 40;
    
    const panelWidth = (width - margin.left - margin.right - panelGap) / 2;
    const panelHeight = height - margin.top - margin.bottom;
    
    const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height);

    const xScale = d3.scaleLinear()
      .domain([-4, 4])
      .range([0, panelWidth]);
    
    // 3. PDF Panel (Left)
    const g1 = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
      
    const yScale1 = d3.scaleLinear()
      .range([panelHeight, 0]);

    const bins = d3.histogram()
      .domain(xScale.domain())
      .thresholds(xScale.ticks(40))
      (standardizedMeans);
      
    const maxDensity = d3.max(bins, d => d.length / numSimulations / (d.x1 - d.x0));
    yScale1.domain([0, Math.max(0.45, maxDensity * 1.15)]);

    g1.selectAll("rect")
      .data(bins)
      .join("rect")
        .attr("x", 1)
        .attr("transform", d => `translate(${xScale(d.x0)}, ${yScale1(d.length / numSimulations / (d.x1 - d.x0))})`)
        .attr("width", d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
        .attr("height", d => panelHeight - yScale1(d.length / numSimulations / (d.x1 - d.x0)))
        .style("fill", "skyblue");

    const line1 = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale1(d.y));

    g1.append("path")
      .datum(xScale.ticks(100).map(x => ({x: x, y: normal.pdf(x)})))
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 2)
      .attr("d", line1);

    g1.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(xScale));

    g1.append("g")
      .call(d3.axisLeft(yScale1).ticks(5));

    g1.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Distribution of Standardized Sample Mean (PDF)");

    // 4. CDF Panel (Right)
    const g2 = svg.append("g")
      .attr("transform", `translate(${margin.left + panelWidth + panelGap},${margin.top})`);

    const yScale2 = d3.scaleLinear()
      .domain([0, 1])
      .range([panelHeight, 0]);

    const sortedMeans = standardizedMeans.slice().sort(d3.ascending);
    const ecdfData = [];
    if (sortedMeans.length > 0) {
        ecdfData.push({x: sortedMeans[0], y: 0});
        for (let i = 0; i < sortedMeans.length; i++) {
          ecdfData.push({x: sortedMeans[i], y: i / sortedMeans.length});
          ecdfData.push({x: sortedMeans[i], y: (i + 1) / sortedMeans.length});
        }
    }

    const line2 = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale2(d.y));

    g2.append("path")
      .datum(ecdfData)
      .attr("fill", "none")
      .attr("stroke", "skyblue")
      .attr("stroke-width", 3)
      .attr("d", line2);
    
    const cdfLine = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale2(d.y));

    g2.append("path")
      .datum(xScale.ticks(100).map(x => ({x: x, y: normal.cdf(x)})))
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 2)
      .attr("d", cdfLine);

    g2.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(xScale));

    g2.append("g")
      .call(d3.axisLeft(yScale2));
      
    g2.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("CDF of Standardized Sample Mean");

    return svg.node();
  }

  return {
    createVisualization
  };
}
