// AI Summary: Interactive bivariate distribution demo with marginal and conditional distributions.
// Renders joint distribution as contour plot with dynamic conditioning lines.
// Clean implementation with consistent coordinate systems throughout.

export function bivariateDemo(d3) {
  // Define the bivariate function (same as Python version)
  function f(x, y) {
    return -10 * Math.pow(y - x*x, 2) - y*y;
  }

  // Create grids
  const xgrid = d3.range(-2, 2.05, 0.05);
  const ygrid = d3.range(-2, 4.05, 0.05);
  const dx = xgrid[1] - xgrid[0];
  const dy = ygrid[1] - ygrid[0];

  // Calculate joint distribution values (Z matrix)
  // Important: d3.contours expects data in row-major order where rows go from top to bottom
  // Since SVG y-axis goes top-to-bottom but our mathematical y-axis goes bottom-to-top,
  // we need to reverse the y ordering
  const Z = ygrid.slice().reverse().map(y => xgrid.map(x => Math.exp(f(x, y))));

  // Calculate marginal distributions
  const marginal_x = xgrid.map((x, i) => {
    let sum = 0;
    for (let j = 0; j < ygrid.length; j++) {
      sum += Z[j][i];
    }
    return sum * dy;
  });

  const marginal_y = ygrid.map((y, j) => {
    // The Z matrix's rows are ordered by descending y-values (due to `ygrid.slice().reverse()`),
    // but we are building `marginal_y` to correspond to ascending y-values (matching `ygrid`).
    // To align them, we must access Z's rows in reverse order.
    const z_row_index = ygrid.length - 1 - j;
    let sum = 0;
    for (let i = 0; i < xgrid.length; i++) {
      sum += Z[z_row_index][i];
    }
    return sum * dx;
  });

  // Function to calculate conditional distributions
  function getConditionalDistributions(x_value, y_value) {
    // p(x|y)
    const conditional_x_given_y = xgrid.map(x => Math.exp(f(x, y_value)));
    const sum_x = conditional_x_given_y.reduce((a, b) => a + b, 0) * dx;
    const normalized_x_given_y = conditional_x_given_y.map(v => v / sum_x);

    // p(y|x)
    const conditional_y_given_x = ygrid.map(y => Math.exp(f(x_value, y)));
    const sum_y = conditional_y_given_x.reduce((a, b) => a + b, 0) * dy;
    const normalized_y_given_x = conditional_y_given_x.map(v => v / sum_y);

    return {
      x_given_y: normalized_x_given_y,
      y_given_x: normalized_y_given_x
    };
  }

  // Create visualization function
  function createVisualization(x_value, y_value, show_conditional_x, show_conditional_y) {
    const width = 640;
    const height = 600;
    const margin = {top: 20, right: 20, bottom: 40, left: 40};
    
    // Create SVG
    const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Define panel dimensions
    const panelWidth = (width - margin.left - margin.right) / 2 - 10;
    const panelHeight = (height - margin.top - margin.bottom) / 2 - 10;
    
    // Get conditional distributions
    const conditionals = getConditionalDistributions(x_value, y_value);
    
    // Top-left: Joint distribution contour plot
    const g1 = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    const xScale1 = d3.scaleLinear()
      .domain([-2, 2])
      .range([0, panelWidth]);
    
    const yScale1 = d3.scaleLinear()
      .domain([-2, 4])
      .range([panelHeight, 0]);
    
    // Create contour data for filled density plot
    const maxZ = d3.max(Z.flat());
    const minZ = d3.min(Z.flat());
    
    // Create more contour levels for smooth appearance
    const nLevels = 20;
    const thresholds = d3.range(nLevels).map(i => 
      minZ + (maxZ - minZ) * i / (nLevels - 1)
    );
    
    const contours = d3.contours()
      .size([xgrid.length, ygrid.length])
      .thresholds(thresholds)
      (Z.flat());
    
    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([minZ, maxZ]);
    
    // Draw filled contours
    // d3.contours generates paths in grid index space (0 to length-1)
    // We need to scale to panel coordinates
    const xScale = panelWidth / (xgrid.length - 1);
    const yScale = panelHeight / (ygrid.length - 1);
    
    g1.append("g")
      .selectAll("path")
      .data(contours)
      .join("path")
      .attr("d", d3.geoPath(d3.geoIdentity()
        .scale(Math.min(xScale, yScale))  // Use uniform scale
        .translate([0, 0])))
      .attr("transform", `scale(${xScale / Math.min(xScale, yScale)}, ${yScale / Math.min(xScale, yScale)})`)
      .attr("fill", d => colorScale(d.value))
      .attr("stroke", "none");

    // NEW: Calculate and draw the 99% Highest Density Region (HDR) contour
    const z_flat = Z.flat();
    const total_mass = z_flat.reduce((a, b) => a + b, 0);

    // Sort z values in descending order to find the HDR
    const sorted_z_values = [...z_flat].sort(d3.descending);
    
    let cumulative_mass = 0;
    let hdr_threshold = 0;
    for (const z_value of sorted_z_values) {
        cumulative_mass += z_value;
        if (cumulative_mass / total_mass >= 0.99) {
            hdr_threshold = z_value; // This is the lowest density value within the 99% HDR
            break;
        }
    }

    if (hdr_threshold > 0 & false) {
        const hdrContour = d3.contours()
            .size([xgrid.length, ygrid.length])
            .thresholds([hdr_threshold]) // Use the calculated HDR threshold
            (z_flat);

        g1.append("g")
            .selectAll("path")
            .data(hdrContour)
            .join("path")
            .attr("d", d3.geoPath(d3.geoIdentity()
                .scale(Math.min(xScale, yScale))
                .translate([0, 0])))
            .attr("transform", `scale(${xScale / Math.min(xScale, yScale)}, ${yScale / Math.min(xScale, yScale)})`)
            .attr("fill", "none")
            .attr("stroke", "#e6550d") // A bright, distinct orange
            .attr("stroke-width", 2);
    }

    // Add a subtle border
    g1.append("rect")
      .attr("width", panelWidth)
      .attr("height", panelHeight)
      .attr("fill", "none")
      .attr("stroke", "#ccc")
      .attr("stroke-width", 1);
    
    // Add axes
    g1.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(xScale1));
    
    g1.append("g")
      .call(d3.axisLeft(yScale1));
    
    g1.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -5)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text("Joint Distribution p(x,y)");
    
    // Add conditioning lines
    if (show_conditional_x) {
      g1.append("line")
        .attr("x1", xScale1(-2))
        .attr("x2", xScale1(2))
        .attr("y1", yScale1(y_value))
        .attr("y2", yScale1(y_value))
        .attr("stroke", "red")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5");
    }
    
    if (show_conditional_y) {
      g1.append("line")
        .attr("x1", xScale1(x_value))
        .attr("x2", xScale1(x_value))
        .attr("y1", yScale1(-2))
        .attr("y2", yScale1(4))
        .attr("stroke", "red")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5");
    }
    
    // Bottom-left: Marginal/Conditional p(x)
    const g2 = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top + panelHeight + 20})`);
    
    const xScale2 = d3.scaleLinear()
      .domain([-2, 2])
      .range([0, panelWidth]);
    
    const x_data = show_conditional_x ? conditionals.x_given_y : marginal_x;
    
    const yScale2 = d3.scaleLinear()
      .domain([0, d3.max(x_data) * 1.1])
      .range([panelHeight, 0]);
    
    const line = d3.line()
      .x((d, i) => xScale2(xgrid[i]))
      .y(d => yScale2(d));
    
    g2.append("path")
      .datum(x_data)
      .attr("fill", "none")
      .attr("stroke", show_conditional_x ? "red" : "steelblue")
      .attr("stroke-width", 2)
      .attr("d", line);
    
    // Fill area under curve
    const area = d3.area()
      .x((d, i) => xScale2(xgrid[i]))
      .y0(panelHeight)
      .y1(d => yScale2(d));
    
    g2.append("path")
      .datum(x_data)
      .attr("fill", show_conditional_x ? "rgba(255,0,0,0.1)" : "rgba(70,130,180,0.1)")
      .attr("d", area);
    
    g2.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(xScale2));
    
    g2.append("g")
      .call(d3.axisLeft(yScale2).ticks(5));
    
    g2.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -5)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text(show_conditional_x ? `p(x | y=${y_value.toFixed(1)})` : "Marginal p(x)");
    
    // Top-right: Marginal/Conditional p(y)
    const g3 = svg.append("g")
      .attr("transform", `translate(${margin.left + panelWidth + 20},${margin.top})`);
    
    const y_data = show_conditional_y ? conditionals.y_given_x : marginal_y;
    
    const xScale3 = d3.scaleLinear()
      .domain([0, d3.max(y_data) * 1.1])
      .range([0, panelWidth]);
    
    const yScale3 = d3.scaleLinear()
      .domain([-2, 4])
      .range([panelHeight, 0]);
    
    const lineY = d3.line()
      .x(d => xScale3(d))
      .y((d, i) => yScale3(ygrid[i]));
    
    g3.append("path")
      .datum(y_data)
      .attr("fill", "none")
      .attr("stroke", show_conditional_y ? "red" : "steelblue")
      .attr("stroke-width", 2)
      .attr("d", lineY);
    
    // Fill area
    const areaY = d3.area()
      .x0(0)
      .x1(d => xScale3(d))
      .y((d, i) => yScale3(ygrid[i]));
    
    g3.append("path")
      .datum(y_data)
      .attr("fill", show_conditional_y ? "rgba(255,0,0,0.1)" : "rgba(70,130,180,0.1)")
      .attr("d", areaY);
    
    g3.append("g")
      .attr("transform", `translate(0,${panelHeight})`)
      .call(d3.axisBottom(xScale3).ticks(5));
    
    g3.append("g")
      .call(d3.axisLeft(yScale3));
    
    g3.append("text")
      .attr("x", panelWidth / 2)
      .attr("y", -5)
      .attr("text-anchor", "middle")
      .style("font-weight", "bold")
      .text(show_conditional_y ? `p(y | x=${x_value.toFixed(1)})` : "Marginal p(y)");
    
    // Add labels
    g2.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (panelHeight / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Density");
    
    g2.append("text")
      .attr("transform", `translate(${panelWidth/2}, ${panelHeight + 35})`)
      .style("text-anchor", "middle")
      .text("x");
    
    g3.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (panelHeight / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("y");
    
    g3.append("text")
      .attr("transform", `translate(${panelWidth/2}, ${panelHeight + 35})`)
      .style("text-anchor", "middle")
      .text("Density");
    
    return svg.node();
  }

  return {
    createVisualization,
    marginal_x,
    marginal_y,
    xgrid,
    ygrid,
    Z
  };
}