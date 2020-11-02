
// defining global constants
var w = 345., h = 350., padding = 30.; // width height
var p = 0.5, lambda = 1.;
var threshold = 1.; // 0 or 1 ?

var tMin = 0;
var tMax = 1;
var maxValue = 5.;

function cost(x, p, lambda) {
    return -p * Math.log(x) -lambda * (1-p) * Math.log(1-x);
}

var xScaleDev = d3.scale.linear()
    .domain([tMin, tMax])
    .range([0., w]);

var yScaleDev = d3.scale.linear()
    .domain([0, maxValue])
    .range([h, 0]);

var xScale = d3.scale.linear()
    .domain([0, 1])
    .range([0., w]);

var yScale = d3.scale.linear()
    .domain([0, 1])
    .range([h, 0]);

var div_left = d3.select("#renderer").append("div");

div_left.attr("class", "block");

var svg = div_left
    .append("svg")
    .attr("width", w + 2 * padding)
    .attr("height", h + 2 * padding);

svg = svg.append('g')
    .attr("transform", "translate(" + padding + "," + padding + ")")

// Define X axis
var xAxis = d3.svg.axis()
    .scale(xScale)
    .orient("bottom")
    .ticks(5);

// Define Y axis
var yAxis = d3.svg.axis()
    .scale(yScale)
    .orient("left")
    .ticks(5);

//Create X axis
svg.append("g")
    .attr("class", "axis")
    .attr("transform", "translate(" + 0 + "," + h + ")")
    .call(xAxis);

//Create Y axis
svg.append("g")
    .attr("class", "axis")
    .attr("transform", "translate(" + 0 + ",0)")
    .call(yAxis);

var cost_function_line = d3.svg.line() // Use cost function
    .x(function (d) {
        return xScale(d);
    })
    .y(function (d) {
        return yScale(cost(d, p, lambda));
    });

var cost_function_path = svg
    .append("path")
    .attr("class", "line")
    .attr("stroke", "green");

var circle = svg // use X from min function, Y from cost function value
    .append("circle")
    .attr("cx", xScale(threshold))
    .attr("cy", yScale(cost(threshold, p, lambda)))
    .attr("r", 4);

function draw() {
    // parsing arguments
    p = parseFloat($('#p').val());
    lambda = parseFloat($('#lambda').val());


    var thresholds = [];

    tMin = 0.0001 // cost function undefined on endpoints
    tMax = 0.9999
    var N = 1000;
    var delta = (tMax - tMin) / N;
    var argMin = p / (p + lambda * (1-p));
    thresholds[0] = tMin;

    for (var i = 1; i <= N; i++) {
        thresholds[i] = thresholds[i - 1] + delta;
    }


    maxValue = Math.max(cost(tMin, p, lambda), cost(tMax, p, lambda)) * 1.05; // changed to scale cost function
    var minValue = Math.min(cost(thresholds, p, lambda)); // can replace with cost(argMin, p, lambda)

    cost_function_path.attr("d", cost_function_line(thresholds)); // changed to cost function

    xScaleDev.domain([tMin, tMax]);

    yScaleDev.domain([0, maxValue]);

    //Define X axis
    var xAxisDev = d3.svg.axis()
        .orient("bottom")
        .scale(xScaleDev)
        .ticks(5);

    //Define Y axis
    var yAxisDev = d3.svg.axis()
        .orient("left")
        .scale(yScaleDev)
        .ticks(5);

    xAxDev.call(xAxisDev);
    yAxDev.call(yAxisDev);

    thresholdLine
        .attr("x1", xScaleDev(threshold))
        .attr("y1", yScaleDev(0))
        .attr("x2", xScaleDev(threshold))
        .attr("y2", yScaleDev(maxValue));

    circle
        .attr("cx", xScale(argMin))
        .attr("cy", yScale(minValue));
}

draw();
