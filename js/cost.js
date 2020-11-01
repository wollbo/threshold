
// defining global constants
var w = 345., h = 350., padding = 30.; // width height
var p = 0.5, lambda = 1.;

var tMin = -3;
var tMax = 5.;
var maxValue = 2.;

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

var roc_curve_line = d3.svg.line() // Use cost function
    .x(function (d) {
        return xScale(1. - cdf(d, mean1, var1));
    })
    .y(function (d) {
        return yScale(1. - cdf(d, mean2, var2));
    });

var roc_curve_path = svg
    .append("path")
    .attr("class", "line")
    .attr("stroke", "green");

var circle = svg // use X from min function, Y from cost function value
    .append("circle")
    .attr("cx", xScale(1 - cdf(threshold, mean1, var1)))
    .attr("cy", yScale(1 - cdf(threshold, mean2, var2)))
    .attr("r", 4);

function draw() {
    // parsing arguments
    mean1 = parseFloat($('#mean1').val());
    mean2 = parseFloat($('#mean2').val());
    var1 = parseFloat($('#var1').val());
    var2 = parseFloat($('#var2').val());


    var thresholds = [];
    var right_thresholds = [];

    tMin = Math.min(mean1 - 3 * Math.sqrt(var1), mean2 - 3 * Math.sqrt(var2));
    tMax = Math.max(mean1 + 3 * Math.sqrt(var1), mean2 + 3 * Math.sqrt(var2));
    var N = 1000;
    var delta = (tMax - tMin) / N;

    thresholds[0] = tMin;
    right_thresholds[0] = threshold;

    for (var i = 1; i <= N; i++) {
        thresholds[i] = thresholds[i - 1] + delta;
        right_thresholds[i] = right_thresholds[i - 1] + delta;
    }


    maxValue = Math.max(pdf(mean1, mean1, var1), pdf(mean2, mean2, var2)) * 1.05;

    roc_curve_path.attr("d", roc_curve_line(thresholds));

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

    var deviation1 = d3.svg.line()
        .x(function (d) {
            return xScaleDev(d);
        })
        .y(function (d) {
            return yScaleDev(pdf(d, mean1, var1));
        });

    var deviation2 = d3.svg.line()
        .x(function (d) {
            return xScaleDev(d);
        })
        .y(function (d) {
            return yScaleDev(pdf(d, mean2, var2));
        });


    pdf_path1.attr("d", deviation1(thresholds));

    pdf_path2.attr("d", deviation2(thresholds));

    var _area1 = d3.svg.area()
        .x(function (d) {
            return xScaleDev(d);
        })
        .y0(function (d) {
            return yScaleDev(d * 0);
        })
        .y1(function (d) {
            return yScaleDev(pdf(d, mean1, var1));
        });

    var _area2 = d3.svg.area()
        .x(function (d) {
            return xScaleDev(d);
        })
        .y0(function (d) {
            return yScaleDev(d * 0);
        })
        .y1(function (d) {
            return yScaleDev(pdf(d, mean2, var2));
        });

    pdf_area1.attr('d', _area1(right_thresholds));

    pdf_area2.attr('d', _area2(right_thresholds));

    thresholdLine
        .attr("x1", xScaleDev(threshold))
        .attr("y1", yScaleDev(0))
        .attr("x2", xScaleDev(threshold))
        .attr("y2", yScaleDev(maxValue));

    circle
        .attr("cx", xScale(1 - cdf(threshold, mean1, var1)))
        .attr("cy", yScale(1 - cdf(threshold, mean2, var2)));
}

draw();
