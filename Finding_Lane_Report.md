# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

### Reflection

### 1. Pipeline

My pipeline consisted of 9 steps:
(1) Make grayscale image
(2) Execute edge detection using canny process
(3) Set region wrapping entire left-right lanes
(4) Use Hough transformation to detect lines 
(5) Separate left and right side lines based on their slope
(6) Calculate average slope and y-intercept value of left lines and right lines
(7) Extrapolate new line based on average slope and y-intercept values
(8) Draw two lines calculated before as highway lanes
(9) Combine lines image with original image


### 2. Potential Shortcomings
Potential shortcoming from my algorithms are:
(1) If the curvature of roadway turning has small R, 
    there is possibilty that the slopes of right and left side line 
    will be both negative or positive, so that the separation will not be accurated 
    with current algorithm. 
(2) When the dashed lane is almost cannot be recognized by perception process 
    as a line (Hough lines), then average lines cannot be calculated accurately.

### 3. Suggest possible improvements to your pipeline

Improvement for pipeline
(a) Improvement for reducing impact of Potential shortcoming(1)
    Use not only slope  to separate left and right lane, 
    but also point coordinates of the lines to separate them.
(b) Improvement for reducing impact of Potential shortcoming(2)
    We know that lane parameters (slope, intercept) will not be change dramatically 
    within 2 or 3 frames of video. So in order to calculate average lines with smaller error,
    we have to compare current average line with average lines calculated in 2     
    until 3 frames before and make decision wether it's better to use current calculation
    result, or the result in previous frames, or make average of them. 