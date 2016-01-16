<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<head>
<META http-equiv="Content-Style-Type" content="text/css">
<META http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Automated Solar Flare Forecast</title>
</head>

<body>

Realtime Solar X-ray Flux Forecast using Deep Learning
=====

<DIV align="center" style="font-size:200%;">
<img src='https://cloud.githubusercontent.com/assets/512367/7226747/632f7350-e786-11e4-85e7-9cd7c2423b3a.png' width='25%'>
</DIV>

We present the 24-hour forecast of GOES X-ray flux,
based on realtime GOES data and HMI-720s Near-Real-Time data.

The forecast is produced by regression of the time series
using Long-Short Temporal Memory (LSTM) neural network.

The feature vector is produced from (1) GOES X-ray flux and (2) wavelet analyses of
HMI images, as described in Muranushi et al (2015):
[http://arxiv.org/abs/1507.08011](http://arxiv.org/abs/1507.08011) .

The source code is available under MIT license at
[https://github.com/nushio3/UFCORIN/tree/master/script](https://github.com/nushio3/UFCORIN/tree/master/script) .

<DIV align="center" style="font-size:200%;">
<table style="font-size:x-large;"><tr>
<td>Largest flare in next 24 hours:</td><td>{{GOES_FLUX}} W/m²</td>
</tr><tr>
<td>Flare category forecast:</td>
<td>{{FLARE_CLASS}}</td>
</tr>
</table>
</DIV>




<img src='prediction-result.png' width='80%'>

The above figure is updated every 12 minutes.

The <font color='blue'>blue curve</font> is the observed data:
The <font color='green'>green curve</font> is 24-hour forecast.
The <font color='red'>red bars</font> are predicted maxima of next *n*-hours where *n* in [1, .., 24] .


Below is the comparison of the past predictions with the reality.

<img src='review-forecast.png' width='80%'>

Deep Learning Powered by 
<a href='http://chainer.org'><img src='https://raw.githubusercontent.com/pfnet/chainer/gh-pages/images/logo.png' height=32px></a>

</body>
