Realtime Solar X-ray Flux Forecast using Deep-LSTM Network
=====

We present the 24-hour forecast of GOES X-ray flux,
based on realtime GOES data and HMI-720s Near-Real-Time data.

The forecast is produced by regression of the time series
using Long-Short Temporal Memory (LSTM) neural network.

The feature vector is produced from (1) GOES X-ray flux and (2) wavelet analyses of
HMI images, as described in Muranushi et al (2015):
[http://arxiv.org/abs/1507.08011](http://arxiv.org/abs/1507.08011) .

The source code is available under MIT license at
[https://github.com/nushio3/UFCORIN/tree/master/script](https://github.com/nushio3/UFCORIN/tree/master/script) .



<img src='prediction-result.png' width='80%'>

The above figure is updated every 12 minutes.

The blue curve is the observed data:
The green curve is 24-hour forecast.
The red bars are predicted maxima of next *n*-hours where *n* in [1, .., 24] .


Deep Learning Powered by [Chainer](http://chainer.org)
