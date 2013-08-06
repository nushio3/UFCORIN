set term png size 1800,1600
set out "test-wavelet.png"
set pm3d
set pm3d map
set xrange [0:256]
set yrange [0:256]
set cbrange [-10000:10000]
set pal gray
set size ratio -1
splot "test-wavelet.txt"
