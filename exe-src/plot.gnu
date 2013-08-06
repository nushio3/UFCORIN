set term png size 1800,1600
set out "test-wavelet.png"
set pm3d
set pm3d map
set xrange [0:1024]
set yrange [0:1024]
set cbrange [-10000:10000]
set pal gray
set size ratio -1
splot "./dist/bwd-S-bspl0-103-DS1.txt"
