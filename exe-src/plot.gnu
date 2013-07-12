set term png size 2560,1600
set out "test.png"
set pm3d
set pm3d map
set xrange [0:1024]
set yrange [0:1024]
set cbrange [-1000:1000]
set pal gray
set size ratio -1
splot "test.txt"
