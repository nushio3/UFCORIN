set term postscript enhanced color solid 15
set pm3d
set pm3d map
set palette define (-1 "red", 0 "white",1  "blue")
set cbrange [-1:1]

set xrange [0:64]
set yrange [0:64]
set size ratio -1

set xtics (0,32,64)
set ytics (0,32,64)

set out 'demosun-real.eps'
splot 'demosun-real.txt' t ''

set xtics (0,2,4,8,16,32,64)
set ytics (0,2,4,8,16,32,64)

set out 'demosun-WS.eps'
splot 'demosun-WS.txt' t ''

set out 'demosun-WN.eps'
splot 'demosun-WN.txt' t ''


set xtics (0,32,64)
set ytics (0,32,64)

set out 'tp-S-real.eps'
splot 'tp-S-real.txt' t ''

set xtics (0,2,4,8,16,32,64)
set ytics (0,2,4,8,16,32,64)

set out 'tp-S-WS.eps'
splot 'tp-S-WS.txt' t ''

set out 'tp-S-WN.eps'
splot 'tp-S-WN.txt' t ''


set xtics (0,32,64)
set ytics (0,32,64)

set out 'tp-N-real.eps'
splot 'tp-N-real.txt' t ''

set xtics (0,2,4,8,16,32,64)
set ytics (0,2,4,8,16,32,64)

set out 'tp-N-WS.eps'
splot 'tp-N-WS.txt' t ''

set out 'tp-N-WN.eps'
splot 'tp-N-WN.txt' t ''

set xtics (0,32,64)
set ytics (0,32,64)

set out 'bs-N-real.eps'
splot 'bs-N-real.txt' t ''

set xtics (0,2,4,8,16,32,64)
set ytics (0,2,4,8,16,32,64)

set out 'bs-N-WS.eps'
splot 'bs-N-WS.txt' t ''

set out 'bs-N-WN.eps'
splot 'bs-N-WN.txt' t ''
