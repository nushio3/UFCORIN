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

set out 'tp-S-real.eps'
splot 'tp-S-real.txt' t ''

set out 'tp-N-real.eps'
splot 'tp-N-real.txt' t ''

set out 'bs-N-real.eps'
splot 'bs-N-real.txt' t ''


set xtics (0,2,4,8,16,32,64)
set ytics (0,2,4,8,16,32,64)

set style arrow 1 filled nohead front linewidth 1 linecolor rgb "green"
set arrow  from  0,32 to 64,32  arrowstyle 1
set arrow  from 32, 0 to 32,64  arrowstyle 1
set arrow  from  0,16 to 32,16  arrowstyle 1
set arrow  from 16, 0 to 16,32  arrowstyle 1
set arrow  from  0, 8 to 16, 8  arrowstyle 1
set arrow  from  8, 0 to  8,16  arrowstyle 1
set arrow  from  0, 4 to  8, 4  arrowstyle 1
set arrow  from  4, 0 to  4, 8  arrowstyle 1
set arrow  from  0, 2 to  4, 2  arrowstyle 1
set arrow  from  2, 0 to  2, 4  arrowstyle 1
set arrow  from  0, 1 to  2, 1  arrowstyle 1
set arrow  from  1, 0 to  1, 2  arrowstyle 1


set out 'tp-N-WN.eps'
splot 'tp-N-WN.txt' t ''

set out 'bs-N-WN.eps'
splot 'bs-N-WN.txt' t ''



set out 'demosun-WN.eps'
splot 'demosun-WN.txt' t ''


set out 'tp-S-WN.eps'
splot 'tp-S-WN.txt' t ''



set arrow  from  0,32 to 64,32  arrowstyle 1
set arrow  from 32, 0 to 32,64  arrowstyle 1
set arrow  from  0,16 to 64,16  arrowstyle 1
set arrow  from 16, 0 to 16,64  arrowstyle 1
set arrow  from  0, 8 to 64, 8  arrowstyle 1
set arrow  from  8, 0 to  8,64  arrowstyle 1
set arrow  from  0, 4 to 64, 4  arrowstyle 1
set arrow  from  4, 0 to  4,64  arrowstyle 1
set arrow  from  0, 2 to 64, 2  arrowstyle 1
set arrow  from  2, 0 to  2,64  arrowstyle 1
set arrow  from  0, 1 to 64, 1  arrowstyle 1
set arrow  from  1, 0 to  1,64  arrowstyle 1



set out 'demosun-WS.eps'
splot 'demosun-WS.txt' t ''




set out 'tp-S-WS.eps'
splot 'tp-S-WS.txt' t ''





set out 'tp-N-WS.eps'
splot 'tp-N-WS.txt' t ''



set out 'bs-N-WS.eps'
splot 'bs-N-WS.txt' t ''
