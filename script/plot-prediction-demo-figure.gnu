set term postscript enhanced color 25
set out 'demo.eps'
set grid
#   set xrange [10000:12000]
#
set xrange [3800:4200]
# set xrange [0:17544]
# set xtics 0,5000


set yrange [-7.5:-2.5]
set xlabel 'hours since 2011/01/01 00:00'
set ylabel 'Solar X-ray flux [W m^{-2}]'
set format y '10^{%1.0l}'
set ytics -7,1
plot \
     './forecast-features/forecast-goes-0.txt' u 2:(log($5)/log(10)) w l lt 1 lc rgb 'green' lw 2 t 'observation', \
     './forecast-features/forecast-goes-24.txt' u 2:(log($5)/log(10)) w l lt 1 lc rgb 'blue' lw 2 t '24h forecast', \
     './survey3/haarC-2-S-0016-0032-0001-0001-regres.txt' w p pt 6 lc rgb 'red' t 'prediction'
