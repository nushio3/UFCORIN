set term postscript enhanced color 25
set out 'demo.eps'
set grid
#   set xrange [10000:12000]
#
 set xrange [3850:4050]
#set xrange [4850:5050]
#set xrange [0:17544]
#set xtics 0,2400

# f(-5.2) -> -4
# f(-5.45) -> -5
# f(-5.8) -> -6

xthre= -5.2
mthre= -5.45
cthre= -5.8

#f(x) = x>mthre? (x-mthre)/(xthre-mthre)-5 :  -(x-mthre)/(cthre-mthre)-5
f(x)=x

set yrange [-7.5:-2.5]
set xlabel 'hours since 2011/01/01 00:00'
set ylabel 'Solar X-ray flux [W m^{-2}]'
set format y '10^{%1.0l}'
set ytics -7,1
plot \
     './forecast-features/forecast-goes-0.txt' u 2:(log($5)/log(10)) w l lw 4 lt 1 lc rgb 'green'  t 'observation', \
     './forecast-features/forecast-goes-24.txt' u 2:(log($5)/log(10)) w l lw 4 lt 1 lc rgb 'blue'  t '24h future max'  ,\
     './survey3/haarC-2-S-0016-0032-0001-0001-regres.txt' u 1:(f($2)) w p pt 6 ps 1 lc rgb 'red' t 'prediction'
