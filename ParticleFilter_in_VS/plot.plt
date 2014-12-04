#角度のみをプロット
#plot "result.dat" u 1 w lines t "kalman(Angle)", "result.dat" u 2 w lines t "PF(MMSE)(Angle)"#,"result.dat" u 4 w lines t "PF(ML)","result.dat" u 3 w points t "Meas", 
#角速度のみをプロット
plot "result.dat" u 7 w lines t "true(Acc)","result.dat" u 8 w lines t "kalman(Acc)", "result.dat" u 9 w lines t "PF(MMSE)(Acc)"
pause -1 "press [Enter] key or [OK] button to quit"
#角度と角速度をプロット
# ----upper plot------
set multiplot
set lmargin at screen 0.20
set rmargin at screen 0.70
set bmargin at screen 0.50
set tmargin at screen 0.90
plot "result.dat" u 3 w points t "Meas","result.dat" u 1 w lines t "kalman(Angle)", "result.dat" u 2 w lines t "PF(MMSE)(Angle)"#,"result.dat" u 4 w lines t "PF(ML)",, 
# ----bottom plot -----
set lmargin at screen 0.20
set rmargin at screen 0.70
set bmargin at screen 0.10
set tmargin at screen 0.50
plot "result.dat" u 5 w lines t "kalman(Acc)", "result.dat" u 6 w lines t "PF(MMSE)(Acc)"

unset multiplot
reset
#pause -1 "press [Enter] key or [OK] button to quit"
