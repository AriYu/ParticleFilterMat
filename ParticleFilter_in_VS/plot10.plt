##--------------------------------
## This script for test_PFM10, test_PFM11, test_PFM12
##--------------------------------
#xと観測値のみをプロット
plot "result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "first sensor(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#推定値、観測値とMMSEをプロット
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 3 w lines t "PF[MMSE](x1)"
pause -1 "press [Enter] key or [OK] button to quit"

# #推定値、観測値とMLをプロット
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 6 w lines lw 1.5 t "ML(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#推定値、観測値とEPVGMをプロット
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 4 w lines t "EPVGM(x1)"# ,"result1.dat" u 2 w lines t "first sensor(x1)"
pause -1 "press [Enter] key or [OK] button to quit"


#推定値、観測値とPFMAPをプロット
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 5 w lines t "PFMAP(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#推定値をプロット
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 3 w lines t "PF[MMSE](x1)","result1.dat" u 4 w lines t "EPVGM(x1)","result1.dat" u 5 w lines t "PFMAP(x1)"
pause -1 "press [Enter] key or [OK] button to quit"
reset