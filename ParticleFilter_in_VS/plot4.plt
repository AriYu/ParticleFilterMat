#pause -1 "press [Enter] key or [OK] button to quit"
#reset
#set isosamples 50
#splot "result_particle.dat" pt 7 ps 0.5 w lines
#reset
#pause -1 "press [Enter] key or [OK] button to quit"

reset
cd "./graph/particles"
set terminal pdf
do for[j=0:499]{
	set output sprintf("particles-%d.pdf",j)
	plot "../../result_particle.dat" ind j u 2:3 smooth unique w linespoints pt 7 ps 1 t sprintf("particles%d",j)
	set output
}

reset
#x1‚Ì‚İ‚ğƒvƒƒbƒg
#plot "../../result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "observed(x1)" ,"result1.dat" u 3 w lines t "PF[MMSE](x1)"
