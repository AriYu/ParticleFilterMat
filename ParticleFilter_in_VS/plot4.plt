#pause -1 "press [Enter] key or [OK] button to quit"
#reset
#set isosamples 50
#splot "result_particle.dat" pt 7 ps 0.5 w lines
#reset
#pause -1 "press [Enter] key or [OK] button to quit"

reset
cd "./graph/particles"
set terminal pdf
do for[j=0:249]{
	set output sprintf("particles-%d.pdf",j)
	plot "../../result_particle.dat" ind j u 1:2 smooth unique w linespoints pt 7 ps 1 t sprintf("particles%d",j)
	set output
}

# set dgrid3d 30, 30
# set hidden3d
# set pm3d                           ## 3次元カラー表示
# set pm3d map                       ## カラーマップ表示
# set ticslevel 0
# #set cbrange[0:0.05]
# #set palette defined ( 0 "black", 1 "white")
# set nokey
# set tics font 'Times,14'
# set size square

# do for[j=0:249]{
# 	set pm3d interpolate 10, 10          ## 補間
# 	set output sprintf("particles-%d.pdf",j)
# 	splot "../../result_particle.dat" ind j u 1:2:3 t sprintf("particles%d",j) with pm3d
# 	set output
# }


# reset

# cd "../viterbi"
# set terminal pdf
# do for[j=0:249]{
# 	set output sprintf("viterbi-%d.pdf",j)
# 	plot "../../epvgm.dat" ind j u 2:3 smooth unique w linespoints pt 7 ps 1 t sprintf("viterbi%d",j)
# 	set output
# }

#x1のみをプロット
#plot "../../result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "observed(x1)" ,"result1.dat" u 3 w lines t "PF[MMSE](x1)"
