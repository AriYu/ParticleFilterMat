#pause -1 "press [Enter] key or [OK] button to quit"
#reset
#set isosamples 50
#splot "result_particle.dat" pt 7 ps 0.5 w lines
#reset
#pause -1 "press [Enter] key or [OK] button to quit"

reset
cd "./graph/densities"
set xtics nomirror
set ytics nomirror

set terminal pdf


do for[j=0:101]{
    set output sprintf("densities-%d.pdf",j)
	plot "../../result_particle.dat" ind j u 1:2 pt 7 ps 0.3 t sprintf("weight %d",j),"../../result_particle.dat" ind j u 7:3 pt 7 ps 0.3 t sprintf("density %d",j),"../../result_particle.dat" ind j u 7:4 pt 7 ps 0.3 t sprintf ("maps %d",j),"../../result_particle.dat" ind j u 1:5 pt 7 ps 0.3 t sprintf("likelihoods %d",j) # ,"../../result_particle.dat" ind j u 1:6 pt 7 ps 0.3 t sprintf("last weight %d",j)
	set output
}


# set dgrid3d 30, 30
# set hidden3d
# set pm3d                           ## 3�����J���[�\��
# set pm3d map                       ## �J���[�}�b�v�\��
# set ticslevel 0
# #set cbrange[0:0.05]
# #set palette defined ( 0 "black", 1 "white")
# set nokey
# set tics font 'Times,14'
# set size square

# do for[j=0:99]{
# 	set pm3d interpolate 10, 10          ## ���
# 	set output sprintf("particles-%d.pdf",j)
# 	splot "../../result_particle.dat" ind j u 1:2:3 t sprintf("particles%d",j) with pm3d
# 	set output
# }


# reset

#x1�݂̂��v���b�g
#plot "../../result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "observed(x1)" ,"result1.dat" u 3 w lines t "PF[MMSE](x1)"
