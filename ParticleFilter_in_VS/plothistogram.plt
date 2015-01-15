cd "./graph/histogram"
set terminal pdf

filter(x,y)=int(x/y)*y

do for[i=0:99]{
particle_ind=i

set output sprintf("cluster_%d.pdf", particle_ind)
plot "../../result_particle.dat" ind particle_ind u (filter($1,0.1)):(1) smooth frequency with boxes t "prior","../../result_after_particle.dat" ind particle_ind u (filter($1,0.1)):(1) smooth frequency with boxes t "posteriori", sprintf("../../clustered_files/clustered_%d.dat",particle_ind) ind 0 t "class 0","" ind 1 t "class 1","" ind 2 t "class 2", "" ind 3 t "class 3", "" ind 4 t "class 4"
}