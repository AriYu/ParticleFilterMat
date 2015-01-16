# ParticleFilterMat

```bash
$ git clone https://github.com/AriYu/ParticleFilterMat.git .
$ cd ParticleFilterMat
$ mkdir bin
$ mkdir -p graph/particles
$ mkdir -p graph/viterbi
$ cd bin
$ cmake ..
$ make
$ cd ..
$ ./testPFM*
```
## test_PFM12
非線形モデル

## plotparticles.plt
サンプリング後のパーティクルの位置と更新後の重みをプロット

## plotlastparticles.plt
サンプリング後のパーティクルの位置と更新前の重みをプロット

## plothistogram.plt
リサンプリング前とリサンプリング後のパーティクルのヒストグラムをプロット
クラスタリング後のパーティクルもプロット

## plotestimation.plt
各推定手法による推定結果をプロット

## ヒストグラムをプロット
```bash
$ filter(x,y)=int(x/y)*y
$ plot "result_particle.dat" ind 29 u (filter($1,0.1)):(1) smooth frequency with boxes, "result_particle.dat" ind 29 u 1:2
```

## 事前分布と事後分布のヒストグラムを同時にプロット
```bash
$ filter(x,y)=int(x/y)*y
$ plot "result_particle.dat" ind 77 u (filter($1,1)):(1) smooth frequency with boxes t "proposal","result_after_particle.dat" ind 77 u (filter($1,1)):(1) smooth frequency with boxes t "posteriori"
```

## 事前分布と事後分布のヒストグラムとクラスタリング結果を同時に表示
```bash
$ filter(x,y)=int(x/y)*y
$ plot "result_particle.dat" ind 30 u (filter($1,0.1)):(1) smooth frequency with boxes t "proposal","result_after_particle.dat" ind 30 u (filter($1,0.1)):(1) smooth frequency with boxes t "posteriori", "clustered_files/clustered_30.dat" ind 0 t "class 0","" ind 1 t "class 1","" ind 2 t "class 2", "" ind 3 t "class 3", "" ind 4 t "class 4"
```

## 推定結果をpdfに保存
```bash
$ gnuplot plotsave2pdf.plt
```

## 任意の回のパーティクルの分布をpdfに保存
中の`particle_ind = `のところを修正すること.
```bash
$ gnuplot plothistogram.plt
```

# TODO
- リサンプリング後のパーティクルのヒストグラムを表示する
- 事後分布をクラスタリングしてヒストグラムを表示する
- 事後分布が多峰性かどうかを判定する
- 多峰性の場合はクラスタリングを行って、パーティクルの多いクラスタの重み付き平均を取る
