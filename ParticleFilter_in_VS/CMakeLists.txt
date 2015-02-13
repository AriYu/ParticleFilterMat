cmake_minimum_required(VERSION 2.8)
project( particle_filter_mat )
find_package( OpenCV REQUIRED )


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
ADD_DEFINITIONS(
  -g
  -O0
  -Wall
  -std=c++11
  )

add_executable( 
  test_PFM9 test_PFM9.cpp 
  ParticleFilter.cpp
  mean_shift_clustering.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM9 ${OpenCV_LIBS} )

add_executable( 
  test_PFM10 test_PFM10.cpp 
  mean_shift_clustering.cpp
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM10 ${OpenCV_LIBS} )
add_executable( 
  test_PFM11 test_PFM11.cpp
  mean_shift_clustering.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM11 ${OpenCV_LIBS} )
add_executable( 
  test_PFM12 test_PFM12.cpp
  mean_shift_clustering.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  unscented_kalman_filter.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM12 ${OpenCV_LIBS} )

add_executable( 
  test_PFM13 test_PFM13.cpp
  mean_shift_clustering.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  unscented_kalman_filter.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM13 ${OpenCV_LIBS} )
add_executable( 
  test_PFM14 test_PFM14.cpp
  mean_shift_clustering.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  unscented_kalman_filter.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM14 ${OpenCV_LIBS} )

add_executable( 
  test_density test_density.cpp
  mean_shift_clustering.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  EPViterbiMAPAlpha.cpp
  pfMapMat.cpp
  unscented_kalman_filter.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_density ${OpenCV_LIBS} )
