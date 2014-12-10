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
  test_PFM test_PFM.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM ${OpenCV_LIBS} )

add_executable( 
  test_PFM2 test_PFM2.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM2 ${OpenCV_LIBS} )

# add_executable( 
#   test_PFM3 test_PFM3.cpp 
#   ParticleFilter.cpp
#   EPViterbiMAP.cpp
#   pfMapMat.cpp
#   RootMeanSquareError.cpp
#   )
# target_link_libraries( test_PFM3 ${OpenCV_LIBS} )

# add_executable( 
#   test_PFM4 test_PFM4.cpp 
#   ParticleFilter.cpp
#   EPViterbiMAP.cpp
#   pfMapMat.cpp
#   RootMeanSquareError.cpp
#   )
# target_link_libraries( test_PFM4 ${OpenCV_LIBS} )

# add_executable( 
#   test_PFM5 test_PFM5.cpp 
#   ParticleFilter.cpp
#   EPViterbiMAP.cpp
#   pfMapMat.cpp
#   RootMeanSquareError.cpp
#   )
# target_link_libraries( test_PFM5 ${OpenCV_LIBS} )

add_executable( 
  test_PFM7 test_PFM7.cpp 
  ParticleFilter.cpp
  EPViterbiMAP.cpp
  pfMapMat.cpp
  RootMeanSquareError.cpp
  )
target_link_libraries( test_PFM7 ${OpenCV_LIBS} )




