#ifndef MEAN_SHIFT_CLUSTERING
#define MEAN_SHIFT_CLUSTERING

#include <iostream>
#include <vector>
#include <cmath>

#include "ParticleFilter.h"

class MeanShiftClustering
{
 public:
  MeanShiftClustering(std::vector< PStateMat >&points,
					  int dim,
					  double sigma = 0.1);
  ~MeanShiftClustering();
  int Clustering(std::vector<int> &indices, double threshold);
 private:
  PStateMat MeanShiftProcedure(PStateMat initX,
							   double threshold = 3e-5);
  double EuclideanDistance(PStateMat p1,
						   PStateMat p2, double h);
  std::vector< PStateMat > points_;
  int dim_;
  double sigma_;
  double num_points_;
};


#endif // MEAN_SHIFT_CLUSTERING
