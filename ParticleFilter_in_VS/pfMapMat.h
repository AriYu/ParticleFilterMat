#ifndef PFMAPMAT_H_
#define PFMAPMAT_H_

#include "ParticleFilter.h"

class pfMapMat
{
 public:
  pfMapMat(ParticleFilterMat &particle_filter);
  ~pfMapMat();

  virtual void Initialization(
			      ParticleFilterMat &particle_filter);
  virtual void Update(
		      ParticleFilterMat &particle_filter,
		      void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
		      void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
		      double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
		      double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean),
		      const double &ctrl_input,
		      const cv::Mat &observed);
  virtual cv::Mat GetEstimation();

 public:
  ParticleFilterMat last_particlefilter;
  std::vector<double> map;
  std::vector<double> p_xx_vec;
  std::vector<double> p_yx_vec;
};

#endif // PFMAPMAP_H_
