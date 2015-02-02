#ifndef UNSCENTED_KALMAN_FILTER_H_
#define UNSCENTED_KALMAN_FILTER_H_

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <new>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class UnscentedKalmanFilter
{
 public:
  UnscentedKalmanFilter(int dimX, int dimY, cv::Mat &x0, cv::Mat &p0);
  UnscentedKalmanFilter(const UnscentedKalmanFilter& x);
  ~UnscentedKalmanFilter();

  virtual void SetProcessNoise(cv::Mat &Cov);
  virtual void SetObservationNoise(cv::Mat &Cov);
  virtual void Update(void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, 
										  const double &input,
										  const cv::Mat &processrnd),
					  void(*obsmodel)(cv::Mat &z, const cv::Mat &x,const cv::Mat &obsrnd),
					  cv::Mat &observation);
  virtual cv::Mat GetEstimation();
 private:
  virtual void  Cholesky(cv::Mat &P, cv::Mat &S);

  int dimX_;
  int dimY_;
  cv::Mat ProcessNoiseCov_;
  cv::Mat ObsNoiseCov_;
  cv::Mat xhat_; // 更新前の状態推定値 xhat(k-1)
  cv::Mat xhat_new_; // 更新後の状態推定値 xhat(k)
  cv::Mat P_; // 更新前の共分散行列 P(k-1)
  cv::Mat P_new_; // 更新後の共分散行列 P(k)
  cv::Mat G_; // カルマンゲイン
};



#endif
