#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class KalmanFilter
{
 public :
  KalmanFilter(double A, double b, double c, double x_hat0, double p0);
  //  virtual void SetObserved(double y);
  virtual void SetNoiseParam(double variance_s, double variance_w);
  virtual void Calc(double input, double obs_y);
  virtual double GetEstimation();
  ~KalmanFilter();
 protected :
  double m_A;
  double m_b;
  double m_c;
  double m_s_v; // システム雑音
  double m_s_w; // 観測雑音
  double x_bar; // 事前状態推定値
  double x_hat; // 状態推定値
  double g_k; // カルマンゲイン
  double p_bar; // 事前誤差共分散行列
  double p;// 事後誤差共分散行列
  double m_y;

};



#endif // EXTENDED_KALMAN_FILTER_H_
