#ifndef EXTENDED_KALMAN_FILTER_H_
#define EXTENDED_KALMAN_FILTER_H_

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class EKFilter
{
 public :
  EKFilter(double (*state_eqn)(double, double), double (*state_diff_eqn)(double, double), double (*obs_eqn)(double), double (*obs_diff_eqn)(double), double x_hat0 = 0, double p0 = 0);
  //  virtual void SetObserved(double y);
  virtual void SetNoiseParam(double variance_s, double variance_w);
  virtual void Calc(double delta_t, double obs_y);
  virtual double GetEstimation();
  ~EKFilter();
 protected :
	 double c_k; // 線形近似（観測方程式を微分）

	 double g_k; // カルマンゲイン
	 double m_s_v; // システム雑音
	 double m_s_w; // 観測雑音
	 double x_bar; // 事前状態推定値
	 double x_hat; // 状態推定値
	 double a_k; // 線形近似（システム方程式を微分）
  
  double p_bar; // 事前誤差共分散行列
  double p;// 事後誤差共分散行列
  double m_y;
  double (*m_state_eqn)(double, double); // state equation function pointer
  double (*m_state_diff_eqn)(double, double); // differential state equation function pointer
  double (*m_obs_eqn)(double); // observation equation function pointer
  double (*m_obs_diff_eqn)(double); // differential observation equation function pointer
};

#endif // EXTENDED_KALMAN_FILTER_H_
