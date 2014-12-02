#include "KalmanFilter.h"

KalmanFilter::KalmanFilter(double A, double b, double c, double x_hat0, double p0)
{
  m_A = A;
  m_b = b;
  m_c = c;
  x_hat			 = x_hat0;
  p				 = p0;
}
KalmanFilter::~KalmanFilter()
{
}

void KalmanFilter::SetNoiseParam(double variance_s, double variance_w){
    m_s_v = variance_s;
    m_s_w = variance_w;
}
// void KalmanFilter::SetObserved(double y)
// {
//     m_y = y;
// }
void KalmanFilter::Calc(double input, double obs_y)
{
    m_y = obs_y;

    x_bar = m_A * x_hat + m_b * input; // x_hat(k-1)
	p_bar = m_A * p * m_A + (m_s_v*m_s_v) * (m_b*m_b);
    g_k = (p_bar * m_c)/(m_c*p_bar*m_c + (m_s_w*m_s_w));
    x_hat = x_bar + g_k * (m_y - m_c*x_bar);
    p = (1-g_k*m_c)*p_bar;
	//	std::cout << "[p] : " << p << std::endl;
}
double KalmanFilter::GetEstimation()
{
    return x_hat;
}
