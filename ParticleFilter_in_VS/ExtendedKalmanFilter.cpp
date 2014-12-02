#include "ExtendedKalmanFilter.h"

EKFilter::EKFilter(double (*state_eqn)(double, double), double (*state_diff_eqn)(double, double), double (*obs_eqn)(double), double (*obs_diff_eqn)(double), double x_hat0, double p0)
{
    m_state_eqn		 = state_eqn;
    m_state_diff_eqn = state_diff_eqn;
    m_obs_eqn		 = obs_eqn;
    m_obs_diff_eqn	 = obs_diff_eqn;
    x_hat			 = x_hat0;
    p				 = p0;
}
EKFilter::~EKFilter()
{
}

void EKFilter::SetNoiseParam(double variance_s, double variance_w){
    m_s_v = variance_s;
    m_s_w = variance_w;
}
// void EKFilter::SetObserved(double y)
// {
//     m_y = y;
// }
void EKFilter::Calc(double delta_t, double obs_y)
{
    m_y = obs_y;
 
    x_bar = m_state_eqn(x_hat, delta_t); // x_hat(k-1)
    a_k = m_state_diff_eqn(x_hat, delta_t); // x_hat(k-1)
    c_k = m_obs_diff_eqn(x_bar); // x_bar(k)
    p_bar = pow(a_k, 2.0) * p + m_s_v;
    g_k = (p_bar * c_k)/(pow(c_k, 2.0)*p_bar + m_s_w);
    x_hat = x_bar + g_k * (m_y - m_obs_eqn(x_bar));
    p = (1-g_k*c_k)*p_bar;
	//	std::cout << "[p] : " << p << std::endl;
}
double EKFilter::GetEstimation()
{
    return x_hat;
}
