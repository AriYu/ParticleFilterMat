#ifndef EP_Viterbi_MAP_H_
#define EP_Viterbi_MAP_H_
#include <iostream>
#include "ParticleFilter.h"

class EPViterbiMat
{
public:
	EPViterbiMat(ParticleFilterMat &particle_filter);
	~EPViterbiMat();

	virtual void Initialization(
		ParticleFilterMat &particle_filter,
		void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
		double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
		double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean),
		const cv::Mat &observed);
	virtual void Recursion(
		ParticleFilterMat &particle_filter,
		void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
		void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
		double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
		double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean),
		const double &ctrl_input,
		const cv::Mat &observed);
	virtual cv::Mat GetEstimation();

protected:
	ParticleFilterMat last_particlefilter;
	bool is_inited_;
	std::vector<double> delta;
	std::vector<double> last_delta;
	std::vector<double> g_yx_vec;
	std::vector<double> f_xx_vec;
	std::vector<double> last_g_yx_vec;
	std::vector<double> max;
	int it_;
	std::ofstream epvgm_output;
};

 

#endif //EP_Viterbi_MAP_H_
