#ifndef EP_Viterbi_MAP_H_
#define EP_Viterbi_MAP_H_

#include "ParticleFilter.h"

class EPViterbiMat
{
public:
	EPViterbiMat(ParticleFilterMat &particle_filter);
	~EPViterbiMat();

	virtual void Initialization(
		ParticleFilterMat &particle_filter,
		void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
		double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
		const cv::Mat &observed);
	virtual void Recursion(
		ParticleFilterMat &particle_filter,
		void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
		void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
		double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
		const double &ctrl_input,
		const cv::Mat &observed);
	virtual cv::Mat GetEstimation();

protected:
	bool _is_inited;
	ParticleFilterMat last_particlefilter;
	std::vector<double> delta;
	std::vector<double> last_delta;
	int _it;
};

 

#endif //EP_Viterbi_MAP_H_
