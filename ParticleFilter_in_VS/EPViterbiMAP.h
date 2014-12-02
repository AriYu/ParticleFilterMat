#ifndef EP_Viterbi_MAP_H_
#define EP_Viterbi_MAP_H_

#include "ParticleFilter.h"

class EPViterbiMat
{
public:
	EPViterbiMat(ParticleFilterMat &particle_filter);
	~EPViterbiMat();

	virtual void Initialization(ParticleFilterMat &particle_filter, cv::Mat &observed);
	virtual void Recursion(ParticleFilterMat &particle_filter, cv::Mat &observed, double &input);
	virtual cv::Mat GetEstimation();

protected:
	bool _is_inited;
	ParticleFilterMat last_particlefilter;
	std::vector<double> delta;
	std::vector<double> last_delta;
	int _it;
};

 

#endif //EP_Viterbi_MAP_H_
