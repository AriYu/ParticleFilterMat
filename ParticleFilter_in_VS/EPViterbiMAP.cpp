#include "EPViterbiMAP.h"
#include <algorithm>
using namespace std;


EPViterbiMat::EPViterbiMat(ParticleFilterMat &particle_filter)
	: last_particlefilter(particle_filter), _is_inited(false)
{
	this->delta.resize(particle_filter._samples);
	this->last_delta.resize(particle_filter._samples);
}

EPViterbiMat::~EPViterbiMat()
{
}

void EPViterbiMat::Initialization(
	ParticleFilterMat &particle_filter,
	void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
	double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
	const cv::Mat &observed)
{
	double g_yx = 0;
	double f_xx = 0;
	double d = 0;

	for (int i = 0; i < particle_filter._samples; i++){

		d = 0;
		g_yx = 0;
		f_xx = 0;

		//=============================================
		// calc g(y_1 | x_1[i])
		//=============================================
		{
			cv::Mat obshat = observed.clone();
			cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

			obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
			g_yx = log(likelihood(observed, obshat, particle_filter._ObsNoiseCov, particle_filter._ObsNoiseMean));
			
		}
		//=============================================q
		// calc f(x_1[i])
		//=============================================
		{
			cv::Mat est_state
				= particle_filter.filtered_particles[i]._state;
			cv::Mat last_state = cv::Mat::zeros(est_state.rows, est_state.cols, CV_64F);
			f_xx = log(likelihood(est_state, last_state, particle_filter._ProcessNoiseCov, particle_filter._ObsNoiseMean));
		}
		delta[i] = f_xx + g_yx;
		last_delta[i] = 0.0;
		last_particlefilter.predict_particles[i]
			= particle_filter.predict_particles[i];
		last_particlefilter.filtered_particles[i]
			= particle_filter.filtered_particles[i];

	}
	_is_inited = true;
}

void EPViterbiMat::Recursion(
	ParticleFilterMat &particle_filter,
	void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
	void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
	double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),	
	const double &ctrl_input,
	const cv::Mat &observed)
{
	double g_yx = 0;
	double f_xx = 0;
	double d = 0;
	double max = 0;
	double tmp = 0;

	if (!_is_inited){
		Initialization(particle_filter, obsmodel, likelihood, observed);
	}
	else
	{
		//cout << "[Recursion]" << endl;
		for (int i = 0; i < particle_filter._samples; i++){

			g_yx = 0;
			{
				cv::Mat obshat = observed.clone();
				cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

				obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
				g_yx = log(likelihood(observed, obshat, particle_filter._ObsNoiseCov, particle_filter._ObsNoiseMean));
				//g_yx = log(particle_filter.filtered_particles[i]._weight);
			}
			for (int j = 0; j < particle_filter._samples; j++){

				f_xx = 0;
				{
					cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
					cv::Mat est_state = particle_filter.filtered_particles[i]._state.clone();
					processmodel(est_state, last_particlefilter.filtered_particles[j]._state, ctrl_input, rnd_num);
					f_xx = log(
						likelihood(est_state,
						particle_filter.filtered_particles[i]._state,
						particle_filter._ProcessNoiseCov,
						particle_filter._ProcessNoiseMean));
				}
				if (j == 0){
					max = last_delta[j] + f_xx;
					delta[i] = g_yx + max;
				}
				else{
					tmp = last_delta[j] + f_xx;
					if (tmp > max){
						max = tmp;
						delta[i] = g_yx + max;
					}
				}
			}
			last_delta[i] = delta[i];
			last_particlefilter.predict_particles[i]
				= particle_filter.predict_particles[i];
			last_particlefilter.filtered_particles[i]
				= particle_filter.filtered_particles[i];
		}
	}
}

cv::Mat EPViterbiMat::GetEstimation()
{
	double max = 0;
	double tmp = 0;

	for (int i = 0; i < last_particlefilter._samples; i++){
		if (i == 0){
			max = delta[0];
			_it = i;
		}
		else{
			if (delta[i] > max){
				max = delta[i];
				_it = i;
			}
		}
	}
	return last_particlefilter.filtered_particles[_it]._state;
}

