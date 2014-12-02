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

void EPViterbiMat::Initialization(ParticleFilterMat &particle_filter, cv::Mat &observed)
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
			cv::Mat error
				= last_particlefilter._C.t() * particle_filter.filtered_particles[i]._state;
			error = error - last_particlefilter._C.t() * observed;
			for (int ii = 0; ii < error.rows; ii++){
				for (int jj = 0; jj < error.cols; jj++){
					d = pow(error.at<double>(ii, jj), 2);
					g_yx += -d / (2.0*last_particlefilter._ObsNoiseCov.at<double>(ii, jj))
						- log(sqrt(2.0*CV_PI*last_particlefilter._ObsNoiseCov.at<double>(ii, jj)));
				}
			}
		}
		//=============================================
		// calc f(x_1[i])
		//=============================================
		{
			cv::Mat error
				= particle_filter.filtered_particles[i]._state - 0;
			for (int ii = 0; ii < error.rows; ii++){
				for (int jj = 0; jj < error.cols; jj++){
					d = pow(error.at<double>(ii, jj), 2);
					f_xx += -d / (2.0*last_particlefilter._ProcessNoiseCov.at<double>(ii, jj))
						- log(sqrt(2.0*CV_PI*last_particlefilter._ProcessNoiseCov.at<double>(ii, jj)));
				}
			}
		}

		last_delta[i] = f_xx + g_yx;
		last_particlefilter.predict_particles[i]
			= particle_filter.predict_particles[i];
		last_particlefilter.filtered_particles[i]
			= particle_filter.filtered_particles[i];

	}
	_is_inited = true;
}

void EPViterbiMat::Recursion(ParticleFilterMat &particle_filter, cv::Mat &observed, double &input)
{
	double g_yx = 0;
	double f_xx = 0;
	double d = 0;
	double max = 0;
	double tmp = 0;

	if (!_is_inited){
		Initialization(particle_filter, observed);
	}
	else
	{
		//cout << "[Recursion]" << endl;
		for (int i = 0; i < particle_filter._samples; i++){
			
			g_yx = 0;
			{
				cv::Mat error
					= last_particlefilter._C.t() * particle_filter.filtered_particles[i]._state;
				error = error - last_particlefilter._C.t() * observed;
				for (int ii = 0; ii < error.rows; ii++){
					for (int jj = 0; jj < error.cols; jj++){
						d = pow(error.at<double>(ii, jj), 2);
						g_yx += -d / (2.0*last_particlefilter._ObsNoiseCov.at<double>(ii, jj))
							- log(sqrt(2.0*CV_PI*last_particlefilter._ObsNoiseCov.at<double>(ii, jj)));
					}
				}
			}
			for (int j = 0; j < particle_filter._samples; j++){
	
				f_xx = 0;
				{
					cv::Mat error1
						= (last_particlefilter._A*last_particlefilter.filtered_particles[i]._state)
						+ last_particlefilter._B*input;
					error1 = error1 - particle_filter.filtered_particles[i]._state; 
					for (int ii = 0; ii < error1.rows; ii++){
						for (int jj = 0; jj < error1.cols; jj++){
							d = pow(error1.at<double>(ii, jj), 2);
							f_xx += -d / (2.0*last_particlefilter._ProcessNoiseCov.at<double>(ii, jj))
								- log(sqrt(2.0*CV_PI*last_particlefilter._ProcessNoiseCov.at<double>(ii, jj)));
						}
					}
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
			max = delta[i];
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

