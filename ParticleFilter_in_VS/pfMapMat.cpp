#include "pfMapMat.h"

pfMapMat::pfMapMat(ParticleFilterMat &particle_filter)
	: last_particlefilter(particle_filter)
{
}

pfMapMat::~pfMapMat()
{
}

void pfMapMat::Initialization(
	ParticleFilterMat &particle_filter)
{
	map.resize(particle_filter._samples);
        p_xx_vec.resize(particle_filter._samples);
        p_yx_vec.resize(particle_filter._samples);
}


void pfMapMat::Update(
					  ParticleFilterMat &particle_filter,
					  void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
					  void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
					  double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
					  double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean),
					  const double &ctrl_input,
					  const cv::Mat &observed)
{
  double sum = 0.0;
  for(int i = 0; i < particle_filter._samples; i++){
	cv::Mat obshat = observed.clone();
	cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

	obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
	p_yx_vec[i] = obs_likelihood(observed, obshat, 
								 particle_filter._ObsNoiseCov, 
								 particle_filter._ObsNoiseMean);
	sum = logsumexp(sum, p_yx_vec[i], (i==0));
  }
  for(int i = 0; i < particle_filter._samples; i++){
	p_yx_vec[i] = p_yx_vec[i] - sum;
  }

  for (int i = 0; i < particle_filter._samples; i++){
	map[i] = 0.0;
	sum = 0.0;
	for(int j = 0; j < particle_filter._samples; j++){
	  cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
	  cv::Mat est_state = particle_filter.filtered_particles[j]._state.clone();
	  processmodel(est_state, last_particlefilter.filtered_particles[j]._state, 
				   ctrl_input, rnd_num);
	  p_xx_vec[j] = trans_likelihood(est_state,
									 particle_filter.filtered_particles[i]._state,
									 particle_filter._ProcessNoiseCov,
									 particle_filter._ProcessNoiseMean);
	  sum = logsumexp(sum, p_xx_vec[j], (j == 0));
	}
	for(int j = 0; j < particle_filter._samples; j++){
	  p_xx_vec[j] = p_xx_vec[j] - sum;
	  //p_xx_vec[j] = last_particlefilter.filtered_particles[i]._weight;
	  //p_xx_vec[j] = 0.0;
	}
	sum = 0;
	double tmp = 0;
	for (int j = 0; j < particle_filter._samples; j++){
	  tmp = (p_xx_vec[j] + last_particlefilter.filtered_particles[j]._weight );
	  map[i] += exp(tmp);
	}
	map[i] = exp(p_yx_vec[i]) * map[i];
  }
  last_particlefilter = particle_filter;
}

cv::Mat pfMapMat::GetEstimation()
{
    double max = 0;
    int max_i = 0;
    int size = map.size();
    max = map[0];
    max_i = 0;
    for (int i = 0; i < size; i++){
        if (map[i] > max){
            max = map[i];
            max_i = i;
        }
    }
    return last_particlefilter.filtered_particles[max_i]._state;
}
