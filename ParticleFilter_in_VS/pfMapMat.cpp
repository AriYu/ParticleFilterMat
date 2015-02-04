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
	map.resize(particle_filter.samples_);
        p_xx_vec.resize(particle_filter.samples_);
        p_yx_vec.resize(particle_filter.samples_);
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
  cv::Mat obshat = observed.clone(); // メモリの確保
  cv::Mat obs_rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F); // メモリの確保
  for(int i = 0; i < particle_filter.samples_; i++){
	obsmodel(obshat, particle_filter.filtered_particles[i].state_, obs_rnd_num);
	p_yx_vec[i] = obs_likelihood(observed, obshat, 
								 particle_filter.ObsNoiseCov_, 
								 particle_filter.ObsNoiseMean_);
	sum = logsumexp(sum, p_yx_vec[i], (i==0));
  }
  for(int i = 0; i < particle_filter.samples_; i++){
  	p_yx_vec[i] = p_yx_vec[i] - sum;
  }

  cv::Mat est_rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
  cv::Mat est_state = particle_filter.filtered_particles[0].state_.clone();
  for (int i = 0; i < particle_filter.samples_; i++){
	map[i] = 0.0;
	sum = 0.0;
	for(int j = 0; j < particle_filter.samples_; j++){
	  processmodel(est_state, particle_filter.predict_particles[j].state_, 
				   ctrl_input, est_rnd_num);
	  // processmodel(est_state, last_particlefilter.filtered_particles[j].state_, 
	  // 			   ctrl_input, est_rnd_num);

	  p_xx_vec[j] = trans_likelihood(est_state,
	  								 particle_filter.filtered_particles[i].state_,
	  								 particle_filter.ProcessNoiseCov_,
	  								 particle_filter.ProcessNoiseMean_);

	  // if(i == j){
	  // 	p_xx_vec[j] = log(1.0);
	  // }else{
	  // 	p_xx_vec[j] = log(0.1);
	  // }
	  // sum = logsumexp(sum, p_xx_vec[j], (j == 0));
	}
	// for(int j = 0; j < particle_filter.samples_; j++){
	//    p_xx_vec[j] = p_xx_vec[j] - sum;
	// }

	double log_weight = 0;
	for (int j = 0; j < particle_filter.samples_; j++){
	  // log_weight = (p_xx_vec[j] + last_particlefilter.filtered_particles[j].weight_ );
	  // log_weight = (p_xx_vec[j] + particle_filter.predict_particles[j].weight_);
	  log_weight = (p_xx_vec[j] + particle_filter.last_filtered_particles[j].weight_);
	  map[i] += exp(log_weight);
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
    return last_particlefilter.filtered_particles[max_i].state_;
}
