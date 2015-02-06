#include "EPViterbiMAP.h"
#include <algorithm>

#define DEBUG

using namespace std;


EPViterbiMat::EPViterbiMat(ParticleFilterMat &particle_filter)
	: last_particlefilter(particle_filter), is_inited_(false)
{
	// this->delta.resize(particle_filter.samples_);
	// this->last_delta.resize(particle_filter.samples_);
	this->g_yx_vec.resize(particle_filter.samples_);
	this->f_xx_vec.resize(particle_filter.samples_);
	this->last_g_yx_vec.resize(particle_filter.samples_);
	this->max.resize(particle_filter.samples_);
#ifdef DEBUG
        epvgm_output.open("epvgm.dat", ios::out);
        if(!epvgm_output.is_open()){ std::cout << "epvgm output open failed" << endl;}
#endif // DEBUG
}

EPViterbiMat::~EPViterbiMat()
{
}

void EPViterbiMat::Initialization(
								  ParticleFilterMat &particle_filter,
								  void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
								  double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
								  double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean),
								  const cv::Mat &observed)
{
  double sum = 0;

  //=============================================
  // calc g(y_1 | x_1[i])
  //=============================================
  for (int i = 0; i < particle_filter.samples_; i++){
	cv::Mat obshat = observed.clone();
	cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

	obsmodel(obshat, particle_filter.filtered_particles[i].state_, rnd_num);
	g_yx_vec[i] = obs_likelihood(observed,
								 obshat, 
								 particle_filter.ObsNoiseCov_, 
								 particle_filter.ObsNoiseMean_);
	sum = logsumexp(sum, g_yx_vec[i], (i==0));
  }
    
  // ===============================================
  // g(y_1 | x_1)の正規化
  for(int i = 0; i < particle_filter.samples_; i++){
	g_yx_vec[i] = g_yx_vec[i] - sum;
  }


  //=============================================
  // calc f(x_1[i])
  //=============================================
  sum = 0.0;
  for (int i = 0; i < particle_filter.samples_; i++){
	cv::Mat est_state
	  = particle_filter.filtered_particles[i].state_;
	cv::Mat last_state = cv::Mat::zeros(est_state.rows, est_state.cols, CV_64F);
	f_xx_vec[i] = trans_likelihood(est_state, last_state, 
								   particle_filter.ProcessNoiseCov_, 
								   particle_filter.ObsNoiseMean_);
	sum = logsumexp(sum, f_xx_vec[i], (i==0));
  }
  for(int i = 0; i < particle_filter.samples_; i++){
  	f_xx_vec[i] = f_xx_vec[i] - sum;
  	//f_xx_vec[i] = 0.0;
  }

  //=============================================
  // log(f(x)) + log(g(y1 | x1))
  for(int i = 0; i < particle_filter.samples_; i++){
	// delta[i] = f_xx_vec[i] + g_yx_vec[i];
	// last_delta[i] = delta[i];
	particle_filter.delta[i] = f_xx_vec[i] + g_yx_vec[i];
	particle_filter.last_delta[i] = particle_filter.delta[i];
	last_g_yx_vec[i] = g_yx_vec[i];
	last_particlefilter.predict_particles[i]
	  = particle_filter.predict_particles[i];
	last_particlefilter.filtered_particles[i]
	  = particle_filter.filtered_particles[i];
#ifdef DEBUG
	epvgm_output << i << " " 
				 << particle_filter.filtered_particles[i].state_.at<double>(0,0) << " " 
				 << g_yx_vec[i] << " "
				 << particle_filter.last_delta[i] << " "
				 << 0 << " " // max
				 << 0 << " " // max fxx
				 << 0 << " " // max lastdelta
				 << particle_filter.delta[i] << endl;
#endif // DEBUG

  }


  epvgm_output << endl; epvgm_output << endl;
  is_inited_ = true;
}

void EPViterbiMat::Recursion(
							 ParticleFilterMat &particle_filter,
							 void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
							 void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
							 double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
							 double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean), 
							 const double &ctrl_input, 
							 const cv::Mat &observed)
{
  

#ifdef DEBUG
  std::vector<double> maxfxx(particle_filter.samples_);
  std::vector<double> maxlastdelta(particle_filter.samples_);
#endif

  //    double tmp = 0;

  if (! is_inited_){
	Initialization(particle_filter, obsmodel, obs_likelihood, trans_likelihood,  observed);
  }
  else
    {
	  cv::Mat obshat = observed.clone(); // メモリの確保
	  cv::Mat obs_rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F); // メモリの確保

	  // ================================================
	  // calc p(y_k | x_k)
	  double sum = 0.0;
	  for(int i = 0; i < particle_filter.samples_; i++){
		obsmodel(obshat, particle_filter.filtered_particles[i].state_, obs_rnd_num);
		g_yx_vec[i] = obs_likelihood(observed, 
									 obshat, 
									 particle_filter.ObsNoiseCov_, 
									 particle_filter.ObsNoiseMean_);
		sum = logsumexp(sum, g_yx_vec[i], (i==0));
	  }
	  // ===============================================
	  // p(y_k | x_k)の正規化
	  for(int i = 0; i < particle_filter.samples_; i++){
		g_yx_vec[i] = g_yx_vec[i] - sum;
	  }

	  cv::Mat est_rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
	  cv::Mat est_state = particle_filter.filtered_particles[0].state_.clone();
	  std::vector<double> lastdelta_fxx(particle_filter.samples_);
	  for(int i = 0; i < particle_filter.samples_; i++){
		// ================================================
		// calc p(x_k(i) | x_k-1(j))
		sum = 0.0;
		for (int j = 0; j < particle_filter.samples_; j++){
		  // processmodel(est_state, 
		  //              last_particlefilter.filtered_particles[j].state_, 
		  //              ctrl_input, est_rnd_num);
		  processmodel(est_state, 
					   particle_filter.predict_particles[j].state_, 
					   ctrl_input, est_rnd_num);
		  f_xx_vec[j] = trans_likelihood(est_state,
										 particle_filter.filtered_particles[i].state_,
										 particle_filter.ProcessNoiseCov_,
										 particle_filter.ProcessNoiseMean_);
		  // if(i == j){
		  // 	f_xx_vec[j] = log(1.0);
		  // }else{
		  // 	f_xx_vec[j] = log(0.1);
		  // }
		  //sum = logsumexp(sum, f_xx_vec[j], (j==0));
		}

		// ===============================================
		//p(x_k(i) | x_k-1(j))の正規化
		// for(int j = 0; j < particle_filter.samples_; j++){
		//   f_xx_vec[j] = f_xx_vec[j] - sum;
		// }


		// ===============================================
		// Search max(delta_k-1 + log(p(x_k(i) | x_k-1(j))))
		for(int j = 0; j < particle_filter.samples_; j++){
		  //lastdelta_fxx[j] = last_delta[j] + f_xx_vec[j];
		  lastdelta_fxx[j] = particle_filter.last_delta[j] + f_xx_vec[j];
		  //lastdelta_fxx[j] = particle_filter.last_delta[j] + f_xx_vec[j] + g_yx_vec[i];
		}
		max[i] = *max_element( lastdelta_fxx.begin(), lastdelta_fxx.end() );

		// delta[i] = g_yx_vec[i] + max[i];
		particle_filter.delta[i] = g_yx_vec[i] + max[i];
		//particle_filter.delta[i] = max[i];
	  }

       

	  for (int i = 0; i < particle_filter.samples_; i++){
#ifdef DEBUG
		epvgm_output << i << " " // [1]
					 << particle_filter.filtered_particles[i].state_.at<double>(0,0) << " " 
					 << g_yx_vec[i] << " " //[3]
					 << particle_filter.last_delta[i] << " " //[4]
					 << max[i] << " "//[5]
					 << maxfxx[i] << " "//[6]
					 << maxlastdelta[i] << " "//[7]
					 << particle_filter.delta[i] << endl;//[8]
#endif // DEBUG
		last_g_yx_vec[i] = g_yx_vec[i];
		//particle_filter.last_delta[i] = particle_filter.delta[i];
		last_particlefilter.filtered_particles[i].weight_
		  = particle_filter.filtered_particles[i].weight_;
		last_particlefilter.predict_particles[i]
		  = particle_filter.predict_particles[i];
		last_particlefilter.filtered_particles[i]
		  = particle_filter.filtered_particles[i];
	  }

#ifdef DEBUG
	  epvgm_output << endl; epvgm_output << endl;
#endif // DEBUG

    }
}

cv::Mat EPViterbiMat::GetEstimation(ParticleFilterMat &particle_filter)
{
    //=====================================================
    double max_ = 0;	
    for (int i = 0; i < last_particlefilter.samples_; i++){
    	if (i == 0){
		  max_ = particle_filter.delta[0];
		  it_ = i;
    	}
    	else{
            if (particle_filter.delta[i] > max_){
			  max_ = particle_filter.delta[i];
			  it_ = i;
            }
    	}
    }
    return last_particlefilter.filtered_particles[it_].state_;
    //========================================================

}

