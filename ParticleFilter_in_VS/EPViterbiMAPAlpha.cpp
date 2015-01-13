#include "EPViterbiMAPAlpha.h"
#include <algorithm>

#define DEBUG

using namespace std;


EPViterbiMatAlpha::EPViterbiMatAlpha(ParticleFilterMat &particle_filter)
	: last_particlefilter(particle_filter), _is_inited(false)
{
	this->delta.resize(particle_filter.samples_);
	this->last_delta.resize(particle_filter.samples_);
        this->g_yx_vec.resize(particle_filter.samples_);
        this->f_xx_vec.resize(particle_filter.samples_);
#ifdef DEBUG
        epvgm_output.open("epvgm_a.dat", ios::out);
        if(!epvgm_output.is_open()){ std::cout << "epvgm output open failed" << endl;}
#endif // DEBUG
}

EPViterbiMatAlpha::~EPViterbiMatAlpha()
{
}

void EPViterbiMatAlpha::Initialization(
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
        g_yx_vec[i] = particle_filter.filtered_particles[i].weight_;
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
        // f_xx_vec[i] = 1.0 / particle_filter.samples_;
        logsumexp(sum, f_xx_vec[i], (i==0));
        //sum += f_xx_vec[i];
    }
    for(int i = 0; i< particle_filter.samples_; i++){
        f_xx_vec[i] = f_xx_vec[i] - sum;
    }


    //=============================================
    // log(f(x)) + log(g(y1 | x1))
    for(int i = 0; i < particle_filter.samples_; i++){
        delta[i] = f_xx_vec[i] + g_yx_vec[i];
        last_delta[i] = 0.0;
        last_particlefilter.predict_particles[i]
            = particle_filter.predict_particles[i];
        last_particlefilter.filtered_particles[i]
            = particle_filter.filtered_particles[i];
    }

#ifdef DEBUG
    for (int i = 0; i < particle_filter.samples_; i++){
        epvgm_output << i << " " 
                     << particle_filter.filtered_particles[i].state_.at<double>(0,0) << " " 
                     << delta[i] << endl;
    }
#endif // DEBUG

    epvgm_output << endl; epvgm_output << endl;
    _is_inited = true;
}

void EPViterbiMatAlpha::Recursion(
    ParticleFilterMat &particle_filter,
    void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
    void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
    double(*obs_likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
    double(*trans_likelihood)(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean), 
    const double &ctrl_input, 
    const cv::Mat &observed)
{
    double max = 0;
    double tmp = 0;

    if (! _is_inited){
        Initialization(particle_filter, obsmodel, obs_likelihood, trans_likelihood,  observed);
    }
    else
    {
        // ================================================
        // calc p(y_k | x_k)
        double sum = 0;
        for(int i = 0; i < particle_filter.samples_; i++){
            g_yx_vec[i] = particle_filter.filtered_particles[i].weight_;
            //cout << "g_yx_vec[" << i << "]" << g_yx_vec[i] << endl;
        }

        for(int i = 0; i < particle_filter.samples_; i++){
            // ================================================
            // calc p(x_k(i) | x_k-1(j))
            sum = 0.0;
            for (int j = 0; j < particle_filter.samples_; j++){
                cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
                cv::Mat est_state = particle_filter.filtered_particles[i].state_.clone();
                processmodel(est_state, 
                             last_particlefilter.filtered_particles[j].state_, 
                             ctrl_input, rnd_num);
                f_xx_vec[j] = trans_likelihood(est_state,
                                               particle_filter.filtered_particles[i].state_,
                                               particle_filter.ProcessNoiseCov_,
                                               particle_filter.ProcessNoiseMean_);
                sum = logsumexp(sum, f_xx_vec[j], (j == 0));
            }
            //cout << "sum : " << sum << endl;
            // ===============================================
            // p(x_k(i) | x_k-1(j))の正規化
            for(int j = 0; j < particle_filter.samples_; j++){
                f_xx_vec[j] = f_xx_vec[j] - sum;
            }

            // ===============================================
            // Search max(delta_k-1 + log(p(x_k(i) | x_k-1(j))))
            for(int j = 0; j < particle_filter.samples_; j++){
                if (j == 0){
                    max = last_delta[j] + f_xx_vec[j];
                    delta[i] = g_yx_vec[i] + max;
                }
                else{
                    tmp = last_delta[j] + f_xx_vec[j];
                    if (tmp > max){
                        max = tmp;
                        delta[i] = g_yx_vec[i] +  max;
                    }
                }
            }
        }



        for (int i = 0; i < particle_filter.samples_; i++){
            last_delta[i] = delta[i];
            last_particlefilter.predict_particles[i]
                = particle_filter.predict_particles[i];
            last_particlefilter.filtered_particles[i]
                = particle_filter.filtered_particles[i];
#ifdef DEBUG
            epvgm_output << i << " " 
                         << particle_filter.filtered_particles[i].state_.at<double>(0,0) << " " 
                         << delta[i] << endl;
#endif // DEBUG
        }

#ifdef DEBUG
        epvgm_output << endl; epvgm_output << endl;
#endif // DEBUG

    }
}

cv::Mat EPViterbiMatAlpha::GetEstimation()
{
    //=====================================================
    double max = 0;	
    for (int i = 0; i < last_particlefilter.samples_; i++){
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
    return last_particlefilter.filtered_particles[_it].state_;
    //========================================================

}

