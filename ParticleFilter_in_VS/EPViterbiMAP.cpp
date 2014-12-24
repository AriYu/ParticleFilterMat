#include "EPViterbiMAP.h"
#include <algorithm>

//#define DEBUG

using namespace std;


EPViterbiMat::EPViterbiMat(ParticleFilterMat &particle_filter)
	: last_particlefilter(particle_filter), _is_inited(false)
{
	this->delta.resize(particle_filter._samples);
	this->last_delta.resize(particle_filter._samples);
        this->g_yx_vec.resize(particle_filter._samples);
        this->f_xx_vec.resize(particle_filter._samples);
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
    for (int i = 0; i < particle_filter._samples; i++){
        cv::Mat obshat = observed.clone();
        cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

        obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
        g_yx_vec[i] = obs_likelihood(observed,
                                 obshat, 
                                 particle_filter._ObsNoiseCov, 
                                 particle_filter._ObsNoiseMean);
        sum = logsumexp(sum, g_yx_vec[i], (i==0));
        //sum += g_yx_vec[i];
    }
    for(int i = 0; i < particle_filter._samples; i++){
        //g_yx_vec[i] = g_yx_vec[i] / sum;
        g_yx_vec[i] = g_yx_vec[i] - sum;
    }


    //=============================================
    // calc f(x_1[i])
    //=============================================
    sum = 0.0;
    for (int i = 0; i < particle_filter._samples; i++){
        cv::Mat est_state
            = particle_filter.filtered_particles[i]._state;
        cv::Mat last_state = cv::Mat::zeros(est_state.rows, est_state.cols, CV_64F);
        f_xx_vec[i] = trans_likelihood(est_state, last_state, 
                                 particle_filter._ProcessNoiseCov, 
                                 particle_filter._ObsNoiseMean);
        //f_xx_vec[i] = 1.0 / particle_filter._samples;
        //sum += f_xx_vec[i];
        sum = logsumexp(sum, f_xx_vec[i], (i==0));
    }
    for(int i = 0; i < particle_filter._samples; i++){
        //f_xx_vec[i] = f_xx_vec[i] / sum;
        //f_xx_vec[i] = f_xx_vec[i] - sum;
        f_xx_vec[i] = 0.0;
    }

    //=============================================
    // log(f(x)) + log(g(y1 | x1))
    for(int i = 0; i < particle_filter._samples; i++){
        delta[i] = f_xx_vec[i] + g_yx_vec[i];
        last_delta[i] = 0.0;
        last_particlefilter.predict_particles[i]
            = particle_filter.predict_particles[i];
        last_particlefilter.filtered_particles[i]
            = particle_filter.filtered_particles[i];
    }

#ifdef DEBUG
    for (int i = 0; i < particle_filter._samples; i++){

    epvgm_output << i << " " 
                 << particle_filter.filtered_particles[i]._state.at<double>(0,0) << " " 
                 << delta[i] << endl;
}
#endif // DEBUG

    epvgm_output << endl; epvgm_output << endl;
    _is_inited = true;
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
        for(int i = 0; i < particle_filter._samples; i++){
            cv::Mat obshat = observed.clone();
            cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
            obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
            g_yx_vec[i] = obs_likelihood(observed, 
                                         obshat, 
                                         particle_filter._ObsNoiseCov, 
                                         particle_filter._ObsNoiseMean);
            //sum += g_yx_vec[i];
            sum = logsumexp(sum, g_yx_vec[i], (i==0));
        }
        // ===============================================
        // p(y_k | x_k)の正規化
        for(int i = 0; i < particle_filter._samples; i++){
            //g_yx_vec[i] = g_yx_vec[i] / sum;
            g_yx_vec[i] = g_yx_vec[i] - sum;
        }

        for(int i = 0; i < particle_filter._samples; i++){
            // ================================================
            // calc p(x_k(i) | x_k-1(j))
            sum = 0.0;
            for (int j = 0; j < particle_filter._samples; j++){
                cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
                cv::Mat est_state = particle_filter.filtered_particles[i]._state.clone();
                processmodel(est_state, 
                             last_particlefilter.filtered_particles[j]._state, 
                             ctrl_input, rnd_num);
                f_xx_vec[j] = trans_likelihood(est_state,
                                               particle_filter.filtered_particles[i]._state,
                                               particle_filter._ProcessNoiseCov,
                                               particle_filter._ProcessNoiseMean);
                //sum += f_xx_vec[j];
                sum = logsumexp(sum, f_xx_vec[j], (j==0));
            }
            // ===============================================
            // p(x_k(i) | x_k-1(j))の正規化
            for(int j = 0; j < particle_filter._samples; j++){
                //f_xx_vec[j] = f_xx_vec[j] / sum;
                f_xx_vec[j] = f_xx_vec[j] - sum;
            }

            // ===============================================
            // Search max(delta_k-1 + log(p(x_k(i) | x_k-1(j))))
            for(int j = 0; j < particle_filter._samples; j++){
                if (j == 0){
                    max = last_delta[j] +  f_xx_vec[j];
                    delta[i] = (g_yx_vec[i]) + max;
                }
                else{
                    tmp = last_delta[j] + (f_xx_vec[j]);
                    if (tmp > max){
                        max = tmp;
                        delta[i] = (g_yx_vec[i]) +  max;
                    }
                }
            }
        }

        // g_yx = 0;
        // {
        //     cv::Mat obshat = observed.clone();
        //     cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);

        //     obsmodel(obshat, particle_filter.filtered_particles[i]._state, rnd_num);
        //     g_yx = log(likelihood(observed, 
        //                           obshat, 
        //                           particle_filter._ObsNoiseCov, 
        //                           particle_filter._ObsNoiseMean));
        // }

        // for (int j = 0; j < particle_filter._samples; j++){

        //     f_xx = 0;
        //     {
        //         cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
        //         cv::Mat est_state = particle_filter.filtered_particles[i]._state.clone();
        //         processmodel(est_state, 
        //                      last_particlefilter.filtered_particles[j]._state, 
        //                      ctrl_input, rnd_num);
        //         f_xx = log(
        //             likelihood(est_state,
        //                        particle_filter.filtered_particles[i]._state,
        //                        particle_filter._ProcessNoiseCov,
        //                        particle_filter._ProcessNoiseMean));

        //     }
        //     if (j == 0){
        //         max = last_delta[j] + f_xx;
        //         delta[i] = g_yx + max;
        //     }
        //     else{
        //         tmp = last_delta[j] + f_xx;
        //         if (tmp > max){
        //             max = tmp;
        //             delta[i] = g_yx + max;
        //         }
        //     }
        // }


        for (int i = 0; i < particle_filter._samples; i++){
            last_delta[i] = delta[i];
            last_particlefilter.predict_particles[i]
                = particle_filter.predict_particles[i];
            last_particlefilter.filtered_particles[i]
                = particle_filter.filtered_particles[i];
#ifdef DEBUG
            epvgm_output << i << " " 
                         << particle_filter.filtered_particles[i]._state.at<double>(0,0) << " " 
                         << delta[i] << endl;
#endif // DEBUG
        }

#ifdef DEBUG
        epvgm_output << endl; epvgm_output << endl;
#endif // DEBUG

    }
}

cv::Mat EPViterbiMat::GetEstimation()
{
    //=====================================================
    double max = 0;	
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
    //========================================================

}

