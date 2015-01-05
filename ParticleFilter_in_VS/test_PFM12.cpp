///////////////////////////////////////////////
//
// This Program is test for ParticleFilterMat.
// - Random walk model
// - x(k) = x(k-1) + v(k)
// - y(k) = x(k) + w(k)
///////////////////////////////////////////////

#include <iostream>
#include <random>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "ParticleFilter.h"
#include "EPViterbiMAP.h"
#include "EPViterbiMAPAlpha.h"
#include "pfMapMat.h"

#include "RootMeanSquareError.h"

#define	PARTICLE_IO

#define NumOfIterate 1
#define NumOfParticle 100
#define ESSth 50

using namespace std;
using namespace cv;

double       k = 0.0;		//! loop count
const double T = 50.0;         //! loop limit

//----------------------------
// Process Equation
//! x		: state
//! xpre	: last state
//! input	: input
//! rnd		: process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
    x.at<double>(0, 0) =  0.5*xpre.at<double>(0,0) 
	  + 25.0*(xpre.at<double>(0,0) / (1.0 + (xpre.at<double>(0,0)*xpre.at<double>(0,0)))) 
	  +  8.0 * cos(1.2*k)
	  + rnd.at<double>(0, 0);
}


//-------------------------
// Observation Equation
//! z : 観測値
//! x : 状態ベクトル
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
    z.at<double>(0, 0) = (x.at<double>(0, 0) * x.at<double>(0, 0)) / 20.0 
	  + rnd.at<double>(0, 0);
}

//-----------------------------------------------------
// Observation Likelihood function
//! z    : 観測値
//! zhat : 推定観測値
//! cov  : 共分散
//! mena : 平均
double Obs_likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean)
{
    double prod = 0.0, e;

    e = z.at<double>(0, 0) - zhat.at<double>(0, 0) - mean.at<double>(0, 0);
    double tmp = -(e*e) / (2.0*cov.at<double>(0, 0));
    tmp = tmp - log(sqrt(2.0*CV_PI*cov.at<double>(0, 0)));
    return tmp;
}

//-----------------------------------------------------
// Trans Likelihood function
//! x    : 状態量
//! xhat : 推定状態量
//! cov  : 共分散
//! mena : 平均
double Trans_likelihood(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean)
{
    // cv::Mat error = x - xhat;
    // double error_norm = cv::norm(error);
    double e = x.at<double>(0,0) - xhat.at<double>(0,0);
    double tmp = -(e*e) / (2.0*cov.at<double>(0, 0));
    //tmp = tmp - log(sqrt(2.0*CV_PI*cov.at<double>(0, 0)));

    return tmp;
}


int main(void) {

    double ave_mmse = 0;
    double ave_epvgm = 0;
    double ave_pfmap = 0;

    // ==============================
    // Set Process Noise
    // ==============================
    cv::Mat ProcessCov = (cv::Mat_<double>(1, 1) << 10.0);
    std::cout << "ProcessCov  = " << ProcessCov << std::endl << std::endl;
    cv::Mat ProcessMean       = (cv::Mat_<double>(1, 1) << 0.0);
    std::cout << "ProcessMean = " << ProcessMean << std::endl << std::endl;
    

    // ==============================
    // Set Observation Noise
    // ==============================
    cv::Mat ObsCov = (cv::Mat_<double>(1, 1) << 1.0);
    std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
    cv::Mat ObsMean = (cv::Mat_<double>(1, 1) << 0.0);
    std::cout << "ObsMean = " << ObsMean << std::endl << std::endl;


    // ==============================
    // Set Initial Particle Noise
    // ==============================
    cv::Mat initCov = (cv::Mat_<double>(1, 1) << 5.0);
    std::cout << "initCov=" << initCov << std::endl << std::endl;
    cv::Mat initMean = (cv::Mat_<double>(1, 1) << 0.0);
    std::cout << "initMean=" << initMean << std::endl << std::endl;

    std::cout << "Particle filter mat initialized!" << endl;


    for(int loop = 0; loop < NumOfIterate; loop++){

        ofstream output;        // x, y
        output.open("result1.dat", ios::out);
        if (!output.is_open()){ std::cout << "open result output failed" << endl; return -1; }

#ifdef PARTICLE_IO
        ofstream particles_file; // k, x, weight
        particles_file.open("result_particle.dat", ios::out);
        if (!particles_file.is_open()){ std::cout << "open result_particle output failed" << endl; return -1; }
#endif // PARTICLE_IO

        // ==============================
        // Set for Particle filter Mat
        // ==============================
        cv::Mat A       = (cv::Mat_<double>(2, 2) << 1.0, 1.0, 0, 1.0);
        std::cout << "A = " << A << std::endl << std::endl;
        cv::Mat B       = (cv::Mat_<double>(2, 2) << 0, 0, 0, 0);
        std::cout << "B = " << B << std::endl << std::endl;
        cv::Mat C       = (cv::Mat_<double>(2, 1) << 1, 0);
        std::cout << "C = " << C << std::endl << std::endl;
	//! A : 状態遷移行列, B : 制御入力, C : 観測行列, dimX : 状態ベクトルの次元数
        ParticleFilterMat pfm(A, B, C, 1);
        pfm.SetProcessNoise(ProcessCov, ProcessMean);
        pfm.SetObservationNoise(ObsCov, ObsMean);
        pfm.Init(NumOfParticle, initCov, initMean);

        Mat    state             = Mat::zeros(1, 1, CV_64F); /* (x) */
        Mat    last_state        = Mat::zeros(1, 1, CV_64F); /* (x) */
        Mat    processNoise(1, 1, CV_64F);
        Mat    measurement       = Mat::zeros(1, 1, CV_64F);
        Mat    measurementNoise  = Mat::zeros(1, 1, CV_64F);
        double first_sensor      = 0.0;


        // ==============================
        // End Point Viterbi Estimation
        // ==============================
        EPViterbiMat epvgm(pfm);
        EPViterbiMatAlpha epvgm_alpha(pfm);


        // ==============================
        // Particle based MAP Estimation
        // ==============================
        pfMapMat pfmap(pfm);
        pfmap.Initialization(pfm);

        // ==============================
        // Root Mean Square Error
        // ==============================
        RMSE mmse_rmse;
        RMSE epvgm_rmse;
        RMSE epvgm_alpha_rmse;
        RMSE pfmap_rmse;
        RMSE obs_rmse;

        cv::RNG rng;            // random generater

        for (k = 0; k < T; k+=1.0){
            std::cout << "\rloop == " << loop << "\tk == " << k << "\r" << endl;

            // ==============================
            // Generate Actual Value
            // ==============================
            double input = 0.0;
            randn(processNoise, Scalar(0), Scalar::all(sqrt(ProcessCov.at<double>(0, 0))));
            process(state, last_state, input, processNoise);


            // ==============================
            // Generate Observation Value
            // ==============================
            first_sensor = rng.gaussian(sqrt(ObsCov.at<double>(0, 0))) 
                + ObsMean.at<double>(0, 0);
            measurementNoise.at<double>(0, 0) = first_sensor;
            observation(measurement, state, measurementNoise);

            // ==============================
            // Particle Filter Process
            // ==============================
            pfm.Sampling(process, input);
            pfm.CalcLikelihood(observation, Obs_likelihood, measurement);
            pfm.Resampling(measurement, ESSth);

#ifdef PARTICLE_IO
            for (int i = 0; i < pfm._samples; i++){
                particles_file << pfm.filtered_particles[i]._state.at<double>(0, 0) << " " 
                               << exp(pfm.predict_particles[i]._weight) << endl;
            }
            particles_file << endl; particles_file << endl;
#endif // PARTICLE_IO


            // ==============================
            // EP-VGM Process
            // ==============================
            epvgm.Recursion(pfm, process, observation, 
                            Obs_likelihood, Trans_likelihood, input, measurement);
            //epvgm_alpha.Recursion(pfm, process, observation, Obs_likelihood, Trans_likelihood, input, measurement);

            // ==============================
            // Particle Based MAP Process
            // ==============================
            pfmap.Update(pfm, process, observation, 
                         Obs_likelihood, Trans_likelihood, input, measurement);

            // ==============================
            // Get Estimation
            // ==============================
            Mat    predictionPF    = pfm.GetMMSE();
            double predict_x_pf    = predictionPF.at<double>(0, 0);
            Mat    predictionEPVGM = epvgm.GetEstimation();
            double predict_x_epvgm = predictionEPVGM.at<double>(0, 0);
            Mat    predictionEPVGMAlpha = epvgm_alpha.GetEstimation();
            double predict_x_epvgm_alpha = predictionEPVGMAlpha.at<double>(0, 0);

            Mat    predictionPFMAP = pfmap.GetEstimation();
            double predict_x_pfmap = predictionPFMAP.at<double>(0, 0);

            // ==============================
            // for RMSE
            // ==============================
            mmse_rmse.storeData(state.at<double>(0, 0), predict_x_pf);
            epvgm_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm);
            epvgm_alpha_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm_alpha);
            pfmap_rmse.storeData(state.at<double>(0, 0), predict_x_pfmap);
            cv::Mat actual_obs = measurement.clone();
            cv::Mat rnd_num = cv::Mat::zeros(actual_obs.rows, actual_obs.cols, CV_64F);
            observation(actual_obs, state, rnd_num);
            obs_rmse.storeData(actual_obs.at<double>(0,0), measurement.at<double>(0,0));
			
		
            // ==============================
            // Save Estimated State
            // ==============================
            output << state.at<double>(0, 0) << " " // [1] true state
                   << measurement.at<double>(0, 0) << " " // [2] first sensor
                   << predict_x_pf << " " // [3] predicted state by PF(MMSE)
                   << predict_x_epvgm << " " // [4] predicted state by EPVGM
                   << predict_x_pfmap << " " // [5] predicted state by PFMAP
                   << predict_x_epvgm_alpha << endl; // [6] predicted state by EPVGMAlpha

            // ==============================
            // Particle Filter Process
            // ==============================
            //pfm.Resampling(measurement, ESSth);

            last_state = state;


        }

        mmse_rmse.calculationRMSE();
        epvgm_rmse.calculationRMSE();
        epvgm_alpha_rmse.calculationRMSE();
        pfmap_rmse.calculationRMSE();
        obs_rmse.calculationRMSE();

        std::cout << "RMSE(MMSE)  : " << mmse_rmse.getRMSE() << endl;
        std::cout << "RMSE(EPVGM) : " << epvgm_rmse.getRMSE() << endl;
        std::cout << "RMSE(PFMAP) : " << pfmap_rmse.getRMSE() << endl;
        std::cout << "RMSE(Obs) : " << obs_rmse.getRMSE() << endl;

        output.close();
        ave_mmse += mmse_rmse.getRMSE();
        ave_epvgm += epvgm_rmse.getRMSE();
        ave_pfmap += pfmap_rmse.getRMSE();
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "nonlinear, multimodal model" << endl;
    cout << "Particles : " <<    NumOfParticle << endl;
    std::cout << "ProcessCov  = " << ProcessCov << std::endl << std::endl;
    std::cout << "ObsCov      =" << ObsCov << std::endl << std::endl;
    std::cout << "RMSE(MMSE)  : " <<  ave_mmse / (double)NumOfIterate << endl;
    std::cout << "RMSE(EPVGM) : " << ave_epvgm / (double)NumOfIterate << endl;
    std::cout << "RMSE(PFMAP) : " << ave_pfmap / (double)NumOfIterate << endl;
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    
    //std::system("wgnuplot -persist plot4.plt");
    //std::system("gnuplot -persist plot10.plt");

    return 0;
}
