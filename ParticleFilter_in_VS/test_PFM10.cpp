///////////////////////////////////////////////
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

//#define	PARTICLE_IO

#define NumOfIterate 1
#define NumOfParticle  100
#define ESSTH 16

using namespace std;
using namespace cv;

double       k = 0.0;		//! loop count
const double T = 200.0;         //! loop limit

//----------------------------
// Process Equation
//! x		: state
//! xpre	: last state
//! input	: input
//! rnd		: process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
  //double last = xpre.at<double>(0, 0);
  x.at<double>(0, 0) =  xpre.at<double>(0, 0) + rnd.at<double>(0, 0);
  //= 0.5*last + (25.0*last / (1.0 + last*last)) + 8.0 * cos(1.2*k) + rnd.at<double>(0, 0);
  // x.at<double>(1, 0)
  // 	= xpre.at<double>(1, 0) + rnd.at<double>(1, 0);
  //= exp(-1*0.01*last)  + 1.0 + rnd.at<double>(0, 0);
}
//-------------------------
// Observation Equation
//! z : �ϑ��l
//! x : ��ԃx�N�g��
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
  z.at<double>(0, 0) = x.at<double>(0, 0) + rnd.at<double>(0, 0);
  //= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0;// +rnd.at<double>(0, 0);
  //= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0;// +rnd.at<double>(0, 0);
  //=  x.at<double>(0, 0);// +rnd.at<double>(0, 0);
}
//-----------------------------------------------------
// Observation Likelihood function
//! z    : �ϑ��l
//! zhat : ����ϑ��l
//! cov  : �����U
//! mena : ����
double Obs_likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean)
{
    double e = 0.0;

    e = z.at<double>(0, 0) - zhat.at<double>(0, 0) - mean.at<double>(0, 0);
    double tmp = -(e*e) / (2.0*cov.at<double>(0, 0));
    tmp = tmp - log(sqrt(2.0*CV_PI*cov.at<double>(0, 0)));
    return tmp;
}

//-----------------------------------------------------
// Trans Likelihood function
//! x    : ��ԗ�
//! xhat : �����ԗ�
//! cov  : �����U
//! mena : ����
double Trans_likelihood(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean)
{
    // cv::Mat error = x - xhat;
    // double error_norm = cv::norm(error);
    double error = x.at<double>(0,0) - xhat.at<double>(0,0);
    double error_norm = pow(error, 2.0);
    double tmp = -error_norm / (2.0*cov.at<double>(0, 0));
    tmp = tmp - log(sqrt(2.0*CV_PI*cov.at<double>(0, 0)));
    return tmp;
}


int main(void) {

    double ave_mmse = 0;
    double ave_epvgm = 0;
    double ave_pfmap = 0;

    // ==============================
    // Set Process Noise
    // ==============================
    cv::Mat ProcessCov = (cv::Mat_<double>(1, 1) << 1.0);
    std::cout << "ProcessCov  = " << ProcessCov << std::endl << std::endl;
    cv::Mat ProcessMean       = (cv::Mat_<double>(1, 1) << 0.0);
    std::cout << "ProcessMean = " << ProcessMean << std::endl << std::endl;
    

    // ==============================
    // Set Observation Noise
    // ==============================
    cv::Mat ObsCov = (cv::Mat_<double>(1, 1) << 5.0);
    std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
    cv::Mat ObsMean = (cv::Mat_<double>(1, 1) << 0.0);
    std::cout << "ObsMean = " << ObsMean << std::endl << std::endl;


    // ==============================
    // Set Initial Particle Noise
    // ==============================
    cv::Mat initCov = (cv::Mat_<double>(1, 1) << 0.01);
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
	//! A : ��ԑJ�ڍs��, B : �������, C : �ϑ��s��, dimX : ��ԃx�N�g���̎�����
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


#ifdef PARTICLE_IO
            for (int i = 0; i < pfm._samples; i++){
                particles_file << pfm.filtered_particles[i]._state.at<double>(0, 0) << " " << exp(pfm.filtered_particles[i]._weight) << endl;
            }
            particles_file << endl; particles_file << endl;
#endif // PARTICLE_IO

            
            // ==============================
            // EP-VGM Process
            // ==============================
            epvgm.Recursion(pfm, process, observation, Obs_likelihood, Trans_likelihood, input, measurement);
            //epvgm_alpha.Recursion(pfm, process, observation, Obs_likelihood, Trans_likelihood, input, measurement);

            // ==============================
            // Particle Based MAP Process
            // ==============================
            pfmap.Update(pfm, process, observation, Obs_likelihood, Trans_likelihood, input, measurement);

            // ==============================
            // Get Estimation
            // ==============================
            Mat    predictionPF    = pfm.GetMMSE();
            double predict_x_pf    = predictionPF.at<double>(0, 0);
            Mat    predictionEPVGM = epvgm.GetEstimation(pfm);
            double predict_x_epvgm = predictionEPVGM.at<double>(0, 0);
            Mat    predictionML = pfm.GetML();
            double predict_x_ml = predictionML.at<double>(0, 0);

            Mat    predictionPFMAP = pfmap.GetEstimation();
            double predict_x_pfmap = predictionPFMAP.at<double>(0, 0);

            // ==============================
            // for RMSE
            // ==============================
            mmse_rmse.storeData(state.at<double>(0, 0), predict_x_pf);
            epvgm_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm);
            epvgm_alpha_rmse.storeData(state.at<double>(0, 0), predict_x_ml);
            pfmap_rmse.storeData(state.at<double>(0, 0), predict_x_pfmap);
		
            // ==============================
            // Save Estimated State
            // ==============================
            output << state.at<double>(0, 0) << " " // [1] true state
                   << measurement.at<double>(0, 0) << " " // [2] first sensor
                   << predict_x_pf << " " // [3] predicted state by PF(MMSE)
                   << predict_x_epvgm << " " // [4] predicted state by EPVGM
                   << predict_x_pfmap << " " // [5] predicted state by PFMAP
                   << predict_x_ml << endl; // [6] predicted state by EPVGMAlpha

            // ==============================
            // Particle Filter Process
            // ==============================
            pfm.Resampling(measurement, ESSTH);

            last_state = state;
        }

        mmse_rmse.calculationRMSE();
        epvgm_rmse.calculationRMSE();
        epvgm_alpha_rmse.calculationRMSE();
        pfmap_rmse.calculationRMSE();

        std::cout << "RMSE(MMSE)  : " << mmse_rmse.getRMSE() << endl;
        std::cout << "RMSE(EPVGM) : " << epvgm_rmse.getRMSE() << endl;
        std::cout << "RMSE(ML): " << epvgm_alpha_rmse.getRMSE() << endl;
        std::cout << "RMSE(PFMAP) : " << pfmap_rmse.getRMSE() << endl;

        output.close();
        ave_mmse += mmse_rmse.getRMSE();
        ave_epvgm += epvgm_rmse.getRMSE();
        ave_pfmap += pfmap_rmse.getRMSE();
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "linear, mono modul model" << endl;
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













