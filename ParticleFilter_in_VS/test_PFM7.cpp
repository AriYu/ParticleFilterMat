///////////////////////////////////////////////
//
// This Program is test for ParticleFilterMat.
// - Two sensor model
///////////////////////////////////////////////

#include <iostream>
#include <random>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "ParticleFilter.h"
#include "EPViterbiMAP.h"
#include "pfMapMat.h"

#include "RootMeanSquareError.h"

//#include "KalmanFilter.h"

#define	PARTICLE_IO

#define NumOfParticle	100

using namespace std;
using namespace cv;

double k = 0.0;			//! loop count
const double T = 250.0; //! loop limit

//----------------------------
// Process Equation
//! x		: state
//! xpre	: last state
//! input	: input
//! rnd		: process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
	double last = xpre.at<double>(0, 0);
	x.at<double>(0, 0)
		//= 0.5*last + (25.0*last / (1.0 + last*last)) + 8.0 * cos(1.2*k) + rnd.at<double>(0, 0);
		= xpre.at<double>(0, 0) + xpre.at<double>(1, 0) + rnd.at<double>(0, 0);
	x.at<double>(1, 0)
		= xpre.at<double>(1, 0) + rnd.at<double>(1, 0);
		//= exp(-1*0.01*last)  + 1.0 + rnd.at<double>(0, 0);
}
//-------------------------
// Observation Equation
//! z : 観測値
//! x : 状態ベクトル
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
	z.at<double>(0, 0)
		//= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0;// +rnd.at<double>(0, 0);

		= x.at<double>(0, 0);// +rnd.at<double>(0, 0);
	//z.at<double>(1, 0)
		//= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0;// +rnd.at<double>(0, 0);
		//=  x.at<double>(0, 0);// +rnd.at<double>(0, 0);


}
//-----------------------------------------------------
// Likelihood function
//! z    : 観測値
//! zhat : 推定観測値
//! cov  : 共分散
//! mena : 平均
double likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean)
{
	double prod = 0.0, e;

	//for (int i = 0; i < z.rows; ++i)
	//{
	//	e = z.at<double>(i, 0) - zhat.at<double>(i, 0) -mean.at<double>(i, 0);
	//	double tmp = exp((-pow(e, 2.0) / (2.0*cov.at<double>(i, 0))));
	//	tmp = tmp / sqrt(2.0*CV_PI*cov.at<double>(i, 0));
	//	prod += tmp;
	//	//cout << "prod[" << i << "]:" << prod << endl;
	//	//cout << endl;
	//}
	for (int i = 0; i < z.rows; ++i)
	{
		e = z.at<double>(i, 0) - zhat.at<double>(0, 0) - mean.at<double>(i, 0);
		double tmp = exp((-pow(e, 2.0) / (2.0*cov.at<double>(i, 0))));
		tmp = tmp / sqrt(2.0*CV_PI*cov.at<double>(i, 0));
		prod += tmp;
		//cout << "prod[" << i << "]:" << prod << endl;
		//cout << endl;
	}
	return prod;
}


int main(void) {

	double mmse_estimated = 0;

	ofstream output;         // x, y
	output.open("result1.dat", ios::out);
	if (!output.is_open()){ std::cout << "open result output failed" << endl; return -1; }

#ifdef PARTICLE_IO
	ofstream particles_file;         // k, x, weight
	particles_file.open("result_particle.dat", ios::out);
	if (!particles_file.is_open()){ std::cout << "open result_particle output failed" << endl; return -1; }
#endif // PARTICLE_IO

	// set for Particle filter Mat
	cv::Mat A = (cv::Mat_<double>(2, 2) << 1.0, 1.0, 0, 1.0);
	std::cout << "A=" << A << std::endl << std::endl;
	cv::Mat B = (cv::Mat_<double>(2, 2) << 0, 0, 0, 0);
	std::cout << "B=" << B << std::endl << std::endl;
	cv::Mat C = (cv::Mat_<double>(2, 1) << 1, 0);
	std::cout << "C=" << C << std::endl << std::endl;
	ParticleFilterMat pfm(A, B, C, 2);

	cv::Mat ProcessCov = (cv::Mat_<double>(2, 1) << 1e-5, 1e-5);
	std::cout << "ProcessCov=" << ProcessCov << std::endl << std::endl;
	cv::Mat ProcessMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "ProcessMean=" << ProcessMean << std::endl << std::endl;
	pfm.SetProcessNoise(ProcessCov, ProcessMean);

	cv::Mat ObsCov = (cv::Mat_<double>(2, 1) << 1e-1, 1e-1);
	std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
	cv::Mat ObsMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "ObsMean=" << ObsMean << std::endl << std::endl;
	pfm.SetObservationNoise(ObsCov, ObsMean);

	cv::Mat initCov = (cv::Mat_<double>(2, 1) << 0.1, 0.1);
	std::cout << "initCov=" << initCov << std::endl << std::endl;
	cv::Mat initMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "initMean=" << initMean << std::endl << std::endl;
	pfm.Init(NumOfParticle, initCov, initMean);
	std::cout << "Particle filter mat initialized!" << endl;

	Mat state = Mat::zeros(2, 1, CV_64F); /* (phi, delta_phi) */
	std::cout << "state = " << endl; std::cout << state << endl;
	Mat last_state = Mat::zeros(2, 1, CV_64F); /* (phi, delta_phi) */
	std::cout << "last_state = " << endl; std::cout << last_state << endl;
	Mat processNoise(2, 1, CV_64F);
	Mat measurement = Mat::zeros(2, 1, CV_64F);
	Mat measurementNoise = Mat::zeros(2, 1, CV_64F);
	double first_sensor = 0.0;
	double second_sensor = 0.0;

	char code = (char)-1;

	// EP-Viterbi Estimation
	EPViterbiMat epvgm(pfm);

	// Particle based MAP Estimation
	pfMapMat pfmap(pfm);
	pfmap.Initialization(pfm);

	RMSE mmse_rmse;
	RMSE epvgm_rmse;
	RMSE pfmap_rmse;

	cv::RNG rng; // random generater

	for (k = 0; k < T; k+=1.0){
		std::cout << "k==" << k << endl;
		if (k == 0){
			last_state.at<double>(0, 0) = 0.05;
			last_state.at<double>(1, 0) = 0.05;
		}
		// 真値を生成
		double input = 0.0;
		randn(processNoise, Scalar(0), Scalar::all(sqrt(ProcessCov.at<double>(0, 0))));
		process(state, last_state, input, processNoise);

		// 観測値を生成
		//randn(measurementNoise, Scalar::all(0), Scalar::all(sqrt(ObsCov.at<double>(0))));
		first_sensor = rng.gaussian(sqrt(ObsCov.at<double>(0, 0))) + ObsMean.at<double>(0, 0);
		second_sensor = rng.gaussian(sqrt(3.0*ObsCov.at<double>(1, 0))) + ObsMean.at<double>(1, 0);
		//observation(measurement, state, measurementNoise);
		measurement.at<double>(0, 0) = state.at<double>(0, 0);
		measurement.at<double>(1, 0) = state.at<double>(0, 0);
		measurementNoise.at<double>(0, 0) = first_sensor;
		measurementNoise.at<double>(1, 0) = second_sensor + 1.0;
		measurement += measurementNoise;

		// Particle Filter
		pfm.Sampling(process, input);
		pfm.CalcLikelihood(observation, likelihood, measurement);

#ifdef PARTICLE_IO
		for (int i = 0; i < pfm._samples; i++){
			particles_file << k << " " << pfm.filtered_particles[i]._state.at<double>(0, 0) << " " << pfm.filtered_particles[i]._weight << endl;
		}
		particles_file << endl; particles_file << endl;
#endif // PARTICLE_IO

		// EP-VGM
		epvgm.Recursion(pfm, process, observation, likelihood, input, measurement);

		// Particle Based MAP
		pfmap.Update(pfm, process, observation, likelihood, input, measurement);

		// Get Estimation
		Mat predictionPF = pfm.GetMMSE();
		double predict_x_pf = predictionPF.at<double>(0, 0);
		Mat predictionEPVGM = epvgm.GetEstimation();
		double predict_x_epvgm = predictionEPVGM.at<double>(0, 0);
		Mat predictionPFMAP = pfmap.GetEstimation();
		double predict_x_pfmap = predictionPFMAP.at<double>(0, 0);

		// for RMSE
		mmse_rmse.storeData(state.at<double>(0, 0), predict_x_pf);
		epvgm_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm);
		pfmap_rmse.storeData(state.at<double>(0, 0), predict_x_pfmap);
		

		// 推定値を保存
		output << state.at<double>(0, 0) << " "		// [1] true state
			<< measurement.at<double>(0, 0) << " "	// [2] first sensor
			<< measurement.at<double>(1, 0) << " "	// [3] second sensor
			<< predict_x_pf << " "					// [4] predicted state by PF(MMSE)
			<< predict_x_epvgm << " "				// [5] predicted state by EPVGM
			<< predict_x_pfmap << endl;				// [6] predicted state by PFMAP

		last_state = state;

		// リサンプリング
		pfm.Resampling(measurement);
	}

	mmse_rmse.calculationRMSE();
	epvgm_rmse.calculationRMSE();
	pfmap_rmse.calculationRMSE();

	std::cout << "RMSE(MMSE) : " << mmse_rmse.getRMSE() << endl;
	std::cout << "RMSE(EPVGM): " << epvgm_rmse.getRMSE() << endl;
	std::cout << "RMSE(PFMAP): " << pfmap_rmse.getRMSE() << endl;

	output.close();

	//std::system("wgnuplot -persist plot4.plt");
	std::system("wgnuplot -persist plot5.plt");

	return 0;
}
