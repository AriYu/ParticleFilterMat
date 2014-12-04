///////////////////////////////////////////////
//
// This Program is test for ParticleFilterMat.
//
///////////////////////////////////////////////

#include <iostream>
#include <random>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "ParticleFilter.h"
//#include "KalmanFilter.h"

#define	PARTICLE_IO

#define NumOfParticle	100

using namespace std;
using namespace cv;

double k = 0.0;			//! loop count
const double T = 10.0; //! loop limit

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
		= 0.5*last + 25.0*last / (1.0 + last*last) + 8.0 * cos(1.2*k) + rnd.at<double>(0, 0);
}
//-------------------------
// Observation Equation
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
	z.at<double>(0, 0)
		= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0;// +rnd.at<double>(0, 0);

}
//-----------------------------------------------------
// Likelihood is a t-distribution with nu = 10
double likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov)
{
	double prod = 0.0, e;
	for (int i = 0; i < z.rows; ++i)
	{
		e = z.at<double>(i, 0) - zhat.at<double>(i, 0);
		double tmp = exp((-pow(e, 2.0) / (2.0*cov.at<double>(i, 0))));
		tmp = tmp / sqrt(2.0*CV_PI*cov.at<double>(i, 0));
		/*double tmp = 1.5 * pow(1.0 + ((e*e) / 10.0), -5.5);*/
		prod += tmp;
	}
	return prod;
}


int main(void) {

	double mmse_estimated = 0;

	ofstream output;         // x, y
	output.open("result1.dat", ios::out);
	if (!output.is_open()){ cout << "open result output failed" << endl; return -1; }

#ifdef PARTICLE_IO
	ofstream particles_file;         // k, x, weight
	particles_file.open("result_particle.dat", ios::out);
	if (!particles_file.is_open()){ cout << "open result_particle output failed" << endl; return -1; }
#endif // PARTICLE_IO

	// set for Particle filter Mat
	cv::Mat A = (cv::Mat_<double>(2, 2) << 1.0, 1.0, 0, 1.0);
	std::cout << "A=" << A << std::endl << std::endl;
	cv::Mat B = (cv::Mat_<double>(2, 2) << 0, 0, 0, 0);
	std::cout << "B=" << B << std::endl << std::endl;
	cv::Mat C = (cv::Mat_<double>(2, 1) << 1, 0);
	std::cout << "C=" << C << std::endl << std::endl;
	ParticleFilterMat pfm(A, B, C, 1);

	cv::Mat ProcessCov = (cv::Mat_<double>(1, 1) << sqrt(10.0));
	std::cout << "ProcessCov=" << ProcessCov << std::endl << std::endl;
	cv::Mat ProcessMean = (cv::Mat_<double>(1, 1) << 0.0);
	std::cout << "ProcessMean=" << ProcessMean << std::endl << std::endl;
	pfm.SetProcessNoise(ProcessCov, ProcessMean);

	cv::Mat ObsCov = (cv::Mat_<double>(1, 1) << sqrt(10));
	std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
	cv::Mat ObsMean = (cv::Mat_<double>(1, 1) << 0.0);
	std::cout << "ObsMean=" << ObsMean << std::endl << std::endl;
	pfm.SetObservationNoise(ObsCov, ObsMean);

	cv::Mat initCov = (cv::Mat_<double>(1, 1) << 0.1);
	std::cout << "initCov=" << initCov << std::endl << std::endl;
	cv::Mat initMean = (cv::Mat_<double>(1, 1) << 0.0);
	std::cout << "initMean=" << initMean << std::endl << std::endl;
	pfm.Init(NumOfParticle, initCov, initMean);
	cout << "Particle filter mat initialized!" << endl;

	Mat state = Mat::zeros(1, 1, CV_64F); /* (phi, delta_phi) */
	cout << "state = " << endl; cout << state << endl;
	Mat last_state = Mat::zeros(1, 1, CV_64F); /* (phi, delta_phi) */
	cout << "last_state = " << endl; cout << last_state << endl;
	Mat processNoise(1, 1, CV_64F);
	Mat measurement = Mat::zeros(1, 1, CV_64F);
	Mat measurementNoise = Mat::zeros(1, 1, CV_64F);
	char code = (char)-1;

	for (k = 0; k < T; k+=1.0){
		cout << "k==" << k << endl;
		if (k == 0){
			last_state.at<double>(0, 0) = 0.1;
		}
		// 真値を生成
		double input = 0.0;
		randn(processNoise, Scalar(0), Scalar::all(sqrt(ProcessCov.at<double>(0, 0))));
		process(state, last_state, input, processNoise);

		// 観測値を生成
		randn(measurementNoise, Scalar::all(0), Scalar::all(ObsCov.at<double>(0)));
		observation(measurement, state, measurementNoise);
		measurement += measurementNoise;

		// Particle Filter
		pfm.Sampling(process, input);
		pfm.CalcLikelihood(observation, likelihood, measurement);

#ifdef PARTICLE_IO
		for (int i = 0; i < pfm._samples; i++){
			particles_file << k << " " << pfm.filtered_particles[i]._state.at<double>(0, 0) << " " << pfm.filtered_particles[i]._weight << endl;
		}
#endif // PARTICLE_IO

		Mat predictionPF = pfm.GetMMSE();
		double predict_x_pf = predictionPF.at<double>(0, 0);

		// 推定値を保存
		output << state.at<double>(0, 0) << " "		// [1] true state
			<< measurement.at<double>(0, 0) << " "	// [2] observed state
			<< predict_x_pf << endl;				// [3] predicted state by PF(MMSE)

		last_state = state;

		pfm.Resampling(measurement);
	}

	output.close();

	system("wgnuplot -persist plot4.plt");

	return 0;
}
