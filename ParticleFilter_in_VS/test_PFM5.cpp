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

#define	DATAOUTPUT

#define NumOfParticle	1024
#define NumOfIterate	500

#define STATE_EQN_MODE	2
#define OBS_EQN_MODE	STATE_EQN_MODE

#define SAMPLING_TIME	0.01
#define TIME_CONSTANT	1.0

#define SYSTEM_GAIN		1.0
#define STEP_GAIN		1.0

#define LOOP			20

using namespace std;
using namespace cv;

double k = 0.0;
double T = 100.0;

//----------------------------
// Process Equation
//! x		: state
//! xpre	: last state
//! input	: input
//! rnd : process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
	double last = xpre.at<double>(0, 0);
	x.at<double>(0, 0)
		= 0.5*last + 25.0*last / (1 + last*last) + 8.0 * cos(1.2*k) + rnd.at<double>(0, 0);
}
//-------------------------
// Observation Equation
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
	z.at<double>(0, 0)
		= x.at<double>(0, 0) * x.at<double>(0, 0) / 20.0 + rnd.at<double>(0, 0);

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
		double tmp = 1.5 * pow(1.0 + ((e*e) / 10.0), -5.5);
		prod += tmp;
	}
	return prod;
}


int main(void) {

	double phi = 0;
	double delta_phi = 0;
	double x_old = 0;
	double y = 0;
	double input = 0;

	double mmse_estimated = 0;

	ofstream output;         // x, y
	output.open("result1.dat", ios::out);
	if (!output.is_open()){ cout << "open result output failed" << endl; return -1; }

	// set for Particle filter Mat
	double delta_t = SAMPLING_TIME;
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

	cv::Mat ObsCov = (cv::Mat_<double>(1, 1) << sqrt(1));
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

	for (k = 0; k < T; ++k){
		cout << "k==" << k << endl;
		if (k == 0){
			//randn(last_state, Scalar::all(0), Scalar::all(1.0));
			last_state.at<double>(0, 0) = 0.1;
		}
		// 真値を生成
		double input = 0.0;
		randn(processNoise, Scalar(0), Scalar::all(sqrt(ProcessCov.at<double>(0, 0))));
		//cout << "Generate processNoise." << endl;
		process(state, last_state, input, processNoise);
		/*cout << "state=" << endl; cout << state << endl;
		cout << "last_state=" << endl; cout << last_state << endl;
		cout << "input=" << endl; cout << input << endl;
		cout << "processNoise=" << endl;  cout << processNoise << endl;*/
		//cout << "Generate real state." << endl;

		// 観測値を生成
		randn(measurementNoise, Scalar::all(0), Scalar::all(ObsCov.at<double>(0)));
		//cout << "Generate measurementNoise." << endl;
		observation(measurement, state, measurementNoise);
		//cout << "Generate measurement." << endl;

		// Particle Filter
		pfm.Sampling(process, input);
		//cout << "[pfm]Sampling step." << endl;
		pfm.CalcLikelihood(observation, likelihood, measurement);
		//cout << "[pfm]Calculation likelihood step." << endl;

		Mat predictionPF = pfm.GetMMSE();
		double predict_x_pf = predictionPF.at<double>(0, 0);
		//cout << "predict_x_PF : " << predict_x_pf << endl;

		// 推定値を保存
		output << state.at<double>(0, 0) << " "		// [1] true state
			<< measurement.at<double>(0, 0) << " "	// [2] observed state
			<< predict_x_pf << endl;				// [3] predicted state by PF(MMSE)

		last_state = state;

		pfm.Resampling(measurement);
		//cout << "[pfm]Resampling step." << endl;
	}

	output.close();

	system("wgnuplot -persist plot4.plt");

	return 0;
}
