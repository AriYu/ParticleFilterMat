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

static inline Point calcPoint(Point2f center, double R, double angle);
void printParticle(cv::Mat &img, Point2f &center, float &R, ParticleFilterMat &pfm);

int main(void) {

	double input = 0;
	double time = 0;
	double mmse_estimated = 0;

	ofstream output;         // x, y
	output.open("result.dat", ios::out);
	if (!output.is_open()){ cout << "open result output failed" << endl; return -1; }

	// set for Particle filter Mat
	double delta_t = SAMPLING_TIME;
	cv::Mat A = (cv::Mat_<double>(3, 3) 
		<< 1.1269, -0.4940, 0.1129, 1.0, 0.0, 0.0,0.0,1.0,0.0);
	std::cout << "A=" << A << std::endl << std::endl;
	cv::Mat B = (cv::Mat_<double>(3, 1) << -0.3832, 0.5919, 0.5191);
	std::cout << "B=" << B << std::endl << std::endl;
	cv::Mat C = (cv::Mat_<double>(3, 1) << 1.0, 0.0, 0.0);
	std::cout << "C=" << C << std::endl << std::endl;
	ParticleFilterMat pfm(A, B, C, 3);
	
	cv::Mat ProcessCov = (cv::Mat_<double>(3, 1) << sqrt(1.0), sqrt(1.0), sqrt(1.0));
	std::cout << "ProcessCov=" << ProcessCov << std::endl << std::endl;
	cv::Mat ProcessMean = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	std::cout << "ProcessMean=" << ProcessMean << std::endl << std::endl;
	pfm.SetProcessNoise(ProcessCov, ProcessMean);

	cv::Mat ObsCov = (cv::Mat_<double>(3, 1) << sqrt(1.0), sqrt(1.0), sqrt(1.0));
	std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
	cv::Mat ObsMean = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	std::cout << "ObsMean=" << ObsMean << std::endl << std::endl;
	pfm.SetObservationNoise(ObsCov, ObsMean);
	
	cv::Mat initCov = (cv::Mat_<double>(3, 1) << 0.1, 0.1, 0.1);
	std::cout << "initCov=" << initCov << std::endl << std::endl;
	cv::Mat initMean = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	std::cout << "initMean=" << initMean << std::endl << std::endl;
	pfm.Init(NumOfParticle, initCov, initMean);

	Mat img(600, 600, CV_8UC3);
	KalmanFilter KF(3, 3, 3, CV_64F);
    //Mat state(3, 1, CV_32F); /* (x1, x2, x3) */
	Mat state = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	Mat processNoise = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	Mat measurement = Mat::zeros(1, 1, CV_64F);
	char code = (char)-1;

	for (;;)
	{
		randn(state, Scalar::all(0), Scalar::all(0.1));
		KF.transitionMatrix = *(Mat_<double>(3, 3) 
			<< 1.1269, -0.4940, 0.1129, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
		cout << "Kalman transitionMatrix : " << endl;
		cout << KF.transitionMatrix << endl;

		setIdentity(KF.measurementMatrix);
		cout << "Kalman measurementMatrix : " << endl;
		cout << KF.measurementMatrix << endl;

		setIdentity(KF.processNoiseCov, Scalar::all(1.0));
		cout << "Kalman processNoiseCov : " << endl;
		cout << KF.processNoiseCov << endl;

		setIdentity(KF.measurementNoiseCov, Scalar::all(1.0));
		cout << "Kalman measurementNoiseCov : " << endl;
		cout << KF.measurementNoiseCov << endl;

		setIdentity(KF.errorCovPost, Scalar::all(1.0));
		cout << "Kalman errorCovPost : " << endl;
		cout << KF.errorCovPost << endl;

		randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

		for (;;)
		{
			// 真値を生成
			double x1 = state.at<double>(0);
			double x2 = state.at<double>(1);
			double x3 = state.at<double>(2);
			
			// 観測値を生成
			randn(measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<double>(0)));
			measurement += KF.measurementMatrix*state;
			double measState = measurement.at<double>(0);

			// 点の位置を推定
			// Kalman Filter
			Mat prediction = KF.predict(B*input);
			double predictKalman_x1 = prediction.at<double>(0);
			double predictKalman_x2 = prediction.at<double>(1);
			double predictKalman_x3 = prediction.at<double>(2);
			
			// Particle Filter
			pfm.Sampling(input);
			Mat predictionPF = pfm.GetMMSE();
			double predictPF_x1 = predictionPF.at<double>(0, 0);
			double predictPF_x2 = predictionPF.at<double>(1, 0);
			double predictPF_x3 = predictionPF.at<double>(2, 0);

			// 推定値を保存
			output << x1 << " "				// [1] true x1
				<< x2 << " "				// [2] true x2
				<< x3 << " "				// [3] true x3
				<< predictKalman_x1 << " "	// [4] Kalman x1
				<< predictKalman_x2 << " "	// [5] Kalman x2
				<< predictKalman_x3 << " "	// [6] Kalman x3
				<< predictPF_x1 << " "		// [7] PF x1
				<< predictPF_x2 << " "		// [8] PF x2
				<< predictPF_x3 << endl;	// [9] PF x3

			// 状態の更新
			if (theRNG().uniform(0, 4) != 0){
				 //Kalman filter
				KF.correct(measurement);
			}
			// Particle filter
			cv::Mat observed = (cv::Mat_<double>(3, 1) << measState, 0.0, 0.0);
			pfm.CalcLikehood(input, observed);
			pfm.Resampling(observed);
			
			input = 2.0*sin(time/5.0);

			randn(processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<double>(0, 0))));
			
			cv::Mat newInput = B*input;
			
			//state = KF.transitionMatrix*state + newInput + processNoise;
			state = A*state + newInput + processNoise;

			time += 1.0;
			cout << "time : " << time << endl;
			imshow("window", img);
			code = (char)waitKey(100);
			if (time > 100){
				code = 27;
				break;
			}
			if (code > 0)
				break;
		}
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}


	output.close();

	system("wgnuplot -persist plot2.plt");

	return 0;
}

static inline Point calcPoint(Point2f center, double R, double angle)
{
	return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}
