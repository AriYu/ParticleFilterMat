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


//----------------------------
// Process Equation
// rnd : process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
	x.at<double>(0, 0)
		= xpre.at<double>(0, 0) + xpre.at<double>(1, 0) + rnd.at<double>(0, 0);
	x.at<double>(1, 0) = xpre.at<double>(1, 0) + rnd.at<double>(1, 0);
}
//-------------------------
// Observation Equation
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
	z.at<double>(0, 0) = x.at<double>(0, 0) + rnd.at<double>(0, 0);
	z.at<double>(1, 0) = 0.0;
}
//-----------------------------------------------------
// Likelihood is a t-distribution with nu = 10
double likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov)
{
	double prod = 0.0, e;
	for (int i = 0; i < z.rows; ++i)
	{
		e = z.at<double>(i,0) - zhat.at<double>(i,0);
		double tmp = exp((-pow(e, 2.0) / (2.0*cov.at<double>(i, 0))));
		tmp = tmp / sqrt(2.0*CV_PI*cov.at<double>(i, 0));
		//double tmp = 1.5 * pow(1.0 + ((e*e) / 10.0), -5.5);
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
	output.open("result.dat", ios::out);
	if (!output.is_open()){ cout << "open result output failed" << endl; return -1; }

	// set for Particle filter Mat
	double delta_t = SAMPLING_TIME;
	cv::Mat A = (cv::Mat_<double>(2, 2) << 1.0, 1.0, 0, 1.0);
	std::cout << "A=" << A << std::endl << std::endl;
	cv::Mat B = (cv::Mat_<double>(2, 2) << 0, 0, 0, 0);
	std::cout << "B=" << B << std::endl << std::endl;
	cv::Mat C = (cv::Mat_<double>(2, 1) << 1, 0);
	std::cout << "C=" << C << std::endl << std::endl;
	ParticleFilterMat pfm(A, B, C, 2);

	cv::Mat ProcessCov = (cv::Mat_<double>(2, 1) << sqrt(1e-5), sqrt(1e-5));
	std::cout << "ProcessCov=" << ProcessCov << std::endl << std::endl;
	cv::Mat ProcessMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "ProcessMean=" << ProcessMean << std::endl << std::endl;
	pfm.SetProcessNoise(ProcessCov, ProcessMean);

	cv::Mat ObsCov = (cv::Mat_<double>(2, 1) << sqrt(1), sqrt(1));
	std::cout << "ObsCov=" << ObsCov << std::endl << std::endl;
	cv::Mat ObsMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "ObsMean=" << ObsMean << std::endl << std::endl;
	pfm.SetObservationNoise(ObsCov, ObsMean);

	cv::Mat initCov = (cv::Mat_<double>(2, 1) << 0.1, 0.1);
	std::cout << "initCov=" << initCov << std::endl << std::endl;
	cv::Mat initMean = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
	std::cout << "initMean=" << initMean << std::endl << std::endl;
	pfm.Init(NumOfParticle, initCov, initMean);

	Mat img(600, 600, CV_8UC3);
	KalmanFilter KF(2, 1, 0);
	Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
	Mat processNoise(2, 1, CV_32F);
	Mat measurement = Mat::zeros(1, 1, CV_32F);
	char code = (char)-1;

	for (;;)
	{
		randn(state, Scalar::all(0), Scalar::all(0.1));
		KF.transitionMatrix = *(Mat_<float>(2, 2) << 1, 1, 0, 1);

		setIdentity(KF.measurementMatrix);
		cout << "KF.measurementMatrix" << KF.measurementMatrix << endl;
		setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
		setIdentity(KF.measurementNoiseCov, Scalar::all(1));
		setIdentity(KF.errorCovPost, Scalar::all(1));

		randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

		for (;;)
		{
			// 真値を生成
			Point2f center(img.cols*0.5f, img.rows*0.5f);
			float R = img.cols / 3.f;
			double stateAngle = state.at<float>(0);
			double stateAcc = state.at<float>(1);
			Point statePt = calcPoint(center, R, stateAngle);

			// 観測値を生成
			randn(measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
			measurement += KF.measurementMatrix*state;

			double measAngle = measurement.at<float>(0);
			Point measPt = calcPoint(center, R, measAngle);

			// 点の位置を推定
			// Kalman Filter
			Mat prediction = KF.predict();
			double predictAngle = prediction.at<float>(0);
			double predictAcc = prediction.at<float>(1);
			cout << "predictAngleKF : " << predictAngle << endl;
			Point predictPt = calcPoint(center, R, predictAngle);
			
			// Particle Filter
			double input = 0.0;
			//pfm.Sampling(input);
			pfm.Sampling(process, input);
			cv::Mat observed = (cv::Mat_<double>(2, 1) << measAngle, 0.0);
			//pfm.CalcLikehood(input, observed);
			pfm.CalcLikelihood(observation, likelihood, observed);
			
			Mat predictionPF = pfm.GetMMSE();
			double predictAnglePF = predictionPF.at<double>(0, 0);
			double predictAccPF = predictionPF.at<double>(1, 0);
			cout << "predictAnglePF : " << predictAnglePF << endl;
			Point predictPtPF = calcPoint(center, R, predictAnglePF);
			
			// 状態の更新
			if (theRNG().uniform(0, 4) != 0){
				// Kalman filter
				KF.correct(measurement);
			}
			
			// 推定値を保存
			output << stateAngle - predictAngle << " "	// [1] true Angle - kalman Angle
				<< stateAngle - predictAnglePF << " "	// [2] true Angle - PF(MMSE) Angle
				<< stateAngle - measAngle << " "		// [3] true Angle - meas Angel
				<< stateAngle - 0 << " "				// [4] true Angle - PF(ML) Angle
				<< stateAcc - predictAcc << " "			// [5] true Acc - Kalman Acc
				<< stateAcc - predictAccPF << " "		// [6] true Acc - PF(MMSE) Acc
				<< stateAcc << " "						// [7] true Acc
				<< predictAcc << " "					// [8] kalman Acc
				<< predictAccPF << endl;				// [9] PF(MMSE) Acc

			// plot points
#define drawCross( center, color, d )                                 \
                line( img, Point( center.x - d, center.y - d ),                \
                             Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                \
                             Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

			img = Scalar::all(0);
			printParticle(img, center, R, pfm);
			drawCross(statePt, Scalar(255, 255, 255), 3);
			drawCross(measPt, Scalar(0, 0, 255), 3);
			drawCross(predictPt, Scalar(0, 255, 0), 3);
			drawCross(predictPtPF, Scalar(255, 0, 255), 3);
			line(img, statePt, measPt, Scalar(0, 0, 255), 3, CV_AA, 0);
			line(img, statePt, predictPt, Scalar(0, 255, 255), 3, CV_AA, 0);
			line(img, statePt, predictPtPF, Scalar(255, 0, 255), 3, CV_AA, 0);


			randn(processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
			state = KF.transitionMatrix*state + processNoise;

			imshow("Kalman vs PF", img);
			code = (char)waitKey(10);

			pfm.Resampling(observed);

			if (code > 0)
				break;
		}
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}


	output.close();

	system("wgnuplot -persist plot.plt");

	return 0;
}

static inline Point calcPoint(Point2f center, double R, double angle)
{
	return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

void printParticle(cv::Mat &img, Point2f &center, float &R, ParticleFilterMat &pfm)
{
	int size = pfm._samples;
	for (int i = 0; i < size; i++){
		Mat particle = pfm.filtered_particles[i]._state;
		double particleAngle = particle.at<double>(0, 0);
		//cout << "predictAnglePF : " << predictAnglePF << endl;
		Point particlePt = calcPoint(center, R, particleAngle);
		drawCross(particlePt, Scalar(255, 255, 255),
			((int)500 * pfm.filtered_particles[i]._weight));
	}
}