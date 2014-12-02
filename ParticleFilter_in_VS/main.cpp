#include <iostream>
#include <random>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ParticleFilter.h"
#include "KalmanFilter.h"

#define	DATAOUTPUT

#define NumOfParticle	1000
#define NumOfIterate	500

#define STATE_EQN_MODE	2
#define OBS_EQN_MODE	STATE_EQN_MODE

#define SAMPLING_TIME	0.01
#define TIME_CONSTANT	1.0

#define SYSTEM_GAIN		1.0
#define STEP_GAIN		1.0

#define LOOP			20

using namespace std;

double state_equation(double x_old, double i);
double observation_equation(double x);
double state_diff_equation(double x_old, double i);
double observation_diff_equation(double x);

int main(void) {

	double x = 0;
	double x_old = 0;
	double y = 0;
	double input = 0;

	double mmse_estimated = 0;
	double kalman_estimated = 0;

	double(*state_eqn)(double, double) = state_equation;
	double(*state_diff_eqn)(double, double) = state_diff_equation;
	double(*obs_eqn)(double) = observation_equation;
	double(*obs_diff_eqn)(double) = observation_diff_equation;

	const double first_particle_variance = sqrt(2.0); // Init particle variance
	const double variance_s = sqrt(0.10); // variance of system noise(w_k)
	const double variance_w = sqrt(100.0); // variance of observation noise(v_k)
	const double likehood_variance = variance_w;
	const double mean_s = 0.0; // average of system noise
	const double mean_w = 0.0; // average of observation noise
	PFilter pfilter(NumOfParticle, first_particle_variance, likehood_variance, state_eqn, obs_eqn);
	pfilter.SetNoiseParam(mean_s, variance_s, mean_w, variance_w);

	double A = 1.0; //exp(-(1 / SYSTEM_GAIN)*SAMPLING_TIME); // System	matrix
	double B = 1.0; // (1 / SYSTEM_GAIN); // input	matrix
	double C = 1.0; // Obserbve	matrix
	double initX0 = 10;
	double initP0 = 10;
	KalmanFilter kalmanfilter(A, B, C, initX0, initP0);
	kalmanfilter.SetNoiseParam(variance_s, variance_w);

	random_device rdev;
	mt19937       engine_s(rdev());
	mt19937       engine_w(rdev());
	normal_distribution<> sigma_s(mean_s, variance_s);
	normal_distribution<> sigma_w(mean_w, variance_w);

	ofstream output;         // x, y
	output.open("result.dat", ios::out);
	if (!output.is_open()){ cout << "open result output failed" << endl; return -1; }

	for (int i = 0; i < NumOfIterate; i++){

		input = 1.0;

		output << x << " " << y << " "
			<< mmse_estimated << " " << kalman_estimated << endl;

		x = state_equation(x_old, input) + sigma_s(engine_s);
		y = observation_equation(x) + sigma_w(engine_w);
		x_old = x;

		auto startTime = chrono::system_clock::now();

		pfilter.Sampling((double)input);
		pfilter.CalcLikehood(y);
		mmse_estimated = pfilter.GetMMSE();
		pfilter.Resampling();

		kalmanfilter.Calc(input, y);
		kalman_estimated = kalmanfilter.GetEstimation();

		auto endTime = chrono::system_clock::now();
		auto timeSpan = endTime - startTime;
		cout << "Iterate : " << i << "\tˆ—ŽžŠÔ : " << chrono::duration_cast<chrono::milliseconds>(timeSpan).count() << "[ms]" << flush;
		cout << "\r" << flush;

	}
	output.close();

	system("wgnuplot -persist plot.plt");

	return 0;
}

double state_equation(double x_old, double i)
{
	switch (STATE_EQN_MODE){
	case 1:
		return (1.0 / 2.0) * x_old + (25.0 * x_old) / (1.0 + pow(x_old, 2.0)) + 8.0*cos(1.2 * i); // [1]
	case 2:
		return x_old;           // [2]
	case 3:
		return x_old + 3.0 * cos(x_old / 10.0); // [3]
	case 4:
		return (1 / SYSTEM_GAIN)*i + exp(-(1 / SYSTEM_GAIN)*SAMPLING_TIME)*x_old; // [4]
	case 5:
		return (SYSTEM_GAIN*STEP_GAIN) / TIME_CONSTANT - (SYSTEM_GAIN*STEP_GAIN)*TIME_CONSTANT*exp(-SAMPLING_TIME*i / TIME_CONSTANT);
	default:
		return 0;
	}
}

double state_diff_equation(double x_old, double i)
{
	switch (STATE_EQN_MODE){
	case 1:
		return (1.0 / 2.0) - (25.0*(pow(x_old, 2.0) - 1)) / (1.0 + pow(x_old, 2.0)); // [1]
	case 2:
		return 1.0;             // [2]
	case 3:
		return 1.0 - (3.0 / 10.0)* sin(x_old / 10.0); // [3]
	case 4:
		return (-1 / pow(SYSTEM_GAIN, 2.0))*i + exp(-(1 / SYSTEM_GAIN)*SAMPLING_TIME)*x_old; // [4]
	case 5:
		return (SYSTEM_GAIN*STEP_GAIN)*exp(-SAMPLING_TIME*i / TIME_CONSTANT);
	default:
		return 0;
	}
}

double observation_equation(double x)
{
	switch (OBS_EQN_MODE){
	case 1:
		return pow(x, 2.0) / 20.0; // [1]
	case 2:
		return x;               // [2]
	case 3:
		return pow(x, 3.0);     // [3]
	case 4:
		return x;               //[4]
	case 5:
		return x;
	default:
		return 0.0;
	}
}

double observation_diff_equation(double x)
{
	switch (OBS_EQN_MODE){
	case 1:
		return x / 10.0;          // [1]
	case 2:
		return 1;               // [2]
	case 3:
		return 3.0 * pow(x, 2.0); // [3]
	case 4:
		return 1;               //[4]
	case 5:
		return 1;
	default:
		return 0.0;
	}
}