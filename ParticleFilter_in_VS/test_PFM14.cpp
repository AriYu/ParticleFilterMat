///////////////////////////////////////////////
// This Program is test for ParticleFilterMat.
// x(k) = [ x(k), v(k) ]
// - x(k) = [1 Δ] x(k-1) + [ξ1(k)]
//          [0 1]          [ξ2(k)]
// - y(k) = [1 0] x(k) + ζ(k)
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

#include "unscented_kalman_filter.h"

#include "RootMeanSquareError.h"

#include "mean_shift_clustering.h"

#include "measure_time.h"

#define	PARTICLE_IO

#define NumOfIterate 1
#define NumOfParticle 1000
#define ESSth 10
using namespace std;
using namespace cv;

double       k = 0.0;		//! loop count
const double T = 500.0;          //! loop limit

const int state_dimension = 2;
const int observation_dimension = 5;


//----------------------------
// Process Equation
//! x		: state
//! xpre	: last state
//! input	: input
//! rnd		: process noise
void process(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd)
{
  // x.at<double>(0, 0) =  0.5*xpre.at<double>(0,0) 
  //   + 25.0*(xpre.at<double>(0,0) / (1.0 + (xpre.at<double>(0,0)*xpre.at<double>(0,0)))) 
  //   +  8.0 * cos(1.2*k)
  //   + rnd.at<double>(0, 0);
  //x.at<double>(0,0) = xpre.at<double>(0,0) + 3.0 * cos(xpre.at<double>(0,0)/10) + rnd.at<double>(0,0);
  x.at<double>(0, 0) = xpre.at<double>(0, 0) + 0.01*xpre.at<double>(1, 0) + rnd.at<double>(0, 0);
  x.at<double>(1, 0) = xpre.at<double>(1, 0) + rnd.at<double>(1, 0);
}


//-------------------------
// Observation Equation
//! z : 観測値
//! x : 状態ベクトル
void observation(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd)
{
  // z.at<double>(0, 0) = (x.at<double>(0, 0) * x.at<double>(0, 0)) / 20.0 
  //     + rnd.at<double>(0, 0);
  // z.at<double>(0, 0) = pow(x.at<double>(0, 0),3.0) + rnd.at<double>(0,0);
  // z.at<double>(1, 0) = pow(x.at<double>(0, 0),3.0) + rnd.at<double>(1,0);
  for(int i = 0; i < observation_dimension; i++){
	z.at<double>(i, 0) = x.at<double>(0, 0) + rnd.at<double>(i,0);
  }
  //z.at<double>(1, 0) = x.at<double>(0, 0) + rnd.at<double>(1,0);
}


//-----------------------------------------------------
// Observation Likelihood function
//! z    : 観測値
//! zhat : 推定観測値
//! cov  : 共分散
//! mena : 平均
double Obs_likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean)
{
	double sum = 0;
	std::vector<double> errors(observation_dimension, 0.0);
	std::vector<double> tmps(observation_dimension, 0.0);
	for(int i = 0; i< observation_dimension; i++){
	  errors[i] = z.at<double>(i, 0) - zhat.at<double>(i, 0) - mean.at<double>(i, 0);
	  tmps[i] = -(errors[i]*errors[i]) / (2.0 * cov.at<double>(i, 0));
	  //sum = logsumexp(sum, tmps[i], (i == 0));
	  sum += exp(tmps[i]);
	}
	sum = log(sum);
	//sum = log(exp(tmp1) + exp(tmp2));
	
    return sum;
}

//-----------------------------------------------------
// Trans Likelihood function
//! x    : 状態量
//! xhat : 推定状態量
//! cov  : 共分散
//! mena : 平均
double Trans_likelihood(const cv::Mat &x, const cv::Mat &xhat, const cv::Mat &cov, const cv::Mat &mean)
{
  static double e1, e2, tmp1, tmp2, sum = 0.0;
  e1 = x.at<double>(0, 0) - xhat.at<double>(0, 0);
  e2 = x.at<double>(1, 0) - xhat.at<double>(1, 0);
  tmp1 = -(e1*e1) / (2.0*cov.at<double>(0, 0));
  tmp2 = -(e2*e2) / (2.0*cov.at<double>(1, 0));
  sum = exp(tmp1) + exp(tmp2);
  return log(sum);
}


int main(void) {

  double ave_mmse  = 0;
  double ave_ml    = 0;
  double ave_epvgm = 0;
  double ave_pfmap = 0;
  double ave_ms    = 0;
  double ave_ukf   = 0;


  // ==============================
  // Set Process Noise
  // ==============================
  cv::Mat ProcessCov        = (cv::Mat_<double>(state_dimension, 1) << 0.05, 0.05);
  std::cout << "ProcessCov  = " << ProcessCov << std::endl << std::endl;
  cv::Mat ProcessMean       = (cv::Mat_<double>(state_dimension, 1) << 0.0, 0.0);
  std::cout << "ProcessMean = " << ProcessMean << std::endl << std::endl;
    
  // ==============================
  // Set Observation Noise
  // ==============================
  cv::Mat ObsCov        = (cv::Mat_<double>(observation_dimension, 1) 
						   << 0.05, 0.05, 0.05, 0.05, 0.05); // fifth sensor model.
  std::cout << "ObsCov  = " << ObsCov << std::endl << std::endl;
  cv::Mat ObsMean       = (cv::Mat_<double>(observation_dimension, 1) 
						   << 0.0, 0.0, 0.0, 0.0, 0.0); // fifth sensor model.
  std::cout << "ObsMean = " << ObsMean << std::endl << std::endl;

  // ==============================
  // Set Initial Particle Noise
  // ==============================
  cv::Mat initCov        = (cv::Mat_<double>(state_dimension, 1) << 3.0, 3.0);
  std::cout << "initCov  = " << initCov << std::endl << std::endl;
  cv::Mat initMean       = (cv::Mat_<double>(state_dimension, 1) << 0.0, 0.0);
  std::cout << "initMean = " << initMean << std::endl << std::endl;

  std::cout << "Particle filter mat initialized!" << endl;


  for(int loop = 0; loop < NumOfIterate; loop++){

	ofstream output;        // x, y
	output.open("result1.dat", ios::out);
	if (!output.is_open()){ std::cout << "open result output failed" << endl; return -1; }
	ofstream output_diff;        // x, y
	output_diff.open("result2.dat", ios::out);
	if (!output_diff.is_open()){ std::cout << "open result output failed" << endl; return -1; }

#ifdef PARTICLE_IO
	ofstream particles_file; // k, x, weight
	particles_file.open("result_particle.dat", ios::out);
	if (!particles_file.is_open()){ 
	  std::cout << "open result_particle output failed" << endl; return -1; }
	ofstream last_particles_file; // k, x, weight
	last_particles_file.open("result_last_particle.dat", ios::out);
	if (!last_particles_file.is_open()){ 
	  std::cout << "open result_particle output failed" << endl; return -1; }
	ofstream particles_after_file; // k, x, weight
	particles_after_file.open("result_after_particle.dat", ios::out);
	if (!particles_after_file.is_open()){ 
	  std::cout << "open result_particle output failed" << endl; return -1; }
	std::vector<ofstream> clustered_file((int)T); // x
	for(int i = 0; i < (int)T; i++){
	  string filename = "clustered_files/clustered_" + std::to_string(i) + ".dat";
	  clustered_file[i].open(filename.c_str(), ios::out);
	  if (!clustered_file[i].is_open()){ 
		std::cout << "open clustered failed" << endl; return -1; }
	}
#endif // PARTICLE_IO

	// ==============================
	// Set for Particle filter Mat
	// ==============================
	//! dimX : 状態ベクトルの次元数
	ParticleFilterMat pfm( state_dimension );
	pfm.SetProcessNoise(ProcessCov, ProcessMean);
	pfm.SetObservationNoise(ObsCov, ObsMean);
	pfm.Init(NumOfParticle, initCov, initMean);

	Mat    state             = Mat::zeros(state_dimension, 1, CV_64F); /* (x) */
	Mat    last_state        = Mat::zeros(state_dimension, 1, CV_64F); /* (x) */
	Mat    processNoise      = Mat::zeros(state_dimension, 1, CV_64F);
	Mat    measurement       = Mat::zeros(observation_dimension, 1, CV_64F);
	Mat    measurementNoise  = Mat::zeros(observation_dimension, 1, CV_64F);
	std::vector<double> sensors(observation_dimension,0.0);

	// ==============================
	// End Point Viterbi Estimation
	// ==============================
	EPViterbiMat epvgm(pfm);

	// ==============================
	// Particle based MAP Estimation
	// ==============================
	pfMapMat pfmap(pfm);
	pfmap.Initialization(pfm);

	// ==============================
	// Unscented Kalman Filter
	// ==============================
	cv::Mat x0 = (cv::Mat_<double>(state_dimension, 1) << 0.0);
	cv::Mat p0 = (cv::Mat_<double>(state_dimension, 1) << 1.0);
	UnscentedKalmanFilter ukf(state_dimension, 1, x0, p0);
	ukf.SetProcessNoise(ProcessCov);
	ukf.SetObservationNoise(ObsCov);

	// ==============================
	// Root Mean Square Error
	// ==============================
	RMSE mmse_rmse;
	RMSE epvgm_rmse;
	RMSE ml_rmse;
	RMSE pfmap_rmse;
	RMSE ms_rmse;
	RMSE obs_rmse;
	RMSE ukf_rmse;

	//cv::RNG rng((unsigned)time(NULL));            // random generater
	static random_device rdev;
    static mt19937 engine(rdev());
	normal_distribution<> processNoiseGen1(ProcessMean.at<double>(0, 0)
										  , sqrt(ProcessCov.at<double>(0, 0)));
	normal_distribution<> processNoiseGen2(ProcessMean.at<double>(1, 0)
										  , sqrt(ProcessCov.at<double>(1, 0)));

	normal_distribution<> obsNoiseGen1(ObsMean.at<double>(0, 0)
									  , sqrt(ObsCov.at<double>(0, 0)));
	normal_distribution<> obsNoiseGen2(ObsMean.at<double>(1, 0)
									  , sqrt(ObsCov.at<double>(1, 0)));
	normal_distribution<> obsNoiseGen3(ObsMean.at<double>(2, 0)
									  , sqrt(ObsCov.at<double>(2, 0)));
	normal_distribution<> obsNoiseGen4(ObsMean.at<double>(3, 0)
									  , sqrt(ObsCov.at<double>(3, 0)));
	normal_distribution<> obsNoiseGen5(ObsMean.at<double>(4, 0)
									   , sqrt(ObsCov.at<double>(4, 0)));
	

	double input = 0.0;
	MeasureTime timer;

	for (k = 0; k < T; k += 1.0){
	  std::cout << "\rloop == " << loop << " / " <<  NumOfIterate 
				<< "\tk == " << k << "\r" << endl;

	  // ==============================
	  // Generate Actual Value
	  // =============================
	  processNoise.at<double>(0, 0) = processNoiseGen1(engine);
	  processNoise.at<double>(1, 0) = processNoiseGen2(engine);
	  process(state, last_state, input, processNoise);

	  // ==============================
	  // Generate Observation Value
	  // ==============================
	  sensors[0] = obsNoiseGen1(engine)
		+ ObsMean.at<double>(0, 0);
	  sensors[1] = obsNoiseGen2(engine)
		+ ObsMean.at<double>(1, 0);
	  sensors[2] = obsNoiseGen1(engine)
		+ ObsMean.at<double>(3, 0);
	  sensors[3] = obsNoiseGen2(engine)
		+ ObsMean.at<double>(4, 0) + 1.0*sin(0.1*k);
	  sensors[4] = obsNoiseGen1(engine)
		+ ObsMean.at<double>(5, 0) + 1.0*cos(0.2*k);

	  for(int i = 0; i < observation_dimension; i++){
		measurementNoise.at<double>(i, 0) = sensors[i];
	  }
	  observation(measurement, state, measurementNoise);

	  // ==============================
	  // Particle Filter Process
	  // ==============================
	  pfm.Sampling(process, input);     
       
#ifdef PARTICLE_IO
	  for (int i = 0; i < pfm.samples_; i++){
		last_particles_file << pfm.filtered_particles[i].state_.at<double>(0, 0) << " " 
							<< exp(pfm.filtered_particles[i].weight_) << endl;
	  }
	  last_particles_file << endl; last_particles_file << endl;
#endif // PARTICLE_IO

	  pfm.CalcLikelihood(observation, Obs_likelihood, measurement);

#ifdef PARTICLE_IO
	  for (int i = 0; i < pfm.samples_; i++){
		particles_file << pfm.filtered_particles[i].state_.at<double>(0, 0) << " " 
					   << exp(pfm.filtered_particles[i].weight_) << endl;
	  }
	  particles_file << endl; particles_file << endl;
#endif // PARTICLE_IO

            
	  // ==============================
	  // EP-VGM Process
	  // ==============================
	  timer.start();
	  epvgm.Recursion(pfm, process, observation, 
	  				  Obs_likelihood, Trans_likelihood, input, measurement);
	  timer.stop();
	  std::cout << "EP-VGM time :" << timer.getElapsedTime() << std::endl;

	  // ==============================
	  // Particle Based MAP Process
	  // ==============================
	  timer.start();
	  pfmap.Update(pfm, process, observation, 
	  			   Obs_likelihood, Trans_likelihood, input, measurement);
	  timer.stop();
	  std::cout << "pf-MAP time :" << timer.getElapsedTime() << std::endl;


	  // ==============================
	  // MeanShift method
	  // ==============================
	  Mat predictionMeanshiftEst = Mat::zeros(state_dimension, 1, CV_64F);
	  timer.start();
	  std::vector< std::vector<PStateMat> > clusters;
	  // int num_of_cluster = pfm.GetClusteringEstimation(clusters, predictionMeanshiftEst);
	  // int num_of_cluster = pfm.GetClusteringEstimation2(clusters, predictionMeanshiftEst,
	  // 													process, Trans_likelihood);
	  int num_of_cluster = 0; 
	  timer.stop();
	  std::cout << "ms-PF time  :" << timer.getElapsedTime() << std::endl;
	  

	  // ==================================
	  // Unscented Kalman Filter Process
	  // ==================================
	 // ukf.Update(process, observation, measurement);
	  cv::Mat ukf_est = ukf.GetEstimation();
		  
	  // ==============================
	  // Get Estimation
	  // ==============================
	  Mat    predictionPF    = pfm.GetMMSE();
	  double predict_x_pf    = predictionPF.at<double>(0, 0);
	  Mat    predictionEPVGM = epvgm.GetEstimation(pfm);
	  double predict_x_epvgm = predictionEPVGM.at<double>(0, 0);
	  Mat    predictionML    = pfm.GetML();
	  double predict_x_ml    = predictionML.at<double>(0, 0);
	  Mat    predictionPFMAP = pfmap.GetEstimation();
	  double predict_x_pfmap = predictionPFMAP.at<double>(0, 0);
	  double predict_x_ms    = predict_x_pf;//predictionMeanshiftEst.at<double>(0,0);
	
	  //=======================================
	  // Resampling step(Particle Filter Step)
	  //=======================================
	  pfm.Resampling(measurement, ESSth);


#ifdef PARTICLE_IO
	  for (int i = 0; i < pfm.samples_; i++){
		particles_after_file << pfm.predict_particles[i].state_.at<double>(0, 0) << " " 
							 << exp(pfm.filtered_particles[i].weight_) << endl;
	  }
	  particles_after_file << endl; particles_after_file << endl;
#endif // PARTICLE_IO

#ifdef PARTICLE_IO
	  for(int cluster = 0; cluster < (int)clusters.size(); cluster++){
		for(int number = 0; number < (int)clusters[cluster].size(); number++){
		  clustered_file[k] << clusters[cluster][number].state_.at<double>(0,0)
							  << " " << 0.0 << endl;
		}
		clustered_file[k] << endl; clustered_file[k] << endl;
	  }
#endif // PARTICLE_IO

	  // ==============================
	  // for RMSE
	  // ==============================
	  mmse_rmse.storeData(state.at<double>(0, 0), predict_x_pf);
	  epvgm_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm);
	  ml_rmse.storeData(state.at<double>(0, 0), predict_x_ml);
	  pfmap_rmse.storeData(state.at<double>(0, 0), predict_x_pfmap);
	  ms_rmse.storeData(state.at<double>(0,0), predict_x_ms);
	  obs_rmse.storeData(state.at<double>(0,0), measurement.at<double>(0,0));
	  ukf_rmse.storeData(state.at<double>(0,0), ukf_est.at<double>(0,0));

	  // ==============================
	  // Save Estimated State
	  // ==============================
	  output << state.at<double>(0, 0) << " "       // [1] true state (x)
			 << measurement.at<double>(0, 0) << " " // [2] first sensor
			 << predict_x_pf << " "                 // [3] Est PF(MMSE)  (x)
			 << predict_x_epvgm << " "              // [4] Est PF(EP-VGM) (x)
			 << predict_x_pfmap << " "              // [5] Est PF(pf-MAP) (x)
			 << predict_x_ml << " "                 // [6] Est PF(Max weight) (x)
			 << predict_x_ms << " "                 // [7] Est PF(MS) (x)
			 << measurement.at<double>(1, 0) << " " // [8] second sensor
			 << measurement.at<double>(2, 0) << " " // [9] third sensor
			 << measurement.at<double>(3, 0) << " " // [10] forth sensor
			 << measurement.at<double>(4, 0) << " " // [11] fifth sensor
			 << state.at<double>(1,0) << " "        // [12] true state (v)
			 << predictionPF.at<double>(1, 0) << " "    // [13] Est PF(MMSE)  (v)
			 << predictionEPVGM.at<double>(1, 0) << " " // [14] Est PF(EP-VGM) (v)
			 << predictionPFMAP.at<double>(1, 0) << " " // [15] Est PF(pf-MAP) (v)
			 << predictionML.at<double>(1, 0) << " "    // [16] Est PF(Max weight) (v)
			 << predictionMeanshiftEst.at<double>(1,0)  // [17] Est PF(MS) (v)
			 << endl; 
	  output_diff << state.at<double>(0, 0) - predict_x_pf << " "
				  << state.at<double>(0, 0) - predict_x_ms << endl;
	  last_state = state;

	  cout << endl;
	  
	}

	mmse_rmse.calculationRMSE();
	epvgm_rmse.calculationRMSE();
	ml_rmse.calculationRMSE();
	pfmap_rmse.calculationRMSE();
	obs_rmse.calculationRMSE();
	ms_rmse.calculationRMSE();
	ukf_rmse.calculationRMSE();

	std::cout << "---------------------------------------------" << std::endl;
	std::cout << "RMSE(MMSE)  : " << mmse_rmse.getRMSE()  << endl;
	std::cout << "RMSE(MS)    : " << ms_rmse.getRMSE()    << endl;
	std::cout << "RMSE(EPVGM) : " << epvgm_rmse.getRMSE() << endl;
	std::cout << "RMSE(PFMAP) : " << pfmap_rmse.getRMSE() << endl;
	std::cout << "RMSE(UKF)   : " << ukf_rmse.getRMSE()   << endl;
	std::cout << "RMSE(Obs)   : " << obs_rmse.getRMSE()   << endl;
	std::cout << "---------------------------------------------" << std::endl;

	ave_mmse  += mmse_rmse.getRMSE();
	ave_epvgm += epvgm_rmse.getRMSE();
	ave_pfmap += pfmap_rmse.getRMSE();
	ave_ml    += ml_rmse.getRMSE();
	ave_ms    += ms_rmse.getRMSE();
	ave_ukf   += ukf_rmse.getRMSE();


	output.close();
  }

  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  std::cout << "nonlinear, multi sensor model" << std::endl;
  std::cout << "- x(k) = x(k-1) + v(k)"        << std::endl;
  std::cout << "- y(k) = x(k) + w(k) "         << std::endl;
  std::cout << "Particles   : " << NumOfParticle << std::endl;
  std::cout << "ESS th      : " << ESSth         << std::endl;
  std::cout << "ProcessCov  = " << ProcessCov    << std::endl << std::endl;
  std::cout << "ObsCov      ="  << ObsCov        << std::endl << std::endl;
  std::cout << "RMSE(MMSE)  : " << ave_mmse  / (double)NumOfIterate << endl;
  std::cout << "RMSE(MS)    : " << ave_ms    / (double)NumOfIterate << endl;
  std::cout << "RMSE(ML)    : " << ave_ml    / (double)NumOfIterate << endl;
  std::cout << "RMSE(EPVGM) : " << ave_epvgm / (double)NumOfIterate << endl;
  std::cout << "RMSE(PFMAP) : " << ave_pfmap / (double)NumOfIterate << endl;
  std::cout << "RMSE(UKF)   : " << ave_ukf   / (double)NumOfIterate << endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

  //std::system("wgnuplot -persist plot4.plt");
  //std::system("gnuplot -persist plot10.plt");

  return 0;
}
