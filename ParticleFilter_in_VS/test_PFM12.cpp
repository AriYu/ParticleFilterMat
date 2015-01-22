///////////////////////////////////////////////
// This Program is test for ParticleFilterMat.
// nonlinear, multimodal model
// - x(k) = 0.5*x(k-1) + 25*( x(k-1) / (1.0 + x(k-1)*x(k-1)) ) + 8*cos(1.2k) +v(k)
// - y(k) = ( x(k)*x(k) ) / 20 + w(k)
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

#include "mean_shift_clustering.h"

#include "measure_time.h"

#define	PARTICLE_IO

#define NumOfIterate 10
#define NumOfParticle 1000
#define ESSth 5
using namespace std;
using namespace cv;

double       k = 0.0;		//! loop count
const double T = 50.0;          //! loop limit

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
    // z.at<double>(0, 0) = (x.at<double>(0, 0)) 
    //       + rnd.at<double>(0, 0);
}

//-----------------------------------------------------
// Observation Likelihood function
//! z    : 観測値
//! zhat : 推定観測値
//! cov  : 共分散
//! mena : 平均
double Obs_likelihood(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean)
{
    double e = 0.0 ;

    e = z.at<double>(0, 0) - zhat.at<double>(0, 0) - mean.at<double>(0, 0);
    double tmp = -(e*e) / (2.0*cov.at<double>(0, 0));
    //tmp = tmp - log(sqrt(2.0*CV_PI*cov.at<double>(0, 0)));
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
  double ave_ml = 0;
  double ave_epvgm = 0;
  double ave_pfmap = 0;
  double ave_ms = 0;
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
  cv::Mat ObsCov = (cv::Mat_<double>(1, 1) << 3.0);
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
	const int state_dimension = 1;
	ParticleFilterMat pfm( state_dimension );
	pfm.SetProcessNoise(ProcessCov, ProcessMean);
	pfm.SetObservationNoise(ObsCov, ObsMean);
	pfm.Init(NumOfParticle, initCov, initMean);

	Mat    state             = Mat::zeros(state_dimension, 1, CV_64F); /* (x) */
	Mat    last_state        = Mat::zeros(state_dimension, 1, CV_64F); /* (x) */
	Mat    processNoise      = Mat::zeros(state_dimension, 1, CV_64F);
	Mat    measurement       = Mat::zeros(1, 1, CV_64F);
	Mat    measurementNoise  = Mat::zeros(1, 1, CV_64F);
	double first_sensor      = 0.0;

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
	// Root Mean Square Error
	// ==============================
	RMSE mmse_rmse;
	RMSE epvgm_rmse;
	RMSE ml_rmse;
	RMSE pfmap_rmse;
	RMSE ms_rmse;
	RMSE obs_rmse;

	//cv::RNG rng((unsigned)time(NULL));            // random generater
	static random_device rdev;
    static mt19937 engine(rdev());
	normal_distribution<> processNoiseGen(ProcessMean.at<double>(0, 0)
										  , sqrt(ProcessCov.at<double>(0, 0)));
	normal_distribution<> obsNoiseGen(ObsMean.at<double>(0, 0)
									  , sqrt(ObsCov.at<double>(0, 0)));

	double input = 0.0;
	MeasureTime timer;

	std::cout << "\rloop == " << loop  << endl;
	for (k = 0; k < T;k += 1.0){
	  std::cout << "\rloop == " << loop << "\tk == " << k << "\r" << endl;

	  // ==============================
	  // Generate Actual Value
	  // =============================
	  //randn(processNoise, Scalar(0), Scalar::all(sqrt(ProcessCov.at<double>(0, 0))));
	  processNoise.at<double>(0,0) = processNoiseGen(engine);
	  process(state, last_state, input, processNoise);

	  // ==============================
	  // Generate Observation Value
	  // ==============================
	  // first_sensor = rng.gaussian(sqrt(ObsCov.at<double>(0, 0))) 
	  // 	+ ObsMean.at<double>(0, 0);
	  first_sensor = obsNoiseGen(engine)
		+ ObsMean.at<double>(0, 0);
	  measurementNoise.at<double>(0, 0) = first_sensor;
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
	  // timer.start();
	  // epvgm.Recursion(pfm, process, observation, 
	  // 				  Obs_likelihood, Trans_likelihood, input, measurement);
	  // timer.stop();
	  // std::cout << "EP-VGM time :" << timer.getElapsedTime() << std::endl;
	  // ==============================
	  // Particle Based MAP Process
	  // ==============================
	  // timer.start();
	  // pfmap.Update(pfm, process, observation, 
	  // 			   Obs_likelihood, Trans_likelihood, input, measurement);
	  // timer.stop();
	  // std::cout << "pf-MAP time :" << timer.getElapsedTime() << std::endl;

	  // ==============================
	  // Get Estimation
	  // ==============================
	  Mat    predictionPF    = pfm.GetMMSE();
	  double predict_x_pf    = predictionPF.at<double>(0, 0);
	  Mat    predictionEPVGM = epvgm.GetEstimation();
	  double predict_x_epvgm = 0;//predictionEPVGM.at<double>(0, 0);
	  Mat    predictionML    = pfm.GetML();
	  double predict_x_ml    = predictionML.at<double>(0, 0);
	  Mat    predictionPFMAP = pfmap.GetEstimation();
	  double predict_x_pfmap = 0;//predictionPFMAP.at<double>(0, 0);
	  // ------------------------------
	  Mat predictionMeanshiftEst = Mat::zeros(state_dimension, 1, CV_64F);
	  timer.start();
	  std::vector< std::vector<PStateMat> > clusters;
	  int num_of_cluster = pfm.GetClusteringEstimation(clusters, predictionMeanshiftEst);
	  timer.stop();
	  std::cout << "ms-PF time  :" << timer.getElapsedTime() << std::endl;
	  double predict_x_ms    = predictionMeanshiftEst.at<double>(0,0);

	  // Resampling step
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
#endif
	  // ==============================
	  // for RMSE
	  // ==============================
	  mmse_rmse.storeData(state.at<double>(0, 0), predict_x_pf);
	  epvgm_rmse.storeData(state.at<double>(0, 0), predict_x_epvgm);
	  ml_rmse.storeData(state.at<double>(0, 0), predict_x_ml);
	  pfmap_rmse.storeData(state.at<double>(0, 0), predict_x_pfmap);
	  ms_rmse.storeData(state.at<double>(0,0), predict_x_ms);
	  obs_rmse.storeData(state.at<double>(0,0), measurement.at<double>(0,0));
					
	  // ==============================
	  // Save Estimated State
	  // ==============================
	  output << state.at<double>(0, 0) << " "       // [1] true state
			 << measurement.at<double>(0, 0) << " " // [2] first sensor
			 << predict_x_pf << " "                 // [3] predicted state by PF(MMSE)
			 << predict_x_epvgm << " "              // [4] predicted state by EPVGM
			 << predict_x_pfmap << " "              // [5] predicted state by PFMAP
			 << predict_x_ml << " "                 // [6] predicted state by PF(ML)
			 << predict_x_ms << endl;               // [7] predicted state by PF(MS)
	  last_state = state;

	  cout << endl;
	  
	}

	mmse_rmse.calculationRMSE();
	epvgm_rmse.calculationRMSE();
	ml_rmse.calculationRMSE();
	pfmap_rmse.calculationRMSE();
	obs_rmse.calculationRMSE();
	ms_rmse.calculationRMSE();

	std::cout << "RMSE(MMSE)  : " << mmse_rmse.getRMSE() << endl;
	std::cout << "RMSE(MS)    : " << ms_rmse.getRMSE() << endl;
	std::cout << "RMSE(EPVGM) : " << epvgm_rmse.getRMSE() << endl;
	std::cout << "RMSE(PFMAP) : " << pfmap_rmse.getRMSE() << endl;
	std::cout << "RMSE(Obs)   : " << obs_rmse.getRMSE() << endl;
	ave_mmse  += mmse_rmse.getRMSE();
	ave_epvgm += epvgm_rmse.getRMSE();
	ave_pfmap += pfmap_rmse.getRMSE();
	ave_ml    += ml_rmse.getRMSE();
	ave_ms    += ms_rmse.getRMSE();

	output.close();
  }
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  std::cout << "nonlinear, multimodal model" << endl;
  std::cout << "Particles   : " << NumOfParticle << endl;
  std::cout << "ProcessCov  = " << ProcessCov << std::endl << std::endl;
  std::cout << "ObsCov      ="  << ObsCov << std::endl << std::endl;
  std::cout << "RMSE(MMSE)  : " << ave_mmse / (double)NumOfIterate << endl;
  std::cout << "RMSE(MS)    : " << ave_ms / (double)NumOfIterate << endl;
  std::cout << "RMSE(ML)    : " << ave_ml / (double)NumOfIterate << endl;
  std::cout << "RMSE(EPVGM) : " << ave_epvgm / (double)NumOfIterate << endl;
  std::cout << "RMSE(PFMAP) : " << ave_pfmap / (double)NumOfIterate << endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    
  //std::system("wgnuplot -persist plot4.plt");
  //std::system("gnuplot -persist plot10.plt");

  return 0;
}
