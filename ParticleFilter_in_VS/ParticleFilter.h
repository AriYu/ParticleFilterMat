#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <new>

#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>



double bm_rand(double sigma, double m);
double bm_rand_v2(double sigma, double m);
double GetRandom(double min, double max);
double logsumexp(double x, double y, bool flg);


typedef struct{
  double x;
  double weight;
} PState;

class PStateMat
{
 public : 
  // x : 状態ベクトル
 PStateMat() : weight_(0){}
 PStateMat(cv::Mat x) : weight_(0)
	{
	  this->state_ = x.clone();
	}
 PStateMat(const PStateMat& x) : weight_(x.weight_)
	{
	  this->state_ = x.state_.clone();
	}
 PStateMat(int dimX, double weight) : weight_(weight)
  {
	assert(dimX > 0);
	//state_ = cv::Mat_<double>(dimX, 1);
	state_ = cv::Mat::zeros(dimX, 1, CV_64F);
  }
  ~PStateMat(){}
  cv::Mat state_;
  double weight_;
};

class ParticleFilterMat
{
 public : 
  //! A : 状態遷移行列, B : 制御入力, C : 観測行列, dimX : 状態ベクトルの次元数
  ParticleFilterMat(cv::Mat A, cv::Mat B, cv::Mat C, int dimX);
  ParticleFilterMat(int dimX);
  ParticleFilterMat(const ParticleFilterMat& x);
  ~ParticleFilterMat();
	
  //! samples : パーティクルの数, initCov : 初期パーティクルの分散, initMean : 初期パーティクルの平均
  virtual void Init(int samples, cv::Mat initCov, cv::Mat initMean);

  //! Cov : 共分散行列, Mean : 平均
  virtual void SetProcessNoise(cv::Mat Cov, cv::Mat Mean);
	
  //! Cov : 共分散行列, Mean : 平均
  virtual void SetObservationNoise(cv::Mat Cov, cv::Mat Mean);
	
  //! delta_t : サンプリング時間, input : 制御入力
  virtual void Sampling(double input);
  virtual void Sampling(void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, 
											const double &input, const cv::Mat &rnd),
						const double &input);
	
  //! observed : 観測値
  virtual void CalcLikehood(double input, cv::Mat observed);
  virtual void CalcLikelihood(void(*obsmodel)(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd),
							  double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, 
												  const cv::Mat &cov, const cv::Mat &mean),
							  const cv::Mat &observed);

  virtual void Resampling(cv::Mat observed, double ESSth=16.0);

  virtual cv::Mat GetMMSE();
  virtual cv::Mat GetML();
  int GetClusteringEstimation(std::vector< std::vector<PStateMat> > &clusters, cv::Mat &est);
  int GetClusteringEstimation2(std::vector< std::vector<PStateMat> > &clusters,
							   cv::Mat &est,
							   void(*processmodel)(cv::Mat &x, 
												   const cv::Mat &xpre, 
												   const double &input, 
												   const cv::Mat &rnd),
							   double(*trans_likelihood)(const cv::Mat &x,
														 const cv::Mat &xhat,
														 const cv::Mat &cov,
														 const cv::Mat &mean));
  int GetClusteringEstimation3(std::vector< std::vector<PStateMat> > &clusters,
							   cv::Mat &est,
							   void(*processmodel)(cv::Mat &x, 
												   const cv::Mat &xpre, 
												   const double &input, 
												   const cv::Mat &rnd),
							   double(*trans_likelihood)(const cv::Mat &x,
														 const cv::Mat &xhat,
														 const cv::Mat &cov,
														 const cv::Mat &mean),
							   double sigma, double cls_th);
  double density(PStateMat x, PStateMat xhat);
  int KernelDensityEstimation( cv::Mat &est,
							   std::vector<double> &densities,
							   std::vector<double> &maps,
							   void(*processmodel)(cv::Mat &x, 
												   const cv::Mat &xpre, 
												   const double &input, 
												   const cv::Mat &rnd),
							   double(*trans_likelihood)(const cv::Mat &x,
														 const cv::Mat &xhat,
														 const cv::Mat &cov,
														 const cv::Mat &mean));
 public : 
  cv::Mat A_; 
  cv::Mat B_;
  cv::Mat C_;
  cv::Mat ProcessNoiseCov_;	// process noise の共分散行列
  cv::Mat ObsNoiseCov_;		// observation noise の共分散行列
  cv::Mat ProcessNoiseMean_;	// process noise の平均
  cv::Mat ObsNoiseMean_;		// observation noise の平均
  int dimX_;	// 状態ベクトルの次元数
  bool isSetProcessNoise_;
  bool isSetObsNoise_;
  bool isResampled_;
  int samples_; // パーティクルの数
  std::vector< double > likelihoods_;
  std::vector< PStateMat > predict_particles;
  std::vector< PStateMat > filtered_particles;
  std::vector< PStateMat > last_filtered_particles;


  // For Viterbi Algorithm
  std::vector<double> delta;
  std::vector<double> last_delta;
};

class PFilter
{
 public :
  PFilter();
  PFilter(int num, double variance1, double likehood, double (*state_eqn)(double, double), double (*obs_eqn)(double));
  ~PFilter();
  virtual void SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w);
  virtual void Sampling(double delta_t);
  virtual void CalcLikehood(double observed);
  virtual void Resampling();
  virtual double GetPredictiveX(int n);
  virtual double GetPosteriorX(int n);
  virtual double GetWeight(int n);
  virtual double GetMMSE();
  virtual double GetMaxWeightX();
  std::vector< PState > m_c_particle; // Predictive particles
  std::vector< PState > m_n_particle; // Filtered particles
 protected :
  int m_num; // Num of particle
  double m_mean_s; // average of system noize
  double m_variance_s; // variance of system noize
  double m_mean_w; // average of observation noize
  double m_variance_w; // variance of observation noize
  double (*m_state_eqn)(double, double); // state equation function pointer
  double (*m_obs_eqn)(double); // observation equation function pointer
  double likehood_variance;
};

class ViterbiMapEstimation
{
 public:
  ViterbiMapEstimation(int num, double ss, double (*state_eqn)(double, double), double (*obs_eqn)(double));
  ~ViterbiMapEstimation();
  //void test();
  virtual void SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w);
  virtual void Initialization(PFilter &pfilter, double obs_y);
  virtual void Recursion(double t, PFilter &pfilter, double obs_y);
  virtual double GetEstimation(double t);
  virtual double GetEstimationAlpha(int t);
  virtual void Backtracking(int t);
  virtual double GetBacktrackingEstimation(int i);

 protected:
  int m_num;
  int loop;
  double m_ss;
  int m_i_t;
  std::vector<PState> m_subparticle;
  std::vector<double> m_observed_y;
  std::vector<double> m_subdelta;
  std::vector<int> m_subphi;

  std::vector< std::vector<PState> > m_particles;
  std::vector< std::vector<double> > m_delta;
  std::vector< std::vector<int> > m_phi;

  std::vector< double > m_x_map;
  std::vector< int > m_i;

  double m_mean_s; // average of system noize
  double m_variance_s; // variance of system noize
  double m_mean_w; // average of observation noize
  double m_variance_w; // variance of observation noize
  double (*m_state_eqn)(double, double); // state equation function pointer
  double (*m_obs_eqn)(double); // observation equation function pointer

  std::fstream savefile;
};

class pfMAP
{
 public:
  pfMAP(double (*state_eqn)(double, double), double (*obs_eqn)(double));
  ~pfMAP();
  virtual void SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w);
  virtual void Initialization(PFilter &pfilter);
  virtual double GetEstimation(PFilter &pfilter, double t, double obs_y);

 protected:
  PFilter last_pfilter;

  double m_mean_s; // average of system noize
  double m_variance_s; // variance of system noize
  double m_mean_w; // average of observation noize
  double m_variance_w; // variance of observation noize

  double (*m_state_eqn)(double, double); // state equation function pointer
  double (*m_obs_eqn)(double); // observation equation function pointer

};
  
double calculationESS(std::vector<PStateMat> &states);

#endif // PARTICLE_FILTER
