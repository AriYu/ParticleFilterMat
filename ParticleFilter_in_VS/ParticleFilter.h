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
	PStateMat() : _weight(0){}
	PStateMat(cv::Mat x) : _weight(0)
	{
		this->_state = x.clone();
	}
	PStateMat(const PStateMat& x) : _weight(x._weight)
	{
		this->_state = x._state.clone();
	}
	PStateMat(int dimX, double weight) : _weight(weight)
	{
		assert(dimX > 0);
		_state = cv::Mat_<double>(dimX, 1);
	}
	~PStateMat(){}
	cv::Mat _state;
	double _weight;
};

class ParticleFilterMat
{
public : 
	//! A : 状態遷移行列, B : 制御入力, C : 観測行列, dimX : 状態ベクトルの次元数
	ParticleFilterMat(cv::Mat A, cv::Mat B, cv::Mat C, int dimX);
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
	virtual void Sampling(void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
							const double &input);
	
	//! observed : 観測値
	virtual void CalcLikehood(double input, cv::Mat observed);
	virtual void CalcLikelihood(void(*obsmodel)(cv::Mat &z, const cv::Mat &x),
								double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov),
								const cv::Mat &observed);

	virtual void Resampling(cv::Mat observed);

	virtual cv::Mat GetMMSE();
	virtual cv::Mat GetML();

public : 
	cv::Mat _A; 
	cv::Mat _B;
	cv::Mat _C;
	cv::Mat _ProcessNoiseCov;	// process noise の共分散行列
	cv::Mat _ObsNoiseCov;		// observation noise の共分散行列
	cv::Mat _ProcessNoiseMean;	// process noise の平均
	cv::Mat _ObsNoiseMean;		// observation noise の平均
	bool _isSetProcessNoise;
	bool _isSetObsNoise;
	int _dimX;	// 状態ベクトルの次元数
	int _samples; // パーティクルの数
	std::vector< PStateMat > predict_particles;
	std::vector< PStateMat > filtered_particles;
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
  

#endif // PARTICLE_FILTER
