#ifndef PARTICLE_FILTER_K
#define PARTICLE_FILTER_K

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

#include "ParticleFilter.h"

class ParticleFilterMatK
{
public : 
	//! A : 状態遷移行列, B : 制御入力, C : 観測行列, dimX : 状態ベクトルの次元数
	ParticleFilterMatK(cv::Mat A, cv::Mat B, cv::Mat C, int dimX);
	ParticleFilterMatK(const ParticleFilterMatK& x);
	~ParticleFilterMatK();
	
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
	virtual void CalcLikelihood(void(*obsmodel)(cv::Mat &z, const cv::Mat &x, const cv::Mat &rnd),
								double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
								const cv::Mat &observed);

	virtual void Resampling(cv::Mat observed, double ESSth=16.0);

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
	bool _isResampled;
	std::vector< PStateMat > predict_particles;
	std::vector< PStateMat > filtered_particles;
};


#endif // PARTICLE_FILTER
