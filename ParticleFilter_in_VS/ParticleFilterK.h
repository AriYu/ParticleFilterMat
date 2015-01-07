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
	//! A : ��ԑJ�ڍs��, B : �������, C : �ϑ��s��, dimX : ��ԃx�N�g���̎�����
	ParticleFilterMatK(cv::Mat A, cv::Mat B, cv::Mat C, int dimX);
	ParticleFilterMatK(const ParticleFilterMatK& x);
	~ParticleFilterMatK();
	
	//! samples : �p�[�e�B�N���̐�, initCov : �����p�[�e�B�N���̕��U, initMean : �����p�[�e�B�N���̕���
	virtual void Init(int samples, cv::Mat initCov, cv::Mat initMean);

	//! Cov : �����U�s��, Mean : ����
	virtual void SetProcessNoise(cv::Mat Cov, cv::Mat Mean);
	
	//! Cov : �����U�s��, Mean : ����
	virtual void SetObservationNoise(cv::Mat Cov, cv::Mat Mean);
	
	//! delta_t : �T���v�����O����, input : �������
	virtual void Sampling(double input);
	virtual void Sampling(void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
							const double &input);
	
	//! observed : �ϑ��l
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
	cv::Mat _ProcessNoiseCov;	// process noise �̋����U�s��
	cv::Mat _ObsNoiseCov;		// observation noise �̋����U�s��
	cv::Mat _ProcessNoiseMean;	// process noise �̕���
	cv::Mat _ObsNoiseMean;		// observation noise �̕���
	bool _isSetProcessNoise;
	bool _isSetObsNoise;
	int _dimX;	// ��ԃx�N�g���̎�����
	int _samples; // �p�[�e�B�N���̐�
	bool _isResampled;
	std::vector< PStateMat > predict_particles;
	std::vector< PStateMat > filtered_particles;
};


#endif // PARTICLE_FILTER
