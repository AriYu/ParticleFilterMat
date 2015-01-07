#include "ParticleFilterK.h"
#include "k_means_pp.h"

#include <algorithm>
using namespace std;

static int rand_init_flag = 0; // for rand


///////////////////////////////////////////////////
// Particle Filter Mat
///////////////////////////////////////////////////

ParticleFilterMatK::ParticleFilterMatK(cv::Mat A, cv::Mat B, cv::Mat C, int dimX)
    : _dimX(dimX), _isSetProcessNoise(false), _isSetObsNoise(false), _isResampled(true)
{
	_A = A.clone();
	//assert(_A.cols == _dimX);
	_B = B.clone();
	//assert(_B.rows == _dimX);
	_C = C.clone();
	//assert(_C.cols == _dimX);
}

ParticleFilterMatK::ParticleFilterMatK(const ParticleFilterMatK& x)
{
	this->_A = x._A.clone();
	this->_B = x._B.clone();
	this->_C = x._C.clone();
	this->_dimX = x._dimX;
	this->_samples = x._samples;
	this->_ProcessNoiseCov = x._ProcessNoiseCov.clone();
	this->_ObsNoiseCov = x._ObsNoiseCov.clone();
	this->_ProcessNoiseMean = x._ProcessNoiseMean.clone();
	this->_ObsNoiseMean = x._ObsNoiseMean.clone();
	this->_isSetProcessNoise = x._isSetProcessNoise;
	this->_isSetObsNoise = x._isSetObsNoise;

	this->predict_particles.resize(x.predict_particles.size());
	std::copy(x.predict_particles.begin(),
		x.predict_particles.end(),
		this->predict_particles.begin());

	this->filtered_particles.resize(x.filtered_particles.size());
	std::copy(x.filtered_particles.begin(),
		x.filtered_particles.end(),
		this->filtered_particles.begin());

}

ParticleFilterMatK::~ParticleFilterMatK(){}

// This func must be called after SetProcessNoise() and SetObsNoise().
void ParticleFilterMatK::Init(int samples, cv::Mat initCov, cv::Mat initMean)
{
	assert(_isSetObsNoise);
	assert(_isSetProcessNoise);

	random_device rdev;
	mt19937 engine(rdev());

	_samples = samples;
	assert(_samples > 0);

	this->predict_particles.reserve(_samples);
	this->filtered_particles.reserve(_samples);
	PStateMat particle(_dimX, log(1.0 / _samples));

	for (int i = 0; i < _samples; i++)
	{
		predict_particles.push_back(particle);
		filtered_particles.push_back(particle);
	}

	for (int j = 0; j < _dimX; j++)
	{
		normal_distribution<> sigma(initMean.at<double>(j, 0)
			, sqrt(initCov.at<double>(j, 0)));
		for (int i = 0; i < _samples; i++){
			filtered_particles[i]._state.at<double>(j, 0)
				= sigma(engine);
			filtered_particles[i]._weight = log((1.0 / (double)_samples));
			predict_particles[i]._state.at<double>(j, 0)
				= sigma(engine);
			predict_particles[i]._weight = log((1.0 / (double)_samples));
		}
	}

}

void ParticleFilterMatK::SetProcessNoise(cv::Mat Cov, cv::Mat Mean)
{
	_ProcessNoiseCov = Cov.clone();
	_ProcessNoiseMean = Mean.clone();
	_isSetProcessNoise = true;
}

void ParticleFilterMatK::SetObservationNoise(cv::Mat Cov, cv::Mat Mean)
{
	_ObsNoiseCov = Cov.clone();
	_ObsNoiseMean = Mean.clone();
	_isSetObsNoise = true;
}

void ParticleFilterMatK::Sampling(double input)
{
	static random_device rdev;
	static mt19937 engine(rdev());


	// ノイズを加える
	for (int j = 0; j < _dimX; j++)
	{
		normal_distribution<> sigma(_ProcessNoiseMean.at<double>(j, 0)
			, sqrt(_ProcessNoiseCov.at<double>(j, 0)));
		for (int i = 0; i < _samples; i++){
			predict_particles[i]._state.at<double>(j, 0)
				+= sigma(engine);
		}
	}
	// 状態遷移
	for (int i = 0; i < _samples; i++){
		filtered_particles[i]._state
			= ((_A * predict_particles[i]._state) + (_B * input));
		filtered_particles[i]._weight = predict_particles[i]._weight;
	}
	//// ノイズを加える
	//for (int j = 0; j < _dimX; j++)
	//{
	//	normal_distribution<> sigma(_ProcessNoiseMean.at<double>(j, 0)
	//		, _ProcessNoiseCov.at<double>(j, 0));
	//	for (int i = 0; i < _samples; i++){
	//		filtered_particles[i]._state.at<double>(j, 0)
	//		+= sigma(engine);
	//	}
	//}
}

void ParticleFilterMatK::Sampling(
    void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
    const double &ctrl_input)
{
    static random_device rdev;
    static mt19937 engine(rdev());

    cv::Mat rnd_num = filtered_particles[0]._state.clone();
	
    for (int i = 0; i < _samples; i++){
        for (int r = 0; r < rnd_num.rows; ++r){
            for (int c = 0; c < rnd_num.cols; ++c){
                normal_distribution<> sigma(_ProcessNoiseMean.at<double>(r, c)
                                            , sqrt(_ProcessNoiseCov.at<double>(r, c)));
                rnd_num.at<double>(r, c) = sigma(engine);
            }
        }

        processmodel(
            filtered_particles[i]._state,
            predict_particles[i]._state,
            ctrl_input,
            rnd_num);
        filtered_particles[i]._weight = predict_particles[i]._weight;
    }
}

void ParticleFilterMatK::CalcLikehood(double input, cv::Mat observed)
{
    assert(observed.rows == _C.rows);
    observed = _C.t() * observed;

    double sum = 0;
    double d = 0;
    int j = 0;
    vector<double> l(_samples); // 対数重み

    cv::Mat estimate_obs = observed.clone();
    cv::Mat estimate_error = observed.clone();


    for (int i = 0; i < _samples; i++)
    {
        l[i] = 0.0;
        //====================================
        // calculate p(y_k | x~_k)
        //====================================
        estimate_obs = _C.t() * filtered_particles[i]._state;
        estimate_error = observed - estimate_obs;
        for (int ii = 0; ii < estimate_error.rows; ii++)
        {
            for (int jj = 0; jj < estimate_error.cols; jj++)
            {
                estimate_error.at<double>(ii, jj) = pow(estimate_error.at<double>(ii, jj), 2.0);
            }
        }
        double weightsum = 0.0;
        double cnt = 0.0;
        for (int ii = 0; ii < estimate_error.rows; ii++)
        {
            for (int jj = 0; jj < estimate_error.cols; jj++)
            {
                double tmp = exp((-estimate_error.at<double>(ii, jj)) / (2.0*pow(_ObsNoiseCov.at<double>(ii, jj), 2)));
                tmp = tmp / sqrt(2.0 * CV_PI*pow(_ObsNoiseCov.at<double>(ii, jj), 2.0));
                weightsum += tmp;
                cnt += 1.0;
            }
        }
        filtered_particles[i]._weight *= (weightsum / cnt);

        //sum = logsumexp(sum, l[i], (i == 0));
        sum += filtered_particles[i]._weight;
        //l[i] = 0;
    }

    //====================================
    // normalize weights
    //====================================
    for (int i = 0; i < _samples; i++)
    {
        //filtered_particles[i]._weight = exp(l[i] - sum);
        filtered_particles[i]._weight = filtered_particles[i]._weight / sum;
    }
}


void ParticleFilterMatK::CalcLikelihood(
    void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
    double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
    const cv::Mat &observed)
{
    static random_device rdev;
    static mt19937 engine(rdev());

    double sum = 0;
    cv::Mat obshat = observed.clone();
    cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
    vector<double> l(_samples); // 対数重み
    for (int i = 0; i < _samples; i++){
        obsmodel(obshat, filtered_particles[i]._state, rnd_num);
        l[i] = likelihood(observed, obshat, _ObsNoiseCov, _ObsNoiseMean);
        sum = logsumexp(sum, l[i], (i==0));
    }

    //====================================
    // normalize weights
    //====================================
    double sum2 = 0;
    for (int i = 0; i < _samples; i++)
    {
        l[i] = l[i] - sum; // Normalize weights
        filtered_particles[i]._weight += l[i];
        sum2 = logsumexp(sum2, filtered_particles[i]._weight, (i == 0));
    }
    for (int i = 0; i < _samples; i++)
    {
        filtered_particles[i]._weight = filtered_particles[i]._weight - sum2;
    }

}

/*
//---------------------------------------------------------------------------
// ESS(Effective sample size)について
// 例）
// すべてのパーティクルの重みが均等だった場合：
//		All Particle weight = 1/N -> ESS = N
// ある一つのパーティクルの重みが1で他が0の場合：
//		ESS = 1
// つまり、ESSが小さいほど縮退が起きている。縮退とはパーティクルが散らばって疎になっている状態。
// 縮退が起きているときはリサンプリングを行って尤度の高いパーティクルを
// 増やす操作を行う。
//----------------------------------------------------------------------------
*/
void ParticleFilterMatK::Resampling(cv::Mat observed, double ESSth)
{
    static random_device rdev;
    static mt19937 engine(rdev());
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    double mean = (double)(1.0 / (double)_samples);
    double ESS = 0;
    double tmp = 0;
    for (int i = 0; i < _samples; i++){
        tmp += pow(exp(filtered_particles[i]._weight), 2.0);
    }
    ESS = 1.0 / tmp;

    if ((ESS < (_samples / ESSth))){ // do resampling
        cout << "[Resampled] ESS : " << ESS << " / " << _samples / ESSth << endl;
        _isResampled = true;
        // -------------- prSystematic --------------
        /*int i = 0;
          double c = filtered_particles[0]._weight;
          double r = dist(engine) / (double)_samples;
          for (int m = 0; m < _samples; m++){
          double U = r + ((double)m) * mean;
          while (U > c){
          i = i + 1;
          c = c + filtered_particles[i]._weight;
          }
          predict_particles[m]._state = filtered_particles[i]._state;
          predict_particles[m]._weight = mean;
          }*/
        //---------------------------------------------------

        // -------------- prMultinomial --------------
        vector<double> linW(_samples, 0);
        double linW_SUM = 0.0;
        for (int i = 0; i < _samples; i++){
            linW_SUM += exp(filtered_particles[i]._weight);
        }
        // Normalize weights:
        assert(linW_SUM > 0);
        for (int i = 0; i < _samples; i++){
            linW[i] *= 1.0 / linW_SUM;
        }
        vector<double> Q(_samples);//累積重み
        {
            double last = 0.0;
            const size_t N = linW.size();
            for (size_t i = 0; i < N; i++){
                last = Q[i] = last + exp(filtered_particles[i]._weight);
            }
        }
        Q[_samples - 1] = 1.1;
        vector<double> T(_samples);
        std::uniform_real_distribution<> rndm(0.0, 0.999999);
        for (int i = 0; i < _samples; i++){
            T[i] = rndm(engine);
        }
        T.push_back(1.0);
        sort(T.begin(), T.end());
        int i = 0;
        int j = 0;
        while (i < _samples)
        {
            if (T[i] < Q[j]){
                predict_particles[i]._state = filtered_particles[j]._state;
                predict_particles[i]._weight = log(mean);
                i++;
            }
            else{
                j++;
                if (j >= _samples){
                    j = _samples - 1;
                }
            }
        }
        //---------------------------------------------------------
    }
    else{ // do not resampling
        _isResampled = false;
        for (int i = 0; i < _samples; i++){
            predict_particles[i] = filtered_particles[i];
        }
    }
}



cv::Mat ParticleFilterMatK::GetMMSE()
{
    cv::Mat mmse = cv::Mat_<double>(_dimX, 1);
    double tmp = 0;
    for (int j = 0; j < _dimX; j++){
        mmse.at<double>(j, 0) = 0.0;
        for (int i = 0; i < _samples; i++)
        {
            tmp = (filtered_particles[i]._state.at<double>(j, 0) 
                   * exp(filtered_particles[i]._weight));
            mmse.at<double>(j, 0) += tmp;
        }
    }
    return mmse;
}

cv::Mat ParticleFilterMatK::GetML()
{
	double max = filtered_particles[0]._weight;
	cv::Mat ml = cv::Mat_<double>(_dimX, 1);
	int num = 0;
	for (int i = 1; i < _samples; i++){
		if (max < filtered_particles[i]._weight){
			max = filtered_particles[i]._weight;
			num = i;
		}
	}
	for (int i = 0; i < _dimX; i++){
		ml.at < double >(i, 0) = filtered_particles[num]._state.at<double>(i, 0);
	}
	return ml;
}
