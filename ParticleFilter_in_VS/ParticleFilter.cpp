#include "ParticleFilter.h"
#include <algorithm>
using namespace std;

static int rand_init_flag = 0; // for rand

PFilter::PFilter()
{
}

PFilter::PFilter(int num, double variance1, double likehood, double(*state_eqn)(double, double), double(*obs_eqn)(double))
{
	random_device rdev;
	mt19937 engine(rdev());
	normal_distribution<> sigma(0, variance1);

	m_num = num;
	likehood_variance = likehood;
	m_state_eqn = state_eqn;
	m_obs_eqn = obs_eqn;
	m_c_particle.resize(m_num);
	m_n_particle.resize(m_num);
	for (int i = 0; i < m_num; i++){
		m_c_particle[i].x = sigma(engine);
		m_c_particle[i].weight = (double)(1.0 / m_num);
		m_n_particle[i].x = sigma(engine);
		m_n_particle[i].weight = (double)(1.0 / m_num);
	}
}

PFilter::~PFilter()
{
}

void PFilter::SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w)
{
	m_mean_s = mean_s;
	m_variance_s = variance_s;
	m_mean_w = mean_w;
	m_variance_w = variance_w;
}
void PFilter::Sampling(double delta_t)
{
	static random_device rdev;
	static mt19937 engine_s(rdev());
	static normal_distribution<> sigma_s(m_mean_s, m_variance_s);
	for (int i = 0; i < m_num; i++){
		// addition of noise
		m_n_particle[i].x = m_state_eqn(m_c_particle[i].x, delta_t) + sigma_s(engine_s);
	}
}

void PFilter::CalcLikehood(double observed)
{
	double sum = 0;
	double d = 0;
	vector<double> l(m_num); // 対数重み
	for (int i = 0; i < m_num; i++){
		d = pow(m_obs_eqn(m_n_particle[i].x) - observed, 2.0);
		l[i] = -d / (2.0*likehood_variance*likehood_variance); // - log(sqrt(2.0*CV_PI*likehood_variance)); // 式(1)のlogをとる
		//sum = m_n_particle[i].weight;
		sum = logsumexp(sum, l[i], (i == 0));
	}
	for (int i = 0; i < m_num; i++){
		m_n_particle[i].weight = exp(l[i] - sum);
	}
}

void PFilter::Resampling()
{
	static random_device rdev;
	static mt19937 engine(rdev());
	static std::uniform_real_distribution<> dist(0.0, (1.0 / (double)m_num));
	double mean = (double)(1.0 / m_num);
	double ESS = 0;
	double tmp = 0;
	for (int i = 0; i < m_num; i++){
		tmp += pow(m_n_particle[i].weight, 2.0);
	}
	ESS = 1.0 / tmp;
	if (ESS < (m_num / 2.0)){ // do resampling
		int i = 0;
		double c = m_n_particle[0].weight;
		// double r = GetRandom(0, (double)(1.0/m_num));
		double r = dist(engine);
		for (int m = 0; m < m_num; m++){
			double U = r + (m)* mean;
			while (U > c){
				i = i + 1;
				c = c + m_n_particle[i].weight;
			}
			m_c_particle[m] = m_n_particle[i];
		}
	}
	else{ // do not resampling
		for (int i = 0; i < m_num; i++){
			m_c_particle[i] = m_n_particle[i];
		}
	}
}

double PFilter::GetPredictiveX(int n)
{
	return m_c_particle[n].x;
}
double PFilter::GetPosteriorX(int n)
{
	return m_n_particle[n].x;
}
double PFilter::GetMMSE()
{
	double mmse = 0;
	for (int i = 0; i < m_num; i++){
		mmse += m_n_particle[i].x * m_n_particle[i].weight;
	}
	return mmse;
}

double PFilter::GetWeight(int n)
{
	return m_n_particle[n].weight;
}

double PFilter::GetMaxWeightX()
{
	double max = m_n_particle[0].weight;
	int num = 0;
	for (int i = 1; i < m_num; i++){
		if (max < m_n_particle[i].weight){
			max = m_n_particle[i].weight;
			num = i;
		}
	}
	return m_n_particle[num].x;
}

///////////////////////////////////////////////////
// Particle Filter Mat
///////////////////////////////////////////////////

ParticleFilterMat::ParticleFilterMat(cv::Mat A, cv::Mat B, cv::Mat C, int dimX)
    : _dimX(dimX), _isSetProcessNoise(false), _isSetObsNoise(false), _isResampled(true)
{
	_A = A.clone();
	//assert(_A.cols == _dimX);
	_B = B.clone();
	//assert(_B.rows == _dimX);
	_C = C.clone();
	//assert(_C.cols == _dimX);
}

ParticleFilterMat::ParticleFilterMat(const ParticleFilterMat& x)
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

ParticleFilterMat::~ParticleFilterMat(){}

// This func must be called after SetProcessNoise() and SetObsNoise().
void ParticleFilterMat::Init(int samples, cv::Mat initCov, cv::Mat initMean)
{
	assert(_isSetObsNoise);
	assert(_isSetProcessNoise);

	random_device rdev;
	mt19937 engine(rdev());

	_samples = samples;
	assert(_samples > 0);

	this->predict_particles.reserve(_samples);
	this->filtered_particles.reserve(_samples);
	PStateMat particle(_dimX, 1 / _samples);

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
			filtered_particles[i]._weight = (1.0 / (double)_samples);
			predict_particles[i]._state.at<double>(j, 0)
				= sigma(engine);
			predict_particles[i]._weight = (1.0 / (double)_samples);
		}
	}

	//for (int i = 0; i < _samples; i++)
	//{
	//	std::cout << "constructor filtered[" << i << "]" <<
	//		filtered_particles[i]._state
	//		<< std::endl << std::endl;
	//}

}

void ParticleFilterMat::SetProcessNoise(cv::Mat Cov, cv::Mat Mean)
{
	_ProcessNoiseCov = Cov.clone();
	_ProcessNoiseMean = Mean.clone();
	_isSetProcessNoise = true;
}

void ParticleFilterMat::SetObservationNoise(cv::Mat Cov, cv::Mat Mean)
{
	_ObsNoiseCov = Cov.clone();
	_ObsNoiseMean = Mean.clone();
	_isSetObsNoise = true;
}

void ParticleFilterMat::Sampling(double input)
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

void ParticleFilterMat::Sampling(
	void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, const cv::Mat &rnd),
	const double &ctrl_input)
{
	static random_device rdev;
	static mt19937 engine(rdev());

	//cv::Mat rnd_num = cv::Mat_<double>(_dimX, 1);
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

void ParticleFilterMat::CalcLikehood(double input, cv::Mat observed)
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


void ParticleFilterMat::CalcLikelihood(
	void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
	double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, const cv::Mat &mean),
	const cv::Mat &observed)
{
	static random_device rdev;
	static mt19937 engine(rdev());

	double sum = 0;
	cv::Mat obshat = observed.clone();
	cv::Mat rnd_num = observed.clone();

	for (int i = 0; i < _samples; i++){
		for (int j = 0; j < rnd_num.cols; j++){
			normal_distribution<> sigma(_ObsNoiseMean.at<double>(j, 0)
				, sqrt(_ObsNoiseCov.at<double>(j, 0)));
			rnd_num.at<double>(j, 0) = sigma(engine);
		}
		obsmodel(obshat, filtered_particles[i]._state, rnd_num);
		filtered_particles[i]._weight
			= filtered_particles[i]._weight*likelihood(observed, obshat, _ObsNoiseCov, _ObsNoiseMean);
		
		sum += filtered_particles[i]._weight;
	}

	//====================================
	// normalize weights
	//====================================
	for (int i = 0; i < _samples; i++)
	{
		filtered_particles[i]._weight = filtered_particles[i]._weight / sum;
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
void ParticleFilterMat::Resampling(cv::Mat observed)
{
    static random_device rdev;
    static mt19937 engine(rdev());
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    double mean = (double)(1.0 / (double)_samples);
    double ESS = 0;
    double tmp = 0;
//	double ESSth = (double)_samples / 40.0;
    //double ESSth = (double)_samples / 10.0;
    //double ESSth = 100.0;
    double ESSth = 16.0;
    for (int i = 0; i < _samples; i++){
        tmp += pow(filtered_particles[i]._weight, 2.0);
    }
    ESS = 1.0 / tmp;



    if ((ESS < (_samples / ESSth)) /*&& error_sum < 2.0*/){ // do resampling
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
            linW_SUM += filtered_particles[i]._weight;
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
                last = Q[i] = last + filtered_particles[i]._weight;
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
                predict_particles[i]._weight = mean;
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

cv::Mat ParticleFilterMat::GetMMSE()
{
	cv::Mat mmse = cv::Mat_<double>(_dimX, 1);
	double tmp = 0;
	for (int j = 0; j < _dimX; j++){
		mmse.at<double>(j, 0) = 0.0;
		for (int i = 0; i < _samples; i++)
		{
			tmp = (filtered_particles[i]._state.at<double>(j, 0) * filtered_particles[i]._weight);
			mmse.at<double>(j, 0) += tmp;
		}
	}
	return mmse;
}

cv::Mat ParticleFilterMat::GetML()
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
///////////////////////////////////////////////////
// Normal distribution random genelator
//---------------------------------------
// Box-Muller transform に基づく方法
// 二つの乱数を発生可能だが、一つしか返していない
// ---------------------------------------
// sigma : variance of Gaussian distribution
// m : baias
///////////////////////////////////////////////////
double bm_rand(double sigma, double m)
{
	static double x, y;
	static CvRNG rng = cvRNG(-1);

	x = cvRandReal(&rng);
	y = cvRandReal(&rng);

	//sigma*sqrt(-2*log(x))*(sin(2*CV_PI*y))+m;
	return sigma*sqrt(-2 * log(x))*(cos(2 * CV_PI*y)) + m;
}

double bm_rand_v2(double sigma, double m)
{
	if (rand_init_flag == 0){
		rand_init_flag = 1;
		srand((unsigned int)time(NULL));
	}
	double x = rand() / 2147483647.1; // RAND_MAX == 2147483647
	double y = rand() / 2147483647.1;

	//sigma*sqrt(-2*log(x))*(sin(2*CV_PI*y))+m;
	return sigma*sqrt(-2 * log(x))*(cos(2 * CV_PI*y)) + m;
}

double GetRandom(double min, double max)
{
	if (rand_init_flag == 0){
		rand_init_flag = 1;
		srand((unsigned int)time(NULL));
	}
	return min + (rand()*(max - min) / (RAND_MAX));
}

double logsumexp(double x, double y, bool flg)
{
	double vmin = 0;
	double vmax = 0;
	if (flg){
		return y;
	}
	if (x == y){
		return x + 0.69314718055; // log(2)
	}
	if (x <= y){
		vmin = x;
		vmax = y;
	}
	else{
		vmin = y;
		vmax = x;
	}
	if (vmax > vmin + 50.0){
		return vmax;
	}
	else{
		return vmax + log(exp(vmin - vmax) + 1.0);
	}
}

ViterbiMapEstimation::ViterbiMapEstimation(int num, double ss,
	double(*state_eqn)(double, double),
	double(*obs_eqn)(double))
{
	m_num = num;          // パーティクルの数
	m_ss = ss;           // 観測ノイズ
	m_i_t = 0;
	m_state_eqn = state_eqn;
	m_obs_eqn = obs_eqn;

	loop = 0;

	m_subparticle.resize(m_num);
	m_subdelta.resize(m_num);
	m_subphi.resize(m_num);
	savefile.open("../data/viterbi.dat", ios::out);
	if (!savefile.is_open()){ cout << "viterbi output failed" << endl; }
}

ViterbiMapEstimation::~ViterbiMapEstimation()
{
}
void ViterbiMapEstimation::SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w)
{
	m_mean_s = mean_s;
	m_variance_s = variance_s;
	m_mean_w = mean_w;
	m_variance_w = variance_w;
}

void ViterbiMapEstimation::Initialization(PFilter &pfilter, double obs_y)
{
	double g_yx = 0;
	double f_xx = 0;
	double d = 0;
	m_particles.push_back(pfilter.m_n_particle);
	m_observed_y.push_back(obs_y);
	m_delta.push_back(m_subdelta);
	m_phi.push_back(m_subphi);

	for (int i = 0; i < m_num; i++){
		m_particles[0][i].x = pfilter.GetPredictiveX(i);

		d = pow(m_obs_eqn(m_particles[0][i].x) - m_observed_y[0], 2.0);
		g_yx = -d / (2.0*m_variance_w) - log(sqrt(2.0*CV_PI*m_variance_w));

		d = pow(m_particles[0][i].x, 2);
		f_xx = -d / (2.0*m_variance_s) - log(sqrt(2.0*CV_PI*m_variance_s));

		m_delta[0][i] = f_xx + g_yx;
		m_phi[0][i] = 0;
	}
	loop += 1;
}

void ViterbiMapEstimation::Recursion(double t, PFilter &pfilter, double obs_y)
{


	double max = 0;
	double tmp = 0;
	double g_yx = 0;
	double f_xx = 0;
	double d = 0;

	vector< PState > PredictiveParticle(m_num);

	m_particles.push_back(m_subparticle);
	m_observed_y.push_back(obs_y);
	m_delta.push_back(m_subdelta);
	m_phi.push_back(m_subphi);

	loop = m_particles.size() - 1;

	for (int i = 0; i < m_num; i++){
		//PredictiveParticle[i].x = pfilter.GetPredictiveX(i);
		PredictiveParticle[i] = pfilter.m_c_particle[i];
		m_particles[loop][i] = pfilter.m_n_particle[i];
		//m_particles[t][i].x = pfilter.GetPosteriorX(i);
	}

	// for(int k = 1; k <= t; k++){
	for (int j = 0; j < m_num; j++){
		d = pow(m_obs_eqn(m_particles.back()[j].x) - m_observed_y.back(), 2.0);
		g_yx = -d / (2.0*m_variance_w) - log(sqrt(2.0*CV_PI*m_variance_w));
		for (int i = 0; i < m_num; i++){
			d = pow(m_particles[loop][j].x - m_state_eqn(m_particles[loop - 1][i].x, t), 2.0);
			//d = pow(m_particles.back()[j].x - m_state_eqn(PredictiveParticle[i].x, t), 2.0);
			f_xx = -d / (2.0*m_variance_s) - log(sqrt(2.0*CV_PI*m_variance_s));
			if (i == 0){
				max = m_delta[loop - 1][i] + f_xx;
				m_phi[loop][j] = i;
				m_delta[loop][j] = g_yx + max;
			}
			else{
				tmp = m_delta[loop - 1][i] + f_xx;
				if (tmp > max){
					max = tmp;
					m_phi[loop][j] = i;
					m_delta[loop][j] = g_yx + max;
				}
			}
		}
	}
	//}

}

double ViterbiMapEstimation::GetEstimation(double t)
{
	double max = 0;
	double tmp = 0;
	loop = m_delta.size() - 1;
	for (int i = 0; i < m_num; i++){
		if (i == 0){
			max = m_delta[loop][i];
			m_i_t = i;
		}
		else{
			//tmp = m_delta[t][i];
			if (m_delta[loop][i] > max){
				max = m_delta[loop][i];
				m_i_t = i;
			}
		}
		savefile << m_particles[loop][i].x << " " << m_delta[loop][i] << endl;
	}
	savefile << endl; savefile << endl;
	m_i.push_back(m_i_t);
	m_x_map.push_back(m_particles[loop][m_i_t].x);
	return m_particles[loop][m_i_t].x;
}


double ViterbiMapEstimation::GetEstimationAlpha(int i)
{
	double sum = 0;
	struct Dist{
		PState state;
		double dist;
		bool operator<(const Dist& right) const {
			return dist < right.dist;
		}
	};

	std::vector<Dist> subarray(m_num);
	size_t size = subarray.size();
	for (unsigned int j = 0; j < size; j++){
		subarray[j].state.x = m_particles.back()[j].x;
		subarray[j].state.weight = m_delta.back()[j];
		subarray[j].dist = fabs(m_particles.back()[m_i_t].x - m_particles.back()[j].x);
	}
	// 距離で昇順ソート
	sort(subarray.begin(), subarray.end());
	// パーティクルの1/10の数の重み付き平均を求める

	for (unsigned int j = 0; j < (unsigned int)(size / 2.0); j++){
		// cout << "[wegit][" << j << "]" << subarray[j].state.weight << endl;
		sum += subarray[j].state.weight;
	}
	double alpha_x = 0;
	//cout << "[sum] : " << sum << endl;
	for (unsigned int j = 0; j < (unsigned int)(size / 2.0); j++){
		alpha_x += subarray[j].state.x * (subarray[j].state.weight / sum);
	}
	return alpha_x;
}


void ViterbiMapEstimation::Backtracking(int t)
{
	for (int k = t - 1; t > 0; t--){
		m_i[k] = m_phi[k + 1][m_i[k + 1]];
		m_x_map[k] = m_particles[k][m_i[k]].x;
	}
}

double ViterbiMapEstimation::GetBacktrackingEstimation(int i)
{
	return m_x_map[i];
}

pfMAP::pfMAP(double(*state_eqn)(double, double), double(*obs_eqn)(double))
{
	m_state_eqn = state_eqn;
	m_obs_eqn = obs_eqn;
}

pfMAP::~pfMAP()
{
}

void pfMAP::SetNoiseParam(double mean_s, double variance_s, double mean_w, double variance_w)
{
	m_mean_s = mean_s;
	m_variance_s = variance_s;
	m_mean_w = mean_w;
	m_variance_w = variance_w;
}

void pfMAP::Initialization(PFilter &pfilter)
{
	last_pfilter = pfilter;
}

double pfMAP::GetEstimation(PFilter &pfilter, double t, double obs_y)
{
	double d = 0;
	double max = 0;
	int max_i = 0;
	unsigned size = pfilter.m_n_particle.size();
	vector<double> map(size);

	for (unsigned int i = 0; i < size; i++){
		map[i] = 0.0;
		for (unsigned int j = 0; j < size; j++){
			d = pow(m_state_eqn(last_pfilter.m_n_particle[j].x, t) - pfilter.m_n_particle[i].x, 2.0);
			// map[i] += (((exp(-1.0*d/(2.0*m_variance_s)))/sqrt(2.0*CV_PI*m_variance_s))
			//            *last_pfilter.m_n_particle[j].weight);
			map[i] += (((exp(-1.0*d / (2.0*m_variance_s))))
				*last_pfilter.m_n_particle[j].weight);
		}
		d = pow(m_obs_eqn(pfilter.m_n_particle[i].x) - obs_y, 2.0);
		// map[i] *= ((exp(-1.0*d/(2.0*m_variance_w)))/sqrt(2.0*CV_PI*m_variance_w));
		map[i] *= (exp(-1.0*d / (2.0*m_variance_w)));
	}

	max = map[0];
	max_i = 0;
	for (unsigned int i = 0; i < size; i++){
		if (map[i] > max){
			max = map[i];
			max_i = i;
		}
	}
	last_pfilter = pfilter;
	return pfilter.m_n_particle[max_i].x;
}

