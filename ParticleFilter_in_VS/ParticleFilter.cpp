#include "ParticleFilter.h"
#include "mean_shift_clustering.h"

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
    : dimX_(dimX), isSetProcessNoise_(false), isSetObsNoise_(false), isResampled_(true)
{
	A_ = A.clone();
	//assert(A_.cols == dimX_);
	B_ = B.clone();
	//assert(B_.rows == dimX_);
	C_ = C.clone();
	//assert(C_.cols == dimX_);
}
ParticleFilterMat::ParticleFilterMat(int dimX)
    : dimX_(dimX), isSetProcessNoise_(false), isSetObsNoise_(false), isResampled_(true)
{
}

ParticleFilterMat::ParticleFilterMat(const ParticleFilterMat& x)
{
	this->A_ = x.A_.clone();
	this->B_ = x.B_.clone();
	this->C_ = x.C_.clone();
	this->dimX_ = x.dimX_;
	this->samples_ = x.samples_;
	this->ProcessNoiseCov_ = x.ProcessNoiseCov_.clone();
	this->ObsNoiseCov_ = x.ObsNoiseCov_.clone();
	this->ProcessNoiseMean_ = x.ProcessNoiseMean_.clone();
	this->ObsNoiseMean_ = x.ObsNoiseMean_.clone();
	this->isSetProcessNoise_ = x.isSetProcessNoise_;
	this->isSetObsNoise_ = x.isSetObsNoise_;

	//	this->indices_.resize(samples_);
	this->predict_particles.resize(x.predict_particles.size());
	std::copy(x.predict_particles.begin(),
		x.predict_particles.end(),
		this->predict_particles.begin());

	this->filtered_particles.resize(x.filtered_particles.size());
	std::copy(x.filtered_particles.begin(),
		x.filtered_particles.end(),
		this->filtered_particles.begin());

	this->last_filtered_particles.resize(x.last_filtered_particles.size());
	std::copy(x.last_filtered_particles.begin(),
		x.last_filtered_particles.end(),
		this->last_filtered_particles.begin());


}

ParticleFilterMat::~ParticleFilterMat(){}

// This func must be called after SetProcessNoise() and SetObsNoise().
void ParticleFilterMat::Init(int samples, cv::Mat initCov, cv::Mat initMean)
{
	assert(isSetObsNoise_);
	assert(isSetProcessNoise_);

	random_device rdev;
	mt19937 engine(rdev());

	samples_ = samples;
	assert(samples_ > 0);


	this->predict_particles.reserve(samples_);
	this->filtered_particles.reserve(samples_);
	this->last_filtered_particles.reserve(samples_);

	PStateMat particle(dimX_, log(1.0 / samples_));

	for (int i = 0; i < samples_; i++)
	{
		predict_particles.push_back(particle);
		filtered_particles.push_back(particle);
		last_filtered_particles.push_back(particle);
	}

	for (int j = 0; j < dimX_; j++)
	{
		normal_distribution<> sigma(initMean.at<double>(j, 0)
			, sqrt(initCov.at<double>(j, 0)));
		for (int i = 0; i < samples_; i++){
			filtered_particles[i].state_.at<double>(j, 0)
				= sigma(engine);
			filtered_particles[i].weight_ = log((1.0 / (double)samples_));
			last_filtered_particles[i].weight_ = filtered_particles[i].weight_;
			predict_particles[i].state_.at<double>(j, 0)
				= sigma(engine);
			predict_particles[i].weight_ = log((1.0 / (double)samples_));
		}
	}

	// For Viterbi Algorithm
	this->delta.resize(samples_);
	this->last_delta.resize(samples_);
}

void ParticleFilterMat::SetProcessNoise(cv::Mat Cov, cv::Mat Mean)
{
	ProcessNoiseCov_ = Cov.clone();
	ProcessNoiseCov_ = ProcessNoiseCov_;
	ProcessNoiseMean_ = Mean.clone();
	isSetProcessNoise_ = true;
}

void ParticleFilterMat::SetObservationNoise(cv::Mat Cov, cv::Mat Mean)
{
	ObsNoiseCov_ = Cov.clone();
	ObsNoiseCov_ = ObsNoiseCov_;
	ObsNoiseMean_ = Mean.clone();
	isSetObsNoise_ = true;
}

void ParticleFilterMat::Sampling(double input)
{
	static random_device rdev;
	static mt19937 engine(rdev());


	// ノイズを加える
	for (int j = 0; j < dimX_; j++)
	{
		normal_distribution<> sigma(ProcessNoiseMean_.at<double>(j, 0)
			, sqrt(ProcessNoiseCov_.at<double>(j, 0)));
		for (int i = 0; i < samples_; i++){
			predict_particles[i].state_.at<double>(j, 0)
				+= sigma(engine);
		}
	}
	// 状態遷移
	for (int i = 0; i < samples_; i++){
		filtered_particles[i].state_
                    = ((A_ * predict_particles[i].state_) + (B_ * input));
		filtered_particles[i].weight_ = predict_particles[i].weight_;
	}
	//// ノイズを加える
	//for (int j = 0; j < dimX_; j++)
	//{
	//	normal_distribution<> sigma(ProcessNoiseMean_.at<double>(j, 0)
	//		, ProcessNoiseCov_.at<double>(j, 0));
	//	for (int i = 0; i < samples_; i++){
	//		filtered_particles[i]._state.at<double>(j, 0)
	//		+= sigma(engine);
	//	}
	//}
}

void ParticleFilterMat::Sampling(
    void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, const double &input, 
						const cv::Mat &rnd),
    const double &ctrl_input)
{
    static random_device rdev;
    static mt19937 engine(rdev());

    cv::Mat rnd_num = filtered_particles[0].state_.clone();
	
    for (int i = 0; i < samples_; i++){
        for (int r = 0; r < rnd_num.rows; ++r){
            for (int c = 0; c < rnd_num.cols; ++c){
                normal_distribution<> sigma(ProcessNoiseMean_.at<double>(r, c)
                                            , sqrt(ProcessNoiseCov_.at<double>(r, c)));
                rnd_num.at<double>(r, c) = sigma(engine);
            }
        }

        processmodel(
            filtered_particles[i].state_,
            predict_particles[i].state_,
            ctrl_input,
            rnd_num);
        filtered_particles[i].weight_ = predict_particles[i].weight_;
    }
}

void ParticleFilterMat::CalcLikehood(double input, cv::Mat observed)
{
    assert(observed.rows == C_.rows);
    observed = C_.t() * observed;

    double sum = 0;
    double d = 0;
    int j = 0;
    vector<double> l(samples_); // 対数重み

    cv::Mat estimate_obs = observed.clone();
    cv::Mat estimate_error = observed.clone();


    for (int i = 0; i < samples_; i++)
    {
        l[i] = 0.0;
        //====================================
        // calculate p(y_k | x~_k)
        //====================================
        estimate_obs = C_.t() * filtered_particles[i].state_;
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
                double tmp = exp((-estimate_error.at<double>(ii, jj)) / (2.0*pow(ObsNoiseCov_.at<double>(ii, jj), 2)));
                tmp = tmp / sqrt(2.0 * CV_PI*pow(ObsNoiseCov_.at<double>(ii, jj), 2.0));
                weightsum += tmp;
                cnt += 1.0;
            }
        }
        filtered_particles[i].weight_ *= (weightsum / cnt);

        //sum = logsumexp(sum, l[i], (i == 0));
        sum += filtered_particles[i].weight_;
        //l[i] = 0;
    }

    //====================================
    // normalize weights
    //====================================
    for (int i = 0; i < samples_; i++)
    {
        //filtered_particles[i]._weight = exp(l[i] - sum);
        filtered_particles[i].weight_ = filtered_particles[i].weight_ / sum;
    }
}


void ParticleFilterMat::CalcLikelihood(
    void(*obsmodel)(cv::Mat &z, const  cv::Mat &x, const cv::Mat &rnd),
    double(*likelihood)(const cv::Mat &z, const cv::Mat &zhat, const cv::Mat &cov, 
						const cv::Mat &mean),
    const cv::Mat &observed)
{
    static random_device rdev;
    static mt19937 engine(rdev());

    double sum = 0;
    cv::Mat obshat = observed.clone();
    cv::Mat rnd_num = cv::Mat::zeros(observed.rows, observed.cols, CV_64F);
    vector<double> l(samples_); // 対数重み
    for (int i = 0; i < samples_; i++){
        obsmodel(obshat, filtered_particles[i].state_, rnd_num);
        l[i] = likelihood(observed, obshat, ObsNoiseCov_, ObsNoiseMean_);
        sum = logsumexp(sum, l[i], (i==0));
    }

    //====================================
    // normalize weights
    //====================================
    double sum2 = 0;
    for (int i = 0; i < samples_; i++)
    {
        l[i] = l[i] - sum; // Normalize weights
        filtered_particles[i].weight_ += l[i];
        sum2 = logsumexp(sum2, filtered_particles[i].weight_, (i == 0));
    }
    for (int i = 0; i < samples_; i++)
    {
        filtered_particles[i].weight_ = filtered_particles[i].weight_ - sum2;
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
void ParticleFilterMat::Resampling(cv::Mat observed, double ESSth)
{
    static random_device rdev;
    static mt19937 engine(rdev());
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    double mean = (double)(1.0 / (double)samples_);
    double ESS = 0;
    double tmp = 0;
    for (int i = 0; i < samples_; i++){
        tmp += pow(exp(filtered_particles[i].weight_), 2.0);
    }
    ESS = 1.0 / tmp;

    if ((ESS < (samples_ / ESSth))){ // do resampling
        cout << "[Resampled] ESS : " << ESS << " / " << samples_ / ESSth << endl;
        isResampled_ = true;
        // -------------- prSystematic --------------
        // int i = 0;
        // double c = exp(filtered_particles[0]._weight);
        // double r = dist(engine) / (double)samples_;
        // for (int m = 0; m < samples_; m++){
        //     double U = r + ((double)m) * mean;
        //     while (U > c){
        //         i = i + 1;
        //         c = c + exp(filtered_particles[i]._weight);
        //     }
        //     predict_particles[m]._state = filtered_particles[i]._state;
        //     predict_particles[m]._weight = log(mean);
        // }
        //---------------------------------------------------

        // -------------- prMultinomial --------------
        vector<double> linW(samples_, 0);
        double linW_SUM = 0.0;
        for (int i = 0; i < samples_; i++){
            linW_SUM += exp(filtered_particles[i].weight_);
        }
        // Normalize weights:
        assert(linW_SUM > 0);
        for (int i = 0; i < samples_; i++){
            linW[i] *= 1.0 / linW_SUM;
        }
        vector<double> Q(samples_);//累積重み
        {
            double last = 0.0;
            const size_t N = linW.size();
            for (size_t i = 0; i < N; i++){
                last = Q[i] = last + exp(filtered_particles[i].weight_);
            }
        }
        Q[samples_ - 1] = 1.1;
        vector<double> T(samples_);
        std::uniform_real_distribution<> rndm(0.0, 0.999999);
        for (int i = 0; i < samples_; i++){
            T[i] = rndm(engine);
        }
        T.push_back(1.0);
        sort(T.begin(), T.end());
        int i = 0;
        int j = 0;
        while (i < samples_)
        {
            if (T[i] < Q[j]){
                predict_particles[i].state_ = filtered_particles[j].state_;
                predict_particles[i].weight_ = log(mean);
				last_filtered_particles[i] = filtered_particles[j];
				last_delta[i] = delta[j]; // For Viterbi Algorithm.
                i++;
            }
            else{
                j++;
                if (j >= samples_){
                    j = samples_ - 1;
                }
            }
        }
        //---------------------------------------------------------
    }
    else{ // do not resampling
        isResampled_ = false;
        for (int i = 0; i < samples_; i++){
            predict_particles[i] = filtered_particles[i];
			last_filtered_particles[i] = filtered_particles[i];
			last_delta[i] = delta[i]; // For Viterbi Algorithm.
        }
    }
}



cv::Mat ParticleFilterMat::GetMMSE()
{
    cv::Mat mmse = cv::Mat_<double>(dimX_, 1);
    double tmp = 0;
    for (int j = 0; j < dimX_; j++){
        mmse.at<double>(j, 0) = 0.0;
        for (int i = 0; i < samples_; i++)
        {
            tmp = (filtered_particles[i].state_.at<double>(j, 0) 
                   * exp(filtered_particles[i].weight_));
            mmse.at<double>(j, 0) += tmp;
        }
    }
    return mmse;
}

int ParticleFilterMat::GetClusteringEstimation(std::vector< std::vector<PStateMat> > &clusters,
											   cv::Mat &est)
{
  int num_of_dimension = dimX_;
  const double sigma = sqrt(ProcessNoiseCov_.at<double>(0,0));
  const double clustering_threshold = (ProcessNoiseCov_.at<double>(0,0));
  std::vector<int> indices;
  std::vector<PStateMat> target_particles;

  // 尤度が一定値以上のパーティクルのみをクラスタリングの対象とする.
  for(int i = 0; i < samples_; i++){
	//	if(exp(filtered_particles[i].weight_) > 0.00005){
	  target_particles.push_back(filtered_particles[i]);
	  //	}
  }

  // クラスタリングする
  MeanShiftClustering cluster(target_particles, num_of_dimension, sigma);
  int num_of_cluster = cluster.Clustering(indices, clustering_threshold);

  // クラスタ数が1つだけもしくはクラスタリングに失敗したら普通にMMSEを計算する.
  cout << "num_of_cluster : " << num_of_cluster << endl;
  if(num_of_cluster == 1 || num_of_cluster == 0){
	est = GetMMSE();
	return num_of_cluster;
  }
  
  // クラスタごとに粒子を分ける
  clusters.resize(num_of_cluster);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < (int)target_particles.size(); i++){
	  if(cluster_ind == indices[i]){
		clusters[cluster_ind].push_back(target_particles[i]);
	  }
	}
  }

  // 各クラスタの重みの和とパーティクルの数を求める
  double sum_of_weight = 0;
  double sum_of_particle = 0;
  std::vector<double> cluster_prob_weight(num_of_cluster, 0.0);
  std::vector<double> cluster_prob_num(num_of_cluster, 0.0);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < (int)clusters[cluster_ind].size(); i++){
	  cluster_prob_weight[cluster_ind] += exp(clusters[cluster_ind][i].weight_);
	}
	sum_of_weight += cluster_prob_weight[cluster_ind];
	cluster_prob_num[cluster_ind] = (double)clusters[cluster_ind].size();
	sum_of_particle += cluster_prob_num[cluster_ind];
  }
  // 各クラスタのパーティクルの数と重みを正規化する．
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	cluster_prob_weight[cluster_ind] = cluster_prob_weight[cluster_ind]/sum_of_weight;
	cluster_prob_num[cluster_ind] = cluster_prob_num[cluster_ind]/sum_of_particle;
  }

  // for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  // 	std::cout << "+--------------------------------------------------------+" << std::endl;
  // 	std::cout << "cluster_prob_weight[" << cluster_ind << "]:" 
  // 			  << cluster_prob_weight[cluster_ind] << endl;
  // 	std::cout << "cluster_prob_num[" << cluster_ind << "]:" 
  // 			  << cluster_prob_num[cluster_ind] << endl;
  // 	std::cout << "connective_prob[" << cluster_ind << "]:" 
  // 			  << cluster_prob_weight[cluster_ind]*cluster_prob_num[cluster_ind] << std::endl;
  // 	std::cout << "+--------------------------------------------------------+" << std::endl;
  // }

  // 重みの和が一番高いクラスタを探す
  double maxprob_of_cluster = cluster_prob_weight[0];
  int maxprob_cluster_ind = 0;
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  	if(maxprob_of_cluster < cluster_prob_weight[cluster_ind]){
  	  maxprob_of_cluster = cluster_prob_weight[cluster_ind];
  	  maxprob_cluster_ind = cluster_ind;
  	}
  }
 
  // 重みの和が一番高いクラスタの重みを正規化する．
  // MMSEを計算する．
  cv::Mat mmse = cv::Mat::zeros(dimX_, 1, CV_64F);
  for (int i = 0; i < (int)clusters[maxprob_cluster_ind].size(); i++)
	{
	  clusters[maxprob_cluster_ind][i].weight_ 
		= exp(clusters[maxprob_cluster_ind][i].weight_) / cluster_prob_weight[maxprob_cluster_ind];
	  mmse.at<double>(0,0) += clusters[maxprob_cluster_ind][i].state_.at<double>(0,0)
		*clusters[maxprob_cluster_ind][i].weight_;
	}
  est = mmse;
  // est = clusters[maxprob_cluster_ind][num].state_;

  return num_of_cluster;
}

int ParticleFilterMat::GetClusteringEstimation2(std::vector< std::vector<PStateMat> > &clusters,
												cv::Mat &est,
												void(*processmodel)(cv::Mat &x, 
																	const cv::Mat &xpre, 
																	const double &input, 
																	const cv::Mat &rnd),
												double(*trans_likelihood)(const cv::Mat &x,
																		  const cv::Mat &xhat,
																		  const cv::Mat &cov,
																		  const cv::Mat &mean))
{
  int num_of_dimension = dimX_;
  const double sigma = sqrt(ProcessNoiseCov_.at<double>(0,0));
  //const double sigma = 1.0;
  const double clustering_threshold = (ProcessNoiseCov_.at<double>(0,0));
  //const double clustering_threshold = 10.0;
  std::vector<int> indices;
  std::vector<PStateMat> target_particles;
  std::vector<PStateMat> target_particles_pre;
  static PStateMat last_state(dimX_, 0.0);

  // 尤度が一定値以上のパーティクルのみをクラスタリングの対象とする.
  for(int i = 0; i < samples_; i++){
	if(exp(filtered_particles[i].weight_) > 0.00005){
	  target_particles.push_back(filtered_particles[i]);
	  //target_particles.push_back(predict_particles[i]);
	  target_particles_pre.push_back(last_filtered_particles[i]);
	 }
  }

  // クラスタリングする
  MeanShiftClustering cluster(target_particles, num_of_dimension, sigma);
  int num_of_cluster = cluster.Clustering(indices, clustering_threshold);

  // クラスタ数が1つだけもしくはクラスタリングに失敗したら普通にMMSEを計算する.
  cout << "num_of_cluster : " << num_of_cluster << endl;
  if(num_of_cluster == 1 || num_of_cluster == 0){
	est = GetMMSE();
	last_state.state_ = est;
	return num_of_cluster;
  }
  
  // クラスタごとに粒子を分ける
  clusters.resize(num_of_cluster);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < (int)target_particles.size(); i++){
	  if(cluster_ind == indices[i]){
		clusters[cluster_ind].push_back(target_particles[i]);
	  }
	}
  }

  // 各クラスタの母分散（variances）を求める（とりあえず一次元のみ）
  // まず平均値（means）を求める
  std::vector<double> means(num_of_cluster,0);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < clusters[cluster_ind].size(); i++){
	  means[cluster_ind] += clusters[cluster_ind][i].state_.at<double>(0,0);
	}
	means[cluster_ind] = means[cluster_ind] / (double)clusters[cluster_ind].size();
  }
  std::vector<double> variances(num_of_cluster, 0);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < clusters[cluster_ind].size(); i++){
	  variances[cluster_ind] += (pow(means[cluster_ind] 
									 - clusters[cluster_ind][i].state_.at<double>(0,0), 2.0));
	}
	variances[cluster_ind] = variances[cluster_ind] / (double)clusters[cluster_ind].size();
  }

  // 各クラスタの重みの和とパーティクルの数を求める
  double sum_of_weight = 0;   // クラスタリングしたすべてのパーティクルの重みの和
  double sum_of_particle = 0; // クラスタリングしたすべてのパーティクルの数
  std::vector<double> cluster_prob_weight(num_of_cluster, 0.0); // 各クラスタの重みの和
  std::vector<double> cluster_prob_num(num_of_cluster, 0.0); // 各クラスタのパーティクルの数
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	for(int i = 0; i < (int)clusters[cluster_ind].size(); i++){
	  cluster_prob_weight[cluster_ind] += exp(clusters[cluster_ind][i].weight_);
	}
	sum_of_weight += cluster_prob_weight[cluster_ind];
	cluster_prob_num[cluster_ind] = (double)clusters[cluster_ind].size();
	sum_of_particle += cluster_prob_num[cluster_ind];
  }
  
  // 各クラスタのMMSEを計算する.
  std::vector<cv::Mat> mmse(num_of_cluster);
  for(int i = 0; i < num_of_cluster; i++){
	mmse[i]  = cv::Mat::zeros(dimX_, 1, CV_64F);// メモリの確保
	for (int j = 0; j < (int)clusters[i].size(); j++)
	  {
		clusters[i][j].weight_ 
		  = exp(clusters[i][j].weight_) / cluster_prob_weight[i];
		for (int l = 0; l < dimX_; l++){
		  mmse[i].at<double>(l,0) += clusters[i][j].state_.at<double>(l,0)
			*clusters[i][j].weight_;
		}
	  }
  }


  // 各クラスタのパーティクルの数と重みを正規化する．
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  	cluster_prob_weight[cluster_ind] = cluster_prob_weight[cluster_ind]/sum_of_weight;
  	cluster_prob_num[cluster_ind] = cluster_prob_num[cluster_ind]/sum_of_particle;
  }


  // 1時刻前の重みの和を計算する
  std::vector<double> fxy(num_of_cluster, 0); // f( x(k) | y(k-1) )
  double sum_fxy = 0.0;
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	bool isFirst = true;
  	for(int i = 0; i < (int)target_particles.size(); i++){
  	  if(cluster_ind == indices[i]){
		double tmp = target_particles_pre[i].weight_;
		fxy[cluster_ind] = logsumexp(fxy[cluster_ind], tmp, isFirst);
		isFirst = false;
  	  }
  	}
	sum_fxy = logsumexp(sum_fxy, fxy[cluster_ind], (cluster_ind == 0));
  }
  for(int i = 0; i < num_of_cluster; i++){
  	fxy[i] = fxy[i] - sum_fxy;
  }

  // 1時刻前の推定値からの遷移確率を計算する
  std::vector<double> fxx(num_of_cluster, 0); // f( x(k) | x(k-1) )
  double sum_fxx = 0.0;
  PStateMat xhatm(last_state); // x^-(k)
  cv::Mat rnd = cv::Mat::zeros(dimX_, 1, CV_PI);
  processmodel(xhatm.state_, last_state.state_, 0, rnd);
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
	bool isFirst = true;
	for(int i = 0; i < (int)target_particles.size(); i++){
	  if(cluster_ind == indices[i]){
		double tmp = trans_likelihood(target_particles[i].state_,
									  xhatm.state_,
									  ProcessNoiseCov_,
									  ProcessNoiseMean_);
		fxx[cluster_ind] = logsumexp(fxx[cluster_ind], tmp, isFirst);
		isFirst = false;
	  }
	}
   	sum_fxx = logsumexp(sum_fxx, fxx[cluster_ind], (cluster_ind == 0));
  }
  for(int i = 0; i < num_of_cluster; i++){
  	fxx[i] = fxx[i] - sum_fxx;
  }

  // for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  // 	std::cout << "+--------------------------------------------------------+" << std::endl;
  // 	std::cout << "cluster_prob_weight[" << cluster_ind << "]:" 
  // 			  << cluster_prob_weight[cluster_ind] << endl;
  // 	std::cout << "cluster_prob_num[" << cluster_ind << "]:" 
  // 			  << cluster_prob_num[cluster_ind] << endl;
  // 	std::cout << "connective_prob[" << cluster_ind << "]:" 
  // 			  << cluster_prob_weight[cluster_ind]*cluster_prob_num[cluster_ind] << std::endl;
  // 	std::cout << "+--------------------------------------------------------+" << std::endl;
  // }


  // 重みが一番高いクラスタを探す
  double maxprob_of_cluster = cluster_prob_weight[0];
  int maxprob_cluster_ind = 0;
  for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  	cout << "cluster prob[" << cluster_ind << "] = " 
  		 << cluster_prob_weight[cluster_ind] << endl;
  	if(maxprob_of_cluster < cluster_prob_weight[cluster_ind]){
  	  maxprob_of_cluster = cluster_prob_weight[cluster_ind];
  	  maxprob_cluster_ind = cluster_ind;
  	}
  }

  //遷移確率と現時刻の重みの和が一番高いクラスタを探す
  // double maxprob_of_cluster = exp(fxx[0]) + cluster_prob_weight[0];
  // int maxprob_cluster_ind = 0;
  // for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  // 	cout << "cluster prob[" << cluster_ind << "] = " 
  // 		 << exp(fxx[cluster_ind]) + cluster_prob_weight[cluster_ind] << endl;
  // 	if(maxprob_of_cluster < (exp(fxx[cluster_ind]) + cluster_prob_weight[cluster_ind])){
  // 	  maxprob_of_cluster = exp(fxx[cluster_ind]) + cluster_prob_weight[cluster_ind];
  // 	  maxprob_cluster_ind = cluster_ind;
  // 	}
  // }


  // パーティクルの数+重みが一番多いクラスタを探索する
  // double maxprob_of_cluster = cluster_prob_weight[0]+cluster_prob_num[0];
  // int maxprob_cluster_ind = 0;
  // for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  // 	cout << "cluster prob[" << cluster_ind << "] = " 
  // 		 << cluster_prob_weight[0]+cluster_prob_num[cluster_ind] << endl;
  // 	if(maxprob_of_cluster < cluster_prob_weight[cluster_ind]+cluster_prob_num[cluster_ind]){
  // 	  maxprob_of_cluster = cluster_prob_weight[cluster_ind]+cluster_prob_num[cluster_ind];
  // 	  maxprob_cluster_ind = cluster_ind;
  // 	}
  // }


  // パーティクルの数+重み+1時刻前の重みが一番多いクラスタを探索する
  // double maxprob_of_cluster =  exp(fxx[0])+
  // 	cluster_prob_weight[0] + exp(fxy[0]);
  // int maxprob_cluster_ind = 0;
  // for(int cluster_ind = 0; cluster_ind < num_of_cluster; cluster_ind++){
  // 	cout << "cluster prob[" << cluster_ind << "] = " 
  // 	     << exp(fxx[cluster_ind]) + 
  // 	  cluster_prob_weight[cluster_ind] +exp(fxy[cluster_ind])
  // 		 << endl;
  // 	if(maxprob_of_cluster < 
  // 	   exp(fxx[cluster_ind]) +
  // 	   cluster_prob_weight[cluster_ind] + exp(fxy[cluster_ind])
  // 	   ){
  	  
  // 	  maxprob_of_cluster = 
  // 		exp(fxx[cluster_ind])  + 
  // 		cluster_prob_weight[cluster_ind] + exp(fxy[cluster_ind]);
  // 	  maxprob_cluster_ind = cluster_ind;
  // 	}
  // }

  est = mmse[maxprob_cluster_ind];  //est = mmse[minprob_cluster_ind];
  last_state.state_ = est;
  return num_of_cluster;
}

cv::Mat ParticleFilterMat::GetML()
{
	double max = filtered_particles[0].weight_;
	cv::Mat ml = cv::Mat_<double>(dimX_, 1);
	int num = 0;
	for (int i = 1; i < samples_; i++){
		if (max < filtered_particles[i].weight_){
			max = filtered_particles[i].weight_;
			num = i;
		}
	}
	for (int i = 0; i < dimX_; i++){
		ml.at < double >(i, 0) = filtered_particles[num].state_.at<double>(i, 0);
	}
	return ml;
}

double calculationESS(std::vector<PStateMat> &states)
{
  double ESS = 0;
  double tmp = 0;
  for (int i = 0; i < states.size(); i++){
	tmp += pow(exp(states[i].weight_), 2.0);
  }
  ESS = 1.0 / tmp;
  return ESS;
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

// ---------------------------------------------
// This is log sum exp problem.
// (http://tsujimotter.info/2012/07/27/logsumexp/)
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

