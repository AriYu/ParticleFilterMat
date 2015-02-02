#include "unscented_kalman_filter.h"

using namespace std;

UnscentedKalmanFilter::UnscentedKalmanFilter(int dimX, int dimY, cv::Mat &x0, cv::Mat &p0)
  : dimX_(dimX), dimY_(dimY)
{
  this->xhat_ = x0.clone();
  this->P_ = p0.clone();
  this->xhat_new_ = x0.clone();
  this->P_new_ = x0.clone();
  this->G_ = P_.clone();
}

UnscentedKalmanFilter::UnscentedKalmanFilter(const UnscentedKalmanFilter& x)
{
  this->dimX_			 = x.dimX_;
  this->dimY_            = x.dimY_;
  this->ProcessNoiseCov_ = x.ProcessNoiseCov_.clone();
  this->ObsNoiseCov_	 = x.ObsNoiseCov_.clone();
  this->xhat_			 = x.xhat_.clone();
  this->xhat_new_		 = x.xhat_new_.clone();
  this->P_				 = x.P_.clone();
  this->P_new_			 = x.P_new_.clone();
  this->G_				 = x.G_.clone();
}

UnscentedKalmanFilter::~UnscentedKalmanFilter()
{
}

void UnscentedKalmanFilter::SetProcessNoise(cv::Mat &Cov)
{
  this->ProcessNoiseCov_ = Cov.clone();
}

void UnscentedKalmanFilter::SetObservationNoise(cv::Mat &Cov)
{
  this->ObsNoiseCov_ = Cov.clone();
}

//----------------------------------------
// Pはn x nの正定値対称行列を仮定している
void UnscentedKalmanFilter::Cholesky(cv::Mat &P, cv::Mat &S)
{
  for(int i = 0; i < P.rows; i++){
	double square_sum = 0;
	for(int j = 0; j < i; j++){
	  square_sum += pow(S.at<double>(i, j), 2.0);
	}
	S.at<double>(i, i) = sqrt(P.at<double>(i, i) - square_sum);
	for(int j = 0; j < P.cols; j++){
	  if(j < i){
		S.at<double>(j, i) = 0;
	  }else if (j > i){
		double sum = 0;
		for(int k = 0; k < i; k++){
		  sum += S.at<double>(j, k) * S.at<double>(i, k);
		}
		S.at<double>(j, i) = (P.at<double>(j, i) - sum) / S.at<double>(i, i);
	  }
	}
  }
}


void UnscentedKalmanFilter::Update(void(*processmodel)(cv::Mat &x, const cv::Mat &xpre, 
													   const double &input,const cv::Mat &processrnd),
								   void(*obsmodel)(cv::Mat &z, const cv::Mat &x,const cv::Mat &obsrnd),
								   cv::Mat &observation)
{
  //----------------------------------------------------------
  // 1時刻前の状態推定値と共分散行列を使ってシグマポイントを計算
  //----------------------------------------------------------
  // シグマポイントの計算
  double kappa = 3.0 - (double)dimX_; // スケーリングパラメータ
  std::vector<double> w(2*dimX_ + 1, 0.0);
  w[0] = kappa / ((double)dimX_ + kappa); // 重み
  double sum_weight = w[0];
  for(int i = 1; i < (int)w.size(); i++){
	w[i] = 1.0/(2.0*(dimX_+kappa));
	sum_weight += w[i];
  }

 
  // シグマポイントの生成
  std::vector<cv::Mat> sigma_points(2*dimX_ + 1);
  for(int i = 0; i < (int)sigma_points.size(); i++){
	sigma_points[i] = cv::Mat_<double>(dimX_, 1);  // シグマポイントのメモリ確保
  }

  // 共分散行列をコレスキー分解
  cv::Mat rootP = P_.clone(); // メモリの確保
  Cholesky(P_, rootP); // コレスキー分解
  // シグマポイントの計算
  sigma_points[0] = xhat_;
  for(int i = 1; i <= dimX_; i++){
	sigma_points[i]       = xhat_ + sqrt(dimX_ + kappa)*rootP.col(i-1);
	sigma_points[i+dimX_] = xhat_ - sqrt(dimX_ + kappa)*rootP.col(i-1);
  }

  //------------------------------
  // 予測ステップ
  //------------------------------
  // シグマポイントの更新
  for(int i = 0; i < (int)sigma_points.size(); i++){
	double input = 0;
	cv::Mat process_rnd_num = cv::Mat::zeros(dimX_, 1, CV_64F);
	processmodel(sigma_points[i], sigma_points[i], input, process_rnd_num);
  }

  // 事前状態推定値の計算
  cv::Mat xhatm = cv::Mat::zeros(dimX_, 1, CV_64F); // メモリの確保
  for(int i = 0; i < (int)sigma_points.size(); i++){
	  xhatm += w[i]*sigma_points[i];
  }

  // 事前誤差共分散行列の計算
  cv::Mat Pm = cv::Mat::zeros(P_.rows, P_.cols, CV_64F); // メモリの確保
  cv::Mat diff_x = xhatm.clone(); // 途中計算用
  for(int i = 0; i < (int)sigma_points.size(); i++){
	diff_x = sigma_points[i] - xhatm;
	Pm += w[i] * diff_x * diff_x.t();
  }
  Pm += ProcessNoiseCov_; // システムノイズを考慮

  // シグマポイントの再計算
  Cholesky(Pm, rootP);
  sigma_points[0] = xhatm;
  for(int i = 1; i <= dimX_; i++){
	sigma_points[i]       = xhatm + sqrt(dimX_ + kappa)*rootP.col(i-1);
	sigma_points[i+dimX_] = xhatm - sqrt(dimX_ + kappa)*rootP.col(i-1);
  }

  // 出力のシグマポイントの計算 出力のベクトルは１次元を仮定（dimY == 1）
  std::vector<cv::Mat> output_sigma_points(2*dimX_ + 1);
  cv::Mat obs_rnd_num = cv::Mat::zeros(dimY_, 1, CV_64F);
  for(int i = 0; i < (int)output_sigma_points.size(); i++){
	output_sigma_points[i] = cv::Mat_<double>(dimY_, 1);  // シグマポイントのメモリ確保
	obsmodel(output_sigma_points[i], sigma_points[i], obs_rnd_num);
  }
  
  // 事前出力推定値
  cv::Mat yhatm = cv::Mat::zeros(dimY_, 1, CV_64F);
  for(int i = 0; i < (int)output_sigma_points.size(); i++){
	yhatm += w[i] * output_sigma_points[i];
  }
  
  // 事前出力誤差共分散行列
  cv::Mat Pym = cv::Mat::zeros(dimY_, dimY_, CV_64F); // メモリの確保
  cv::Mat diff_y = Pym.clone(); // 途中計算用
  for(int i = 0; i < (int)output_sigma_points.size(); i++){
	diff_y = output_sigma_points[i] - yhatm;
	Pym += w[i] * diff_y * diff_y.t();
  }

  // 事前状態・出力誤差共分散行列
  cv::Mat Pxym = cv::Mat::zeros(P_.rows, P_.cols, CV_64F); // メモリの確保
  for(int i = 0; i < (int)sigma_points.size(); i++){
	diff_x = sigma_points[i] - xhatm;
	diff_y = output_sigma_points[i] - yhatm;
	Pxym += w[i] * diff_x * diff_y;
  }

  // カルマンゲイン
  G_ = Pxym / (Pym + ObsNoiseCov_);

  //------------------------------
  // フィルタリングステップ
  //------------------------------
  // 状態推定値
  xhat_new_ = xhatm + G_*(observation - yhatm);
  // 事後誤差共分散行列
  P_new_ = Pm - G_*Pxym.t();

  xhat_ = xhat_new_;
  P_ = P_new_;
}

cv::Mat UnscentedKalmanFilter::GetEstimation(){
  return xhat_new_;
}
