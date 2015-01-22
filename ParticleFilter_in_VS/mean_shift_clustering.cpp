#include "mean_shift_clustering.h"
//#define DEBUG

MeanShiftClustering::MeanShiftClustering(std::vector< PStateMat >&points,
										 int dim,
										 double sigma)
  : points_(points), dim_(dim), sigma_(sigma)
{
  num_points_ = points_.size();
}

MeanShiftClustering::~MeanShiftClustering()
{
}

double MeanShiftClustering::EuclideanDistance(PStateMat p1,
											  PStateMat p2)
{
  double distance = 0;
  for(int i = 0; i < dim_; i++){
	distance += pow(p1.state_.at<double>(i,0) - p2.state_.at<double>(i,0), 2.0);
  }
  return sqrt(distance);
  //  return cv::norm(p1.state_, p2.state_);
}

PStateMat MeanShiftClustering::MeanShiftProcedure(PStateMat initX,
												  double threshold)
{
  PStateMat node(initX);
  PStateMat last_node(initX);

  //#ifdef DEBUG
  unsigned long int loop = 0;
  //#endif

  while(true){
	PStateMat sum_gx(dim_, 0.0);
	sum_gx.state_ = cv::Mat::zeros(sum_gx.state_.rows, sum_gx.state_.cols, CV_64F);
	double sum_g = 0;
	for(int i = 0; i < num_points_; i++){
	  double dist = EuclideanDistance(node, points_[i]);
	  dist = sigma_ * dist * dist;
	  double g_i = dist * exp(-dist);
	  sum_g += g_i;
	  sum_gx.state_.at<double>(0,0) = sum_gx.state_.at<double>(0,0) 
		+ g_i * points_[i].state_.at<double>(0,0);
	}
	node.state_.at<double>(0,0) = sum_gx.state_.at<double>(0,0) / sum_g;

	if(EuclideanDistance(node, last_node) < threshold){
	  break;
	}else{
	  #ifdef DEBUG
	  std::cout << "node : " << node.state_ << std::endl;
	  std::cout << "last :" << last_node.state_ << std::endl;
	  std::cout << "EuclideanDistance : " << EuclideanDistance(node, last_node) << std::endl;
	  #endif
	  last_node = node;
	}
	//#ifdef DEBUG
	loop++;
	//std::cout << "loop : " << loop << std::endl;
	if(loop > 10000){ break; } 
	//#endif
  }
  #ifdef DEBUG
  std::cout << std::endl;
  #endif
  return node;
}

int MeanShiftClustering::Clustering(std::vector<int> &indices, double threshold)
{
  if(indices.size() != points_.size()){
	indices.resize(points_.size());
  }
  std::vector< PStateMat > clusters;
  for(int i = 0; i < num_points_; i++){
	PStateMat node = MeanShiftProcedure(points_[i], 3e-4);
	bool is_new_cluster(true);
	for(int j = 0; j < (int)clusters.size(); j++){
	  if(EuclideanDistance(clusters[j], node) < threshold){
		indices[i] = j;
		is_new_cluster = false;
		break;
	  }
	}
	if(is_new_cluster){
	  indices[i] = clusters.size();
	  clusters.push_back(node);
	}
  }
  return clusters.size();
}
