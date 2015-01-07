#ifndef K_MEANS
#define K_MEANS

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <istream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <boost/algorithm/string.hpp>


int ReadDataFromFile(std::string filename, 
					 std::vector< std::vector<double> > &data,
					 int dim,
					 std::string splitchar=",");


class Cluster
{
public:
  Cluster(){}
  Cluster(int cluster_num, int dim){
	centroids.resize(cluster_num);
	copy_centroids.resize(cluster_num);
	clusters.resize(cluster_num);
	for(int i = 0; i < cluster_num; i++){
	  centroids[i].resize(dim);
	  copy_centroids[i].resize(dim);
	}
  }
public:
  std::vector< std::vector<double> > centroids;
  std::vector< std::vector<double> > copy_centroids;
  std::vector< std::vector<int> > clusters;
};

class KmeansPlus
{
 public:
  KmeansPlus(std::vector< std::vector<double> > &points,
		 int cluster_num,
		 int dim);
  ~KmeansPlus();
  void Clustering(Cluster &result_clusters);
 
 private:
  void UpdateCentroid();
  double EuclideanDistance(std::vector<double> p1,
							std::vector<double> p2);
  Cluster cluster_;
  std::vector< std::vector<double> > points_;
  int spep_; // Num of iteration
  int points_num_; // Num of data points
  int cluster_num_;
  int dim_; // Num of data dimension
  std::vector<double> distance;  
};

#endif // K_MEANS
