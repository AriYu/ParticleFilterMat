#include "k_means_pp.h"

using namespace std;

KmeansPlus::KmeansPlus(std::vector< std::vector<double> > &points,
					   int cluster_num,
					   int dim)
  : cluster_(cluster_num, dim), distance(cluster_num)
{
  points_ = points;
  dim_ = dim;
  cluster_num_ = cluster_num;
  points_num_ = points.size();
  srand(time(NULL));
  int index = 0;
  
  // 乱数で最初のクラスタ中心をランダムに選ぶ
  index = rand() % points_num_;
  cluster_.centroids[0] = points[index];
  // 最初に選んだクラスタ中心からの距離を計算
  vector<double> dist(points_num_);
  double min_distance = 0;
  for(int k = 1; k < cluster_num; k++){
	double sum = 0;
	// 最近傍点との距離Dを計算する．
	for(int i = 0; i < points_num_; i++){
	  min_distance = EuclideanDistance(cluster_.centroids[0], points[i]);
	  for(int l = 1; l < k; l++){
		dist[i] = EuclideanDistance(cluster_.centroids[l], points[i]);
		if(dist[i] < min_distance){
		  min_distance = dist[i];
		}
	  }
	  dist[i] = min_distance;
	  sum += min_distance;
	}
	// 距離を正規化して確率化する
	for(int i = 0; i < points_num_; i++){
	  dist[i] = dist[i] / sum;
	}
	// ルーレット選択でクラスタ中心を選ぶ
	int j = 0;
	double c = dist[0];
	double rnd_num = rand() % 100; // 0 ~ 99 の乱数
	rnd_num = rnd_num / 100; // 0 ~ 0.99 の乱数
	while (rnd_num > c){
	  j = j + 1;
	  c = c + dist[j];
	}
	cluster_.centroids[k] = points[j];
  }
}

KmeansPlus::~KmeansPlus()
{
}
double KmeansPlus::EuclideanDistance(std::vector<double> p1,
						 std::vector<double> p2)
{
  double distance = 0;
  for(int i = 0; i < dim_; i++){
	distance += pow(p1[i] - p2[i],2);
  }
  return sqrt(distance);
}
void KmeansPlus::UpdateCentroid()
{
  // update centroid
  vector<double> centroid(dim_);
  for(int i = 0; i < cluster_num_; i++){
	for(int j = 0; j < dim_; j++){
	  centroid[j] = 0;
	}
	cout << "cluster[" << i << "], size : " << cluster_.clusters[i].size() << endl;
	for(int j = 0; j < (int)cluster_.clusters[i].size(); j++){
	  for(int k = 0; k < dim_; k++){
		centroid[k] += points_[cluster_.clusters[i][j]][k];
	  }
	}
	for(int j = 0; j < dim_; j++){
	  centroid[j] = centroid[j] / (double)cluster_.clusters[i].size();
	}
	cluster_.centroids[i] = centroid;
  }
}

void KmeansPlus::Clustering(Cluster &result_clusters)
{

  int flag = true;
  int step = 0;
  while(flag)
  	{
	  step++;
	  for(int i = 0; i < cluster_num_; i++){
		cluster_.clusters[i].clear();
	  }

	  for(int i = 0; i < points_num_; i++){
		for(int j = 0; j < cluster_num_; j++){
		  distance[j] = EuclideanDistance(points_[i] , cluster_.centroids[j]);
		}
		int id = 0;
		double min_distance = distance[0];
		for(int j = 0; j < cluster_num_; j++){
		  if(distance[j] < min_distance){
			min_distance = distance[j];
			id = j;
		  }
		}
		cluster_.clusters[id].push_back(i);
	  }
	  
	  cluster_.copy_centroids = cluster_.centroids;

	  UpdateCentroid();

	  double sum = 0;
	  for(int i = 0; i < cluster_num_; i++){
		sum += EuclideanDistance(cluster_.copy_centroids[i], cluster_.centroids[i]);
	  }

	  if( sum == 0.0){
	  	flag = false;
	  }else{
		cout << "sum : " << sum << endl;
	  }
	}
  result_clusters = cluster_;
  cout << "iteration : " << step << endl;
}


int ReadDataFromFile(string filename, 
					 vector< vector<double> > &data, 
					 int dim,
					 string splitchar)
{
  ifstream infile; 
  string tmpdata;
  vector<double> point(dim);
  vector<string> chardata;

  infile.open(filename.c_str());
  if (!infile)
	{
	  cout << "Unable to open input."<< endl;
	  return 1;
	}
  while(getline(infile, tmpdata)){
	boost::algorithm::split(chardata, tmpdata, boost::algorithm::is_any_of(splitchar.c_str()));
	for(int i = 0; i < dim; i++){
	  point[i] = stod(chardata[i]);
	}
	data.push_back(point);
  }

  infile.close();
  return 0;
} 
