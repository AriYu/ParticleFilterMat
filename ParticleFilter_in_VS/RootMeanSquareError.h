#ifndef ROOT_MEAN_SQUARE_ERROR_H_
#define ROOT_MEAN_SQUARE_ERROR_H_
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class RMSE
{
 public:
  RMSE();
  ~RMSE();
  void storeData(double x, double x_hat);
  void calculationRMSE();
  double getRMSE();

 protected:
  double m_rmse;
  std::vector<double> m_x;
  std::vector<double> m_x_hat;
};


#endif //ROOT_MEAN_SQUARE_ERROR_H_
