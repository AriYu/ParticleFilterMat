#include "RootMeanSquareError.h"

RMSE::RMSE()
{
    m_rmse = 0.0;
}

RMSE::~RMSE()
{
}

void RMSE::storeData(double x, double x_hat)
{
    m_x.push_back(x);
    //std::cout << "[x_hat] : " << x_hat << std::endl;
    m_x_hat.push_back(x_hat);
}

void RMSE::calculationRMSE()
{
    double sum = 0.0;
    unsigned int size = m_x.size();
    for(unsigned int i = 0; i < size; i++){
        sum += pow(m_x[i] - m_x_hat[i], 2.0);
    }
    m_rmse = sqrt(sum/((double)size));
}

double RMSE::getRMSE()
{
    return m_rmse;
}
