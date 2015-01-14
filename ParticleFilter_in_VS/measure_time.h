#ifndef MEASURE_TIME
#define MEASURE_TIME

#include <iostream>
#include <chrono>

class MeasureTime
{
 public:
 MeasureTime() 
   : elapsedTime_(0)
	{
	}
  ~MeasureTime()
	{
	}
  void start()
  {
	startTime_ = std::chrono::system_clock::now();
	elapsedTime_ = 0;
  }
  void stop()
  {
	endTime_ = std::chrono::system_clock::now();
  }
  int getElapsedTime()
  {
	elapsedTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_ - startTime_).count();
	return elapsedTime_;
  }
 private:
  std::chrono::time_point<std::chrono::system_clock> startTime_;
  std::chrono::time_point<std::chrono::system_clock> endTime_;
  int elapsedTime_;
};


#endif
