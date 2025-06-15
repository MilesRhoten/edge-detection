#ifndef MAINH
#define MAINH

#include <opencv2/opencv.hpp>

using namespace cv;

void frametosobel(Mat &input, int ID, Mat &output,
		  sem_t &sem1, sem_t &sem2);

Mat to442_grayscale(Mat &input);

Mat to442_sobel(Mat &input);

#endif
