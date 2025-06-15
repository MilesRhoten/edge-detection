#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <thread>
#include <arm_neon.h>
#include <semaphore.h>

#include "main.h"

#define NUMSEMS 4

using namespace cv;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Give video file name\n");
  }

  VideoCapture cap(argv[1]);
  if (!cap.isOpened()) {
    perror("error in capture");
    exit(-1);
  }

  Mat frame;

  Mat output;

  std::list<Mat> mats(4);
  
  int done = 0;  
  
  std::list<sem_t> semList1(4);
  std::list<sem_t> semList2(4);  
  
  int i;
  for (sem_t sem : semList1) {
    int worked = sem_init(&sem, 0, 0);
    if (worked == -1) {
      perror("sem failure\n");
      exit(-1);
    }
  }

  for (sem_t sem : semList2) {
    int worked = sem_init(&sem, 0, 0);
    if (worked == -1) {
      perror("sem failure\n");
      exit(-1);
    }
  }

  auto sem1 = semList1.begin();
  auto sem2 = semList2.begin();
  auto mat = mats.begin();
  i = 0;
  
  std::list<std::thread> threads;
  while (sem1 != semList1.end()) {
    threads.push_back(std::thread(frametosobel, *mat,
				  i, std::ref(output), *sem1,
				  *sem2, std::ref(done)));    
    ++sem1;
    ++sem2;
    ++mat;
    ++i;
  }
  
    
  while (1) {
    bool end = cap.read(frame);
    if (!end) {
      done = 1;
      break;
    }

    // split up frame
    // divide to 4
  
    int rows = frame.rows;
    int cols = frame.cols;
    
    // split into 4 chunks, all the full width and a quarter height
    int row1 = rows / 4;
    int midRow = rows / 2;
    int row3 = midRow + row1;
    
    int height = rows / 4;
    
    int i = 0;
    for (Mat mat : mats) {
      if (i == 0) {
	mat = frame(Rect(0, 0, cols, height + 1));
      }
      if (i == 1) {
	mat = frame(Rect(0, row1 - 1, cols, height + 2));
      }
      if (i == 2) {
	mat = frame(Rect(0, midRow - 1, cols, height + 2));
      }
      if (i == 3) {
	mat = frame(Rect(0, row3 - 1, cols, height + 1));
      }
      i += 1;
    }
    
    
    
    output = Mat(frame.rows, frame.cols, CV_8UC1);
    
    
    // update threads
    for (sem_t sem : semList1) {
      int fail = sem_post(&sem);
      if (fail == -1) {
	perror("sem post fail");
	exit(-1);
      }
    }
    
    // wait for threads
    for (sem_t sem : semList2) {
      int fail = sem_wait(&sem);
      if (fail == -1) {
	perror("sem wait fail");
	exit(-1);
      }
    }
    
    
    namedWindow("Frame", WINDOW_NORMAL);
    resizeWindow("Frame", 480, 360);
    cv::imshow("Frame", frame);
    
    namedWindow("Sobel Frame", WINDOW_NORMAL);
    resizeWindow("Sobel Frame", 480, 360);
    cv::imshow("Sobel Frame", output);
    
  }
  
  // wait on 4 threads
  for (auto &thr : threads) {
    thr.join();
  }
  
  return 0;
}


void frametosobel(Mat &input, int ID, Mat &output,
		  sem_t &sem1, sem_t &sem2,
		  int &done) {
  while (done == 0) {
    // wait here for frame
    int fail = sem_wait(&sem1);
    if (fail == -1) {
      perror("sem wait fail in frametosolbe");
      exit(-1);
    }

    Mat gray = to442_grayscale(input);
    Mat sobel =  to442_sobel(gray);
    int xOffset = 1;
    
    // if its the middle 2 rows, take 2 off height
    // if its top or bottom, take 1 off
    int height = sobel.rows - 1;
    
    if ((ID == 0) || (ID == 3)) {
      height = height - 1;
    }
    
    // this is the offset for the final rectangle 
    int yOffset = 1 + (height * ID);
    
    
    // a rect for the section of output to write to
    cv::Rect outRect(xOffset, yOffset, sobel.cols - 2, height);
    
    // rect to trim sobel
    Rect trimSobelRect(1, 1, sobel.cols - 2, height);
    
    // we should remove this line, seems unneccessary
    Mat trimSobel = sobel(trimSobelRect).clone();
    
    
    
    // copy the trimmed sobel to the correct section of output
    trimSobel.copyTo(output(outRect));

    // tell main we are done
    fail = sem_post(&sem2);
    if (fail == -1) {
      perror("sempost");
      exit(-1);
    }
  }
}


cv::Mat to442_grayscale(cv::Mat &input) {
  cv::Mat grayMat(input.rows, input.cols, CV_8UC1);

  uchar *pixelPtr = input.data;
  int channels = input.channels();
  
  int y;
  for (y = 0; y < input.rows; y++) {
    int x;
    for (x = 0; x < input.cols; x++ ) {
      int offset = (y * input.step) + (x * channels);
      
      uint8_t grayVal = static_cast<uint8_t>
	(.2126 * pixelPtr[offset + 2]) +
	(.7152 * pixelPtr[offset + 1]) +
	(.0722 * pixelPtr[offset]);

      //cv::Scalar grayPixel = cvScalar(grayVal);
      grayMat.at<uint8_t>(y, x) = grayVal;
    }
  }

  return grayMat;
}


cv::Mat to442_sobel(cv::Mat& input) {
  cv::Mat sobel(input.rows, input.cols, CV_8UC1);
  
  int sobelX[3][3] = {{-1, 0, 1},
			   {-2, 0, 2},
			   {-1, 0, 1}};
  int sobelY[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};

  cv::Scalar pixel;

  uchar *pixelPtr = input.data;
  
  int y;
  for (y = 1; y < input.rows - 1; y++) {
    int x;
    for (x = 1; x < input.cols - 1; x += 8) {
      int16x8_t sumX = vdupq_n_s16(0);
      int16x8_t sumY = vdupq_n_s16(0);

      int i;
      for (i = -1; i < 2; i++) {
	int j;
	for (j = -1; j < 2; j++) {
	  int offset = ((y + i) * input.step) + (x + j);
	  
	  int8x8_t pixel = vld1_s8((int8_t *) &pixelPtr[offset]);
	  
	  sumX = vmlaq_n_s16(sumX, vmovl_s8(pixel), sobelX[i + 1][j + 1]);
	  sumY = vmlaq_n_s16(sumY, vmovl_s8(pixel), sobelY[i + 1][j + 1]);
	}
      }

      int16x8_t magnitude = vqaddq_s16(vabsq_s16(sumX), vabsq_s16(sumY));
      uint8x8_t sobelResult = vqmovn_u16(vreinterpretq_u16_s16(magnitude));
      
      vst1_u8(&sobel.at<uchar>(y, x), sobelResult);
    }
  }

  return sobel;
}

