#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Give video file name\n");
  }

  std::string imagePath = argv[1];

  Mat image = imread(imagePath, IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Could not open find image: " << imagePath << std::endl;
    return -1;
  }

  imshow("Image viewer", image);

  waitKey(0);

  
  
  return 0;
}
