#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.cpp"

using namespace cv;
using namespace std;

int main()
{
    cout << "OpenCV Version: " << CV_VERSION << endl;
    Mat img = imread("./cat.jpg", 0);
    imshow("img", img);
    waitKey();
    Matrix<uchar> m = img;
    Mat img1 = m.slice(0, m.getRow() / 2, 0, m.getColumn() / 2).toMat(0);
    imshow("img1", img1);
    waitKey();
    return 0;
}