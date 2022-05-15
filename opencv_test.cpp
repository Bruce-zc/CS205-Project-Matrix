#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.cpp"

using namespace cv;
using namespace std;

int main()
{
    cout << "OpenCV Version: " << CV_VERSION << endl;

    Mat img1 = imread("./cat.jpg", IMREAD_GRAYSCALE);
    imshow("img_gray", img1);

    Matrix<uchar> m1 = img1;
    Mat img1_after = m1.slice(m1.getRow() / 4, 3 * m1.getRow() / 4, m1.getColumn() / 4, 3 * m1.getColumn() / 4).toMat(0);
    imshow("img1_after", img1_after);


    Mat img2 = imread("./cat.jpg", IMREAD_COLOR);
    imshow("img_color", img2);

    vector<Mat> img2_channels;
	split(img2, img2_channels);

    Matrix<uchar> mB = img2_channels.at(0);
    Mat img2_after_B = mB.slice(mB.getRow() / 4, 3 * mB.getRow() / 4, mB.getColumn() / 4, 3 * mB.getColumn() / 4).toMat(0);
    Matrix<uchar> mG = img2_channels.at(1);
    Mat img2_after_G = mG.slice(mG.getRow() / 4, 3 * mG.getRow() / 4, mG.getColumn() / 4, 3 * mG.getColumn() / 4).toMat(0);
    Matrix<uchar> mR = img2_channels.at(2);
    Mat img2_after_R = mR.slice(mR.getRow() / 4, 3 * mR.getRow() / 4, mR.getColumn() / 4, 3 * mR.getColumn() / 4).toMat(0);

    Mat img2_after;
    Mat img2_channels_after[3] = {img2_after_B, img2_after_G, img2_after_R};
    merge(img2_channels_after, 3, img2_after);

    imshow("img2_after", img2_after);
    waitKey();
    return 0;
}
