#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.cpp"

using namespace cv;
using namespace std;

int main()
{
    cout << "OpenCV Version: " << CV_VERSION << endl;

    // Read in gray image and slice
    Mat img1 = imread("./cat.jpg", IMREAD_GRAYSCALE);
    imshow("img_gray", img1);
    Matrix<uchar> m1 = img1;
    Mat img1_after = m1.slice(m1.getRow() / 4, 3 * m1.getRow() / 4, m1.getColumn() / 4, 3 * m1.getColumn() / 4).toMat(0);
    imshow("img1_after", img1_after);

    // Read in colorful image and slice
    Mat img2 = imread("./cat.jpg", IMREAD_COLOR);
    imshow("img_color", img2);
    vector<Mat> img2_channels;
    split(img2, img2_channels);
    Matrix<uchar> mB = img2_channels.at(0);
    Matrix<uchar> mG = img2_channels.at(1);
    Matrix<uchar> mR = img2_channels.at(2);
    Mat img2_slice_B = mB.slice(mB.getRow() / 4, 3 * mB.getRow() / 4, mB.getColumn() / 4, 3 * mB.getColumn() / 4).toMat(0);
    Mat img2_slice_G = mG.slice(mG.getRow() / 4, 3 * mG.getRow() / 4, mG.getColumn() / 4, 3 * mG.getColumn() / 4).toMat(0);
    Mat img2_slice_R = mR.slice(mR.getRow() / 4, 3 * mR.getRow() / 4, mR.getColumn() / 4, 3 * mR.getColumn() / 4).toMat(0);
    Mat img2_slice;
    Mat img2_channels_after[3] = {img2_slice_B, img2_slice_G, img2_slice_R};
    merge(img2_channels_after, 3, img2_slice);
    imshow("img2_slice", img2_slice);

    // Convolutions
    Mat destB1;
    img2_channels.at(0).convertTo(destB1, CV_32F, 1 / 255.0);
    Matrix<float> mB1 = destB1;
    Mat destG1;
    img2_channels.at(1).convertTo(destG1, CV_32F, 1 / 255.0);
    Matrix<float> mG1 = destG1;
    Mat destR1;
    img2_channels.at(2).convertTo(destR1, CV_32F, 1 / 255.0);
    Matrix<float> mR1 = destR1;

    Matrix<float> mB2 = mB1.slice(mB1.getRow() / 5, mB1.getRow() / 5 + 512, mB1.getColumn() / 3, mB1.getColumn() / 3 + 512);
    Matrix<float> mG2 = mG1.slice(mG1.getRow() / 5, mG1.getRow() / 5 + 512, mG1.getColumn() / 3, mG1.getColumn() / 3 + 512);
    Matrix<float> mR2 = mR1.slice(mR1.getRow() / 5, mR1.getRow() / 5 + 512, mR1.getColumn() / 3, mR1.getColumn() / 3 + 512);

    Mat img2_conv_B = mB2.toMat(CV_32F);
    Mat img2_conv_G = mG2.toMat(CV_32F);
    Mat img2_conv_R = mR2.toMat(CV_32F);
    Mat img2_conv;
    Mat img21_channels_after[3] = {img2_conv_B, img2_conv_G, img2_conv_R};
    merge(img21_channels_after, 3, img2_conv);
    imshow("Original Image", img2_conv);

    /** Edge Detection (Sobel: Y direction)
     * [  1   2   1
     *    0   0   0
     *   -1  -2  -1 ]
     */
    Mat m = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    Matrix<float> kernal = m;
    img2_conv_B = mB2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_G = mG2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_R = mR2.conv(kernal, 2).toMat(CV_32F);
    img2_conv;
    Mat img22_channels_after[3] = {img2_conv_B, img2_conv_G, img2_conv_R};
    merge(img22_channels_after, 3, img2_conv);
    imshow("Edge Detection (Sobel Operator on Y Direction)", img2_conv);

    /** Edge Detection
     * [ -1  -1  -1
     *   -1   8  -1
     *   -1  -1  -1 ]
     */
    m = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    kernal = m;
    img2_conv_B = mB2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_G = mG2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_R = mR2.conv(kernal, 2).toMat(CV_32F);
    img2_conv;
    Mat img23_channels_after[3] = {img2_conv_B, img2_conv_G, img2_conv_R};
    merge(img23_channels_after, 3, img2_conv);
    imshow("Edge Detection (Laplace Operator)", img2_conv);

    /** Sharpeness
     * [ -1  -1  -1
     *   -1   9  -1
     *   -1  -1  -1 ]
     */
    m = (Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
    kernal = m;
    img2_conv_B = mB2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_G = mG2.conv(kernal, 2).toMat(CV_32F);
    img2_conv_R = mR2.conv(kernal, 2).toMat(CV_32F);
    img2_conv;
    Mat img24_channels_after[3] = {img2_conv_B, img2_conv_G, img2_conv_R};
    merge(img24_channels_after, 3, img2_conv);
    imshow("Sharpness", img2_conv);

    /** Gaussian Filter
     * [ 1   2   1
     *   2   4   2
     *   1   2   1 ] / 16
     */
    m = (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16;
    kernal = m;
    img2_conv_B = mB2.conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).toMat(CV_32F);
    img2_conv_G = mG2.conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).toMat(CV_32F);
    img2_conv_R = mR2.conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).conv(kernal, 2).toMat(CV_32F);
    Mat img25_channels_after[3] = {img2_conv_B, img2_conv_G, img2_conv_R};
    merge(img25_channels_after, 3, img2_conv);
    imshow("Gaussian Filtering (5 times)", img2_conv);

    waitKey();
    return 0;
}
