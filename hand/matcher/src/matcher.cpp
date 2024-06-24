#include "matcher.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include "math.h"

using namespace std;
using namespace cv;

Mat Matcher::getLBP()
{
    Mat img = __roi.clone();
    Mat gray_src;
    gray_src = img;
    //    cvtColor(img, gray_src, COLOR_BGR2GRAY);

    // 定义LBP图像的长宽，由于最外围一圈无8领域，所以长宽相比于原图各减少2
    int width = img.cols - 2;
    int hight = img.rows - 2;

    // 初始化一全为0的矩阵
    Mat lbpImg = Mat::zeros(hight, width, CV_8UC1);

    for (int row = 1; row < img.rows - 1; row++)
    {
        for (int col = 1; col < img.cols - 1; col++)
        {
            uchar c = gray_src.at<uchar>(row, col);
            uchar code = 0;

            // 对于八个邻域值做处理
            //|= 按位或，a |= b 和 a = a | b 等价;
            // 左移<<就是将二进制的每一个数都往左移动一位,高位舍去，低位补0（超过存储上限8位的属于高位，需舍去）
            code |= (gray_src.at<uchar>(row - 1, col - 1) > c) << 7;
            code |= (gray_src.at<uchar>(row - 1, col) > c) << 6;
            code |= (gray_src.at<uchar>(row - 1, col + 1) > c) << 5;
            code |= (gray_src.at<uchar>(row, col + 1) > c) << 4;
            code |= (gray_src.at<uchar>(row + 1, col + 1) > c) << 3;
            code |= (gray_src.at<uchar>(row + 1, col) > c) << 2;
            code |= (gray_src.at<uchar>(row + 1, col - 1) > c) << 1;
            code |= (gray_src.at<uchar>(row, col) > c) << 0;
            // 赋值操作，注意row和col是从0开始的；
            lbpImg.at<uchar>(row - 1, col - 1) = code;
        }
    }

    return lbpImg;
}
