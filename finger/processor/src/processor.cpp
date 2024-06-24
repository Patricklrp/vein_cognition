#include "processor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

bool findMax(vector<int> &gray_vals, int index, int last_val = 0, int margin = 20)
{
    int startIndex = (index - margin < 0) ? 0 : index - margin;
    int endIndex = (index + margin > gray_vals.size() - 1) ? gray_vals.size() : index + margin;
    for (int i = startIndex; i != endIndex + 1; i++)
    {
        if (gray_vals[index] < gray_vals[i])
            return false;
        if (gray_vals[index] == last_val)
            return false;
    }
    return true;
}

// 获取最大轮廓
void Processor::getMaxContour(Mat src, vector<Point> &max_contour)
{
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // 提取contours中面积最大的轮廓
    int max_index = 0;
    double max_area = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > max_area)
        {
            max_area = area;
            max_index = i;
        }
    }
    max_contour = contours[max_index];
    return;
}

Mat Processor::rotateImage(const Mat &source, double angle)
{
    // 获取图像的维度（高度和宽度）
    int width = source.cols;
    int height = source.rows;

    // 指定旋转中心，通常是图像的中心
    Point2f center(static_cast<float>(width) / 2, static_cast<float>(height) / 2);

    // 指定缩放比例
    double scale = 1.0;

    // 获取旋转矩阵
    Mat rotMat = getRotationMatrix2D(center, angle, scale);

    // 计算旋转后的图像尺寸
    Rect2f bbox = RotatedRect(Point2f(), source.size(), angle).boundingRect2f();

    // 调整旋转矩阵以考虑平移
    rotMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // 应用旋转矩阵
    Mat rotated;
    warpAffine(source, rotated, rotMat, bbox.size());

    return rotated;
}

Mat Processor::deleteSmallArea(cv::Mat src, int min_area = 500)
{
    // 进行连通组件分析
    Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // 创建输出图像
    Mat dst = Mat::zeros(src.size(), CV_8UC1);

    // 遍历所有标签，剔除小区域
    for (int i = 1; i < num_labels; i++)
    {
        if (stats.at<int>(i, CC_STAT_AREA) >= min_area)
        {
            dst.setTo(255, labels == i);
        }
    }
    return dst;
}

Mat Processor::preprocess()
{
    // 灰度化
    Mat gray;
    cvtColor(__src, gray, COLOR_BGR2GRAY);

    // 二值化
    Mat binary;
    int otsu_val = threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    otsu_val *= 0.8;
    threshold(gray, binary, otsu_val, 255, THRESH_BINARY);
    // 形态学滤波
    Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    morphologyEx(binary, binary, cv::MORPH_OPEN, element);
    morphologyEx(binary, binary, cv::MORPH_CLOSE, element);

    // 删除小连通域
    binary = this->deleteSmallArea(binary, 10000);

    return binary;
}

Mat Processor::getROI(Mat binary)
{
    vector<Point> max_contour;
    this->getMaxContour(binary, max_contour);
    // 使用最小二乘法拟合直线
    cv::Vec4f line;
    cv::fitLine(max_contour, line, cv::DIST_L2, 0, 0.01, 0.01);

    // 绘制拟合的直线
    float vx = line[0];
    float vy = line[1];
    float x = line[2];
    float y = line[3];
    int rows = binary.rows;
    int cols = binary.cols;
    int lefty = int((-x * vy / vx) + y);
    int righty = int(((cols - x) * vy / vx) + y);

    // 测试
    Mat test;
    cvtColor(binary, test, COLOR_GRAY2BGR);
    cv::line(test, cv::Point(cols - 1, righty), cv::Point(0, lefty), cv::Scalar(0, 255, 0), 2);

    // 计算line的斜率
    double k = vy / vx;
    // cout << "斜率为:" << k << endl;

    // 弧度制转角度值
    double angle = atan(k) * 180 / CV_PI;

    // 旋转图像
    __src = rotateImage(__src, angle);
    binary = rotateImage(binary, angle);
    cvtColor(binary, test, COLOR_GRAY2BGR);

    // 重新计算
    this->getMaxContour(binary, max_contour);

    cv::fitLine(max_contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    vx = line[0];
    vy = line[1];
    x = line[2];
    y = line[3];

    // 计算上下边界
    int top_y = 0;
    int bottom_y = rows - 1;

    for (int i = 0; i < max_contour.size(); i++)
    {
        Point p = max_contour[i];
        // cout << p.x << endl;
        if (p.x < (float)cols * 0.05 || p.x > (float)cols * 0.95)
            continue;

        // cout << "flag" << endl;
        if (p.y < y)
        {
            if (p.y > top_y)
                top_y = p.y;
        }
        else
        {
            if (p.y < bottom_y)
                bottom_y = p.y;
        }
    }

    // 绘制上下边界
    // cv::line(test, cv::Point(0, top_y), cv::Point(cols, top_y), cv::Scalar(0, 0, 255), 2);
    // cv::line(test, cv::Point(0, bottom_y), cv::Point(cols, bottom_y), cv::Scalar(0, 255, 0), 2);

    // 截取出整个手指
    Rect target(Point2f((float)cols * 0.05, top_y), Point2f((float)cols * 0.95, bottom_y));
    __roi = __src(target).clone();

    // 根据亮度截取出关节内手指
    Mat roi_gray;
    cvtColor(__roi, roi_gray, COLOR_BGR2GRAY);

    // 获取灰度图中线处灰度变化曲线
    int middle_y = roi_gray.rows / 2;
    vector<int> gray_vals;

    for (int i = 0; i < roi_gray.cols; i++)
        gray_vals.push_back(roi_gray.at<uchar>(middle_y, i));

    // 提取两个关节点
    vector<int> max_index;
    int margin = 32;
    while (max_index.size() < 2)
    {
        max_index.clear();
        margin /= 2;
        for (int i = 0; i < gray_vals.size(); i++)
        {
            int last_val = (max_index.size() == 0) ? 0 : gray_vals[max_index.back()];
            if (findMax(gray_vals, i, last_val, margin))
            {
                max_index.push_back(i);
            }
        }
    }

    // 若极值点数大于2
    if (max_index.size() > 2)
    {
        sort(max_index.begin(), max_index.end(), [=](int a, int b)
             { return gray_vals[a] > gray_vals[b]; });
        int option = 1;
        while (max_index.size() > 2 && option < max_index.size())
        {
            if (abs(max_index[option] - max_index[0]) < __roi.cols / 2)
            {
                auto it = max_index.begin() + option;
                max_index.erase(it);
            }
            else
            {
                option++;
            }
        }
        sort(max_index.begin(), max_index.end(), [=](int a, int b)
             { return gray_vals[a] > gray_vals[b]; });
    }
    // cout << max_index[0] << endl;
    // cout << "size = " << max_index.size() << endl;
    // cout << max_index[0] << " " << max_index[1] << endl;
    target = Rect(Point2f(max_index[1], 0), Point2f(max_index[0], __roi.rows - 1));
    __roi = __roi(target).clone();

    // // // 绘制灰度变化曲线
    // int width = roi_gray.cols;
    // cout << "width = " << width << endl;
    // Mat plot = Mat::zeros(Size(width, 300), CV_8UC1); // 创建空白图像用于绘制曲线
    // for (int col = 0; col < width; ++col)
    // {
    //     // cout << gray_vals[col] << endl;
    //     plot.at<uchar>(255 - gray_vals[col], col) = 255;
    // }
    // // for (auto index : max_index)
    // // {
    // // cv::line(plot, Point(index, 0), Point(index, plot.rows - 1), Scalar(255), 1);
    // // }
    // cv::line(plot, Point(max_index[0], 0), Point(max_index[0], plot.rows - 1), Scalar(255), 1);
    // cv::line(plot, Point(max_index[1], 0), Point(max_index[1], plot.rows - 1), Scalar(255), 1);

    // imshow("Gray Curve", plot);
    // waitKey(-1);

    return __roi.clone();
    // return test;
}

// 创建 Gabor 滤波器
cv::Mat Processor::createGaborKernel(cv::Size ksize, double sigma, double theta, double lambd, double gamma, double psi)
{
    int ksizeY = ksize.height;
    int ksizeX = ksize.width;
    cv::Mat kernel(ksizeY, ksizeX, CV_32F);

    double sigmaX = sigma;
    double sigmaY = sigma / gamma;

    double nstds = 3.0;
    double xmax = std::max(std::fabs(nstds * sigmaX * std::cos(theta)), std::fabs(nstds * sigmaY * std::sin(theta)));
    xmax = std::ceil(std::max(1.0, xmax));
    double ymax = std::max(std::fabs(nstds * sigmaX * std::sin(theta)), std::fabs(nstds * sigmaY * std::cos(theta)));
    ymax = std::ceil(std::max(1.0, ymax));

    double xmin = -xmax;
    double ymin = -ymax;

    for (int y = 0; y < ksizeY; y++)
    {
        for (int x = 0; x < ksizeX; x++)
        {
            double xp = (x - ksizeX / 2) * std::cos(theta) + (y - ksizeY / 2) * std::sin(theta);
            double yp = -(x - ksizeX / 2) * std::sin(theta) + (y - ksizeY / 2) * std::cos(theta);

            double gauss = std::exp(-0.5 * (std::pow(xp / sigmaX, 2.0) + std::pow(yp / sigmaY, 2.0)));
            double wave = std::cos(2.0 * CV_PI * xp / lambd + psi);

            kernel.at<float>(y, x) = static_cast<float>(gauss * wave);
        }
    }

    return kernel;
}

Mat Processor::reinforce(const Mat &src)
{
    Mat img;
    if (src.type() != CV_8UC1)
        cvtColor(src, img, COLOR_BGR2GRAY);
    else
        img = src.clone();

    // 参数设置
    cv::Size ksize(3, 3); // 滤波器大小
    double sigma = 10.0;  // 高斯核的标准差
    double theta = 0;     // 滤波器方向
    double lambd = 10.0;  // 波长
    double gamma = 1;     // 纵横比
    double psi = 0;       // 相位偏移
                          // for (int i = 1; i < 20; i++)
                          // {
    Mat dst = img.clone();
    // for (int i = 1; i < 30; i += 3)
    // {

    //     dst = img.clone();
    //     theta = 0;
    for (; theta < CV_PI; theta += CV_PI / 8)
    {
        cv::Mat kernel = createGaborKernel(ksize, sigma, theta, lambd, gamma, psi);
        // 滤波
        cv::filter2D(dst, dst, CV_32F, kernel);
        // 归一化到0-255范围内
        cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
        dst.convertTo(dst, CV_8U);
        // cout << to_string(i) << endl;
    }
    //     imshow(to_string(i), dst);
    // }
    // cout << dst.channels() << endl;

    return dst;
}
