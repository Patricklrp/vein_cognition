#include "processor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

bool findMax(int index, vector<double> dist, int margin = 20)
{
    int total_num = dist.size();
    for (int i = 0; i < margin * 2 + 1; i++)
    {
        int compare_index = (index + total_num - margin + i) % total_num;
        if (dist[compare_index] > dist[index])
            return false;
    }
    return true;
}

bool findMin(int index, vector<double> dist, int margin = 20)
{
    int total_num = dist.size();
    for (int i = 0; i < margin * 2 + 1; i++)
    {
        int compare_index = (index + total_num - margin + i) % total_num;
        if (dist[compare_index] < dist[index])
            return false;
    }
    return true;
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

Point2f rotatePoint(Point2f p, Size rotated_size, double angle, Size origin)
{
    // cout << angle << endl;
    Point2f origin_to_center(-origin.width / 2, -origin.height / 2);
    Point2f center(static_cast<float>(origin.width) / 2, static_cast<float>(origin.height) / 2);
    double scale = 1.0;
    Mat rotMat = getRotationMatrix2D(center, angle, scale);
    Matx22f rotMatx(static_cast<double>(rotMat.at<double>(0, 0)), static_cast<double>(rotMat.at<double>(0, 1)),
                    static_cast<double>(rotMat.at<double>(1, 0)), static_cast<double>(rotMat.at<double>(1, 1)));
    Point2f rotated = rotMatx * (p + origin_to_center);
    // cout << rotMatx << endl;
    // cout << rotated.x << " " << rotated.y << endl;
    Point2f center_to_origin = Point2f(rotated_size.width / 2, rotated_size.height / 2);
    rotated = rotated + center_to_origin;
    return rotated;
}

Mat Processor::deleteWrist(Mat src, int L, Point2f centroid)
{
    int limit = min(src.rows, int(centroid.y + L));
    for (int row = limit; row < src.rows; row++)
    {
        uchar *rowPtr = src.ptr<uchar>(row);
        for (int col = 0; col < src.cols; col++, rowPtr++)
            *rowPtr = 0;
    }
    return src;
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
    otsu_val *= 0.5;
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
    // Mat test;
    // cvtColor(binary, test, COLOR_GRAY2BGR);

    // 获取手掌边缘点
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    float max_area = 0;
    vector<Point> hand;
    float area;
    for (auto &contour : contours)
    {
        area = contourArea(contour);
        // cout << area << endl;
        // cout << area << endl;
        if (area > max_area)
        {
            hand = contour;
            max_area = area;
        }
    }

    // 计算质心
    auto M = moments(hand);
    __centroid = Point2f(M.m10 / M.m00, M.m01 / M.m00);

    // 提取关键点
    // 去除手腕
    int L = 130;
    this->deleteWrist(binary, L, __centroid);
    // circle(binary, __centroid, 2, Scalar(0), -1);
    // imshow("test", binary);
    // waitKey(-1);
    binary = this->deleteSmallArea(binary, 10000);

    // 提取底部中点
    int limit = __centroid.y + L;
    int row = limit > binary.rows ? binary.rows : limit - 1;
    int left = 0;
    int right = 0;
    uchar *rowPtr = binary.ptr<uchar>(row);
    for (int col = 0; col < binary.cols; col++, rowPtr++)
        if (*rowPtr != 0)
            left == right ? (left = col) : (right = col);
    Point middle = Point((left + right) / 2, row);

    // 计算距离
    vector<vector<Point>> new_contours;
    findContours(binary, new_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    max_area = 0;
    vector<Point> new_hand;
    for (auto &contour : new_contours)
    {
        area = contourArea(contour);
        if (area > max_area)
        {
            new_hand = contour;
            max_area = area;
        }
    }
    vector<double> dist;
    for (auto &p : new_hand)
        dist.push_back(norm(p - middle));

    // 检索初始点
    int startIndex = 0;
    for (int i = 0; i < new_hand.size(); i++)
    {
        if (abs(new_hand[i].x - left) < 1.5 && new_hand[i].y == row)
        {
            startIndex = i;
            break;
        }
    }
    // cout << startIndex << endl;

    // 寻找手指峰谷点
    vector<int> keyPointsIndex;
    vector<Point> keyPoints;
    for (int i = 0; i < new_hand.size(); i++)
    {
        int index = (startIndex + i) % new_hand.size();
        if (new_hand[index].y == row)
            continue;
        if (keyPointsIndex.size() == 9)
            break;
        else if (keyPointsIndex.size() % 2 == 0)
        {
            if (findMax(index, dist, 8))
                keyPointsIndex.push_back(index);
        }
        else
        {
            if (findMin(index, dist, 8))
                keyPointsIndex.push_back(index);
        }
    }
    // cout << "size = " << keyPoints.size() << endl;
    for (auto &index : keyPointsIndex)
        keyPoints.push_back(new_hand[index]);

    // // For test
    // Mat test;
    // cvtColor(binary, test, COLOR_GRAY2BGR);
    // for (auto &p : keyPoints)
    //     circle(test, p, 2, Scalar(0, 0, 255), -1);
    // // circle(test, new_hand[startIndex], 2, Scalar(0, 255, 0), -1);
    // // drawContours(test, new_contours, -1, Scalar(255, 0, 0));
    // imshow("test", test);
    // waitKey(-1);

    // cout << "flag" << endl; // for debugging
    // 获取关键点
    Point p1, p2;
    RotatedRect ROI_Rect;                                                      // p1 为食指根，p2 为小指根
    if (norm(keyPoints[5] - keyPoints[7]) > norm(keyPoints[3] - keyPoints[1])) // 左手
    {
        Point aux = keyPoints[5] * 2 - keyPoints[3];
        p1 = (aux + keyPoints[5]) / 2;
        aux = keyPoints[1] * 2 - keyPoints[3];
        p2 = (aux + keyPoints[1]) / 2;

        // 构造正方形
        Vec2f side = static_cast<Point2f>(p2 - p1);
        Matx22f rotate_mat(0, -1, 1, 0);
        auto new_side = rotate_mat * side;
        Point2f p3 = Point2f(p2.x + new_side[0], p2.y + new_side[1]);
        // line(binary, p1, p2, Scalar(0));
        // line(binary, bottom_right, p2, Scalar(0));
        ROI_Rect = RotatedRect(p1, p2, p3);
    }
    else
    {
        Point aux = keyPoints[3] * 2 - keyPoints[5];
        p1 = (aux + keyPoints[3]) / 2;
        aux = keyPoints[7] * 2 - keyPoints[5];
        p2 = (aux + keyPoints[7]) / 2;

        // 构造正方形
        Vec2f side = static_cast<Point2f>(p1 - p2);
        Matx22f rotate_mat(0, -1, 1, 0);
        auto new_side = rotate_mat * side;
        Point2f p3 = Point2f(p2.x + new_side[0], p2.y + new_side[1]);
        ROI_Rect = RotatedRect(p1, p2, p3);
        // line(binary, p1, p2, Scalar(0));
        // line(binary, bottom_right, p2, Scalar(0));
    }

    Point2f rect_points[4];
    ROI_Rect.points(rect_points);
    float rect_angle = ROI_Rect.angle;
    Mat rotated_mat = rotateImage(binary, rect_angle);
    __rotated_src = rotateImage(__src, rect_angle);
    vector<Point2f> rotated_points;
    for (auto &p : rect_points)
        rotated_points.push_back(rotatePoint(p, __rotated_src.size(), rect_angle, __src.size()));
    Rect target(rotated_points[0], rotated_points[2]);
    Mat __roi = __rotated_src(target).clone();
    if (norm(keyPoints[5] - keyPoints[7]) < norm(keyPoints[3] - keyPoints[1])) // 右手
        flip(__roi, __roi, 1);
    // for (auto &p : rotated_points)
    // circle(rotated_mat, p, 2, Scalar(0), -1);
    // for (auto &p : rect_points)
    // circle(binary, p, 2, Scalar(0), -1);

    // cout << p.x  << " " << p.y << endl;

    // for (int i = 0; i < 4; i++)
    // line(binary, rect_points[i], rect_points[(i + 1) % 4], Scalar(0));

    return __roi;
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
