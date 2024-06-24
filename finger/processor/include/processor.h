#pragma once

#include <opencv2/core.hpp>

#include <memory>

class Processor
{
private:
    cv::Mat __src;
    cv::Point2f __centroid;
    cv::Mat __rotated_src;
    cv::Mat __roi;

    // Gabor滤波器
    cv::Mat createGaborKernel(cv::Size ksize, double sigma, double theta, double lambd, double gamma, double psi);

public:
    // 构造函数
    Processor() = default;
    Processor(cv::Mat src) { __src = src; }

    void changeSrc(cv::Mat src)
    {
        __src = src;
        __roi = cv::Mat::zeros(__roi.size(), CV_8UC3);
    };
    // 预处理
    cv::Mat preprocess();

    // 过滤出目标
    cv::Mat deleteSmallArea(cv::Mat src, int min_area);

    // 获取最大轮廓
    void getMaxContour(cv::Mat src, std::vector<cv::Point> &max_contour);

    // 获取ROI
    cv::Mat getROI(cv::Mat binary);

    // 图像旋转
    cv::Mat rotateImage(const cv::Mat &source, double angle);

    // 图像特征增强
    cv::Mat reinforce(const cv::Mat &src);
};
using Processor_ptr = std::shared_ptr<Processor>;