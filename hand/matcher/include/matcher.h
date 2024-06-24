#include <opencv2/core.hpp>

#include <memory>

class Matcher
{
private:
    cv::Mat __roi;

public:
    Matcher() = delete;
    Matcher(const cv::Mat &roi) { __roi = roi; }

    // 提取LBP特征
    cv::Mat getLBP();
};
using Matcher_ptr = std::shared_ptr<Matcher>;
