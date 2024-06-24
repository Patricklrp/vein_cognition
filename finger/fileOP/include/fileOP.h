#include <opencv2/core.hpp>

class FileOP
{
private:
    // 单例模型
    FileOP() = default;
    FileOP(const FileOP &FO) = delete;
    ~FileOP() = default;
    const FileOP &operator=(const FileOP &FO) = delete;

public:
    // 读取图片文件及其标签
    static void readFiles(const std::string root, std::vector<std::vector<cv::Mat>> &imgs, std::vector<std::string> &labels);

    // 保存ROI图像
    static void saveFiles(std::vector<std::vector<cv::Mat>> rois, std::vector<std::string>labels, std::string root);
};
