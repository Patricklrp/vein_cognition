#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "processor.h"

using namespace std;
using namespace cv;

int main()
{
    // 读入图片
    Mat image = imread("/home/liu/桌面/vein_recognition/finger/test.bmp");

    // 预处理
    Processor_ptr ps = make_shared<Processor>(image);
    Mat binary = ps->preprocess();

    // 获取ROI
    Mat roi = ps->getROI(binary);

    // 滤波增强特征
    Mat reinforce = ps->reinforce(roi);

    imshow("test", image);
    // imshow("binary", binary);
    // imshow("roi", roi);
    // imshow("reinforce", reinforce);
    waitKey(-1);
    return 0;
}