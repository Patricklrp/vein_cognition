#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

#include "processor.h"
#include "fileOP.h"

using namespace std;
using namespace cv;

string root_path = "/home/liu/文档/机器视觉创新实践资料/第二批12-1早/第二批12-1早-手指/109-200";
string save_path = "";

int main()
{
    // // 读入图像
    // vector<vector<Mat>> imgs;
    // vector<string> labels;
    // FileOP::readFiles(root_path, imgs, labels);

    // // 变量初始化
    // vector<vector<Mat>> rois = imgs;
    // Processor_ptr ps = make_shared<Processor>();

    // // 处理图像
    // for (int each_person = 0; each_person < imgs.size(); each_person++)
    // {
    //     for (int each_img = 0; each_img < imgs[each_person].size(); each_img++)
    //     {
    //         try
    //         {
    //             Mat img = imgs[each_person][each_img];
    //             ps->changeSrc(img);

    //             // 图像预处理
    //             Mat binary = ps->preprocess();

    //             // ROI提取
    //             Mat roi = ps->getROI(binary);
    //             // 增强
    //             Mat reinforced = ps->reinforce(roi);

    //             // 暂存
    //             rois[each_person][each_img] = reinforced;
    //         }
    //         catch (const std::exception &e)
    //         {
    //             cout << "label = " << each_person << endl;
    //             cout << "num = " << each_img << endl;
    //         }
    //     }
    // }

    // // 保存roi
    // FileOP::saveFiles(rois, labels, root_path);

    // 读入图片
    Mat image = imread("/home/liu/文档/机器视觉创新实践资料/第二批12-1早/第二批12-1早-手指/109-200/109/109-3-1.bmp");

    // 预处理
    Processor_ptr ps = make_shared<Processor>(image);
    Mat binary = ps->preprocess();

    // 获取ROI
    Mat roi = ps->getROI(binary);

    // 滤波增强特征
    Mat reinforce = ps->reinforce(roi);

    // imshow("test", image);
    // imshow("binary", binary);
    // imshow("roi", roi);
    imshow("reinforce", reinforce);
    waitKey(-1);
    return 0;
}