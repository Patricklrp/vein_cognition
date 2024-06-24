/**
 * @file main.cpp
 * @author 刘锐平 (liu19120353430@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-06-05
 *
 * @copyright Copyright (c) 2024
 *
 */

// #include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "processor.h"
#include "matcher.h"
#include "fileOP.h"

using namespace std;
using namespace cv;

string root_path = "/home/liu/文档/机器视觉创新实践资料/第二批12-1早/第二批12-1早-手掌";
string save_path = "";

int main(int argc, char *argv[])
{

    // string test_path = "/home/liu/桌面/vein_recognition/C++/test.jpg";
    // Mat img = imread(test_path);

    // 读入图像
    vector<vector<Mat>> imgs;
    vector<string> labels;
    FileOP::readFiles(root_path, imgs, labels);

    // 变量初始化
    vector<vector<Mat>> rois = imgs;
    Processor_ptr ps = make_shared<Processor>();

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

    // 保存
    // FileOP::saveFiles(rois,labels,root_path);

    Mat img = imgs[2][2];
    // Mat img = imgs[each_person][each_img];
    ps->changeSrc(img);
    // 图像预处理
    Mat binary = ps->preprocess();

    // ROI提取
    Mat roi = ps->getROI(binary);

    // 增强
    Mat reinforced = ps->reinforce(roi);

    // // 数据库匹配
    // Matcher_ptr mp = make_shared<Matcher>(reinforced);
    // Mat LBPMat = mp->getLBP();

    //

    // 输出结果
    // imshow("origin", img);
    // imshow("binary", binary);
    // cout << reinforced.channels() << endl;
    // imshow("roi", roi);
    // imshow("reinforce", reinforced);
    // imshow("LBP", LBPMat);

    waitKey(-1);

    return 0;
}