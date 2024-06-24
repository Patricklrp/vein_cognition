#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "fileOP.h"

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

void FileOP::readFiles(const string str_root, vector<vector<Mat>> &imgs, vector<string> &labels)
{

    fs::path root(str_root);

    // 错误路径判断
    if (!fs::exists(root) || !fs::is_directory(root))
    {
        cerr << "The provided path is not a directory or does not exist." << endl;
        return;
    }

    // 遍历得到所有子文件夹路径并记录文件夹名
    int i = 0;
    for (const auto &entry : fs::directory_iterator(root))
    {
        if (fs::is_directory(entry.status()))
        {

            // cout << entry.path().string() << std::endl;
            // 保存标签
            labels.push_back(entry.path().filename().string());

            // 读取所有图片
            Mat img;
            imgs.push_back(vector<Mat>());
            fs::path subdir = entry.path();
            for (const auto &file : fs::directory_iterator(subdir))
            {
                // cout << file.path().string() << endl;
                img = imread(file.path().string());
                imgs[i].push_back(img);
            }
            i++;
        }
    }

    return;
}

void FileOP::saveFiles(vector<vector<Mat>> rois, vector<string> labels, string root)
{
    // 切分训练集和测试集
    random_device rd;
    mt19937 g(rd());
    float ratio = 0.7; // 训练集比例
    fs::path root_path(root);
    root_path = root_path.parent_path();
    root_path += "/data";
    fs::path train_path(root_path.string() + "/train");
    fs::path test_path(root_path.string() + "/test");

    for (int i = 0; i < labels.size(); i++)
    {
        string label = labels[i];
        int n = rois[i].size();
        shuffle(rois[i].begin(), rois[i].end(), g); // 随机打乱
        string train_label_path = train_path.string() + "/" + label;
        string test_label_path = test_path.string() + "/" + label;
        fs::create_directories(train_label_path);
        fs::create_directories(test_label_path);

        Mat temp;
        for (int j = 0; j < n; j++)
        {
            // cout << rois[i][j].channels() << endl;
            cvtColor(rois[i][j], temp, COLOR_GRAY2BGR);
            resize(temp,temp,Size(128,60));
            if (j < n * ratio) // 属于训练集
                imwrite(train_label_path + "/" + to_string(j) + ".bmp", temp);
            else
            {
                // cout << (test_label_path + "/" + to_string(int(j - n * ratio)) + ".bmp") << endl;
                imwrite(test_label_path + "/" + to_string(int(j - n * ratio)) + ".bmp", temp);
            }
        }
    }
}
