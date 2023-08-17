// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0  // 畸变矫正因子

// 边线点和平面点匹配关系
int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;  // 扫描周期  laserOdometry模块publish的位姿是10Hz
constexpr double DISTANCE_SQ_THRESHOLD = 25;  // 定义最近邻点的最小距离阈值
constexpr double NEARBY_SCAN = 2.5;  // 附近范围内的扫描线

int skipFrameNum = 5;  //用于控制帧之间的跳跃间隔，表示在处理输入数据时要跳过的帧的数量
bool systemInited = false;  // 系统初始化，跳过第一帧

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

// 声明并初始化了三个指针对象
// pcl::PointCloud<PointType>::Ptr 表示一个指向 PointCloud 类模板的指针对象
// new pcl::PointCloud<PointType>() 创建了一个新的空的 PointCloud 对象，并返回其指针。
// laserCloudCornerLast、laserCloudSurfLast 和 laserCloudFullRes三个指针分别指向这三个新创建的 PointCloud 对象。
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
// 设置旋转和平移的初始值
// 当前帧相对于世界坐标系的变换关系，即激光雷达里程计的姿态和位置信息。
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
// 点云特征匹配时的优化变量，优化的是两帧点云之间的相对位姿增量
// 当前帧位姿与上一帧位姿的位姿增量
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

// 下面的2个分别是优化变量para_q和para_t的映射：表示的是两个world坐标系下的位姿P之间的增量，例如△P = P0.inverse() * P1
// 将相对位姿增量通过Map映射为世界坐标系下的当前帧到上一帧之间的位姿变换关系
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// 5个消息队列
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf; // 极大边线点
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf; // 次极大边线点
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf; // 极小平面点
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;  // 次极小平面点
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;  // 有序点云
// 使用互斥锁
std::mutex mBuf;

// undistort lidar point
// 当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在点云扫描开始位置静止扫描得到的点云
// 计算当前点在上一帧lidar坐标系下的坐标，即将当前帧点云变换到上一帧坐标系下
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    // 调整去畸变因子
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    // 创建一个四元数 q_point_last，使用 slerp 函数对上一帧的四元数 q_last_curr 进行球面线性插值。
    // 这个插值操作是为了根据 s 的值（范围在[0, 1]之间）计算出当前点在上一帧坐标系中的旋转姿态。
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    // 通过将上一帧的平移向量 t_last_curr 乘以 s，得到当前点在上一帧坐标系中的平移量。
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);  // 定义一个三维向量 point，包含当前点在当前帧坐标系中的坐标
    // 得到当前点在上一帧坐标系中的坐标 un_point。
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    // po为去畸变后的点
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;  // 强度信息为scanid
}

// transform all lidar points to the start of the next frame
// 函数 TransformToEnd 的作用是将当前帧中的点云坐标转换到当前帧的结束时刻
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp); // 首先对点进行去畸变

    // 得到当前帧中的点在上一帧坐标系中的坐标
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    // 通过将 un_point 减去平移向量 t_last_curr，并使用上一帧到当前帧的旋转四元数 q_last_curr 的逆来对其进行旋转变换，
    // 得到在当前帧结束时刻的坐标 point_end。
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();// 获取互斥锁mBuf，以确保在对队列进行操作时不会发生并发访问的问题。
    // 用于处理接收到的点云数据并将其添加到消息队列中（添加到队列末尾）
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock(); //
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");  // 初始化ROS节点
    ros::NodeHandle nh; // nh是一个ROS节点句柄(NodeHandle)，在创建ROS节点后，会使用节点句柄来访问ROS的各种功能和服务。

    // 设定里程计的帧率（每跳过skipFrameNum帧采样1帧）
    // mapping_skip_frame指定了在建图过程中跳过的帧数，默认值为2表示每隔两帧才进行一次建图操作，中间的帧会被跳过。
    // mapping_skip_frame是要获取的参数名称，skipFrameNum用于存储获取的参数值，2为默认值
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    // 从scanRegistration订阅的消息话题
    // 极大边线点  次极大边线点   极小平面点  次极小平面点
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    // 简单处理后的有序点云
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    // 发布上一帧的边线点
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    // 发布上一帧的平面点
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    // 发布全部有序点云，就是从scanRegistration订阅来的点云，未经其他处理
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // 发布帧间的位姿变换
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布帧间的平移运动
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    // nav_msgs::Path是ROS中定义的一种消息类型，用于表示路径信息。
    // laserPath是一个对象，用于存储激光雷达的运动轨迹。轨迹由一系列的路径点组成，每个路径点包含了位置和姿态信息。
    // 每当进行激光数据处理和建图时，新的路径点会被添加到laserPath中，以记录激光雷达的轨迹。
    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);  // 控制循环频率，即每秒的循环次数

    // ros::ok()是一个函数，用于检查ROS节点是否仍在运行。只有在ROS节点处于运行状态时，返回值为true，否则返回值为false。
    // 循环中的代码会在每次循环迭代时执行一次，直到ros::ok()返回false，即ROS节点停止运行或收到终止信号。
    while (ros::ok())
    {
        ros::spinOnce();
        // 用于处理ROS节点的回调函数和消息。它会检查并执行尚未处理的回调函数，并处理接收到的消息。
        
        // 如果所有订阅的话题都收到了
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            // 这里是5个queue(队列)记录了最新的极大/次极大边线点，极小/次极小平面点，全部点云
            // 用队列头的时间戳，分配时间戳
            // 用于获取队列中第一个点云消息对象的时间戳，并将其转换为秒的表示。
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            // 如果时间不同步，则报错
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                // ROS出现丢包会导致这种情况
                // 在ROS中，丢包指的是在消息传递过程中，接收方未能成功接收到发送方发送的消息。
                printf("unsync messeage!");
                ROS_BREAK(); // 当前循环立即终止
            }
//*************************************************************************************************************//
            // 从队列中取出来对应的数据，并转换成pcl点云数据
            mBuf.lock(); // 使用互斥量锁定
            cornerPointsSharp->clear(); // 清空对象准备接收新的数据
            // 将ROS消息类型的数据转换为PCL库的点云类型数据
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            // 从队列中移除队首元素，即移除已经处理过的ROS消息数据，以准备处理下一个数据
            cornerSharpBuf.pop(); 

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();  // 解锁
//*************************************************************************************************************//
            TicToc t_whole; // 开始计时
            // initializing
            // 对初始帧的操作是，跳过初始帧的位姿估计，直接作为第二帧的上一帧
            // 初始帧不进行匹配，仅仅将 cornerPointsLessSharp 保存至 laserCloudCornerLast
            if (!systemInited)
            {
                // 主要用来跳过第一帧数据
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                // 极大边线点的数量（这里大小指曲率）
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                // 极小平面点的数量
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt; // 开始计时（迭代优化2次）
                // 点到线以及点到面的非线性优化，迭代2次（选择当前优化的位姿的特征点匹配，
                // 并优化位姿（4次迭代），然后重新选择特征点匹配并优化）
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter) // 优化迭代次数
                {
                    corner_correspondence = 0; // 边线点匹配关系
                    plane_correspondence = 0;  // 平面点匹配关系

                    //ceres::LossFunction *loss_function = NULL;
                    // 使用Huber核函数来减少外点的影响，Algorithm 1: Lidar Odometry中有
                    // 设置ceres求解器的配置参数
                    // 创建一个HuberLoss函数作为误差项的损失函数，其中的参数0.1表示损失函数的阈值。
                    // HuberLoss是一种鲁棒性损失函数，用于减小异常值对优化结果的影响。
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    // 创建一个对象，用于对四元数进行参数化，保持单位长度
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    // 创建一个Problem的配置选项对象，用于设置Ceres Solver求解优化问题时的一些参数选项。
                    ceres::Problem::Options problem_options;

                    //创建一个Ceres Problem对象，使用之前设置好的问题选项problem_options
                    ceres::Problem problem(problem_options);
                    // 添加一个参数块到问题中。para_q是一个指向四元数的指针，表示需要优化的四元数变量。4表示该参数块的维度
                    // q_parameterization是之前设置的四元数的参数化方式
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    // 添加另一个参数块到问题中。para_t是一个指向平移向量的指针，表示需要优化的平移变量，3表示维度
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel; // 用于存储具有强度信息的三维点结构
                    std::vector<int> pointSearchInd; //用于存储最近邻搜索时返回的最近邻点的索引
                    std::vector<float> pointSearchSqDis;  //存储最近邻搜索时返回的最近邻点与给定点之间的平方距离

                    TicToc t_data; // 开始计时
                    // find correspondence for corner features
                    // 基于最近邻原理建立corner特征点之间关联，对每一个极大边线点去上一帧的次极大边线点集中找匹配
                    
                    //*************************遍历所有极大边线点-边线点匹配**************************//
                    for (int i = 0; i < cornerPointsSharpNum; ++i)  // 遍历所有极大边线点
                    {
                        // 将当前帧的极大边线点坐标转换到起始帧坐标系下，存储在pointSel中。统一参考坐标系
                        // TransformToStart函数用于将当前帧中的点坐标转换到上一帧Lidar坐标系下。
                        // &(cornerPointsSharp->points[i])：获取当前帧中一个极大边线点的指针，该点的坐标是相机坐标系或雷达坐标系下的。
                        // &pointSel：用于存储转换后的点坐标的pcl::PointXYZI类型的变量。
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);

                        // 使用kdtree对pointSel进行最近邻搜索，找到上一帧中与其最接近的1个点，并存储最近点的索引和最短距离
                        // kdtree中存储的是上一帧的次极大边线点，即在上一帧的次极大边线点集合中寻找当前帧中极大边线点的最近邻点
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点l有效
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];  // 存放最近点索引
                            // 获取上一帧中最近点的scanid
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD; // 最小距离阈值

                            // 从增加扫描线的方向寻找
                            // search in the direction of increasing scan line
                            // 从最近点附近开始，遍历上一帧的边线点，在上一帧中寻找最近点
                            // laserCloudCornerLast存储上一帧中的边线点
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                // 在当前扫描线附近范围内寻找边线点
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                // 如果超出扫描线附近范围则退出当次循环
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                // 距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                
                                // 如果距离小于最小平方距离，则认为找到了上一帧中附近扫描线上的最近点
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            // 从减少扫描线的方向寻找
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        // 至此，by kd-tree从上一帧中找到的最近点的附近扫描线上的最近邻点以唯一确定

                        // 两个最近点都有效，可以构建非线性优化问题
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            // 以笔记中的图为例 当前点为i
                            // 取当前点的坐标存储为三维向量
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            // 上一帧中的最近邻点为j
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            // 上一帧最近点附近扫描线上的最近点
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            // 如果需要进行时间畸变校正，则通过以下计算得到时间补偿因子s的值
                            if (DISTORTION)  // 初始值为0
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0; // 运动补偿系数，kitti数据集的点云已经被补偿过，所以s = 1.0
                            
                            // 用于构建边缘特征的残差项，并将其添加到优化问题中
                            // LidarEdgeFactor::Create 用于创建一个边缘特征的代价函数对象
                            // 这个代价函数对象需要提供当前特征点(curr_point)、上一个关键帧中的两个边缘特征点(last_point_a和last_point_b)，以及时间补偿因子(s)作为输入参数

                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            // 将代价函数添加到优化问题中
                            // 参数分别为代价函数，损失函数，旋转参数和平移参数 （旋转和平移为优化变量）
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            
                            corner_correspondence++;  // 边线点匹配关系
                        }
                    }

                    // find correspondence for plane features
                    //*************************遍历所有极小平面点-平面点匹配**************************//
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        // 统一坐标系，将当前帧坐标系转换到起始坐标系下
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        // kd-tree寻找上一帧中的最近邻点
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        // 3个最近邻点构成一个平面
                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0]; // 上一帧中最近邻点l的索引

                            // get closest point's scan ID
                            // 获取最近邻点l的scanid
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                            
                            // 在两个方向上搜索扫描线附近的最近邻点m和同一条扫描线上的点l
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)  // m和l这两个点都找到了
                            {
                                
                                // 获取点的三维坐标并存储为向量
                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                // 构造平面代价函数并添加残差项，优化变量为旋转和平移
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                // 平面点匹配关系
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());  // 结束计时，输出帧间匹配所需时间

                    // 如果找到的关联点少于10个，则警告
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;  // 开始计时
                    // 设定求解器类型，最大迭代次数，不输出过程信息，优化报告存入summary
                    ceres::Solver::Options options; // 设置优化器选项
                    options.linear_solver_type = ceres::DENSE_QR; // 设置线性求解器类型-QR分解
                    options.max_num_iterations = 4;  // 最大迭代次数
                    options.minimizer_progress_to_stdout = false;  //不打印输出
                    ceres::Solver::Summary summary;  // 创建summary对象用于存储求解器的汇总信息
                    // 基于构建的所有残差项，求解最优的当前帧位姿与上一帧位姿的位姿增量：para_q和para_t
                    ceres::Solve(options, &problem, &summary); // 调用solve函数执行优化求解，并将求解结果保存在summary中
                    printf("solver time %f ms \n", t_solver.toc()); // 输出求解时间
                }
                printf("optimization twice time %f \n", t_opt.toc());  // 迭代优化两次所需时间

                // 更新帧间匹配的结果，得到激光雷达里程计在世界坐标系下的位姿
                // 用最新计算出的位姿增量，更新上一帧的位姿，得到当前帧的位姿，注意这里说的位姿都指的是世界坐标系下的位姿
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

//********************************发布结果并为下一次帧间匹配做准备********************************************//
            TicToc t_pub;  // 开始计时

            // publish odometry
            // 创建nav_msgs::Odometry消息类型，把信息导入，并发布
            nav_msgs::Odometry laserOdometry; // 创建一个激光里程计消息对象
            laserOdometry.header.frame_id = "camera_init";  // 参考坐标系
            laserOdometry.child_frame_id = "/laser_odom";   //子坐标系
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);//设置激光里程计消息的时间戳为当前帧的时间
            // 设置激光里程计消息的姿态信息，即旋转四元数（当前帧相对于世界坐标系的）
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            // 设置激光里程计消息的位置信息，即平移向量（当前帧相对于世界坐标系的）
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            // 发布激光里程计消息，将其发送到相应的ROS话题中，以供其他节点或系统使用。
            pubLaserOdometry.publish(laserOdometry);

            // 创建一个包含位姿信息的激光位姿消息对象。
            geometry_msgs::PoseStamped laserPose;
            // 将激光位姿消息的头部信息、位姿、时间戳设置为与激光里程计消息相同
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            // 将激光位姿消息添加到激光路径消息中
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "camera_init";  // 设置参考坐标系
            // 发布激光路径消息，其他节点或系统可以获取到激光扫描的运动轨迹
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0) // 跳过
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    // 将当前帧中的点云坐标转换到当前帧的结束时刻
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            // 位姿估计完毕之后，当前边线点和平面点就变成了上一帧的边线点和平面点，
            // 把索引和数量都转移过去
            // 实现两帧点云之间的指针交换

            // 创建临时指针laserCloudTemp，赋值为次极大边线点指针
            // 实现交换 cornerPointsLessSharp和laserCloudCornerLast
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;  
            cornerPointsLessSharp = laserCloudCornerLast;  // 存储上一帧边线点
            laserCloudCornerLast = laserCloudTemp;   // 存储次极大边线点

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;  // 存储上一帧平面点
            laserCloudSurfLast = laserCloudTemp;  // 存储次极小平面点

            // 上一帧中的边线点和平面点
            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            // 设置kd-tree的输入，为上一帧的边线点和平面点
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            // 为了控制最后一个节点的执行频率， 只有整除时才发布结果
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                // 发布次极大边线点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                // 发布次极小平面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                // 原封不动的转发当前帧点云
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());  // 输出发布结果的时间
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc()); //输出整个帧间匹配的时间
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++; // 完成一帧的处理
        }
        rate.sleep();
        // 用于控制循环的执行速率。当调用该函数时，它会根据之前设置的频率计算出需要休眠的时间，以确保循环以指定的频率运行。
    }
    return 0;
}