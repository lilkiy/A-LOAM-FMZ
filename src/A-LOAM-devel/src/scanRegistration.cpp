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
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv2/imgproc.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

//扫描周期, velodyne频率10Hz，周期0.1s
const double scanPeriod = 0.1;//1秒扫10次

//初始化控制变量
const int systemDelay = 0; // 初始化延迟阈值，LOAM中弃用前20帧，此处不弃用数据
int systemInitCount = 0;  // 初始化计数器
bool systemInited = false;

//激光雷达线数 此处不是常量
int N_SCANS = 0;

//点云曲率, 40000为一帧点云中点的最大数量 数组
float cloudCurvature[400000];  //cloudCurvature是存储每个点的曲率
//曲率点对应的序号 数组
int cloudSortInd[400000];
//点是否筛选过标志：0-未筛选过，1-筛选过
int cloudNeighborPicked[400000];
//点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

// 创建6个发布器和一个存储发布器对象的容器
ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false; // 是否创建每条扫描线的发布器对象

double MINIMUM_RANGE = 0.1; // 最小有效距离

// 函数模板
// 从输入点云中移除雷达周边过近的点，并将结果保存在输出点云中，并设定距离阈值参数
// pcl::PointCloud<PointT> 表示点云数据结构
// PointT 是点云中点的类型。
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 假如输入输出点云不使用同一个变量，则需要将输出点云的时间戳和容器大小与输入点云同步
    if (&cloud_in != &cloud_out)
    {
        // 头部信息通常包含点云的时间戳、坐标系、帧号等元数据。
        cloud_out.header = cloud_in.header;
        // 将输出点云的容器大小调整为与输入点云中的点数相同
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    // 遍历输入点云中的点
    // 把点云距离小于给定阈值的去除掉,也就是去除距离雷达过近的点
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue; // 去除距离过小的点，即不把该点添加到输出点云中
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    // 说明有点被去除了，需要重新调整输出点云的大小
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    // 这里是对每条扫描线上的点云进行直通滤波，因此设置点云的高度为1（表示单行），宽度为数量，稠密点云
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

// 激光点云回调函数，处理订阅器对象获得的消息
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 如果系统没有初始化的话，就等几帧
    //------------1、点云预处理----------------//
    if (!systemInited)
    { 
        systemInitCount++; // 系统初始化计数器
        if (systemInitCount >= systemDelay)  // 如果达到了初始化延迟阈值（这里为0），则认为完成了初始化
        {
            systemInited = true;
        }
        else
            return;
    }

    // 作者自己设计的计时类，以构造函数为起始时间，以toc()函数为终止时间，并返回时间间隔(ms)
    TicToc t_whole;  // 开始计时（创建对象时，以构造函数为起始时间）
    TicToc t_prepare;

    //记录每个scan（一次扫描）有曲率的点的开始和结束索引，初始值为0
    // 一次扫描包含了多条激光线的扫描过程
    //分别用scanStartInd数组和scanEndInd数组记录
    // N_SCANS为激光雷达线数
    std::vector<int> scanStartInd(N_SCANS, 0); //有曲率点的开始索引和结束索引
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn; // 存储三维点云数据
    // 把点云从ros格式转到pcl的格式
    // *laserCloudMsg获取点云消息
    // 将 ROS 消息类型的点云数据 laserCloudMsg 转换为 PCL 中的点云对象 laserCloudIn
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    // 去除掉点云中的nan点  // 移除指定索引处的空点
    // 在点云数据中，NaN点通常表示无效的或缺失的数据，这可能是由于传感器错误、测量错误、遮挡、反射不足等原因导致的。
    // 去除NaN点的方法通常涉及遍历点云中的每个点，检查其坐标值是否为NaN，并将非NaN点保留在新的点云中。
    // 如果提供了索引向量 indices，那么只有在对应索引处的点才会进行检查和移除操作。
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 去除距离小于阈值的点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);
    
    // 获取点云中点的数量
    // laserCloudIn 是一个点云对象，而 points 是该对象中存储点数据的容器。
    // 通过调用 size() 函数，可以获取该容器（点云）中点的数量
    int cloudSize = laserCloudIn.points.size();

    // ---------------------2、计算点云角度范围------------//
    // 计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反就相当于转成了逆时针
    //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
    // 计算了点云数据中第一个点的起始角度
    // atan2反三角函数用于计算给定点的反正切值，计算了点 (x, y) 相对于原点的方位角。
    // 由于起始角度通常以X轴正向为0度，而Y轴正向为90度，因此我们使用 -atan2() 来得到相对于X轴正向的角度。
    // atan2函数返回 y/x 的反正切值，以弧度表示，取值范围为(-π,π]
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;
    // atan2范围是[-Pi,PI]，这里加上2PI是为了保证起始到结束相差2PI符合实际
    //lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi
    // 一个scan扫描周期是2pi，起始角度和结束角度之间相差一个周期

    //结束方位角与开始方位角差值控制在(PI,3*PI)范围,atan2范围是[-Pi,PI]在此基础上加上2*pi
    // 总有一些例外，比如这里大于3PI，和小于PI，就需要做一些调整到合理范围
    //扫描场在右半部分（一、四象限）的情况
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    //扫描场在左半部分（二、三象限）的情况
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    //对于每个点云点计算对应的扫描线，主要掌握其计算方法，实际现在的3d激光雷达驱动中都已经集成了每个点云点的线号、时间戳和角度等信息，不需要再单独计算.

    // ---------------------3、根据角度将点划入不同线中------------//
    bool halfPassed = false; //lidar扫描线是否旋转过半
    int count = cloudSize; // 点云数量
    PointType point;
    // 定义了一个名为 laserCloudScans 的向量，其中每个元素都是 pcl::PointCloud<PointType> 类型的点云数据。向量的大小为 N_SCANS
    // 在激光雷达数据处理中，通常将激光扫描数据按照不同的线进行分组，每个线或层对应一个点云数据。
    // 该向量的每个元素对应一条线的所扫描的点云数据pcl::PointCloud<PointType>
    // 该点云数据是由类型为PointType的元素组成的
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // 遍历每一个点云点
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // 计算当前点的俯仰角
        // 计算垂直视场角，决定这个激光点所在的scanID
        // 通过计算垂直视场角确定激光点在哪个扫描线上（N_SCANS线激光雷达）

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16)
        {
            // +-15°的垂直视场，垂直角度分辨率2°，-15°时的scanID = 0
            /*
             * 垂直视场角，可以算作每个点的
             * 如果是16线激光雷达，结算出的angle应该在-15~15之间
             */
            scanID = int((angle + 15) / 2 + 0.5);
            // scanID(0~15)
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;  // 点云数量减少，说明是无效点
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK(); // 是一个宏定义，通常用于中断（break）ROS（机器人操作系统）中的执行流程。
            // 被用于某些特定的条件或错误处理情况下，用于提前退出或中断当前的ROS操作
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        // 计算水平角，即相邻两个点之间的夹角
        float ori = -atan2(point.y, point.x);  //当前点的角度
         // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        /*
         * 如果此时扫描没有过半，则halfPassed为false
         */ 
        if (!halfPassed)
        { 
            /*
             * 如果ori-startOri小于-0.5pi或大于1.5pi，则调整ori的角度
             */
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            
            // 扫描点是过半则设定halfPassed为true
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else  // 扫描过半
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        /*
         * relTime 是一个0~1之间的小数，代表占用扫描时间的比例，乘以扫描时间得到真实扫描时刻，
         * scanPeriod扫描时间默认为0.1s 即10Hz
         */
        float relTime = (ori - startOri) / (endOri - startOri);
        //1次扫描时长为0.1秒，通过角度占比，来计算当前激光点距离当次扫描开始时刻的时间差
        // scanID通过上面计算垂直角计算得到
        point.intensity = scanID + scanPeriod * relTime; 
        // laserCloudScans存储这一条扫描线上的点云点
        laserCloudScans[scanID].push_back(point); 
    }
    
    cloudSize = count; // count存储的是点云数量
    printf("points size %d \n", cloudSize);

    // ---------------------4、计算点云点曲率------------//
    //创建了一个指向 pcl::PointCloud<PointType> 类型的智能指针 laserCloud,指向一个新创建的空的点云数据集
    //可以通过操作这个指针来添加、修改或访问其中的点云数据。
    //通过 new 运算符动态分配了一个 pcl::PointCloud<PointType> 对象。
    //这个对象表示一个点云数据集,点云数据集通常用于存储激光雷达获取到的点云信息
    // 通过使用智能指针 Ptr，我们可以方便地管理和操作这个点云数据集对象
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        // 这两个数组存放的是每条扫描线上可以计算曲率的初始点和结束点的索引
        // 前5个点和后5个点都无法计算曲率，因为他们不满足左右两侧各有5个点
        scanStartInd[i] = laserCloud->size() + 5;  // 此时的size大小为0
        *laserCloud += laserCloudScans[i]; //将所有的点按照线号从小到大放入一个容器
        scanEndInd[i] = laserCloud->size() - 6;  // 此时的size大小为点云数据总数
    }

    // 将一帧无序点云转换成有序点云消耗的时间
    printf("prepare time %f \n", t_prepare.toc());

    // 理论上去除前后五个点的点云点都应该能够计算曲率
    for (int i = 5; i < cloudSize - 5; i++)  // cloudSiz为点云数量
    { 
        // 去目标点左右两侧各5个点，与目标点的坐标作差
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        // cloudCurvature是存储每个点的曲率
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;  // 存储点云点的序号，也就是点的索引
        cloudNeighborPicked[i] = 0;  // 是否筛选过的标志 0-未筛选
        cloudLabel[i] = 0;  // 点分类标号（基于曲率）  0-曲率很小
    }
    
    // ---------------------5、按照曲率提取特征点------------//
    TicToc t_pts;  // 创建一个计时器对象，用于记录特定代码块的执行时间

    // 各类特征点以点云对象的形式保存
    pcl::PointCloud<PointType> cornerPointsSharp;  // 极大边线点
    pcl::PointCloud<PointType> cornerPointsLessSharp; // 次极大边线点
    pcl::PointCloud<PointType> surfPointsFlat;  // 极小平面点
    pcl::PointCloud<PointType> surfPointsLessFlat;  // 次极小平面点（经过降采样）

    // 对每条线扫scan进行操作（曲率排序，选取对应特征点）
    float t_q_sort = 0;  // 用来记录排序花费的总时间
    for (int i = 0; i < N_SCANS; i++)
    {
        // 如果最后一个可算曲率的点与第一个的差小于6，说明无法分成6个扇区，跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        //创建了一个指向pcl::PointCloud<PointType>类型的智能指针surfPointsLessFlatScan，
        //并通过new关键字动态分配了一个新的pcl::PointCloud<PointType>对象。
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 为了使特征点均匀分布，将一个scan（即一次扫描）分成6个扇区
        for (int j = 0; j < 6; j++)
        {
            // 把每条扫描线划分为相等的6个扇区，计算每个扇区的起始点与结束点索引
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp; //创建一个计时器对象
            // 按照曲率进行升序排序
            // 对cloudSortInd数组中从索引sp到索引ep之间的元素进行排序
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // t_q_sort累计每个扇区曲率排序时间总和
            // tic()函数用于开始计时，记录当前时间点。
            // toc()函数用于结束计时，计算从上次调用tic()到当前时间点之间的时间间隔。
            t_q_sort += t_tmp.toc();

            // 选取极大边线点（2个）和次极大边线点（20个）
            int largestPickedNum = 0;
            // 从最大曲率往最小曲率遍历，寻找边线点，并要求大于0.1
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];   // 获取该点的索引
                // cloudSortInd数组中已经按曲率升序排序完毕

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)  //一开始所有点云数据都默认未筛选过
                {

                    largestPickedNum++; // 满足要求的点云点数目 
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]); // 极大边线点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]); // 次极大边线点
                    }
                    else if (largestPickedNum <= 20)
                    {            
                        //点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小
                        // 超过2个选择点以后，设置为次极大边线点，仅放入次极大边线点容器             
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]); // 次极大边线点
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; // 标志这个点已经被筛选过了
                    // ID为ind的特征点的相邻scan点距离的平方 <= 0.05的点标记为选择过，避免特征点密集分布
                    // 右侧
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1; //距离的平方 <= 0.05的点标记为选择过
                    }
                    // 右侧
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

             // 选取极小平面点（4个）
            int smallestPickedNum = 0;
            // 按曲率从小到大选取
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k]; // 获取该点的索引

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1;  
                    surfPointsFlat.push_back(laserCloud->points[ind]);  // 极小平面点

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; // 标记为已选择过
                    // 左侧
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1; // 将周围点标记为已选择过
                    }
                    // 右侧
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]); // 剩下的都是次极小平面点
                }
            }
        }

        // 对每一条scan线上的次极小平面点进行一次降采样
        // surfPointsLessFlatScan是一个未经降采样的点云数据。
        // surfPointsLessFlatScanDS是降采样后的点云数据。
        // pcl::VoxelGrid<PointType>是一个体素格滤波器，用于进行点云的降采样操作。
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 将要降采样的点云设置为滤波器的输入。
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        // 设置体素格的尺寸，这里是每个维度上的尺寸都是0.2。
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        // 应用滤波器，将降采样后的点云数据存储在surfPointsLessFlatScanDS中。
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        // 每次处理得到新的降采样点云数据时，都将其累积到surfPointsLessFlat中
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);  // 排序时间
    printf("seperate points time %f \n", t_pts.toc());  // 第5步按照曲率提取特征点的时间

    // ---------------------6、发布各类话题------------//
    // 发布有序点云，极大/次极大边线点，极小/次极小平面点，按需发布每条扫描线上的点云
    // 作用：将处理后的点云数据转换为ROS消息格式，并发布到指定话题
    
    // 创建一个sensor_msgs::PointCloud2类型的消息变量laserCloudOutMsg
    sensor_msgs::PointCloud2 laserCloudOutMsg; 
    // 将pcl点云数据*laserCloud转换为ROS消息格式，并存储到laserCloudOutMsg中
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    // 设置laserCloudOutMsg的时间戳与原始点云消息的时间戳一致
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    // 设置laserCloudOutMsg的坐标系为"/camera_init"
    laserCloudOutMsg.header.frame_id = "camera_init";
    // 将ROS消息laserCloudOutMsg发布到名为pubLaserCloud的话题
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scam 按需发布每条扫描线上的点云

    if(PUB_EACH_LINE)   // 是否创建每条扫描线的发布器对象
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);  // 把每条扫描线上的点云数据转换为ROS消息格式
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "camera_init";
            pubEachScan[i].publish(scanMsg); // 发布ROS消息
        }
    }

    // 输出整个扫描注册过程的时间
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

// argc:表示命令行参数的数量；
// argv:是一个指向字符指针数组的指针，用于存储命令行参数的字符串。每个字符串代表一个命令行参数，其中第一个参数是程序的名称,后续参数是用户传递的其他参数。
int main(int argc, char **argv)
{
    // 用于初始化ROS节点的函数,ROS系统会根据提供的命令行参数初始化节点
    // scanRegistration为ROS节点名称
    ros::init(argc, argv, "scanRegistration");

    // 创建了一个名为nh的ros::NodeHandle对象
    // nh是一个ROS节点句柄(NodeHandle)，在创建ROS节点后，会使用节点句柄来访问ROS的各种功能和服务。
    // 该对象用于与当前节点进行通信，允许节点执行各种ROS相关的操作
    ros::NodeHandle nh;

    // 从配置文件中获取激光雷达的线数
    // param<int>是一个模板函数，用于获取参数的值，并指定参数的类型。
    // scan_line 是要获取的参数的名称
    // N_SCANS 是一个变量，用于接收获取到的参数值。
    // 在调用该模板函数时，如果指定的参数存在，则将参数的值赋给接收变量
    // 16是默认值，当要获取的参数scan_line不存在时，使用默认值
    nh.param<int>("scan_line", N_SCANS, 16);

    // 获取最小有效距离
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    // 输出当前激光雷达线数
    printf("scan line number %d \n", N_SCANS);

    // 只有线束是16 32 64的才可以继续
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    
    // 1个订阅话题 点云 (A-LOAM中去除了imu)
    // 主要的回调函数：laserCloudHandler
    // 使用ROS的C++客户端库中的ros::Subscriber类创建了一个名为subLaserCloud的订阅器对象
    // 该订阅器订阅了名为/velodyne_points的sensor_msgs::PointCloud2类型的ROS话题。
    // 通过对象nh调用subscribe函数来创建订阅器，
    // 模板参数<sensor_msgs::PointCloud2>：指定订阅器接收的消息类型。
    // ("/velodyne_points")：指定要订阅的话题名称
    // 100：指定订阅器的消息队列长度，表示可以同时接收和存储的未处理消息的最大数量。
    // laserCloudHandler：是一个回调函数，当接收到"/velodyne_points"话题发布的消息时，将调用该回调函数对消息进行处理。
    // 回调函数是用户自定义的。
    // 总结：创建了一个订阅器对象，当接收到所订阅的话题"/velodyne_points"发布的PointCloud2类型的消息时，调用回调函数对消息进行处理。
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    // 6个发布话题
    // 创建了6个发布器对象，用于发布激光点云消息到6个不同的话题
    // 创建一个名为pubLaserCloud的发布器对象，用于发布激光点云消息。
    // 通过对象nh调用advertise函数来创建发布器
    // sensor_msgs::PointCloud2，指定所发布的消息类型，即激光点云消息类型。
    // ("/velodyne_cloud_2")：指定要发布的话题名称
    // 100：指定发布器的消息队列长度，表示可以同时缓存和发布的消息的最大数量。
    // 总结：创建了一个发布器对象，用于发布激光点云消息到"/velodyne_cloud_2"话题
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    // 如果PUB_EACH_LINE == true ,则发布每条扫描线上的点云点
    if(PUB_EACH_LINE)
    {
        // 遍历每个扫描线
        for(int i = 0; i < N_SCANS; i++)
        {
            // 创建发布器对象，用于发布指定扫描线的激光点云数据。
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            // 将创建的每个扫描线的发布器对象添加到容器中
            pubEachScan.push_back(tmp);
        }
    }
    // ros::spin()进入循环，一直调用回调函数，用户输入Ctrl+C退出
    ros::spin();

    return 0;
}
