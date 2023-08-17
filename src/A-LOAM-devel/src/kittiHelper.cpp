// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

// 定义lidar数据读取函数，返回一个容器，函数参数为激光雷达数据文件的路径
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_helper");//初始化ros节点kitti_helper
    ros::NodeHandle n("~"); //创建节点句柄
    std::string dataset_folder, sequence_number, output_bag_file;
    // 用于获取名为 "dataset_folder" 的参数的值，并将其存储在名为 dataset_folder 的变量中
    n.getParam("dataset_folder", dataset_folder); // 数据集文件夹
    n.getParam("sequence_number", sequence_number); //序列号
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    n.getParam("to_bag", to_bag); // 是否将数据集转换为rosbag格式
    if (to_bag)
        n.getParam("output_bag_file", output_bag_file); //获取bag输出文件路径
    int publish_delay; //发布延迟
    n.getParam("publish_delay", publish_delay);
    publish_delay = publish_delay <= 0 ? 1 : publish_delay; //延迟时间<=0 则设为1

    // ros::Publisher：这是ROS中用于发布消息的类
    // pub_laser_cloud：这是你创建的发布者对象的名称，使用该对象来发布激光点云消息
    // 调用 advertise 函数来创建发布者
    // sensor_msgs::PointCloud2：这是要发布的消息类型，即激光点云消息的类型。
    // "/velodyne_points"：这是发布者要发布消息的话题名称，激光点云消息将会被发布到该话题上
    // 2：这是发布者的消息队列大小，表示最多可以保留两条未发送的消息。
    // /velodyne_points话题是真正对算法有用的话题
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);

    // 创建了一个用于处理图像消息的图像传输对象，n为之前创建的节点句柄
    image_transport::ImageTransport it(n);
    // 创建一个图像发布者，用于发布图像消息，
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2);

    // 创建里程计发布者对象
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    nav_msgs::Odometry odomGT; // 创建一个里程计消息
    // 里程计消息用于描述移动物体的运动信息，包含了位置、方向、线性速度、角速度等信息
    odomGT.header.frame_id = "camera_init"; // 发布的数据是来自camera_init坐标系
    odomGT.child_frame_id = "/ground_truth";

    // 创建路径真值发布者对象
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    nav_msgs::Path pathGT; // 创建路径消息
    pathGT.header.frame_id = "camera_init";// 表示路径消息中的轨迹数据是相对于名为 "camera_init" 的坐标系定义的。

    // 获取时间戳地址文件
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt"; //创建一个字符串代码存储时间戳文件的路径
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);//创建一个输入文件流，用于读取文件内容，打开了dataset_folder+timestamp_path路径下的文件，并设置打开方式为输入

    // 读取轨迹真值文件，对于odometry和SLAM真实值轨迹直接来自于RTK，但是投影到了left camera
    std::string ground_truth_path = "results/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    rosbag::Bag bag_out; // 创建输出bag对象
    if (to_bag) // 是否将数据集转换为rosbag格式
        bag_out.open(output_bag_file, rosbag::bagmode::Write);
    
    // camera 相对于lidar的旋转矩阵
    Eigen::Matrix3d R_transform; //Matrix3d表示三维double类型的矩阵，旋转矩阵
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0; // 设置旋转矩阵初值
    Eigen::Quaterniond q_transform(R_transform); // 旋转矩阵转换为四元数

    std::string line; 
    std::size_t line_num = 0; // 图片编号

    // 10.0：表示期望的循环频率，即每秒执行的次数
    // publish_delay：这是一个变量，用于表示消息发布的时间间隔。根据这个值和期望的循环频率，计算出每次循环的间隔时间，以控制消息发布的频率。
    ros::Rate r(10.0 / publish_delay);// 创建速率对象，用于控制消息发布的频率
    //循环读 时间戳 文件 的 每一行 
    while (std::getline(timestamp_file, line) && ros::ok())
    {
        float timestamp = stof(line); // 把string转成浮点型float
        std::stringstream left_image_path, right_image_path;
        // 设置左右目图片路径
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        // 通过opencv读取图片
        cv::Mat left_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat right_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);

        // 读取轨迹真值文件，将读取的内容转换为位姿矩阵
        std::getline(ground_truth_file, line);//读取 ground_truth的每一行，存储到line中
        std::stringstream pose_stream(line);//创建一个字符串流对象，将之前读取的一行内容 line 转换为一个可供解析的流对象
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose; //声明位姿 矩阵 3行4列
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                // 从 pose_stream 流中提取一个单独的数据项，并将其存储在字符串变量 s 中。提取的数据项以空格作为分隔符。
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }

        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());//提取位姿矩阵的旋转部分，并转换为四元数
        Eigen::Quaterniond q = q_transform * q_w_i;
        q.normalize();
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();

        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        pubOdomGT.publish(odomGT);//发布里程计消息

        geometry_msgs::PoseStamped poseGT; //表示带有时间戳的位姿消息
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT);
        pubPathGT.publish(pathGT);//发布路径真值消息

        // read lidar point cloud
        std::stringstream lidar_data_path;
        // 读取激光雷达点云数据
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        // 调用读雷达数据的函数
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";

        std::vector<Eigen::Vector3d> lidar_points; // 声明雷达点的位置向量
        std::vector<float> lidar_intensities; // 声明雷达点的强度向量
        pcl::PointCloud<pcl::PointXYZI> laser_cloud; // 声明一个pcl的雷达点云
        // 每个点数据占四个float数据，分别是xyz，intensity，存到laser_cloud容器中
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }

        sensor_msgs::PointCloud2 laser_cloud_msg;//声明ros中的点云消息
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);//将pcl点云转换成上一行定义的ros点云消息
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "camera_init";
        pub_laser_cloud.publish(laser_cloud_msg); // 发布ros点云消息

        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();
        // 发布左右目图像
        pub_image_left.publish(image_left_msg);
        pub_image_right.publish(image_right_msg);

        if (to_bag)
        {
            // 写入rosbag
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/path_gt", ros::Time::now(), pathGT);
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);
        }

        line_num ++;//更新行数，相当于更新读取的文件序号
        r.sleep(); //进行延时，控制发布频率
    }
    bag_out.close();
    std::cout << "Done \n";


    return 0;
}