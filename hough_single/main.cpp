#include <eigen3/Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

/*
 * Author: Yushuo
 *
 * Date: 27.01.2025
 * PCL + cv + cpp test
 *
 * */

// short name of point in pcl
typedef pcl::PointXYZ PointT;

int main() {
  std::cout << "Hello, World!" << std::endl;

  /* read the ply file from local*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::string filename = "../hdbscan/label11.ply"; // path as a hypermeter

  if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return -1;
  }

  std::cout << "Loaded " << cloud->points.size() << " points from " << filename
            << std::endl;

  //----------------------------------------------------------------------------////////
  /*visualization of the point cloud*/
  pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
  viewer.setBackgroundColor(0, 0, 0);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      cloud_color_handler(cloud, 255, 255, 255); // White color
  viewer.addPointCloud(cloud, cloud_color_handler, "cloud");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer.addCoordinateSystem(1.0);
  viewer.spinOnce(100);
  //---------------------------------------------------------------/------------------------/----/

  /* project the target to 2D and rasterization
   *
   * main code is from taozijiangzi jun,
   *https://blog.csdn.net/qq_41753052/article/details/120115780
   *
   * */
  std::cout << "Start to project to 2D" << std::endl;

  PointT minpt, maxpt;
  pcl::getMinMax3D(*cloud, minpt, maxpt);
  std::cout << "the xmin is " << minpt.x << " and the ymin is " << minpt.y
            << std::endl;
  std::cout << "the xmax is " << maxpt.x << " and the ymax is " << maxpt.y
            << std::endl;

  /*point to matrix*/
  std::cout << "Start build matrix...";
  float pix_size = 0.04; // a hypermeter need to set
  float residual = 10;
  int row =
      static_cast<int>((maxpt.y + residual - (minpt.y - residual)) / pix_size);
  int col =
      static_cast<int>((maxpt.x + residual - (minpt.x - residual)) / pix_size);
  std::cout << "row is " << row << " and the col is " << col << std::endl;

  /*try my visualization*/
  cv::Mat image = cv::Mat::zeros(row, col, CV_8UC1);
  for (const auto &point : cloud->points) {
    /*
    int x = static_cast<int>(((point.x - minpt.x) / (maxpt.x - minpt.x)) *
                             (col - 1));
    int y = static_cast<int>(((point.y - minpt.y) / (maxpt.y - minpt.y)) *
                             (row - 1));
  */

    int x = static_cast<int>(
        ((point.x - minpt.x + residual) / (maxpt.x - minpt.x + 2 * residual)) *
        (col - 1));
    int y = static_cast<int>((row - 1) - ((point.y - minpt.y + residual) /
                                          (maxpt.y - minpt.y + 2 * residual)) *
                                             (row - 1));
    uchar value = static_cast<uchar>((point.z - std::min(0.0f, point.z)) /
                                     (std::max(0.0f, point.z)) * 255);
    image.at<uchar>(y, x) = value;
  }
  cv::imshow("image", image);
  cv::Mat image_houghp = image.clone();
  // cv::imwrite("../src/image.png", image);
  cv::waitKey(0);
  ///---------------------------------////////////------------------------------------

  /*
  // iterate the matrix
  Eigen::MatrixXf ground_matrix = Eigen::MatrixXf::Zero(row, col);
  int row_n, col_n;
  for (size_t i = 0; i < cloud->size(); i++) {
    row_n = static_cast<int>(
        row - ((cloud->points[i].y - (minpt.y - residual)) / pix_size));
    col_n = static_cast<int>((cloud->points[i].x - (minpt.x - residual)) /
                             pix_size);

    if (row_n >= 0 && row_n < row && col_n >= 0 && col_n < col) {
      ground_matrix(row_n, col_n) += 1;
    }
  }

  float total_sum = ground_matrix.sum();
  std::cout << "Total sum of all elements in the ground matrix: " << total_sum
            << std::endl;

  // binarization of the matrix
  ground_matrix = 255 * ground_matrix / ground_matrix.maxCoeff();

  float bin_threshold = 10;
  for (int i = 0; i < ground_matrix.rows(); i++) {
    for (int j = 0; j < ground_matrix.cols(); j++) {
      if (ground_matrix(i, j) > bin_threshold) {
        ground_matrix(i, j) = 255;
      } else {
        ground_matrix(i, j) = 0;
      }
    }
  }
  */

  /* start the hough*/
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(image, lines, 1, CV_PI / 180, 50, 50, 10);

  for (size_t i = 0; i < lines.size(); i++) {
    cv::Vec4i l = lines[i];
    cv::line(image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
             cv::Scalar(255), 3);
    // Draw endpoints
    cv::circle(image, cv::Point(l[0], l[1]), 5, cv::Scalar(255), cv::FILLED);
    cv::circle(image, cv::Point(l[2], l[3]), 5, cv::Scalar(255), cv::FILLED);
  }

  // cv::imwrite("../src/hough.png", image);

  // Optionally display the image
  cv::imshow("Detected Lines and Endpoints", image);
  cv::waitKey(0);

  //---------------------------------------------------------------------------------///////////////
  /* Probability Hough Transform*/
  double deltaRho = 3, deltaTheta = CV_PI / 180;
  double minVote = 20;
  double minLength = 20.0, maxGap = 25;
  cv::Scalar color = cv::Scalar(255, 0, 0);

  std::vector<cv::Vec4i> lineSegs;
  lineSegs.clear();
  cv::HoughLinesP(image_houghp, lineSegs, deltaRho, deltaTheta, minVote,
                  minLength, maxGap);

  std::vector<cv::Vec4i>::const_iterator it2 = lineSegs.begin();
  std::cout << "find " << lineSegs.size() << " line segments using HoughP!"
            << std::endl;
  while (it2 != lineSegs.end()) {
    cv::Point pt1((*it2)[0], (*it2)[1]);
    cv::Point pt2((*it2)[2], (*it2)[3]);

    cv::line(image_houghp, pt1, pt2, color, 1);
    cv::circle(image_houghp, pt1, 2, cv::Scalar(255), cv::FILLED);
    cv::circle(image_houghp, pt2, 2, cv::Scalar(255), cv::FILLED);

    ++it2;
  }

  cv::imwrite("../src/houghLinesP.png", image_houghp);
  cv::imshow("HoughlineP", image_houghp);
  cv::waitKey(0);

  cv::destroyAllWindows();

  return 0;
}
