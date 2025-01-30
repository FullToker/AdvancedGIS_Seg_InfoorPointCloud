#include <eigen3/Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
// bool to control the window and the save
const bool is_show = true;
const bool is_save = false;

void cvshow(std::string window_name, cv::Mat &image_to_show,
            bool flag = is_show) {
  if (flag) {
    cv::imshow(window_name, image_to_show);
    cv::waitKey(0);
  }
}
void cvsave(std::string save_rpath, cv::Mat &image_to_save,
            bool flag = is_save) {
  if (flag) {
    cv::imwrite(save_rpath, image_to_save);
  }
}

void read_project_binaryImage(std::string filename) {}

/*hough transform aiming to detect lines*/
void hough_test(std::string filename) {
  /* read the ply file from local*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return;
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

  ///////////---------------------------------------//-----------------------------------------//

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

  cv::Mat image_houghp = image.clone();
  // cvwrite("../src/image.png", image);
  cvshow("image", image);

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
  // cv::imshow("Detected Lines and Endpoints", image);
  // cv::waitKey(0);
  cvshow("Detected Lines and Endpoints", image);

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

  cvsave("../src/houghLinesP.png", image_houghp);
  cvshow("Detected Lines and Endpoints", image_houghp);

  cv::destroyAllWindows();
}

/* region extraction using opencv*/
void region_extraction(std::string filename) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return;
  }

  std::cout << "Loaded " << cloud->points.size() << " points from " << filename
            << std::endl;

  //----------------------------------------------------------------------------////////
  /*visualization of the point cloud using PCL */
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

  std::cout << "Start to project to 2D..." << std::endl;

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

  ///////////---------------------------------------//-----------------------------------------//

  /*visualization the image(binary)*/
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

  cvshow("2D projection binary", image);

  /////////-------------------------------------------------------------------///////////////--------------------------------//
  // find the seed point
  int residual_pix = residual / pix_size;

  std::vector<cv::Point> corners = {
      cv::Point(0 + residual_pix, 0 + residual_pix),
      cv::Point(image.cols - 1 - residual_pix, 0 + residual_pix),
      cv::Point(0 + residual_pix, image.rows - 1 - residual_pix),
      cv::Point(image.cols - 1 - residual_pix, image.rows - 1 - residual_pix)};

  std::vector<cv::Point> directions = {cv::Point(1, 1), cv::Point(-1, 1),
                                       cv::Point(1, -1), cv::Point(-1, -1)};

  cv::Mat result = image.clone();
  cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

  cv::Point seed_pt;
  for (int i = 0; i < 4; i++) {
    cv::Point corner = corners[i];
    cv::Point dir = directions[i];
    int count = 0;
    cv::Point current = corner;

    while (current.x >= 0 && current.x < image.cols && current.y >= 0 &&
           current.y < image.rows) {
      if (image.at<uchar>(current.y, current.x) == 255) {
        count++;
      }
      current.x += dir.x;
      current.y += dir.y;
    }
    if (count % 2 == 1) {
      seed_pt = corner + 2 * dir;
      if (seed_pt.x >= 0 && seed_pt.x < image.cols && seed_pt.y >= 0 &&
          seed_pt.y < image.rows) {
        result.at<cv::Vec3b>(seed_pt.y, seed_pt.x) = cv::Vec3b(0, 0, 255);
      }
      break;
    }
  }
  cvshow("seed", result);

  ///------------------------------------------------------------------------------------------/////////
  /* use find contours to find the contour of the 2d projection */

  // do closed operation
  cv::Mat closed_img;
  int morph_size = 1;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
      cv::Point(morph_size, morph_size));
  cv::morphologyEx(image, closed_img, cv::MORPH_CLOSE, element);
  cvshow("closed", closed_img);

  std::cout << "find contours..." << std::endl;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(closed_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);
  std::cout << "find " << contours.size() << " contours in this image"
            << std::endl;

  cv::Mat contour_result(image.size(), CV_8U, cv::Scalar(255));
  /* -1: all contours, 150: color; 1: width of the line*/
  cv::drawContours(contour_result, contours, -1, 150, 1);
  cvshow("contours result", contour_result);

  /*	Approximates a polygonal curve(s) with the specified precision */
  std::vector<cv::Point> poly;
  cv::approxPolyDP(contours[0], poly, 5, true);
  // cv::polylines(contour_result, poly, true, 0, 1);

  cvshow("contours result", contour_result, false);

  // iterate all points in poly
  std::cout << "there are " << poly.size() << " points in poly" << std::endl;
  std::vector<cv::Point>::iterator itc = poly.begin();
  while (itc != poly.end()) {
    std::cout << "x: " << (*itc).x << "; y: " << (*itc).y << std::endl;
    ++itc;
  }

  //-------------------------------------------------------------------////////////////////////
  /* draw the convex hull*/
  std::vector<cv::Point> hull;
  cv::convexHull(contours[0], hull);
  std::cout << "size of the hull is: " << hull.size() << std::endl;
  cv::polylines(contour_result, hull, true, 166, 1);
  cvshow("convex hull", contour_result);

  /*
  // detect the defects
  std::vector<cv::Vec4i> defects;
  cv::convexityDefects(contours[0], hull, defects);
  std::cout << "size of defects: " << defects.size() << std::endl;
  std::cout << "first element : " << defects[0] << std::endl;
  */

  ///-------------------------------------------------------------------------------//--------------------------------//
  /* watershed*/
  cv::Mat markers = cv::Mat::zeros(contour_result.size(), CV_32SC1);

  markers.at<int>(seed_pt.x, seed_pt.y) = 1;

  for (int i = 0; i < contour_result.rows; i++) {
    for (int j = 0; j < contour_result.cols; j++) {
      if (contour_result.at<uchar>(i, j) < 100) {
        markers.at<int>(i, j) = 2;
      }
    }
  }

  cv::watershed(cv::Mat::ones(contour_result.size(), CV_8UC3), markers);

  cv::Mat water_result(contour_result.size(), CV_8UC3);
  for (int i = 0; i < markers.rows; i++) {
    for (int j = 0; j < markers.cols; j++) {
      int index = markers.at<int>(i, j);
      if (index == -1) {
        water_result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
      } else if (index == 1) {
        water_result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
      } else if (index == 2) {
        water_result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255); // 前景为蓝色
      }
    }
  }
  cvshow("watershed", water_result);
}

int main() {
  std::cout << "Hello, World!" << std::endl;

  std::string filename = "/media/fys/T7 "
                         "Shield/AdvancedGIS/rebuild/hdbscan_synth1/"
                         "label21.ply"; // path as a hypermeter
  region_extraction(filename);

  return 0;
}
