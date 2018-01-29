#pragma once

#ifndef MLCAPTURE_PROCESS_H
#define MLCAPTURE_PROCESS_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#define raw_image_width 176 //Define the camera input width
#define raw_image_height 144 //Define the camera input height
#define image_roi_sy 100 //Define the region of interest start y
#define image_width 320 //Define the processing image width
#define image_height (raw_image_height - image_roi_sy) //Define the processing image height

//Filtering options
#define dlane_min_h 0
#define dlane_min_s 0
#define dlane_min_v 0
#define dlane_max_h 165
#define dlane_max_s 255
#define dlane_max_v 45
#define dlane_box_blur 3

//Line detection constants
#define min_length 0.2f * image_height
#define max_connect_length 0.08f * image_height
#define min_sym_diff 0.1f * image_height
#define horz_dead_zone 0.05f * image_width
#define vert_dead_zone 0.05f * image_height
#define dmin_angle 30
#define dmax_angle 90
#define avg_max_angle_diff 60 //The total allowed difference of line angle allowed between each line

//Sobel thresholds
#define abs_sobel_thresh_min_x 40
#define abs_sobel_thresh_max_x 150
#define abs_sobel_thresh_min_y 5
#define abs_sobel_thresh_max_y 150
//#define abs_sobel_thresh_kernel 11

//Angle calculation constants
#define displacement_line 0.45f //Where to put the center line
#define displacement_left displacement_line * image_width
#define displacement_right (1.0f - displacement_line) * image_width


//unsigned int r_image_height = raw_image_height, r_image_width = raw_image_width;

void filter_image(cv::Mat&, cv::Mat&, cv::Mat&, int, int, int, int, int, int, int);
void find_lines(cv::Mat&, std::vector<cv::Vec4i>&);
bool find_rect(cv::Mat&, cv::Mat&, std::vector<std::vector<cv::Point>>&, cv::RotatedRect&);
void find_rects(cv::Mat&, cv::Mat&, std::vector<std::vector<cv::Point>>&, std::vector<cv::RotatedRect>&);
bool rect_center_line(cv::RotatedRect&, cv::Vec4i&);
void rects_to_lines(std::vector<cv::RotatedRect>&, std::vector<cv::Vec4i>&);
void filter_lines(std::vector<cv::Vec4i>&, int, int);
void cluster_similar_lines(std::vector<cv::Vec4i>&);
void get_lanes(std::vector<cv::Vec4i>&, cv::Vec4i&, cv::Vec4i&);
void calc_disp(float&, float&, cv::Vec4i&, cv::Mat&);
void calculate_displacement(float&, float&, cv::Vec4i&, cv::Vec4i&, cv::Mat&);
#endif
