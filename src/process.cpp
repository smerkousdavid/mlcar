#include <cfloat>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "process.h"

const double get_max(cv::Mat &to_get) {
	double min_v, max_v;
	cv::minMaxLoc(to_get, &min_v, &max_v);
	return max_v;
}

const void round_and_scale(cv::Mat &to_scale, cv::Mat &scaled, const double multiplier, const double divisor) {
	//cv::Mat divider(to_scale.rows, to_scale.cols, CV_8UC1, cv::Scalar(divisor)); //Copy the divider to the same size division matrix
	
	cv::Mat rounded(to_scale.rows, to_scale.cols, CV_8UC1); //Create a zero binary matrix (one byte per element)
	for(int row = 0; row < to_scale.rows; ++row) {
		for(int col = 0; col < to_scale.cols; ++col) {
			rounded.at<unsigned char>(row, col) = static_cast<unsigned char>((multiplier * std::abs(to_scale.at<float>(row, col))) / divisor); //Multiply the current matrix element value by the scalar and then divide it using the element wise division matrix
			unsigned char v = rounded.at<unsigned char>(row, col);
		}
	}
	
	rounded.copyTo(scaled); //Copy the new matrix over to the old one
}

template<typename T>
const void thresh_gradient(cv::Mat &to_thresh, cv::Mat &threshed_output, const T min_thresh, const T max_thresh) {
	cv::Mat threshed = cv::Mat::zeros(to_thresh.rows, to_thresh.cols, CV_8UC1); //Create a zero binary matrix (one byte per element)
	for(int row = 0; row < to_thresh.rows; ++row) {
		for(int col = 0; col < to_thresh.cols; ++col) {
			T &v= to_thresh.at<T>(row, col);
			if(v >= min_thresh && v <= max_thresh) {
				threshed.at<unsigned char>(row, col) = 1;
			} // else v = 0;
		}
	}
	threshed.copyTo(threshed_output);
}

const void abs_sobel_thresh(cv::Mat &input, cv::Mat &output, const bool is_x, const unsigned char min_thresh, const unsigned char max_thresh) {
	cv::Mat sobel;
	cv::Sobel(input, sobel, CV_32F, (int) is_x, (int) !is_x); //Run the sobel gradient detection
	sobel.copyTo(output);
	//cv::convertScaleAbs(sobel, output);
	round_and_scale(output, output, 255, get_max(output)); //Scale each element to 255 using the max gradient as a reference
	
	//cv::inRange(output, min_thresh, max_thresh, output);
	thresh_gradient<float>(output, output, min_thresh, max_thresh);
}

const void mag_thresh(cv::Mat &input, cv::Mat &output, const unsigned char kernel, const unsigned char min_thresh, const unsigned max_thresh) {
	cv::Mat sobel_x, sobel_y;
	cv::Sobel(input, output, CV_64F, 1, 0, kernel); //Gradient in the x direction
	cv::Sobel(input, output, CV_64F, 0, 1, kernel); //Gradient in the y direction
	
	//Get the distance between the x and y gradients (for every element)
	cv::Mat mag_thresh_gradient = cv::Mat(input.rows, input.cols, CV_64F);
	cv::Mat mag_binary = cv::Mat::zeros(input.rows, input.cols, CV_8UC1); //Create a zero binary matrix (one byte per element)
	for(int row = 0; row < input.rows; ++row) {
		double *ptr = mag_thresh_gradient.ptr<double>(row);
		for(int col = 0; col < input.cols; col++) {
			double *pixel = ptr;
			
			//Get the sobel pixel pointers
			unsigned char *x_ptr = sobel_x.ptr(row); //Get the sobel x pointer
			unsigned char *y_ptr = sobel_x.ptr(row); //Get the sobel x pointer

			//Set the mag_thresh_gradient pixel value to the x and y sobel gradients
			pixel[col] = std::sqrt((x_ptr[col] * x_ptr[col]) + (y_ptr[col] * y_ptr[col]));
		}
	}

	//Scale all of the values to be between 0 and 255 and use the max gradient as a refernce
	round_and_scale(mag_binary, mag_binary, 255, get_max(mag_binary));

	//Do a binary thresh gradient
	thresh_gradient<unsigned char>(mag_binary, output, min_thresh, max_thresh);
}

const void apply_thresholds(cv::Mat &input, cv::Mat &output_fixed) {
	cv::Mat sobel_x, sobel_y;
	abs_sobel_thresh(input, sobel_x, true, abs_sobel_thresh_min_x, abs_sobel_thresh_max_x);
	abs_sobel_thresh(input, sobel_y, false, abs_sobel_thresh_min_y, abs_sobel_thresh_max_y);

	cv::Mat output = cv::Mat::zeros(sobel_x.rows, sobel_x.cols, CV_8UC1); //Create the base binary image
	
	for(int row = 0; row < output.rows; ++row) {
		for(int col = 0; col < output.cols; ++col) {
			//Set the new gradient value to match that of the x and y gradients 
			output.at<unsigned char>(row, col) = (sobel_x.at<unsigned char>(row, col) && sobel_y.at<unsigned char>(row, col));
			//unsigned char o = output.at<unsigned char>(row, col);
			//if(o != 0) printf("OVERLAP!\n");
		}
	}

	output.copyTo(output_fixed);
}

void filter_image(cv::Mat &input, cv::Mat &midstep, cv::Mat &output, int mh, int ms, int mv, int ah, int as, int av, int blur) {
	//cv::cvtColor(input, output, CV_BGR2GRAY); //Change the image's colorspace
	//apply_thresholds(output, midstep); //Apply the Sobel thresholds

	/*
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	
	cv::Mat grad_x, grad_y;
cv::Mat abs_grad_x, abs_grad_y;


/// Gradient X
cv::Sobel( output, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
/// Gradient Y
cv::Sobel( output, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
convertScaleAbs( grad_y, abs_grad_y );

addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, midstep);
	*/

	/*cv::medianBlur(midstep, output, blur); //Smooth out the gradient binaries
	for(int row = 0; row < input.rows; ++row) {
		unsigned char *ptr = output.ptr(row);
		for(int col = 0; col < input.cols; col++) {
			//Set the new gradient value to match that of the x and y gradients 
			//ptr[col] = (sobel_x.ptr(row)[col] && sobel_y.ptr(row)[col]);
			
			if(ptr[col] == 1) printf("EQUALISH!\n");
		}
	}*/
	cv::cvtColor(input, output, CV_BGR2HSV);
	cv::inRange(output, cv::Scalar(mh, ms, mv), cv::Scalar(ah, as, av), output);
	//cv::blur(output, midstep, cv::Size(blur, blur));
	cv::medianBlur(output, midstep, blur);
	//cv::Canny(midstep, output, 60, 90);
	output = midstep(cv::Rect(cv::Point(0, image_roi_sy), cv::Point(raw_image_width, raw_image_height)));
}

void find_lines(cv::Mat &input, std::vector<cv::Vec4i> &lines) {
	cv::HoughLinesP(input, lines, CV_PI , CV_PI / 180, 20, min_length, max_connect_length);
}

bool find_rect(cv::Mat &input, cv::Mat &midstep, std::vector<std::vector<cv::Point>> &cont, cv::RotatedRect &rect) {
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(input, cont, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	float largest_len = 0;
	int largest_ind = -1;
	for(int i = 0; i < cont.size(); i++) {
		float len = cv::arcLength(cont[i], true);
			if(len > largest_len) {
			largest_len = len;
			largest_ind = i;
		}
	}

	if(largest_ind == -1) return false;

	rect = cv::minAreaRect(cv::Mat(cont[largest_ind]));

	cv::Point2f rect_points[4];
	rect.points(rect_points);

	for(int j = 0; j < 4; j++) {
		cv::line(midstep, rect_points[j], rect_points[(j+1) % 4], cv::Scalar(255, 0, 0), 1, 8);
	}
	return true;
}

void find_rects(cv::Mat &input, cv::Mat &midstep, std::vector<std::vector<cv::Point>> &cont, std::vector<cv::RotatedRect> &rects) {
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(input, cont, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	for(int i = 0; i < cont.size(); i++) {
		cv::RotatedRect rRect = cv::minAreaRect(cv::Mat(cont[i]));
		rects.push_back(rRect);
		
		cv::Point2f rect_points[4];
		rRect.points(rect_points);

		for(int j = 0; j < 4; j++) {
			cv::line(midstep, rect_points[j], rect_points[(j+1) % 4], cv::Scalar(255, 0, 0), 1, 8);
		}
	}
}

float rad_to_degrees(float rad) {
	return (180.0f / CV_PI) * rad;
}

float get_line_angle(const cv::Vec4i &line, bool generic) {
	float x_diff = std::abs(line[2] - line[0]);
	float y_diff = std::abs(line[3] - line[1]);
	float f_y, s_y;
	float offset = 0;
	
	if(line[0] < line[2]) {
		f_y = line[1];
		s_y = line[3];
	} else {
		f_y = line[3];
		s_y = line[1];
	}

	if(generic) {
		if(x_diff == 0) return 90.0f;
		return rad_to_degrees(atan(y_diff / x_diff));
	} else {
		if(f_y < s_y) {
			offset = 360.0f;
			y_diff = -y_diff;
			if(x_diff == 0) return 270.0;
		}
		if(x_diff == 0) return 90.0f;
		return rad_to_degrees(atan(y_diff / x_diff)) + offset;
	}

}

double get_line_length(const cv::Vec4i &line) {
	return std::sqrt(static_cast<double>((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3])));
			//powl(static_cast<double>(line[2] - line[0]), 2.0) + powl(static_cast<double>(line[3] - line[1]), 2.0);
	//return sqrtf(pow(line[2] - line[0], 2) + pow(line[3] - line[1], 2));
}

const void get_line_midpoint(const cv::Vec4i &line, int &x, int &y) {
	x = (line[0] + line[2]) / 2;
	y = (line[1] + line[3]) / 2;
}

bool rect_center_line(cv::RotatedRect &rect, cv::Vec4i &line) {
	cv::Point2f rect_points[4];
	cv::Vec4i rect_lines[4];
	rect.points(rect_points);

	//Convert the four points into the four rectangle lines
	for(int j = 0; j < 4; j++) {
		rect_lines[j] = cv::Vec4i(static_cast<int>(rect_points[j].x),
								static_cast<int>(rect_points[j].y),
								static_cast<int>(rect_points[(j+1) % 4].x),
								static_cast<int>(rect_points[(j+1) % 4].y));
	}

	//Define the new center line points
	int f_x, s_x, f_y, s_y;

	//Check to see which rectange is longer
	if(get_line_length(rect_lines[0]) > get_line_length(rect_lines[1])) {
		get_line_midpoint(rect_lines[1], f_x, f_y);
		get_line_midpoint(rect_lines[3], s_x, s_y);
	} else {
		get_line_midpoint(rect_lines[0], f_x, f_y);
		get_line_midpoint(rect_lines[2], s_x, s_y);
	}

	//Push the new center lines
	line = cv::Vec4i(f_x, f_y, s_x, s_y);

	//Make sure the line is long enough
	return (get_line_length(line) > min_length);
}

void rects_to_lines(std::vector<cv::RotatedRect> &rects, std::vector<cv::Vec4i> &lines) {
	for(int i = 0; i < rects.size(); i++) {
		cv::Point2f rect_points[4];
		cv::Vec4i rect_lines[4];
		rects[i].points(rect_points);

		//Convert the four points into the four rectangle lines
		for(int j = 0; j < 4; j++) {
			rect_lines[j] = cv::Vec4i(static_cast<int>(rect_points[j].x),
										static_cast<int>(rect_points[j].y),
										static_cast<int>(rect_points[(j+1) % 4].x),
										static_cast<int>(rect_points[(j+1) % 4].y));
		}

		//Define the new center line points
		int f_x, s_x, f_y, s_y;

		//Check to see which rectange is longer
		if(get_line_length(rect_lines[0]) > get_line_length(rect_lines[1])) {
			get_line_midpoint(rect_lines[1], f_x, f_y);
			get_line_midpoint(rect_lines[3], s_x, s_y);
		} else {
			get_line_midpoint(rect_lines[0], f_x, f_y);
			get_line_midpoint(rect_lines[2], s_x, s_y);
		}

		//Push the new center lines
		lines.push_back(cv::Vec4i(f_x, f_y, s_x, s_y));
	}
}

//Function to see if two lines are close to each other
bool is_line_close(const cv::Vec4i &o, const cv::Vec4i &t) {
	cv::Vec4i l1(o), l2(t);

	double len1 = get_line_length(l1);
	double len2 = get_line_length(l2);

	double product = (l1[2] - l1[0]) * (l2[2] - l2[0]) + (l1[3] - l1[1]) * (l2[3] - l2[1]);

	if(std::abs(product / (len1 * len2)) < cos(CV_PI / 4)) {
		return false;
	}

	float max_scale = 0.5f;
	int mx1 = static_cast<int>(static_cast<float>(l1[0] + l1[2]) * max_scale);
	int mx2 = static_cast<int>(static_cast<float>(l2[0] + l2[2]) * max_scale);
	int my1 = static_cast<int>(static_cast<float>(l1[1] + l1[3]) * max_scale);
	int my2 = static_cast<int>(static_cast<float>(l2[1] + l2[3]) * max_scale);

	double dist = std::sqrt(static_cast<double>((mx1 - mx2) * (mx1 - mx2) + (my1 - my2) * (my1 - my2)));

	if(dist > std::max(len1, len2) * 0.8f) { // maxScale) {
		return false;
	}

	return true;
}

float get_center_line() {
	return (float) image_width / 2.0f;
}

void filter_lines(std::vector<cv::Vec4i> &lines, int min_angle, int max_angle) {
	float left_dead_zone = horz_dead_zone;
	float right_dead_zone = image_width - left_dead_zone;
	float top_dead_zone = vert_dead_zone;
	float bottom_dead_zone = image_height - top_dead_zone;

	int ind = 0;
	for(std::vector<cv::Vec4i>::iterator it = lines.begin(); it != lines.end(); it++) {
		cv::Vec4i line = *it;
		ind++;
		float avg_x = ((float) line[0] + (float) line[2]) / 2.0f;
		float avg_y = ((float) line[1] + (float) line[3]) / 2.0f;
		if(avg_x < left_dead_zone || avg_x > right_dead_zone || avg_y < top_dead_zone || avg_y > bottom_dead_zone) {
			lines.erase(it);
			it = lines.begin();
			continue;
		}
	
		/*float line_angle = get_line_angle(line, true);
		if(line_angle < min_angle || line_angle > max_angle) {
			lines.erase(it);
			it = lines.begin();
			continue;
		}*/
	}
}

bool is_similar_to(cv::Vec4i &l1, cv::Vec4i &l2) {
	return (l1[0] == l2[0] && l1[1] == l2[1] && l1[2] == l2[2] && l1[3] == l2[3]) ||
			(l1[0] == l2[2] && l1[1] == l2[3] && l1[2] == l2[0] && l1[3] == l2[1]);
}

void cluster_similar_lines(std::vector<cv::Vec4i> &lines) {
	//Parition the common clusters into labeled clusters
	std::vector<int> labels;
	int num_of_clusts = cv::partition(lines, labels, is_line_close);

	std::vector<std::vector<cv::Vec4i> > clusters;

	//Create the new base cluster group
	for(int ind = 0; ind < num_of_clusts; ind++) {
		clusters.push_back(std::vector<cv::Vec4i>());
	}

	//Group the common clusters into new arrays
	for(int clust = 0; clust < labels.size(); clust++) {
		clusters[labels[clust]].push_back(lines[clust]);
	}

	//Get the absolute difference in each line
	std::vector<cv::Vec4i> abs_lines;
	for(std::vector<std::vector<cv::Vec4i> >::iterator it = 
					clusters.begin(); it != clusters.end(); it++) {
		std::vector<cv::Vec4i> n_lines = *it;

		cv::Point lowest(image_width, image_height), highest(0, 0);
		for(std::vector<cv::Vec4i>::iterator lIt = n_lines.begin();
						lIt != n_lines.end(); lIt++) {
			cv::Vec4i line = *lIt;

			int high_y = std::max(line[1], line[3]);
			int low_y = std::min(line[1], line[3]);

			cv::Point higher(((high_y == line[1]) ? line[0] : line[2]), high_y);
			cv::Point lower(((low_y == line[1]) ? line[0] : line[2]), low_y);

			if(higher.y > highest.y) highest = higher;
			if(lower.y < lowest.y) lowest = lower;
		}

		abs_lines.push_back(cv::Vec4i(lowest.x, lowest.y, highest.x, highest.y));
	}

	lines = abs_lines;
}

void get_lanes(std::vector<cv::Vec4i> &lines, cv::Vec4i &left, cv::Vec4i &right) {
	cv::Vec4i left_lane(image_width, 0, image_width, 0), right_lane(0, 0, 0, 0);

	for(std::vector<cv::Vec4i>::iterator it = 
					lines.begin(); it != lines.end(); it++) {
		cv::Vec4i line = *it;

		float avg_x = ((float) line[0] + (float) line[2]) / 2.0f;
		float avg_lx = ((float) left_lane[0] + (float) left_lane[2]) / 2.0f;
		float avg_rx = ((float) right_lane[0] + (float) right_lane[2]) / 2.0f;

		if(avg_x < avg_lx) {
			//if(std::max(line[1], line[3]) > std::max(left_lane[1], left_lane[3]) - min_sym_diff) {
				left_lane = line;
			//}
		}

		if(avg_x > avg_rx) {
			//if(std::max(line[1], line[3]) > std::max(right_lane[1], right_lane[3]) - min_sym_diff) {
				right_lane = line;
			//}
		}
	}

	if(is_similar_to(left_lane, right_lane)) {
		printf("ANGLE: %.2f\n", get_line_angle(right_lane, false));
		if(get_line_angle(right_lane, false) > 270.0f) {
			left = cv::Vec4i(0, image_height, 0, image_height);
			right = right_lane;
		} else {
			left = left_lane;
			right = cv::Vec4i(image_width, image_height, image_width, image_height);
		}
	} else {
		left = left_lane;
		right = right_lane;
	}
}

float get_slope(cv::Vec4i &line) {
	float lX = (float) std::min(line[0], line[2]);
	cv::Point f_point, s_point;
	if(lX == line[0]) {
		f_point = cv::Point(line[0], line[1]);
		s_point = cv::Point(line[2], line[3]);
	} else {
		f_point = cv::Point(line[2], line[3]);
		s_point = cv::Point(line[0], line[1]);
	}
	if((s_point.x - f_point.x) < 0.05 || (s_point.x - f_point.x) > -0.05) return -1.0f;
	return (((float) s_point.y) - ((float) f_point.y)) / (((float) s_point.x) - ((float) f_point.x));
}

float get_perc_from_center(float avg_x) {
	float disp = (avg_x - displacement_left);
	if(disp < 0) {
		disp /= displacement_left;	
	} else {
		disp /= displacement_right;
	}
	return disp; 
}

void calc_disp(float &disp, float &c_disp, cv::Vec4i &line, cv::Mat &img) {
	float l_x, h_x;

	//Get the lowest point (remember that y starts in the top left corner)
	if(line[1] > line[3]) {
		l_x = line[0];
		h_x = line[2];
	} else {
		l_x = line[2];
		h_x = line[0];
	}
	
	cv::line(img, cv::Point(displacement_left, 0), cv::Point(displacement_left, image_height), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
	cv::line(img, cv::Point(h_x, 0), cv::Point(h_x, image_height), cv::Scalar(120, 120, 255), 2, cv::LINE_AA);
	cv::line(img, cv::Point(l_x, 0), cv::Point(l_x, image_height), cv::Scalar(120, 120, 120), 2, cv::LINE_AA);

	disp = get_perc_from_center(h_x);
	c_disp = get_perc_from_center(l_x);
}

void calculate_displacement(float &disp, float &c_disp, cv::Vec4i &left, cv::Vec4i &right, cv::Mat &img) {
	float l_slope = get_slope(left);
	float r_slope = get_slope(right);

	float lb = -1.0f * ((l_slope * (float) left[0]) - (float) left[1]);
	float rb = -1.0f * ((r_slope * (float) right[0]) - (float) right[1]);

	float min_y = (float) std::max(std::min(left[1], left[3]), std::min(right[1], right[3]));
	float max_y = (float) std::min(std::max(left[1], left[3]), std::max(right[1], right[3]));

	float lx = (min_y - lb) / l_slope;
	float rx = (min_y - rb) / r_slope;
	float lcx = (max_y - lb) / l_slope;
	float rcx = (max_y - rb) / r_slope;

	if(l_slope == -1) {
		lx = left[0];
		lcx = lx;
	}

	if(r_slope == -1) {
		rx = right[0];
		rcx = rx;
	}
	
	float avg_x = (lx + rx) / 2.0f;
	float avgcx = (lcx + rcx) / 2.0f;

	//cv::circle(img, cv::Point(lx, min_y), 5, cv::Scalar(0, 255, 255), 5);
	//cv::circle(img, cv::Point(rx, min_y), 5, cv::Scalar(0, 255, 255), 5);
	cv::line(img, cv::Point(displacement_left, 0), cv::Point(displacement_left, image_height), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
	cv::line(img, cv::Point(avg_x, 0), cv::Point(avg_x, image_height), cv::Scalar(120, 120, 255), 2, cv::LINE_AA);
	//cv::circle(img, cv::Point(lcx, max_y), 5, cv::Scalar(0, 120, 255), 5);
	//cv::circle(img, cv::Point(rcx, max_y), 5, cv::Scalar(0, 120, 255), 5);
	cv::line(img, cv::Point(avgcx, 0), cv::Point(avgcx, image_height), cv::Scalar(120, 120, 120), 2, cv::LINE_AA);

	//printf("Disp: %.2f\n", displacement_left);
	//printf("Avg: %.2f\n", avg_x);
	//printf("CAvg: %.2f\n", avgcx);

	disp = get_perc_from_center(avg_x);
	c_disp = get_perc_from_center(avgcx);
}
