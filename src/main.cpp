#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <linux/videodev2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <nanomsg/nn.h>
#include <nanomsg/bus.h>

#include "mjpegserver.h"
#include "process.h"

//Common methods to handle printing and array manipulations
#define LOG(x) printf("%s\n", x)
#define CLEAR(x) memset(&(x), 0, sizeof(x))

//Common device constants
#define device_name "/dev/video1" //Define the camera name
#define server_port 8080
#define frame_update_delay 100 //Minimum time to wait before sending a new frame
#define nn_bind_addr "tcp://*:9090"
#define max_errors 10

//An abstract element for the buffers array
struct buffer {
	void *start;
	size_t length;
};

int fd = -1; //File device descriptor
int nn_s = -1; //nn socket descriptor
buffer *buffers; //The total frame buffers
unsigned int n_buffers, n_errors; //The current frame buffers size and error count
MJPEGServer mjpeg(server_port); //, debug(server_port + 1);
//cv::Mat frame(raw_image_height, raw_image_width, CV_8UC3, cv::Scalar(0, 0, 0)); //Create the new blank image
cv::Mat frame(raw_image_height, raw_image_width, CV_8UC1, cv::Scalar(0)); //Create the new blank image
pthread_t thread_recv, thread_send;
pthread_mutex_t settings_mutex = PTHREAD_MUTEX_INITIALIZER;
struct timeval last_time;

//Image settings
int source_select = 0,
	lane_min_h = dlane_min_h,
	lane_min_s = dlane_min_s,
	lane_min_v = dlane_min_v,
	lane_max_h = dlane_max_h,
	lane_max_s = dlane_max_s,
	lane_max_v = dlane_max_v,
	lane_blur = dlane_box_blur,
	lane_min_angle = dmin_angle,
	lane_max_angle = dmax_angle;

void display_mat(cv::Mat &frame) {
	static auto last_time = std::chrono::high_resolution_clock::now();
	auto current_time = std::chrono::high_resolution_clock::now();

	auto millis_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

	if(millis_diff > frame_update_delay) {
		if(frame.empty()) {
			LOG("Empty Frame!");
			return;
		}
		
		//Only update the frame every x millis
		std::vector<unsigned char> buff;
		std::vector<int> params;
	
		cv::imencode(".jpg", frame, buff, params);
		mjpeg.write(buff.data(), buff.size());

		last_time = current_time; //Update the last time struct
	}
}

void report_findings(float c_disp, float f_disp) {
	char to_report[50];
	int num_write = sprintf(to_report, "{\"cd\":%.5f,\"fd\":%.5f}", c_disp, f_disp);
	nn_send(nn_s, to_report, num_write, NN_DONTWAIT); // , 0);
}

//Callback function when a new image has been processed
void process_image(const void *p, int size) {
	//LOG("NEW IMAGE ARRIVED!");

	//std::vector<unsigned char> rawFrame(charBuff, charBuff + size);

	//Write the rawFrame to the mjpeg server
	//mjpeg.write(rawFrame);

	//Decode the current frame
	//cv::imdecode(rawFrame, 1, &frame); //Read into the frame object as a three channel image

	//Convert the yuyv array into
	frame = cv::Mat(raw_image_height, raw_image_width, CV_8UC2, (void*) p);
	cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUYV);

	//Threshold the image
	cv::Mat mid, proc;
	pthread_mutex_lock(&settings_mutex);
	filter_image(frame, mid, proc, 
					lane_min_h, lane_min_s, lane_min_v,
					lane_max_h, lane_max_s, lane_max_v,
					lane_blur);
	pthread_mutex_unlock(&settings_mutex);

	//Create a blank debugging image
	cv::Mat dbg = cv::Mat::zeros(cv::Size(image_width, image_height), frame.type());

	//Find all of the lines in the image
	//std::vector<cv::Vec4i> lines;
	//find_lines(proc, lines); 

	std::vector<std::vector<cv::Point>> cont;
	//std::vector<cv::RotatedRect> rects;
	//std::vector<cv::Vec4i> lines;
	cv::RotatedRect rect;
	cv::Vec4i center_line;

	//Write the default frame if it was requested
	pthread_mutex_lock(&settings_mutex);
	switch(source_select) {
		case -1:
			//Do nothing
			break;
		case 0:
			display_mat(frame);
			break;
		case 1:
			display_mat(mid);
			break;
		case 2:
			display_mat(proc);
			break;
	}
	pthread_mutex_unlock(&settings_mutex);
	if(!find_rect(proc, proc, cont, rect)) {
		report_findings(-2.0f, -2.0f);
		return;
	}

	if(!rect_center_line(rect, center_line)) {
		report_findings(-2.0f, -2.0f);
		return;
	}

	//find_rects(proc, proc, cont, rects); 
	//rects_to_lines(rects, lines);

	//Filter out the obscure lines
	//pthread_mutex_lock(&settings_mutex);
	//filter_lines(lines, lane_min_angle, lane_max_angle);
	//pthread_mutex_unlock(&settings_mutex);

	/*
	for(size_t ind = 0; ind < lines.size(); ind++) {
		cv::Vec4i line = lines[ind];
		cv::line(dbg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
	}*/
	

	//Group close lines together and attach them into longer more straight lines
	//cluster_similar_lines(lines);

	//Get both of the lanes
	
	//cv::Vec4i left, right;
	//get_lanes(lines, left, right);
	//cv::line(dbg, cv::Point(left[0], left[1]), cv::Point(left[2], left[3]), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	//cv::line(dbg, cv::Point(right[0], right[1]), cv::Point(right[2], right[3]), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

	//Calculate the displacements
	float future_disp = 0.0f, current_disp = 0.0f;
	calc_disp(future_disp, current_disp, center_line, dbg);
	//calculate_displacement(future_disp, current_disp, left, right, dbg);
	//printf("Current: %.2f\nFuture: %.2f\n", current_disp, future_disp);

	//if(isnan(future_disp)) future_disp = -2.0f;
	//if(isnan(current_disp)) current_disp = -2.0f;

	//Write the mjpeg image to the client
	pthread_mutex_lock(&settings_mutex);
	if(source_select == 3) {
		display_mat(dbg);
	}
	pthread_mutex_unlock(&settings_mutex);

	//Send the updated values
	report_findings(current_disp, future_disp);
}

//Report error and exit
void errno_exit(const char *s) {
	fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
	if(n_errors++ > max_errors) exit(EXIT_FAILURE);
}

//Pulling ioctl (wait for the request to not error before returning)
int xioctl(int fh, int request, void *arg) {
	int r; //Define the return value
	
	do {
		r = ioctl(fh, request, arg); //Make the ioctl request
	} while(r == -1 && EINTR == errno); //Wait for a non zero or no error
	
	return r;
}

//Read a frame from the video device
int read_frame() {
	struct v4l2_buffer buf; //Create a new video for linux (v2) image buffer
	unsigned int i;

	CLEAR(buf); //Clear the v4l2_buffer

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; //Define the buffer type as a video capture
	buf.memory = V4L2_MEMORY_MMAP; //Store the buffer in a memory mapped file and not to the disk

	if(xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
		switch(errno) {
			case EAGAIN:
				return 0; //The buffer hasn't been updated
			case EIO:
				//Ignore EIO
			default:
				errno_exit("mmap"); //Report a read error
		}
	}

	assert(buf.index < n_buffers);

	//Send the buffer to the process_image method
	process_image(buffers[buf.index].start, buf.bytesused);


	if(xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
		errno_exit("mmap_qbuf");
	}

	return 1;
}

void mainloop() {
	while(true) {
		fd_set fds;
		struct timeval tv;
		int r;

		FD_ZERO(&fds);
		FD_SET(fd, &fds);

		tv.tv_sec = 2;
		tv.tv_usec = 0;

		r = select(fd + 1, &fds, NULL, NULL, &tv);

		if(r == -1) {
			if(EINTR == errno) {
				continue;
				errno_exit("select");
			}
		}

		if(r == 0) {
			fprintf(stderr, "select timeout\n");
			if(n_errors++ > max_errors) exit(EXIT_FAILURE);
		}

		read_frame();
	}
}

void stop_capturing() {
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if(xioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
		errno_exit("Video stream off!");
	}
}

void start_capturing() {
	//Create the last time object
	gettimeofday(&last_time, NULL);

	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	
	for(unsigned int ind = 0; ind < n_buffers; ind++) {
		struct v4l2_buffer buf;
		
		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = ind;

		if(xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
			errno_exit("mmap_qbuf");
		}
	}

	if(xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
		errno_exit("streamon");
	}
}

void uninit_device() {
	for(unsigned int ind = 0; ind < n_buffers; ind++) {
		if(munmap(buffers[ind].start, buffers[ind].length) == -1) {
			errno_exit("munmap");
		}
	}
}

void init_mmap() {
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if(xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
		if(EINVAL == errno) {
			fprintf(stderr, "%s does not support mmap", device_name);
			exit(EXIT_FAILURE);
		} else {
			errno_exit("Video request bufs");
		}
	}

	if(req.count < 2) {
		fprintf(stderr, "Insufficient buffer memory!");
		exit(EXIT_FAILURE);
	}

	buffers = (buffer *) calloc(req.count, sizeof(*buffers));

	if(!buffers) {
		fprintf(stderr, "Out of memory!");
		exit(EXIT_FAILURE);
	}

	for(n_buffers = 0; n_buffers < req.count; ++n_buffers) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;

		if(xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
			errno_exit("video query buff");
		}

		buffers[n_buffers].length = buf.length;
		buffers[n_buffers].start = mmap(NULL, //Start anywhere
											buf.length,
											PROT_READ | PROT_WRITE,
											MAP_SHARED,
											fd, buf.m.offset);
		if(MAP_FAILED == buffers[n_buffers].start) {
			errno_exit("mmap");
		}
	}
}

void init_device() {
	struct v4l2_capability cap;
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
	struct v4l2_format fmt;
	unsigned int min;

	if(xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
		if(EINVAL == errno) {
			fprintf(stderr, "Device not found");
			exit(EXIT_FAILURE);
		} else {
			errno_exit("video query capture");
		}
	}

	if(!(cap.capabilities & V4L2_CAP_STREAMING)) {
		fprintf(stderr, "Device doesn't support streaming!\n");
		exit(EXIT_FAILURE);
	}

	CLEAR(cropcap);

	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = raw_image_width;
	fmt.fmt.pix.height = raw_image_height;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;//V4L2_PIX_FMT_MJPEG;
	fmt.fmt.pix.field = V4L2_FIELD_NONE; //V4L2_FIELD_INTERLACED;
	//fmt.fmt.pix.colorspace = V4L2_COLORSPACE_JPEG;

	min = fmt.fmt.pix.width * 2;
	if(fmt.fmt.pix.bytesperline < min) {
		fmt.fmt.pix.bytesperline = min;		
	}

	min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
	if(fmt.fmt.pix.sizeimage < min) {
		fmt.fmt.pix.sizeimage = min;
	}

	printf("Requesting\n\tdims: %d x %d\n", raw_image_width, raw_image_height);
	printf("\tformat: %d\n", fmt.fmt.pix.pixelformat);
	printf("\tfield: %d\n", fmt.fmt.pix.field);
	
	if(xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
		errno_exit("video format");
	}

	printf("New\n\tdims: %d x %d\n", fmt.fmt.pix.width, fmt.fmt.pix.height);
	printf("\tformat: %d\n", fmt.fmt.pix.pixelformat);
	printf("\tfield: %d\n", fmt.fmt.pix.field);

	init_mmap();
}

void close_device() {
	if(close(fd) == -1) {
		errno_exit("close");
	}
	fd = -1;
}

void open_device() {
	struct stat st;

	if(stat(device_name, &st) == -1) {
		fprintf(stderr, "Couldn't identify the device\n");
		if(n_errors++ > max_errors) {
			exit(EXIT_FAILURE);
		} else {
			usleep(400000);
			open_device();
		}
	}

	if(!S_ISCHR(st.st_mode)) {
		fprintf(stderr, "Device failure\n");
		if(n_errors++ > max_errors) {
			exit(EXIT_FAILURE);
		} else {
			usleep(400000);
			open_device();
		}
	}

	fd = open(device_name, O_RDWR | O_NONBLOCK, 0);

	if(fd == -1) {
		fprintf(stderr, "Couldn't open the device\n");
		if(n_errors++ > max_errors) {
			exit(EXIT_FAILURE);
		} else {
			usleep(400000);
			open_device();
		}
	}
}

static void *main_recv(void* context) {
	int nn_ss = *((int*) context);

	while(true) {
		char *buf = NULL;
		int recv = nn_recv(nn_ss, &buf, NN_MSG, 0);
		if(recv >= 0) {
			LOG("Got a new update!");
			printf("\n\nData: %s\n\n", buf);
			pthread_mutex_lock(&settings_mutex);
			sscanf(buf, "%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;", 
							&source_select,
							&lane_min_h,
							&lane_min_s,
							&lane_min_v,
							&lane_max_h,
							&lane_max_s,
							&lane_max_v,
							&lane_blur,
							&lane_min_angle,
							&lane_max_angle);
			pthread_mutex_unlock(&settings_mutex);
			nn_freemsg(buf);
		}
		usleep(10000);		
	}

	return NULL;
}

void start_server() {
	unsigned char *emptyFrame = (unsigned char *) "";
	mjpeg.write(emptyFrame, 1);
	mjpeg.start();

	//Create a nanomessage node that pushes data
	nn_s = nn_socket(AF_SP, NN_BUS);
	if(nn_bind(nn_s, nn_bind_addr) < 0) {
		fprintf(stderr, "Couldn't bind nanomessage to port 9090!");
		if(n_errors++ > max_errors) {
			exit(EXIT_FAILURE);
		} else {
			usleep(40000);
			start_server();
		}
	}
	
	//Create the new nanomessage listen thread
	pthread_create(&thread_recv, NULL, main_recv, &nn_s);
	
	//debug.write(emptyFrame);
	//debug.start();
}

void stop_server() {
	mjpeg.stop();

	//Stop the nanomessage server
	nn_shutdown(nn_s, 0);
	//debug.stop();
}

int main(int argc, char **argv) {
	signal(SIGPIPE, SIG_IGN); //Catch sigpipe (thrown by the mjpeg server)

	LOG("Opening the device!");
	open_device();
	init_device();
	start_server();
	start_capturing();
	mainloop();
	uninit_device();
	close_device();
	stop_server();
	return 0; 
}
