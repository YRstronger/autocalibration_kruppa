#include "Frame.h"


namespace Simple_OrbSlam
{
	Frame::Frame(int id, cv::Mat& camera_intrinsic) {
		mnId_ = id;
		mmK_ = camera_intrinsic.clone();
	}

	Frame::Frame(const Frame& f) {
		mnId_ = f.mnId_;
		mmK_ = f.mmK_;
		mmDesp_ = f.mmDesp_;
		mvKpt_ = f.mvKpt_;
	}

	Frame::Frame(int id, std::vector<cv::KeyPoint>& kpt, cv::Mat& desp, cv::Mat& camera_intrinsic) {
		mnId_ = id;
		mmK_ = camera_intrinsic.clone();
		mmDesp_ = desp.clone();
		mvKpt_ = kpt;
	}

	Frame::~Frame() {

	}

	

}