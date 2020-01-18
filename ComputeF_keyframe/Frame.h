#ifndef FRAME_H
#define FRAME_H

// common
#include"common.h"

namespace Simple_OrbSlam
{
	class Frame {
	public:
		// 默认构造函数
		Frame(int id, cv::Mat& camera_intrinsic);

		// 拷贝构造函数
		Frame(const Frame& f);

		// **构造函数
		Frame(int id, std::vector<cv::KeyPoint>& kpt, cv::Mat& desp, cv::Mat& camera_intrinsic);

		// 默认析构函数
		~Frame();


		// 获取当前帧id
		int GetFrameId() {
			return mnId_;
		}

		// 获取当前帧特征点
		std::vector<cv::KeyPoint>& GetKeyPoints() {
			return mvKpt_;
		}

		// 获取当前帧特征点对应的描述子
		cv::Mat GetDescriptors() {
			return mmDesp_;
		}

		// 获取当前帧特征点对应的深度值
		std::vector<double>& GetDepth() {
			return mvDepth_;
		}

		// 获取相机内参数
		cv::Mat GetCameraIntrinsic() {
			return mmK_;
		}


	private:
		// 当前帧id
		int mnId_;

		// 当前帧关键点
		std::vector<cv::KeyPoint> mvKpt_;

		// 对应特征点的描述子
		cv::Mat mmDesp_;

		// 对应特征点的深度值
		std::vector<double> mvDepth_;

		// 相机内参数
		cv::Mat mmK_;

		// 对应有效地图点的索引
		std::vector<int> mvMapPointsId_;
	};
}

#endif // FRAME_H
