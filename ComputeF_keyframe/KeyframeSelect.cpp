#include "KeyframeSelect.h"


namespace Simple_OrbSlam
{
	KeyframeSelect::KeyframeSelect() {

		mbGlobalLoop_ = false;
		mbLocationMode_ = false;
		mnLastRelocId_ = -1;
		mbLocalMapFree_ = true;
		mnKeyFrameNumInQueue_ = 3;

		mnMinKeyframePass_ = 3;
		mnMaxKeyframePass_ = 6;
		mdMaxInlierRatio_ = 0.5;
		mdMinInlierRatio_ = 0.02;
		maxInliers_ = 300;
		mnMinInliers_ = 30;
	}

	KeyframeSelect::~KeyframeSelect() {

	}

	bool KeyframeSelect::CheckKeyframe(Frame& last_keyframe, Frame& new_frame, std::vector<cv::DMatch>& matches) {

		// 检查是否全局闭环中
		if (CheckGlobalLoop()) {
			return false;
		}
		// 检查是否纯定位模式
		if (CheckLocationMode()) {
			return false;
		}
		// 检查是否刚经历过重定位
		int new_frame_id = new_frame.GetFrameId();
		if (new_frame_id < mnLastRelocId_ + 1) {
			return false;
		}

		// 检查内点率
		bool inlierRatio = CheckInliersRatio(last_keyframe,matches);
		//// 两个关键帧间经过了最大帧数
		//bool cond1 = (new_frame.GetFrameId() - last_keyframe.GetFrameId()) >= mnMaxKeyframePass_;
		// 两个关键帧间经过了最小帧数且局部建图线程空闲
		bool cond2 = ((new_frame.GetFrameId() - last_keyframe.GetFrameId()) >= mnMinKeyframePass_) && mbLocalMapFree_;
		// 局部建图关键帧队列中关键帧数不超过3帧
		bool cond3 = mnKeyFrameNumInQueue_ < 3;

		// 综合上述条件
		bool result = (inlierRatio && 1) || (inlierRatio && cond2) || (inlierRatio && cond3);

		return result;
	}

	// 检查内点率
	bool KeyframeSelect::CheckInliersRatio(Frame& last_keyframe, std::vector<cv::DMatch>& matches) {
		// 地图点及其对应特征点的id,可用结构体实现
		// 此外,这个last_keyframe也可以直接用上一个关键帧的地图点表示
		//这个程序就是用上一关键帧的地图点数（可以看作特征点数），来和当前关键帧的特征点数量比较
          
		double num_LastFKP = last_keyframe.GetKeyPoints().size();   //上一关键帧的特征点数量
		double num_Matches = matches.size();    //上一关键帧和当前帧的匹配点数量
		double ratio = num_Matches / num_LastFKP;
	//	std::vector<cv::KeyPoint> vNewFKP = new_frame.GetKeyPoints();   //当前帧的特征点

		if (ratio < mdMinInlierRatio_) {
			return false;
		}

		int inliers = matches.size();
		//// 统计内点数
		//int inliers = 0;
		//for (int i = 0; i < matches.size(); ++i) {
		//	int idx = matches[i].queryIdx;
		//	std::vector<int>::iterator it = std::find(vp3d_id.begin(), vp3d_id.end(), idx);
		//	if (it != vp3d_id.end()) {
		//		++inliers;
		//	}
		//}

		//std::cout << "In CheckInliersRatio, we found ratio" << ratio  << std::endl;
		//std::cout << "In CheckInliersRatio, we found " << inliers<<"inliers" << std::endl;

		// 对比内点数量
		if (ratio < mdMaxInlierRatio_ &&
			ratio > mdMinInlierRatio_ &&
			inliers > mnMinInliers_&&
			inliers < maxInliers_) {
			return true;
		}
		return false;
	}
}