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

		// ����Ƿ�ȫ�ֱջ���
		if (CheckGlobalLoop()) {
			return false;
		}
		// ����Ƿ񴿶�λģʽ
		if (CheckLocationMode()) {
			return false;
		}
		// ����Ƿ�վ������ض�λ
		int new_frame_id = new_frame.GetFrameId();
		if (new_frame_id < mnLastRelocId_ + 1) {
			return false;
		}

		// ����ڵ���
		bool inlierRatio = CheckInliersRatio(last_keyframe,matches);
		//// �����ؼ�֡�侭�������֡��
		//bool cond1 = (new_frame.GetFrameId() - last_keyframe.GetFrameId()) >= mnMaxKeyframePass_;
		// �����ؼ�֡�侭������С֡���Ҿֲ���ͼ�߳̿���
		bool cond2 = ((new_frame.GetFrameId() - last_keyframe.GetFrameId()) >= mnMinKeyframePass_) && mbLocalMapFree_;
		// �ֲ���ͼ�ؼ�֡�����йؼ�֡��������3֡
		bool cond3 = mnKeyFrameNumInQueue_ < 3;

		// �ۺ���������
		bool result = (inlierRatio && 1) || (inlierRatio && cond2) || (inlierRatio && cond3);

		return result;
	}

	// ����ڵ���
	bool KeyframeSelect::CheckInliersRatio(Frame& last_keyframe, std::vector<cv::DMatch>& matches) {
		// ��ͼ�㼰���Ӧ�������id,���ýṹ��ʵ��
		// ����,���last_keyframeҲ����ֱ������һ���ؼ�֡�ĵ�ͼ���ʾ
		//��������������һ�ؼ�֡�ĵ�ͼ���������Կ������������������͵�ǰ�ؼ�֡�������������Ƚ�
          
		double num_LastFKP = last_keyframe.GetKeyPoints().size();   //��һ�ؼ�֡������������
		double num_Matches = matches.size();    //��һ�ؼ�֡�͵�ǰ֡��ƥ�������
		double ratio = num_Matches / num_LastFKP;
	//	std::vector<cv::KeyPoint> vNewFKP = new_frame.GetKeyPoints();   //��ǰ֡��������

		if (ratio < mdMinInlierRatio_) {
			return false;
		}

		int inliers = matches.size();
		//// ͳ���ڵ���
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

		// �Ա��ڵ�����
		if (ratio < mdMaxInlierRatio_ &&
			ratio > mdMinInlierRatio_ &&
			inliers > mnMinInliers_&&
			inliers < maxInliers_) {
			return true;
		}
		return false;
	}
}