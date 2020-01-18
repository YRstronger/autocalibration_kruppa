#ifndef KEYFRAMESELECT_H
#define KEYFRAMESELECT_H

#include "Frame.h"

namespace Simple_OrbSlam 
{
	class Frmae;

	class KeyframeSelect {
	public:
		// Ĭ�Ϲ��캯��
		KeyframeSelect();

		// Ĭ����������
		~KeyframeSelect();

		// ����ؼ�֡
		bool CheckKeyframe(Frame& last_keyframe, Frame& new_frame, std::vector<cv::DMatch>& matches);

		// ����ڵ���
		bool CheckInliersRatio(Frame& last_keyframe, std::vector<cv::DMatch>& matches);

	private:

		// �Ƿ�����ȫ�ֱջ�
		bool CheckGlobalLoop() {
			return mbGlobalLoop_;
		}

		// �Ƿ��Ǵ���λģʽ
		bool CheckLocationMode() {
			return mbLocationMode_;
		}

		// ��һ���ض�λ��Ĺؼ�֡
		int CheckLastRelocId() {
			return mnLastRelocId_;
		}


		// ��һ���ض�λ�Ĺؼ�֡id
		int mnLastRelocId_;

		// �Ƿ�ȫ�ֱջ�
		bool mbGlobalLoop_;

		// �Ƿ񴿶�λģʽ
		bool mbLocationMode_;

		// �ֲ���ͼ�Ƿ����
		bool mbLocalMapFree_;

		// �ֲ���ͼ�еĹؼ�֡�����еĹؼ�֡��Ŀ
		int mnKeyFrameNumInQueue_;

		// -----------------------
		// �����¹ؼ�֡ǰ���پ�������֡
		int mnMinKeyframePass_;

		// �����ؼ�֡���ܹ���������֡��
		int mnMaxKeyframePass_;

		// ����ڵ���,�������ʾ��֡���ص���̫��
		double mdMaxInlierRatio_;

		// ��С�ڵ���,�������ʾ��֡�㹲�Ӳ���̫��
		double mdMinInlierRatio_;

		// ��С�ڵ�����,��Ҫ�������˶�����
		int mnMinInliers_;
		
		//����ڵ�����
		int maxInliers_;
	};
}



#endif // KEYFRAME_SELECT_H
