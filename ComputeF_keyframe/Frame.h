#ifndef FRAME_H
#define FRAME_H

// common
#include"common.h"

namespace Simple_OrbSlam
{
	class Frame {
	public:
		// Ĭ�Ϲ��캯��
		Frame(int id, cv::Mat& camera_intrinsic);

		// �������캯��
		Frame(const Frame& f);

		// **���캯��
		Frame(int id, std::vector<cv::KeyPoint>& kpt, cv::Mat& desp, cv::Mat& camera_intrinsic);

		// Ĭ����������
		~Frame();


		// ��ȡ��ǰ֡id
		int GetFrameId() {
			return mnId_;
		}

		// ��ȡ��ǰ֡������
		std::vector<cv::KeyPoint>& GetKeyPoints() {
			return mvKpt_;
		}

		// ��ȡ��ǰ֡�������Ӧ��������
		cv::Mat GetDescriptors() {
			return mmDesp_;
		}

		// ��ȡ��ǰ֡�������Ӧ�����ֵ
		std::vector<double>& GetDepth() {
			return mvDepth_;
		}

		// ��ȡ����ڲ���
		cv::Mat GetCameraIntrinsic() {
			return mmK_;
		}


	private:
		// ��ǰ֡id
		int mnId_;

		// ��ǰ֡�ؼ���
		std::vector<cv::KeyPoint> mvKpt_;

		// ��Ӧ�������������
		cv::Mat mmDesp_;

		// ��Ӧ����������ֵ
		std::vector<double> mvDepth_;

		// ����ڲ���
		cv::Mat mmK_;

		// ��Ӧ��Ч��ͼ�������
		std::vector<int> mvMapPointsId_;
	};
}

#endif // FRAME_H
