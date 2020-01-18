#include "OrbMatching.h"

#include <cmath>

// Ĭ�Ϲ��캯��,��ȡbow�ʵ�ģ�Ͳ�������ֵ
namespace Simple_OrbSlam {

	//FeatureMatching::FeatureMatching(std::shared_ptr<DBoW3::Vocabulary> voc) :mVoc_(voc) {
	//	// ����bow��ֵ����
	//	mTh_ = 50;
	//	mNNRatio_ = 0.6;
	//}

	// ���ù��캯��ֱ�Ӽ�������ƥ��,��ѡ��ƥ�䷽ʽ
	//FeatureMatching::FeatureMatching(std::shared_ptr<DBoW3::Vocabulary> voc,
	//	cv::Mat& desp1, cv::Mat& desp2,
	//	std::vector<cv::DMatch>& matches, bool use_bow = true) :mVoc_(voc) {
	//	// ����bow��ֵ����
	//	mTh_ = 50;
	//	mNNRatio_ = 0.6;

	//	bool match_result = true;

	//	// ѡ������ƥ�䷽ʽ
	//	if (use_bow) {
	//		match_result = MatchByDBoW(desp1, desp2, matches);
	//		std::cout << "Match By DBoW result is " << match_result << ", since we have "
	//			<< matches.size() << " matching pairs!" << std::endl;
	//	}
	//	else {
	//		match_result = MatchByBruteForce(desp1, desp2, matches);
	//		std::cout << "Match By BruteForce result is " << match_result << ", since we have "
	//			<< matches.size() << " matching pairs!" << std::endl;
	//	}

	//}

	// Ĭ����������
	//FeatureMatching::~FeatureMatching() {
	//	mVoc_ = nullptr;
	//}
	OrbMatching::~OrbMatching() {
	}


	// ���ñ���ƥ�䷨��������ƥ��
	bool OrbMatching::MatchByBruteForce(cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches) {

		matches.clear();

		// ��֤�����ӷǿ�
		if (!(desp1.rows > 0 && desp2.rows > 0)) {
			return false;
		}

		std::vector<cv::DMatch> matches_new;
		cv::BFMatcher bf(cv::NORM_HAMMING);
		// ����ƥ��
		bf.match(desp1, desp2, matches_new);

		// ͳ������ƥ��������С����
		double min_dist = 10000.0, max_dist = 0.;
		for (int i = 0; i < matches_new.size(); ++i) {
			double dist = matches_new[i].distance;
			if (dist < min_dist) {
				min_dist = dist;
			}
			if (dist > max_dist) {
				max_dist = dist;
			}
		}

		// ������ֵ��������ֵ�����ҵ����ʵ�ƥ��
		for (int i = 0; i < matches_new.size(); ++i) {
			double dist = matches_new[i].distance;
			if (dist < std::max(min_dist * 2, 30.0)) {
				matches.push_back(matches_new[i]);
			}
		}

		return matches.size() > 0 ? true : false;
	}

	//  // ���û��ڴʴ�ģ�͵ķ�����������ƥ��
	//bool FeatureMatching::MatchByDBoW(cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches) {

	//	// ��֤�����ӷǿ�
	//	if (!(desp1.rows > 0 && desp2.rows > 0)) {
	//		return false;
	//	}

	//	matches.clear();

	//	// std::cout << "image 1 has " << desp1.rows << " features!" << std::endl;
	//	// std::cout << "image 2 has " << desp2.rows << " features!" << std::endl; 

	//	// ����bow����
	//	ComputeBoWVector(desp1, mBowVec1_, mFeatVec1_);
	//	ComputeBoWVector(desp2, mBowVec2_, mFeatVec2_);

	//	// ���Ƚ���bow����ƥ��
	//	DBoW3::FeatureVector::iterator f1it = mFeatVec1_.begin();
	//	DBoW3::FeatureVector::iterator f2it = mFeatVec2_.begin();
	//	DBoW3::FeatureVector::iterator f1end = mFeatVec1_.end();
	//	DBoW3::FeatureVector::iterator f2end = mFeatVec2_.end();
	//	while (f1it != f1end && f2it != f2end) {
	//		// ����������һ��Ϊbow����id
	//		if (f1it->first == f2it->first) {
	//			const std::vector<unsigned int> vIdx1 = f1it->second;
	//			const std::vector<unsigned int> vIdx2 = f2it->second;
	//			// ���޶���bow�����������������н�����һƥ��
	//			for (int i = 0; i < vIdx1.size(); ++i) {
	//				const size_t idx1 = vIdx1[i];
	//				cv::Mat feat1 = desp1.row(idx1);
	//				int bestDist1 = 256;
	//				int bestDist2 = 256;
	//				int bestIdx = -1;
	//				for (int j = 0; j < vIdx2.size(); ++j) {
	//					const size_t idx2 = vIdx2[j];
	//					cv::Mat feat2 = desp2.row(idx2);
	//					// ���������Ӽ�ľ���
	//					int dist = ComputeMatchingScore(feat1, feat2);

	//					if (dist < bestDist1) {
	//						bestDist2 = bestDist1;
	//						bestDist1 = dist;
	//						bestIdx = idx2;
	//					}
	//					else if (dist < bestDist2) {
	//						bestDist2 = dist;
	//					}
	//				}
	//				// ����ڱ�����
	//				if (bestDist1 <= mTh_) {
	//					if (static_cast<float>(bestDist1) < mNNRatio_ * static_cast<float>(bestDist2)) {
	//						cv::DMatch m;
	//						m.queryIdx = idx1;
	//						m.trainIdx = bestIdx;
	//						m.distance = bestDist1;
	//						matches.push_back(m);
	//					}
	//				}
	//			}
	//			++f1it;
	//			++f2it;
	//		}
	//		else if (f1it->first < f2it->first) {
	//			f1it = mFeatVec1_.lower_bound(f2it->first);
	//		}
	//		else {
	//			f2it = mFeatVec2_.lower_bound(f1it->first);
	//		}
	//	}
	//	return matches.size() > 0 ? true : false;
	//}

	// ����ͶӰ����������ƥ��
	bool OrbMatching::MatchByProject(std::vector<cv::Point3d>& vp3d1, cv::Mat& desp1,
		std::vector<cv::Point2d>& vp2d2, cv::Mat& desp2, double& radian,
		cv::Mat& K, cv::Mat& R, cv::Mat& t, std::vector<cv::DMatch>& matches) {

		matches.clear();

		// ���ο�֡�е���ά��ͶӰ����ǰ֡�в��Һ��ʵ�ƥ���
		for (int i = 0; i < vp3d1.size(); ++i) {
			cv::Mat p3d = (cv::Mat_<double>(3, 1) << vp3d1[i].x, vp3d1[i].y, vp3d1[i].z);   //��ͼ��ά��
			cv::Mat p3d_trans = R * p3d + t;    //�任
			p3d_trans /= p3d_trans.at<double>(2, 0);   //��һ��
			double u = K.at<double>(0, 0)*p3d_trans.at<double>(0, 0) + K.at<double>(0, 2);
			double v = K.at<double>(1, 1)*p3d_trans.at<double>(1, 0) + K.at<double>(1, 2);     //ͶӰ������ƽ��
			if (u < 0 || u > K.at<double>(0, 2) || v < 0 || v > K.at<double>(1, 2)) {
				continue;     //������ͼ��Χ
			}
			// ��ƥ��뾶�в��Һ��ʵ�ƥ���ѡ��
			std::vector<cv::Mat> desp_temp;
			std::vector<int> desp_index;
			for (int j = 0; j < vp2d2.size(); ++j) {
				cv::Point2d p2d = vp2d2[j];
				// u-radian < x < u+radian
				// v-radian < y < v+radian
				if ((u - radian) < p2d.x && (u + radian) > p2d.x &&
					(v - radian) < p2d.y && (v + radian) > p2d.y) {
					desp_temp.push_back(desp2.row(j));
					desp_index.push_back(j);
				}
			}

			// �ں�ѡ���������ҵ�����ʵ�ƥ���
			cv::Mat d1 = desp1.row(i);
			int min_dist = 256;
			int sec_min_dist = 256;
			int best_id = -1;
			for (int k = 0; k < desp_temp.size(); ++k) {
				cv::Mat d2 = desp2.row(desp_index[k]);
				int dist = ComputeMatchingScore(d1, d2);
				if (dist < min_dist) {
					sec_min_dist = min_dist;
					min_dist = dist;
					best_id = desp_index[k];
				}
				else if (dist < sec_min_dist) {
					sec_min_dist = dist;
				}
			}

			// ������ֵ����ɸѡ
			if (min_dist < mTh_) {
				if (min_dist < mNNRatio_*sec_min_dist) {
					cv::DMatch m1;
					m1.queryIdx = i;
					m1.trainIdx = best_id;
					m1.distance = min_dist;
					matches.push_back(m1);
				}
			}
		}

		return matches.size() > 0;
	}

	//// ����bowƥ�����ֵ����
	/*void FeatureMatching::SetThreshold(int threshold, float nn_ratio) {
		mTh_ = threshold;
		mNNRatio_ = nn_ratio;
	}*/

	// ��������ת��Ϊbow����
	//void FeatureMatching::ComputeBoWVector(cv::Mat& desp, DBoW3::BowVector& bowVec, DBoW3::FeatureVector& featVec) {

	//	// ���ԭ������
	//	bowVec.clear();
	//	featVec.clear();

	//	// �����������ӻ��ֳ�һ��һ�������Ӵ���������
	//	std::vector<cv::Mat> allDesp;
	//	allDesp.reserve(desp.rows);
	//	for (int i = 0; i < desp.rows; ++i) {
	//		allDesp.push_back(desp.row(i));
	//	}

	//	// ��ÿ�������Ӷ�ת���ɶ�Ӧ��bow����
	//	mVoc_->transform(allDesp, bowVec, featVec, 4);

	//}

	// �������������Ӽ��ƥ�����
	int OrbMatching::ComputeMatchingScore(cv::Mat& desp1, cv::Mat& desp2) {

		const int *p1 = desp1.ptr<int32_t>();
		const int *p2 = desp2.ptr<int32_t>();

		// ����������ƥ�����
		int dist = 0;

		// λ����,Ŀ�����������������֮���ж��ٸ���ͬ�ĵ�
		for (int i = 0; i < 8; ++i, ++p1, ++p2) {
			unsigned int v = (*p1) ^ (*p2);
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}

		return dist;
	}

}