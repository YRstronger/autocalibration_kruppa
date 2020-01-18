
#ifndef ORBMATCHING_H
#define ORBMATCHING_H

// common
#include"common.h"


// δʹ�ôʵ��
//// dbow3
//#include "DBoW3/src/DBoW3.h"

// c++ 
#include <memory>

namespace Simple_OrbSlam {

	class OrbMatching {

	public:
		// Ĭ�Ϲ��캯��,��ȡbow�ʵ�ģ�Ͳ�������ֵ
	//	FeatureMatching(std::shared_ptr<DBoW3::Vocabulary> voc);

		// ���ù��캯��ֱ�Ӽ�������ƥ��,��ѡ��ƥ�䷽ʽ
	//	FeatureMatching(std::shared_ptr<DBoW3::Vocabulary> voc,
			//cv::Mat& desp1, cv::Mat& desp2,
			//std::vector<cv::DMatch>& matches, bool use_bow);

		// Ĭ����������
		~OrbMatching();

		// ���ñ���ƥ�䷨��������ƥ��
		bool MatchByBruteForce(cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches);

		// ���û��ڴʴ�ģ�͵ķ�����������ƥ��
	//	bool MatchByDBoW(cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches);

		// ����ͶӰ����������ƥ��
		bool MatchByProject(std::vector<cv::Point3d>& vp3d1, cv::Mat& desp1,
			std::vector<cv::Point2d>& vp2d2, cv::Mat& desp2, double& radian,
			cv::Mat& K, cv::Mat& R, cv::Mat& t, std::vector<cv::DMatch>& matches);



		// ����bowƥ�����ֵ����
	//	void SetThreshold(int threshold, float nnratio);

	private:

		// ��������ת��Ϊbow����
		//void ComputeBoWVector(cv::Mat& desp, DBoW3::BowVector& bowVec, DBoW3::FeatureVector& featVec);

		// �������������Ӽ��ƥ�����
		int ComputeMatchingScore(cv::Mat& desp1, cv::Mat& desp2);



		// �ʴ�ģ��
		//std::shared_ptr<DBoW3::Vocabulary> mVoc_ = nullptr;

		// bow����,����id��Ȩ��
	//	DBoW3::BowVector mBowVec1_, mBowVec2_;

		// feature����,������bow���е�id��������id
	//	DBoW3::FeatureVector mFeatVec1_, mFeatVec2_;

		// ����ڱ�������ֵ����
		int mTh_;
		float mNNRatio_;
	};
}

#endif // ORBMATCHING_H#pragma once
