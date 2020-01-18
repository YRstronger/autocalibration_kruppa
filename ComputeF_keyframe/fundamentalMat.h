#ifndef FUNDAMENTALMAT_H
#define FUNDAMENTALMAT_H


#include "common.h"

namespace Simple_OrbSlam
{
	class computeGeofromPointCorr
	{

	public:
		bool InitComputeFM(vector<Point2f> _vpts1, vector<Point2f> _vpts2,  //输入的特征点和匹配关系
			Mat& F21);

		~computeGeofromPointCorr() {}

		void FindFundamental(float &score, cv::Mat &F21);

		Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

		float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

		void Normalize(const vector<cv::Point2f> &vpts, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
protected:
		// points from Reference Frame (Frame 1)
		vector<Point2f> vpts1;
		//points from Current Frame (Frame 2)
		vector<Point2f> vpts2;

		//// Current Matches from Reference to Current
		//vector<DMatch> vMatches12;

		// Standard Deviation and Variance
		float mSigma=1.0, mSigma2=mSigma*mSigma;

		// Ransac max iterations
		int mMaxIterations=5000;

		// Ransac sets
		vector<vector<size_t> > mvSets;

	};

} //namespace Simple_OrbSlam

#endif // 

