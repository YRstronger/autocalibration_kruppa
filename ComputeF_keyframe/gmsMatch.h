#ifndef GMSMATCH_H
#define GMSMATCH_H

// common
#include"common.h"


using namespace std;
using namespace cv;

namespace gmsMatch
{
	class GMSMatcher
	{
	public:
		// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
		GMSMatcher(const vector<KeyPoint>& vkp1, const Size& size1, const vector<KeyPoint>& vkp2, const Size& size2,
			const vector<DMatch>& vDMatches, const double thresholdFactor);


		~GMSMatcher() {}

		// Get Inlier Mask
		// Return number of inliers
		int getInlierMask(vector<bool> &vbInliers, const bool withRotation = false, const bool withScale = false);

		//added by 睿
        //统计每个方格的匹配索引
		vector<vector<int> > pointIdxInPerCell;

	private:
		// Normalized Points
		vector<Point2f> mvP1, mvP2;

		// Matches
		vector<pair<int, int> > mvMatches;

		// Number of Matches
		size_t mNumberMatches;

		// Grid Size
		Size mGridSizeLeft, mGridSizeRight;
		int mGridNumberLeft;
		int mGridNumberRight;

		// x      : left grid idx
		// y      : right grid idx
		// value  : how many matches from idx_left to idx_right
		Mat mMotionStatistics;

		//
		vector<int> mNumberPointsInPerCellLeft;

		// Inldex  : grid_idx_left
		// Value   : grid_idx_right
		vector<int> mCellPairs;

		// Every Matches has a cell-pair
		// first  : grid_idx_left
		// second : grid_idx_right
		vector<pair<int, int> > mvMatchPairs;

		// Inlier Mask for output
		vector<bool> mvbInlierMask;

		//
		Mat mGridNeighborLeft;
		Mat mGridNeighborRight;

		double mThresholdFactor;


		// Assign Matches to Cell Pairs
		void assignMatchPairs(const int GridType, const int rotationtype);

		void convertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches);

		int getGridIndexLeft(const Point2f &pt, const int type);

		int getGridIndexRight(const Point2f &pt);

		vector<int> getNB9(const int idx, const Size& GridSize);

		void initalizeNeighbors(Mat &neighbor, const Size& GridSize);

		void normalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts);

		// Run
		int run(const int rotationType);

		void setScale(const int scale);

		// Verify Cell Pairs
		void verifyCellPairs(const int rotationType);
	};


	void matchGMS(const Size& size1, const Size& size2, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
		const vector<DMatch>& matches1to2, bool EvenFlag, vector<DMatch>& matchesGMS, const bool withRotation, const bool withScale,
		const double thresholdFactor);
}



#endif
