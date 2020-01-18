

#include "fundamentalMat.h"



namespace Simple_OrbSlam
{

	bool computeGeofromPointCorr::InitComputeFM(vector<Point2f> _vpts1, vector<Point2f> _vpts2,  //输入的特征点和匹配关系
		 Mat& F21)  //输出的F矩阵
	{
		vpts1 = _vpts1, vpts2 = _vpts2;

		//如果输入的特征点不正确，或者匹配点小于15，返回错误
		if (vpts1.size() < 1 || vpts2.size() < 1 )
		{
			return false;
		}

		//RANSAC

		const int N = vpts1.size();

		// Generate sets of 8 points for each RANSAC iteration
		mvSets = vector< vector<size_t> >(mMaxIterations, vector<size_t>(8, 0));

		RNG rng;

		for (int it = 0; it < mMaxIterations; it++)
		{

			// Select a minimum set
			for (size_t j = 0; j < 8; j++)
			{
				size_t idx_i;
				for (idx_i = rng.uniform(0, N);
					std::find(mvSets[it].data(), mvSets[it].data() +j, idx_i) != mvSets[it].data() + j;//意味着idx_i不和已有的重复：for循环结束条件是idx_i在这个范围(idx,idx+i)的位置是最后一个idx+i：
					idx_i = rng.uniform(0, N))
				{
				}
				mvSets[it][j] = idx_i;

			}
		}


		//计算F矩阵
		float score;
		FindFundamental(score, F21);
		return true;
	}

	//RANSAC计算F矩阵
	void computeGeofromPointCorr::FindFundamental(float &score, cv::Mat &F21)
	{
		// Number of putative matches
		const int N = vpts1.size();

		// Normalize coordinates
		vector<cv::Point2f> vPn1, vPn2;
		cv::Mat T1, T2;
		Normalize(vpts1, vPn1, T1);
		Normalize(vpts2, vPn2, T2);
		cv::Mat T2t = T2.t();

		// Best Results variables
		score = 0.0;
		vector<bool> vbMatchesInliers = vector<bool>(N, false);

		
		// Iteration variables
		vector<cv::Point2f> vPn1i(8);
		vector<cv::Point2f> vPn2i(8);
		cv::Mat F21i;
		vector<bool> vbCurrentInliers(N, false);
		float currentScore;

		// Perform all RANSAC iterations and save the solution with highest score
		for (int it = 0; it < mMaxIterations; it++)
		{
			// Select a minimum set
			for (int j = 0; j < 8; j++)
			{
				int idx = mvSets[it][j];

				vPn1i[j] = vPn1[idx];
				vPn2i[j] = vPn2[idx];
			}

			cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

			F21i = T2t * Fn*T1;

			currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

			if (currentScore > score)
			{
				F21 = F21i.clone();
				vbMatchesInliers = vbCurrentInliers;
				score = currentScore;
			}
		}
	}


	//8点法和强制秩为2->计算每个含8对点的子样本的F矩阵
	Mat computeGeofromPointCorr::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
	{
		const int N = vP1.size();

		cv::Mat A(N, 9, CV_32F);

		for (int i = 0; i < N; i++)
		{
			const float u1 = vP1[i].x;
			const float v1 = vP1[i].y;
			const float u2 = vP2[i].x;
			const float v2 = vP2[i].y;

			A.at<float>(i, 0) = u2 * u1;
			A.at<float>(i, 1) = u2 * v1;
			A.at<float>(i, 2) = u2;
			A.at<float>(i, 3) = v2 * u1;
			A.at<float>(i, 4) = v2 * v1;
			A.at<float>(i, 5) = v2;
			A.at<float>(i, 6) = u1;
			A.at<float>(i, 7) = v1;
			A.at<float>(i, 8) = 1;
		}

		cv::Mat u, w, vt;

		cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		cv::Mat Fpre = vt.row(8).reshape(0, 3);

		cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		w.at<float>(2) = 0;

		return  u * cv::Mat::diag(w)*vt;
	}

	//前后投影误差计算误差总值
	float computeGeofromPointCorr::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
	{
		const int N = vpts1.size();

		const float f11 = F21.at<float>(0, 0);
		const float f12 = F21.at<float>(0, 1);
		const float f13 = F21.at<float>(0, 2);
		const float f21 = F21.at<float>(1, 0);
		const float f22 = F21.at<float>(1, 1);
		const float f23 = F21.at<float>(1, 2);
		const float f31 = F21.at<float>(2, 0);
		const float f32 = F21.at<float>(2, 1);
		const float f33 = F21.at<float>(2, 2);

		vbMatchesInliers.resize(N);

		float score = 0;

		const float th = 3.841;
		const float thScore = 5.991;

		const float invSigmaSquare = 1.0 / (sigma*sigma);

		for (int i = 0; i < N; i++)
		{
			bool bIn = true;

			const cv::Point2f &p1 = vpts1[i];
			const cv::Point2f &p2 = vpts2[i];

			const float u1 = p1.x;
			const float v1 = p1.y;
			const float u2 = p2.x;
			const float v2 = p2.y;

			// Reprojection error in second image
			// l2=F21x1=(a2,b2,c2)

			const float a2 = f11 * u1 + f12 * v1 + f13;
			const float b2 = f21 * u1 + f22 * v1 + f23;
			const float c2 = f31 * u1 + f32 * v1 + f33;

			const float num2 = a2 * u2 + b2 * v2 + c2;

			const float squareDist1 = num2 * num2 / (a2*a2 + b2 * b2);

			const float chiSquare1 = squareDist1 * invSigmaSquare;

			if (chiSquare1 > th)
				bIn = false;
			else
				score += thScore - chiSquare1;

			// Reprojection error in second image
			// l1 =x2tF21=(a1,b1,c1)

			const float a1 = f11 * u2 + f21 * v2 + f31;
			const float b1 = f12 * u2 + f22 * v2 + f32;
			const float c1 = f13 * u2 + f23 * v2 + f33;

			const float num1 = a1 * u1 + b1 * v1 + c1;

			const float squareDist2 = num1 * num1 / (a1*a1 + b1 * b1);

			const float chiSquare2 = squareDist2 * invSigmaSquare;

			if (chiSquare2 > th)
				bIn = false;
			else
				score += thScore - chiSquare2;

			if (bIn)
				vbMatchesInliers[i] = true;
			else
				vbMatchesInliers[i] = false;
		}

		return score;
	}

	void computeGeofromPointCorr::Normalize(const vector<Point2f> &vpts, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
	{
		float meanX = 0;
		float meanY = 0;
		const int N = vpts.size();

		vNormalizedPoints.resize(N);

		for (int i = 0; i < N; i++)
		{
			meanX += vpts[i].x;
			meanY += vpts[i].y;
		}

		meanX = meanX / N;
		meanY = meanY / N;

		float meanDevX = 0;
		float meanDevY = 0;

		for (int i = 0; i < N; i++)
		{
			vNormalizedPoints[i].x = vpts[i].x - meanX;
			vNormalizedPoints[i].y = vpts[i].y - meanY;

			meanDevX += fabs(vNormalizedPoints[i].x);
			meanDevY += fabs(vNormalizedPoints[i].y);
		}

		meanDevX = meanDevX / N;
		meanDevY = meanDevY / N;

		float sX = 1.0 / meanDevX;
		float sY = 1.0 / meanDevY;

		for (int i = 0; i < N; i++)
		{
			vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
			vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
		}

		T = cv::Mat::eye(3, 3, CV_32F);
		T.at<float>(0, 0) = sX;
		T.at<float>(1, 1) = sY;
		T.at<float>(0, 2) = -meanX * sX;
		T.at<float>(1, 2) = -meanY * sY;
	}



} //namespace Simple_OrbSlam
