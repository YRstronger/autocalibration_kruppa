#include "KeyframeSelect.h"
#include "Frame.h"
#include"OrbExtractor.h"
#include"OrbMatching.h"
#include"AutoCalibra.h"
#include"gmsMatch.h"
#include"optimization.h"
#include"fundamentalMat.h"

// c++
#include <iostream>
#include <string>
#include <sstream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d.hpp>

#define  IMG_NUM 2000

using namespace std;
using namespace cv;
using namespace Eigen;

//自定义的命名空间
using namespace Simple_OrbSlam;


vector<float> mvLevelSigma2;

// 从指定路径中读取深度图和彩色图
void ReadImages(string& path_to_img, vector<Mat>& rgb_imgs);

// 特征提取
void FeatureExtraction(Mat& rgb_img, vector<KeyPoint>& kpt, Mat& desp);


// 特征匹配
void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<KeyPoint> ref_vkeypoint, vector<KeyPoint> cur_vkeypoint, vector<DMatch>& matches);

//使用对极约束筛选部分匹配点对
bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
	const std::vector<float> mmvLevelSigma2);


int main(int argc, char** argv) {

	if (argc < 2) {
		cout << "Please input ./keyframe_select ./path_to_image" << endl;
		return -1;
	}

	// 读取图像
	string path_to_img = argv[1];
	vector<Mat> rgb_img;
	//ReadImages(path_to_img,rgb_img);
	//if (rgb_img.size() == 0) {
	//	cout << "No valid images! please check the path!" << endl;
	//	return -1;
	//}

	Mat K = (Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);

	// 关键帧
	vector<Frame> vKeyframes;
	KeyframeSelect kf_select;

	// 第一帧
	Mat I_rgb,lastImg;
	vector<KeyPoint> vkpts;
	Mat desps;
	stringstream ss;
	string rgb_path;
	ss << path_to_img << "\\data1_" << 0 << ".png";
	//ss << path_to_img << "\\data1_" << 0 << "1.png";
	ss >> rgb_path;
	I_rgb = imread(rgb_path.c_str(), 0);
	FeatureExtraction(I_rgb, vkpts, desps);
	Frame kf(0, vkpts, desps, K);
	// 设置三维地图点(粗略版本!!!)
	vKeyframes.push_back(kf);
	cout << "Add the first keyframe!" << endl;
	cout << "-----------------------" << endl;
	vector<Matrix3d>   vFmatrix;

	// 逐帧处理
	for (int i = 1; i < IMG_NUM; ++i) {
		if (i %40 == 0)
		{
			vkpts.clear();
			stringstream ss;
			string rgb_path;
			ss << path_to_img << "\\data1_" << i << ".png";
			ss >> rgb_path;
			lastImg = I_rgb;
			I_rgb = imread(rgb_path.c_str(), 0);
			FeatureExtraction(I_rgb, vkpts, desps);
			Frame new_frame(i, vkpts, desps, K);

			// 特征匹配
			vector<DMatch> matchBF, matchgms,finalmatch;
			// 最后一个关键帧
			Frame last_keyframe = vKeyframes[vKeyframes.size() - 1];
			Mat last_kf_desp = last_keyframe.GetDescriptors();
			FeatureMatching(last_kf_desp, desps, last_keyframe.GetKeyPoints(), vkpts, matchBF);
			//gms匹配
			gmsMatch::matchGMS(lastImg.size(), I_rgb.size(), last_keyframe.GetKeyPoints(), vkpts, matchBF, true, matchgms, true, false, 5);

			// 查看是否需要插入关键帧
			bool add_new_kf = kf_select.CheckKeyframe(last_keyframe, new_frame, matchgms);
			finalmatch=matchgms;

			////画出gms匹配的结果
			//Mat matchImg1, matchImg2;
			//drawMatches(lastImg, last_keyframe.GetKeyPoints(), I_rgb, vkpts, finalmatch, matchImg1);
			//namedWindow("1st match", 0);
			//cv::resizeWindow("1st match", 960, 540);
			//imshow("1st match", matchImg1);
			//drawMatches(lastImg, last_keyframe.GetKeyPoints(), I_rgb, vkpts, matchBF, matchImg2);
			//namedWindow("bf match", 0);
			//cv::resizeWindow("bf match", 960, 540);
			//imshow("bf match", matchImg2);
			//waitKey(10);


			//if (add_new_kf) {

				//计算两个关键帧之间的F矩阵
				//a.得到匹配点对

				// 计算	F矩阵用
				vector<Point2f> points1,npoints1;   
				vector<Point2f> points2,npoints2;
				Point2d m1c(0, 0), m2c(0, 0);
				double t, scale1 = 0, scale2 = 0;

				for (int j = 0; j < finalmatch.size(); j++)
				{
						points1.push_back(last_keyframe.GetKeyPoints()[finalmatch[j].queryIdx].pt);
						points2.push_back( vkpts[finalmatch[j].trainIdx].pt);

				}
				// compute centers and average distances for each of the two point sets
				for (int i = 0; i < points1.size(); i++)
				{
					m1c += Point2d(points1[i]);
					m2c += Point2d(points2[i]);
				}

				// calculate the normalizing transformations for each of the point sets:
				// after the transformation each set will have the mass center at the coordinate origin
				// and the average distance from the origin will be ~sqrt(2).
				t = 1. / points1.size();
				m1c *= t;
				m2c *= t;

				for (int i = 0; i < points1.size(); i++)
				{
					scale1 += norm(Point2d(points1[i].x - m1c.x, points1[i].y - m1c.y));
					scale2 += norm(Point2d(points2[i].x - m2c.x, points2[i].y - m2c.y));
				}

				scale1 *= t;
				scale2 *= t;

				if (scale1 < FLT_EPSILON || scale2 < FLT_EPSILON)
					return 0;

				scale1 = std::sqrt(2.) / scale1;
				scale2 = std::sqrt(2.) / scale2;

				for (int i = 0; i < points1.size(); i++)
				{
					double x1 = (points1[i].x - m1c.x)*scale1;
					double y1 = (points1[i].y - m1c.y)*scale1;
					double x2 = (points2[i].x - m2c.x)*scale2;
					double y2 = (points2[i].y - m2c.y)*scale2;
					Point2d p1(x1, y1),p2(x2,y2);
					npoints1.push_back(p1);
					npoints2.push_back(p2);

				}

				computeGeofromPointCorr GeoCompute;
				Mat Fun;


				if (!GeoCompute.InitComputeFM(points1, points2, Fun))
				{	
					// 插入新的关键帧 if 点不够
					vKeyframes.push_back(new_frame);
					cout << "feature points are not enough" << endl;
					continue;
					
				}
				Fun = Fun / Fun.at<float>(2,2);
				cout << Fun << endl;
				//b. 使用opencv计算F矩阵
				Mat Fundamentalmatrix2 =
					findFundamentalMat(npoints1, npoints2, FM_RANSAC, 3, 0.999999);
				Matrix3d T1;
				T1<<scale1, 0, -scale1 * m1c.x, 0, scale1, -scale1 * m1c.y, 0, 0, 1;
				Matrix3d T2;
				T2<<scale2, 0, -scale2 * m2c.x, 0, scale2, -scale2 * m2c.y, 0, 0, 1;

				Matrix3d FM;
				FM << Fundamentalmatrix2.at<double>(0, 0), Fundamentalmatrix2.at<double>(0, 1), Fundamentalmatrix2.at<double>(0, 2),
					Fundamentalmatrix2.at<double>(1, 0), Fundamentalmatrix2.at<double>(1, 1), Fundamentalmatrix2.at<double>(1, 2),
					Fundamentalmatrix2.at<double>(2, 0), Fundamentalmatrix2.at<double>(2, 1), Fundamentalmatrix2.at<double>(2, 2);
				//
				FM = T2.transpose()*FM*T1;
				FM = FM / FM(8);

				cout << FM << endl;

				//Mat Fundamentalmatrix3 =
				//	findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.999999);

				//Mat Fundamentalmatrix4 =
				//	findFundamentalMat(points1, points2, FM_7POINT, 3, 0.999999);

				//cout << Fundamentalmatrix3 << endl;
				//cout << Fundamentalmatrix4 << endl;
				//cout << Fundamentalmatrix2 << endl;
				//cout << FM << endl;

				//OptimizedFundam optiF;
				//optiF.build_solve(points1, points2, FM);
				//cout << "optimized: " << FM<<endl;

				Matrix3d FM2;
				FM2 << Fun.at<float>(0, 0), Fun.at<float>(0, 1), Fun.at<float>(0, 2),
					Fun.at<float>(1, 0), Fun.at<float>(1, 1), Fun.at<float>(1, 2),
					Fun.at<float>(2, 0), Fun.at<float>(2, 1), Fun.at<float>(2, 2);
				//
				vFmatrix.push_back(FM2);

				if (vFmatrix.size() > 70)
					break;

				// 插入新的关键帧
				vKeyframes.push_back(new_frame);

				//Mat outImg;
				//drawKeypoints(I_rgb, vkpts, outImg, cv::Scalar::all(-1));
				cout << "Add the " << i << " image as keyframe!" << endl;
				cout << "--------------------------------------" << endl;
		}
	}

	//自标定
	//初始参数顺序：fx，fy，s，cx，cy
	double ini_Kvec[5] = { 2300.0,2300.0,0.0,960.0,540.0 };
	double vecK[5] = { 2300.0,2300.0,0.0,960.0,540.0 };

	//double ini_Kvec[3] = { 2350.0,960.0,540.0 };
	//double vecK[3] = { 2350.0,960.0,540.0 };
	vector<Matrix3d> vFinput;
	for (int i = 0; i < vFmatrix.size(); ++i)
	{
		Matrix3d tempFm = vFmatrix[i];
		if (abs(tempFm(0, 2)) > 0.1 || abs(tempFm(1, 2)) > 0.1) {

			cout << "this F cal error" << endl;
			continue;
		}
		vFinput.push_back(tempFm);
	}
	

	//for (int k = 0; k < 3; ++k) {

		double vecK1[5] = { 2200.0,2200.0,0.0,960.0,540.0 };
	    //double vecK1[4] = { 2200.0,2200.0,960.0,540.0 };
		cout << "the 1th calculation" << endl;
		AutoCalibra ac1(AutoCalibra::ACmethods::SimpliedK);//使用简化kruppa方程
		ac1.CalAC(vecK1, vFinput);

		double vecK2[5] = { 2320.0,2320.0,0.0,960.0,540.0 };
		//double vecK2[4] = { 2300.0,2300.0,960.0,540.0 };
		cout << "the 2th calculation" << endl;
		ac1.CalAC(vecK2, vFinput);

	    double vecK3[5] = { 2100.0,2100.0,0.0,960.0,540.0 };
		//double vecK3[4] = { 2100.0,2100.0,960.0,540.0 };
		cout << "the 3th calculation" << endl;
		ac1.CalAC(vecK3, vFinput);
	    
	//}
	return 0;
}

// 从指定路径中读取深度图和彩色图
void ReadImages(string& path_to_img,vector<Mat>& rgb_imgs) {
	rgb_imgs.clear();

	for (int i = 0; i < 1000; ++i) {
		stringstream ss;
		string rgb_path, depth_path;
		ss << path_to_img << "\\data1_" << i << ".png";
		ss >> rgb_path;
		Mat rgb = imread(rgb_path.c_str(), 0);
		rgb_imgs.push_back(rgb);
		//imshow("1",rgb);
		//waitKey(0);
	}
}

// 特征提取
void FeatureExtraction(Mat& rgb_img, vector<KeyPoint>& kpt, Mat& desp) {
	
	kpt.clear();
	desp.release();
	// parameters-----grid based orb extractor
	int nfeatures = 2000;  //特征点个数？还不清楚这1000代表了什么要求？
	int nlevels = 1;    //图像金字塔层数
	float fscaleFactor = 1.2;
	float fIniThFAST = 20;
	float fMinThFAST = 5;

	Mat grayImg,mask;
	//rgb=
	//imshow("1", rgb_img);
	//waitKey(0);
	//if (rgb_img.empty())
	//	waitKey();
	//cout << rgb_img.channels() << endl;
	//cvtColor(rgb_img, grayImg, COLOR_RGB2GRAY);
	grayImg = rgb_img;//例子中的输入图像为灰度图像
	ORBextractor* pORBextractor;
	pORBextractor = new ORBextractor(nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST);
	(*pORBextractor)(grayImg, mask, kpt, desp);

    mvLevelSigma2 = pORBextractor->GetLevelSigmaSquares();
    
	//Mat outImg;
	//drawKeypoints(grayImg, kpt, outImg, cv::Scalar::all(-1)/*, DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);
	//imshow("GridOrbKpsImg", outImg);
	//waitKey(0);
}



// 特征匹配
void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<KeyPoint> ref_vkeypoint, vector<KeyPoint> cur_vkeypoint, vector<DMatch>& matches) {
	Ptr<BFMatcher> bf_match = new BFMatcher(NORM_HAMMING);
	vector<DMatch> match_temp;
	bf_match->match(ref_desp, cur_desp, match_temp);

	matches.clear();
	double d_min = min_element(match_temp.begin(), match_temp.end(),
		[](DMatch& d1, DMatch& d2) { return d1.distance < d2.distance; })->distance;
	for (int i = 0; i < match_temp.size(); ++i) {

		KeyPoint refkp = ref_vkeypoint[match_temp[i].queryIdx], curkp = cur_vkeypoint[match_temp[i].trainIdx];

		//if (abs(refkp.pt.x - curkp.pt.x) > 150 || abs(refkp.pt.y - curkp.pt.y) > 150)
		//	continue;


		double d = match_temp[i].distance;
		if (d < max(30.0, d_min * 2)) {
			matches.push_back(match_temp[i]);
		}
	}
}

bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
	const std::vector<float> mmvLevelSigma2)
{
	// Epipolar line in second image l = x1'F12 = [a b c]
	//cout << F12.at<double>(0, 0) <<endl;
	const float a = kp1.pt.x*F12.at<double>(0, 0) + kp1.pt.y*F12.at<double>(1, 0) + F12.at<double>(2, 0);
	const float b = kp1.pt.x*F12.at<double>(0, 1) + kp1.pt.y*F12.at<double>(1, 1) + F12.at<double>(2, 1);
	const float c = kp1.pt.x*F12.at<double>(0, 2) + kp1.pt.y*F12.at<double>(1, 2) + F12.at<double>(2, 2);

	const float num = a * kp2.pt.x + b * kp2.pt.y + c;

	const float den = a * a + b * b;

	if (den == 0)
		return false;

	float anglediff = (kp1.angle - kp2.angle) / 180.0;
	


	 const float dsqr = num * num / den;

	//ORBslam原来的
		// const float dsqr = num * num / den;
	//return dsqr < 3.84*mmvLevelSigma2[kp2.octave];

	//2019.12.25 修改：
//	const float dsqr = abs(num);


	//if (anglediff > 0.05)
	//	return false;

	//if(((abs(kp1.pt.x - kp2.pt.x)+abs(kp1.pt.y - kp2.pt.y))<5))
	//	return false;

	if (abs(anglediff) < 0.01)
		return true;

	if (dsqr > 100 * mmvLevelSigma2[kp2.octave])
		return false;
	else
		return true;
}
