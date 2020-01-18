#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

// common
#include"common.h"

namespace Simple_OrbSlam
{
	class OptimizedFundam
	{
	public:

		OptimizedFundam();

		void build_solve(vector<Point2d> vkps1, vector<Point2d> vkps2,Matrix3d& FM);

	};


	//构建<简化Kruppa方程>的代价函数
	class OptiFundambyEpiloar {

	public:
		OptiFundambyEpiloar() {}


		//模板T为jet类型(Ceres导数数据变量)
		template<typename T>
		bool operator()(
			const T* const vec_F,   //F矩阵向量：1*8
			const T* const pointpair,//点对向量：1*4
			T* residual) const {

			typedef Eigen::Matrix<T, 3, 3> Mat3;
			typedef Eigen::Matrix<T, 1, 3> Point_homo;
			typedef Eigen::Matrix<T, 3, 1> vec_homo;
			Mat3 FundamMatrix;
			Point_homo p1, p2;

			//F矩阵和点对初始化
			p1[0] = T(pointpair[0]), p1[1] = T(pointpair[1]), p2[0] = T(pointpair[2]), p2[1] = T(pointpair[3]);
			p1[2] = T(1.0),p2[2] = T(1.0); 
			FundamMatrix(8)= T(1.0);
			for (int i=0; i < 8; ++i)
			{
				FundamMatrix(i) = T(vec_F[i]);
			}

			vec_homo l1,l2;
			l1 = FundamMatrix * p1.transpose();
			l2 =   FundamMatrix * p2.transpose();
			//cout << p2*l1;
			T t1 = (p2 * l1),t2=(p1 * l2);
			//对极误差
			residual[0] = t1*t1 /*/ (l1[0] * l1[0] + l1[1] * l1[1])*/;
			residual[1] = t2*t2/*/ (l2[0] * l2[0] + l2[1] * l2[1])*/;
			return true;

		}

		//
		//private


	};

}


#endif
