#include"optimization.h"

namespace Simple_OrbSlam
{
	OptimizedFundam::OptimizedFundam() {}

	void OptimizedFundam::build_solve(vector<Point2d> vkps1, vector<Point2d> vkps2, Matrix3d& FM)
	{
		Problem problem;
		double vec_F[8];
		LossFunction* loss_function = new ceres::HuberLoss(1.0);

		for (int i = 0; i < vkps1.size(); ++i)
		{
			double pointpair[4] = { vkps1[i].x,vkps1[i].y,vkps2[i].x,vkps2[i].y };

			for (int j = 0; j < 8; j++)
			{
				vec_F[j] = FM(j);
			}
			CostFunction* cost_function =
				new AutoDiffCostFunction<OptiFundambyEpiloar, 2, 8, 4>(
					new OptiFundambyEpiloar());
			problem.AddResidualBlock(cost_function,
				loss_function,
				vec_F,
				pointpair);

		}
			Solver::Options options;
			options.minimizer_progress_to_stdout = true;   //是否显示迭代结果的标志
			//options.initial_trust_region_radius = 100;
			//options.gradient_tolerance = 1e-20;
			//options.function_tolerance = 1e-20;
			//options.parameter_tolerance = 1e-16;
			options.max_num_iterations = 5;
			options.initial_trust_region_radius = 1e+00;

			Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			//	std::cout << summary.BriefReport() << "\n";
			//std::cout << summary.FullReport() << "\n";
			//cout << summary.final_cost << endl;

			//输出F矩阵
			for (int j = 0; j < 8; j++)
			{
				FM(j) = vec_F[j];
			}
		}
}
