
#include "ceres/ceres.h"
#include<vector>
#include <iostream>
#include <list>

//opencv
#include <opencv2/core/core.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include "opencv2/core/hal/hal.hpp"
#include <opencv2/features2d/features2d.hpp>


using namespace Eigen;
using namespace std;
using namespace cv;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

