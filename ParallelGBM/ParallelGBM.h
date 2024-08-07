#pragma once

#ifdef ParalelllGBM_EXPORTS
#define ParalellGBM_API __declspec(dllexport)
#else
#define ParalellGBM_API __declspec(dllimport)
#endif

extern "C" ParalellGBM_API double Paralellgbm(double sigma, double mu, double time, double curr_price, int samples, int steps_into_future, int numOfCores);

extern "C" ParalellGBM_API void GeometricBrownianMotion(double sigma, double mu, double time, double curr_price, int samples, int steps_into_future);
