#ifndef LINEAR_H
#define LINEAR_H

#include <stddef.h>

typedef struct {
    double *weigths;
    double bias;
    double learning_rate;
    size_t feature_cnt;
} LinearRegression;

int learning_init(
    LinearRegression *model,
    size_t feature_cnt,
    double learning_rate
);

void linear_free();

double learning_mse();

double learning_predict_one();

int learn_train();

#endif
