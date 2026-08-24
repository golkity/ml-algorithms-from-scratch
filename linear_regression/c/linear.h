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

void linear_free(LinearRegression *model);

double learning_mse(
    const LinearRegression *model,
    const double *x,
    const double *y,
    size_t sample_cnt
);

double learning_predict_one(
    const LinearRegression *model,
    const double *sample
);

int learn_train(
    LinearRegression *model,
    const double *x,
    const double *y,
    size_t sample_cnt,
    size_t epochs
);

#endif
