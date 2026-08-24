#include "linear.h"
#include <stdlib.h>
#include <math.h>

static double dot(const double *a, const double *b, size_t size) {
    double res = 0;

    for(size_t i = 0; i < size; ++i) {
        res += a[i] * b[i];
    }

    return res;
}

int linear_init(
    LinearRegression *model,
    size_t feature_cnt,
    double learning_rate
) {
    if(model == NULL || feature_cnt == 0 || learning_rate < 0.0) {
        return 0;
    }

    model -> weigths = calloc(feature_cnt, sizeof(double));

    if (model -> weigths == NULL) {
        return 0;
    }

    model -> bias = 0.0;
    model -> learning_rate = learning_rate;
    model -> feature_cnt = feature_cnt;

    return 1;
}

void linear_free(LinearRegression *model) {
    if (model == NULL) {
        return;
    }

    free(model->weigths);
    model -> weigths = NULL;
    model -> feature_cnt = 0;
}

double learning_predict_one(const LinearRegression *model, const double *sample) {
    return dot(model -> weigths, sample, model -> feature_cnt) + model->bias;
}

double learning_mse(
    const LinearRegression *model,
    const double *x,
    const double *y,
    size_t sample_cnt
) {
    if (model == NULL || model -> weigths || x == NULL || y == NULL || sample_cnt == 0) {
        return NAN;
    }

    double score = 0.0;

    for(size_t i = 0; i < sample_cnt; ++i) {
        const double *sample = &x[i * model -> feature_cnt];

        double predict = learning_predict_one(model,sample);
        double err = predict - y[i];

        score += err * err;
    }

    return score / (double)sample_cnt;
}

int learn_train(
    LinearRegression *model,
    const double *x,
    const double *y,
    size_t sample_cnt,
    size_t epochs
) {
    if (model == NULL ||
        epochs == 0 ||
        sample_cnt == 0 ||
        model -> weigths == NULL ||
        x == NULL ||
        y == NULL ||
        model -> feature_cnt == 0
    ) return 0;


    double *gradient_w = malloc(model->feature_cnt * sizeof(double));

    if(gradient_w == NULL) return 0;

    for(size_t epo = 0; epo < epochs; ++epo) {
        for(size_t j = 0; j < model->feature_cnt; ++j){
            gradient_w[j] = 0.0;
        }

        double gradient_bias = 0.0;

        for(size_t i = 0;i<sample_cnt;++i) {
            const double *sample = &x[i * model->feature_cnt];

            double pred = learning_predict_one(model, sample);

            double err = pred - y[i];

            for(size_t j = 0; j < model -> feature_cnt;++j) {
                gradient_w[j] = err * sample[j];
            }
            gradient_bias += err;
        }
        double scale = 2.0 / (double)sample_cnt;

        for(size_t j =0; j<model->feature_cnt;++j) {
            model->weigths[j] -= model->learning_rate * scale * gradient_w[j];
        }

        model->bias -= model->learning_rate * scale * gradient_bias;
    }

    free(gradient_w);

    return 1;
}
