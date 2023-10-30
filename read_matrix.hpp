#pragma once

#include "reduce_comm.hpp"

#include <mlpack/core.hpp>

using namespace std;

using feature_t = vector<double_t>;
using feature_p = shared_ptr<feature_t>;
using feature_matrix_t = vector<feature_p>;
using feature_matrix_p = shared_ptr<feature_matrix_t>;
auto read_matrix(const shared_ptr<json> p_j) -> feature_matrix_p {
    const auto ret = make_shared<feature_matrix_t>();
    for (const auto & obj : (*p_j)["result"]) {
        const auto __vec = obj["feature"];
        const auto _ve = make_shared<feature_t>(__vec.begin(), __vec.end());
        ret->push_back(_ve);
    }
    return ret;
}

using counter_t = vector<size_t>;
using counter_p = shared_ptr<counter_t>;
auto graph_read_cluster_size(const shared_ptr<json> p_j) -> counter_p {
    const auto ret = make_shared<counter_t>();
    for (const auto & obj : (*p_j)["result"]) {
        ret->push_back(obj["size"]);
    }
    return ret;
}

auto graph_read_flow_code(const shared_ptr<json> p_j) -> counter_p {
    const auto ret = make_shared<counter_t>();
    for (const auto & obj : (*p_j)["result"]) {
        ret->push_back(obj["code"]);
    }
    return ret;
}

auto graph_read_edge_code(const shared_ptr<json> p_j) -> counter_p {
    const auto ret = make_shared<counter_t>();
    for (const auto & obj : (*p_j)["result"]) {
        ret->push_back(obj["is_short"].get<bool>());
    }
    return ret;
}

auto read_lens(const shared_ptr<json> p_j) -> feature_p {
    const auto ret = make_shared<feature_t>();
    for (const auto & obj : (*p_j)["result"]) {
        ret->push_back(obj["len"]);
    }
    return ret;
}


void norm_matrix(const feature_matrix_p pmt, const bool log_norm=true, const size_t multiplex=64) {
    const auto ret = make_shared<feature_matrix_t>();
    size_t y = pmt->at(0)->size(), x = pmt->size();
    vector<double_t> max_vec(y, EPS);
    vector<double_t> min_vec(y, HUG);

    mutex wrt_mutex;
    const u_int32_t part_size = ceil(((double) x) / ((double) multiplex));
    vector<pair<size_t, size_t> > _assign;
    for (size_t core = 0, idx = 0; core < multiplex; ++ core, idx = min(x, idx + part_size)) {
        _assign.push_back({idx, min(idx + part_size, x)});
    }

    auto __f1 = [&] (size_t _from, size_t _to) -> void {
    vector<double_t> _max_vec(y, EPS);
    vector<double_t> _min_vec(y, HUG);
    for (size_t i = _from; i < _to; ++ i) {
        for (size_t j = 0; j < y; ++ j) {
            double_t v = pmt->at(i)->at(j);
            if (log_norm) {
                v = log2(max(0.0, v) + 1);
            }
            pmt->at(i)->at(j) = v;
            _max_vec[j] = max(_max_vec[j], v);
            _min_vec[j] = min(_min_vec[j], v);
        }
    }
    wrt_mutex.lock();
    for (size_t j = 0; j < y; ++ j) {
        max_vec[j] = max(max_vec[j], _max_vec[j]);
        min_vec[j] = min(min_vec[j], _min_vec[j]);
    }
    wrt_mutex.unlock();
    };

    vector<thread> vt1;
    for (size_t core = 0; core < multiplex; ++core) {
        vt1.emplace_back(__f1, _assign[core].first, _assign[core].second);
    }
    for (auto & t : vt1)
        t.join();

    vector<double_t> norm_vec(y, 0);
    for (size_t j = 0; j < y; ++ j) {
        norm_vec[j] = max_vec[j] - min_vec[j];
        if (fabs(norm_vec[j]) <= EPS) {
            norm_vec[j] = -1;
        }
    }

    auto __f2 = [&] (size_t _from, size_t _to) -> void {
    for (size_t i = _from; i < _to; ++ i) {
        for (size_t j = 0; j < y; ++ j) {
            double_t v = pmt->at(i)->at(j);
            pmt->at(i)->at(j) = norm_vec[j] < EPS ? 0.0 : (v - min_vec[j]) / norm_vec[j];
        }
    }
    };

    vector<thread> vt2;
    for (size_t core = 0; core < multiplex; ++core) {
        vt2.emplace_back(__f2, _assign[core].first, _assign[core].second);
    }
    for (auto & t : vt2)
        t.join();

}


static auto construct_feature_matrix(const shared_ptr<feature_matrix_t> p_feature_vec,
                const feature_matrix_p fp_feature, const feature_matrix_p tp_feature,
                const int32_t max_num_fp=-1, const int32_t max_num_tp=-1, bool norm_feature=true, bool log_norm=true) -> pair<size_t, size_t> {
    size_t num_fp = 0;
    feature_matrix_t & features_vec = *p_feature_vec;
    if (fp_feature->size() < max_num_fp || max_num_fp == -1) {
        features_vec.assign(fp_feature->begin(), fp_feature->end());
    } else {
        WARN("Partial FP is not supported in the cuerrent version.");
        auto rng = std::default_random_engine {};
        shuffle(fp_feature->begin(), fp_feature->end(), rng);
        features_vec.assign(fp_feature->begin(), fp_feature->begin() + max_num_fp);
    }
    num_fp = features_vec.size();

    size_t num_tp;
    if (tp_feature->size() < max_num_tp || max_num_tp == -1) {
        features_vec.reserve(features_vec.size() + tp_feature->size());
        features_vec.insert(features_vec.end(), tp_feature->begin(), tp_feature->end());
        num_tp = tp_feature->size();
    } else {
        WARN("Partial TP is not supported in the cuerrent version.");
        auto rng = std::default_random_engine {};
        shuffle(tp_feature->begin(), tp_feature->end(), rng);
        features_vec.reserve(features_vec.size() + max_num_tp);
        features_vec.insert(features_vec.end(), tp_feature->begin(), tp_feature->begin() + max_num_tp);
        num_tp = max_num_tp;
    }
    if (norm_feature) {
        norm_matrix(p_feature_vec, log_norm);
    }
    return {num_fp, num_tp};
}


static const auto f_trans_armadillo_mat_T = [] (const vector<vector<double> > & mx, const bool enable_log=true) -> arma::mat {
    size_t x_len = mx.size();
    size_t y_len = mx[0].size();
    arma::mat mxt(y_len, x_len , arma::fill::randu);
    for (size_t i = 0; i < x_len; i ++) {
        for (size_t j = 0; j < y_len; j ++) {
            if (enable_log)
                mxt(j, i) = log2(max(mx[i][j], 0.0) + 1);
            else
                mxt(j, i) = mx[i][j];
        }
    }
    return mxt;
};


const auto f_get_vec_hash = [] (const vector<size_t> & ve) -> size_t {
        size_t seed = ve.size();
        for (auto x: ve) {
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = (x >> 16) ^ x;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
};
