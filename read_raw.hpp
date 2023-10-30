#pragma once

#include "reduce_comm.hpp"

using namespace std;

using json_p = shared_ptr<json>;
auto f_read_positive(const string & path) -> json_p {
    const auto ret_json = make_shared<json>();
    try {
        ifstream fin(path, ios::in);
        fin >> *ret_json;   
    }
    catch(const exception & e) {
        WARN(e.what());
        WARNF("When reading file: %s.", path.c_str());
        FATAL_ERROR("Reading positive file failed.");
    }
    assert(ret_json->count("result") != 0);
#ifdef DISP_DATASET_DETAILS
    LOGF("Read %7ld positive records form %10s.", (*ret_json)["result"].size(), path.c_str());
#endif
    return ret_json;
}


auto f_read_negative(const string & path) -> json_p {
    const auto ret_json = make_shared<json>();
    try {
        ifstream fin(path, ios::in);
        fin >> *ret_json;   
    }
    catch(const exception & e) {
        WARN(e.what());
        WARNF("When reading file: %s.", path.c_str());
        FATAL_ERROR("Reading negative file failed.");
    }
    assert(ret_json->count("result") != 0);
    // LOGF("Read %7ld negative records form %10s.", (*ret_json)["result"].size(), path.c_str());
    return ret_json;
}


#define CHECK_JSON_KEY(J, K) do { \
    if (!J.count(#K)) { \
        WARNF("Json Key %s not found", #K); \
        FATAL_ERROR("Incompleted configuration."); \
    } \
} while(0)

auto read_configuration(const string & path) -> json {
    json config_j;
    try {
        ifstream fin(path, ios::in);
        fin >> config_j;
    }
    catch(const exception& e) {
        WARN(e.what());
        FATAL_ERROR("Reading configuration file failed.");
    }
    CHECK_JSON_KEY(config_j, target);
    if (!config_j.count("data") && !config_j.count("groups")) {
        FATAL_ERROR("Incompleted configuration, data/groups are not found.");
    }

    CHECK_JSON_KEY(config_j, max_num_tp);
    CHECK_JSON_KEY(config_j, max_num_fp);

    const string target = config_j["target"].get<string>();

    const auto _f_check = [target] (const json & j_check) -> void {
        if (target == "Hypervision") {
            CHECK_JSON_KEY(j_check, min_point_short);
            CHECK_JSON_KEY(j_check, min_point_long);
            CHECK_JSON_KEY(j_check, cluster_num);
        } else if (target == "Whisper" || target == "FAE") {
            CHECK_JSON_KEY(j_check, eps);
            CHECK_JSON_KEY(j_check, min_point);
            CHECK_JSON_KEY(j_check, rad);
            CHECK_JSON_KEY(j_check, size_line);
        } else {
            CHECK_JSON_KEY(j_check, eps);
            CHECK_JSON_KEY(j_check, min_point);
            CHECK_JSON_KEY(j_check, rad);

            CHECK_JSON_KEY(j_check, feature_dist);
            CHECK_JSON_KEY(j_check, water_line);
        }
    };

    if (config_j.count("groups")) {
        for (const auto & ref : config_j["groups"]) {
            CHECK_JSON_KEY(ref, data);
            _f_check(ref);
        }
    } else {
        _f_check(config_j);
    }
    return config_j;
}
