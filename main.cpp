#include <gflags/gflags.h>

#include "./read_raw.hpp"
#include "./read_matrix.hpp"
#include "./read_dir.hpp"
// #include "./reduce_freq.hpp"
#include "./reduce_flow.hpp"
// #include "./reduce_graph.hpp"
#include "./gen_metrics.hpp"

#include "reduce_comm.hpp"

using namespace std;

DEFINE_string(config, "../example.json", "Target configuration file.");

mutex m_g_collector;
map<string, metrics_collect_t> global_flow_collector, global_packet_collector;

static const string base_dir = "../datasets/";
static const string log_base_dir = "../log/";
static string log_dir = log_base_dir;

void scrub_main(const string target, const string data, const size_t max_num_fp, const size_t max_num_tp, const json jin) {

    const string fp_addr = base_dir + target + '/' + data + "_FP.json";
    const string tp_addr = base_dir + target + '/' + data + "_TP.json";
    const string fn_addr = base_dir + target + '/' + data + "_FN.json";
    const string tn_addr = base_dir + target + '/' + data + "_TN.json";
    const string save_dir = log_dir + data + ".log";
    FILE * out_fp = fopen(save_dir.c_str(), "w");

    json_p p_fp_json = f_read_positive(fp_addr);
    json_p p_tp_json = f_read_positive(tp_addr);
    json_p p_fn_json = f_read_negative(fn_addr);
    json_p p_tn_json = f_read_negative(tn_addr);

    const auto metrics_flow_initial = generate_metrics_flow(p_fp_json, p_tp_json, p_fn_json, p_tn_json);
    const auto metrics_packet_initial = generate_metrics_packet(p_fp_json, p_tp_json, p_fn_json, p_tn_json);

    index_list_p p_reduced_fp, p_reduced_tp;

    const auto p_m_fp = read_matrix(p_fp_json);
    const auto p_m_tp = read_matrix(p_tp_json);

    if (p_m_fp->size() == 0 && p_m_tp->size() == 0) return;
    tie(p_reduced_fp, p_reduced_tp) = reduce_fp_flow(data, p_m_fp, p_m_tp,
    max_num_fp, max_num_fp, jin["eps"], jin["min_point"], jin["rad"], jin["feature_dist"], jin["water_line"]);

    if (p_reduced_fp && p_reduced_tp) {
        const auto metrics_flow2 = generate_metrics_flow(p_fp_json, p_tp_json, p_fn_json, p_tn_json, p_reduced_fp, p_reduced_tp);
        const auto metrics_packet2 = generate_metrics_packet(p_fp_json, p_tp_json, p_fn_json, p_tn_json, p_reduced_fp, p_reduced_tp);

        const auto reduce_flow = get_reduce_result_map_flow(p_fp_json, p_tp_json, p_reduced_fp, p_reduced_tp);
        const auto reduce_packet = get_reduce_result_map_packet(p_fp_json, p_tp_json, p_reduced_fp, p_reduced_tp);

#ifdef DISP_METRICS_INDIVIDUAL
        show_metrics_flow_packet(data, metrics_flow_initial, metrics_flow2, metrics_packet_initial, metrics_packet2, out_fp);
        show_reduce_flow_packet(data, reduce_flow, reduce_packet, out_fp);
#endif

        const auto final_result_part = get_final_result_map(metrics_flow_initial, metrics_flow2, 
                                                            metrics_packet_initial, metrics_packet2,
                                                            reduce_flow, reduce_packet);
        m_g_collector.lock();
        global_flow_collector.insert({data, final_result_part.first});
        global_packet_collector.insert({data, final_result_part.second});
        m_g_collector.unlock();
    }
}


void expand_dir(const string target_name, const string dataset_name) {
    test_and_create_dir(log_base_dir);
    log_dir += target_name + "/";
    test_and_create_dir(log_dir);
    log_dir += dataset_name + "/";
    test_and_create_dir(log_dir);
    test_and_clear_dir(log_dir);
}


int main(int argc, char * argv[]) {
    __START_FTIMMER__

    google::ParseCommandLineFlags(&argc, &argv, true);

    const json config_j = read_configuration(FLAGS_config);
    const auto target_name = static_cast<string>(config_j["target"]);

    const size_t token_str = FLAGS_config.rfind("/") + 1;
    const string dataset_token = FLAGS_config.substr(token_str, FLAGS_config.find(".json") - token_str);
    expand_dir(target_name, dataset_token);

    vector<thread> tasks_vec;
    if (config_j.count("groups")) {
        for (const auto & ref: config_j["groups"]) {
            const auto data_name_vec = ref["data"];
            for (const auto & obj: data_name_vec) {
                const string data_name = static_cast<string>(obj);
                tasks_vec.emplace_back(
                    scrub_main, target_name, data_name, config_j["max_num_fp"], config_j["max_num_tp"], ref
                );
            }
        }
    } else {
        const auto data_name_vec = config_j["data"];
        for (const auto & obj: data_name_vec) {
            const string data_name = static_cast<string>(obj);
            tasks_vec.emplace_back(
                scrub_main, target_name, data_name, config_j["max_num_fp"], config_j["max_num_tp"], config_j
            );
        }
    }

    for (auto & p: tasks_vec)
        p.join();

#ifdef DISP_OVERALL_METRICS
    FILE * out_fp = fopen((log_dir + "All.log").c_str(), "w");
    show_final_result(global_flow_collector, global_packet_collector, out_fp);
#endif

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return 0;
}
