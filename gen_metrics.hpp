#pragma once

#include "reduce_comm.hpp"

#include "./read_raw.hpp"


using metrics_collect_t = map<string, double_t>;
using index_list_p = shared_ptr<vector<size_t> >;
auto display_string_metrics(const metrics_collect_t & res) -> string {
    stringstream ss;
    for (const auto ref: res) {
        ss << ref.first << ' ' << fixed << setprecision(4) << ref.second << ", ";
    }
    string res_str = ss.str();
    return res_str.substr(0, res_str.length() - 2);
}


static const auto _f_safe_divide = [] (double_t a, double_t b, double_t c=1.0) -> double_t {
    return b == 0 ? c : a / b; 
};


static const set<string> negative_metrics = {"FPR", "EER"};
auto display_string_metrics_diff(const metrics_collect_t & bef, const metrics_collect_t & aft) -> string {
    stringstream ss;
    for (const auto ref: aft) {
        ss << ref.first << ' ' << fixed << setprecision(4);
        if (negative_metrics.find(ref.first) == negative_metrics.end()) {
            ss << _f_safe_divide(ref.second - bef.at(ref.first), bef.at(ref.first), 0.0);
        } else {
            ss << _f_safe_divide(bef.at(ref.first) - ref.second, bef.at(ref.first), 1.0);
        }
        ss << ", ";
    }
    string res_str = ss.str();
    return res_str.substr(0, res_str.length() - 2);
}


void show_metrics_flow_packet(const string data, const metrics_collect_t metrics_flow_initial, const metrics_collect_t metrics_flow2,
                              const metrics_collect_t metrics_packet_initial, const metrics_collect_t metrics_packet2, FILE * ofp=stdout) {
    fprintf(ofp, "[%-10s] Flow Before: %s.\n", data.c_str(), display_string_metrics(metrics_flow_initial).c_str());
    fprintf(ofp, "[%-10s] Flow After: %s.\n", data.c_str(), display_string_metrics(metrics_flow2).c_str());
    fprintf(ofp, "[%-10s] Flow Diff.: %s.\n", data.c_str(), display_string_metrics_diff(metrics_flow_initial, metrics_flow2).c_str());

    fprintf(ofp, "[%-10s] Pkt. Before: %s.\n", data.c_str(), display_string_metrics(metrics_packet_initial).c_str());
    fprintf(ofp, "[%-10s] Pkt. After: %s.\n", data.c_str(), display_string_metrics(metrics_packet2).c_str());
    fprintf(ofp, "[%-10s] Pkt. Diff.: %s.\n", data.c_str(), display_string_metrics_diff(metrics_packet_initial, metrics_packet2).c_str());
}

void show_reduce_flow_packet(const string data, const metrics_collect_t reduce_flow, const metrics_collect_t reduce_packet, FILE * ofp=stdout) {
    fprintf(ofp, "[%-10s] Flow Reduce: [Num. TP %7ld] [Reduced TP %7ld] (%5.4lf); [Num. FP %7ld] [Reduced FP %7ld] (%5.4lf).\n", 
        data.c_str(), 
        (size_t) reduce_flow.at("TP"), 
        (size_t) reduce_flow.at("Reduce_TP"), 
        _f_safe_divide(reduce_flow.at("Reduce_TP"), reduce_flow.at("TP"), 0.0),
        (size_t) reduce_flow.at("FP"), 
        (size_t) reduce_flow.at("Reduce_FP"), 
        _f_safe_divide(reduce_flow.at("Reduce_FP"), reduce_flow.at("FP"), 1.0)
    );

    fprintf(ofp, "[%-10s] Pkt. Reduce: [Num. TP %7ld] [Reduced TP %7ld] (%5.4lf); [Num. FP %7ld] [Reduced FP %7ld] (%5.4lf).\n", 
        data.c_str(), 
        (size_t) reduce_packet.at("TP"), 
        (size_t) reduce_packet.at("Reduce_TP"), 
        _f_safe_divide(reduce_packet.at("Reduce_TP"), reduce_packet.at("TP"), 0.0),
        (size_t) reduce_packet.at("FP"), 
        (size_t) reduce_packet.at("Reduce_FP"), 
        _f_safe_divide(reduce_packet.at("Reduce_FP"), reduce_packet.at("FP"), 1.0)
    );
}


auto get_metrics_diff_map(const metrics_collect_t & bef, const metrics_collect_t & aft) -> metrics_collect_t {
    metrics_collect_t ret;
    for (const auto ref: aft) {
        if (negative_metrics.find(ref.first) == negative_metrics.end()) {
            ret.insert({ref.first, _f_safe_divide(ref.second - bef.at(ref.first), bef.at(ref.first), 0.0)});
        } else {
            ret.insert({ref.first, _f_safe_divide(bef.at(ref.first) - ref.second, bef.at(ref.first), 1.0)});
        }
    }
    return ret;
}


auto get_reduce_result_map_flow(const json_p p_fp_json, const json_p p_tp_json,
                                const index_list_p p_reduced_fp, const index_list_p p_reduced_tp) -> metrics_collect_t {
    metrics_collect_t ret;
    size_t num_fp = (*p_fp_json)["result"].size(), num_tp = (*p_tp_json)["result"].size();
    ret.insert({"FP", num_fp});
    ret.insert({"TP", num_tp});
    ret.insert({"Reduce_FP", p_reduced_fp->size()});
    ret.insert({"Reduce_TP", p_reduced_tp->size()});
    ret.insert({"Reduce_FP_Rate", _f_safe_divide(p_reduced_fp->size(), num_fp, 1.0) });
    ret.insert({"Reduce_TP_Rate", _f_safe_divide(p_reduced_tp->size(), num_tp, 0.0) });
    return ret;
}

auto get_reduce_result_map_packet(const json_p p_fp_json, const json_p p_tp_json,
                                    const index_list_p p_reduced_fp, const index_list_p p_reduced_tp) -> metrics_collect_t {
    size_t num_fp = 0, num_tp = 0;
    for (const auto & ref : (*p_fp_json)["result"]) {
        num_fp += ref["len"].get<size_t>();
    }
    for (const auto & ref : (*p_tp_json)["result"]) {
        num_tp += ref["len"].get<size_t>();
    }

    size_t reduced_fp = 0, reduced_tp = 0;
    for (const auto index: * p_reduced_fp) {
        reduced_fp += (*p_fp_json)["result"][index]["len"].get<size_t>();
    }
    for (const auto index: * p_reduced_tp) {
        reduced_tp += (*p_tp_json)["result"][index]["len"].get<size_t>();
    }
    metrics_collect_t ret;
    ret.insert({"FP", num_fp});
    ret.insert({"TP", num_tp});
    ret.insert({"Reduce_FP", reduced_fp});
    ret.insert({"Reduce_TP", reduced_tp});
    ret.insert({"Reduce_FP_Rate", _f_safe_divide(reduced_fp, num_fp, 1.0) });
    ret.insert({"Reduce_TP_Rate", _f_safe_divide(reduced_tp, num_tp, 0.0) });
    return ret;
}


static auto __f_merge_result_map(const metrics_collect_t ma, const metrics_collect_t mb) -> metrics_collect_t {
    metrics_collect_t ret = mb;
    for (const auto & ref: ma) {
        ret.insert({ref.first, ref.second});
    }
    return ret;
}

static auto _get_final_result_map_flow(const metrics_collect_t metrics_flow_initial, 
                                        const metrics_collect_t metrics_flow2, const metrics_collect_t reduce_flow) -> metrics_collect_t {

    metrics_collect_t metric_res = get_metrics_diff_map(metrics_flow_initial, metrics_flow2);
    return __f_merge_result_map(metric_res, reduce_flow); 
}

static auto _get_final_result_map_packet(const metrics_collect_t metrics_packet_initial, 
                                        const metrics_collect_t metrics_packet2, const metrics_collect_t reduce_packet) -> metrics_collect_t {
    
    metrics_collect_t metric_res = get_metrics_diff_map(metrics_packet_initial, metrics_packet2);
    return __f_merge_result_map(metric_res, reduce_packet);
}

auto get_final_result_map(const metrics_collect_t metrics_flow_initial, const metrics_collect_t metrics_flow2,
                            const metrics_collect_t metrics_packet_initial, const metrics_collect_t metrics_packet2,
                            const metrics_collect_t reduce_flow, const metrics_collect_t reduce_packet) -> pair<metrics_collect_t, metrics_collect_t> {
    return {
        _get_final_result_map_flow(metrics_flow_initial, metrics_flow2, reduce_flow),
        _get_final_result_map_packet(metrics_packet_initial, metrics_packet2, reduce_packet),
    };
}


auto _avg_aggregate_final_result(const map<string, metrics_collect_t> res) -> metrics_collect_t {
    metrics_collect_t ret = cbegin(res)->second;
    for (auto & ref: ret) {
        ref.second = 0;
    }
    for (const auto ref1: res) {
        for (const auto ref2: ref1.second) {
            ret[ref2.first] += ref2.second;
        }
    }
    for (auto & ref: ret) {
        ref.second /= res.size();
    }
    return ret;
}  

void show_final_result(const map<string, metrics_collect_t> global_flow_collector, 
                        const map<string, metrics_collect_t> global_packet_collector, FILE * ofp=stdout) {
    assert(global_flow_collector.size() == global_packet_collector.size());
    const auto agg_flow_collector = _avg_aggregate_final_result(global_flow_collector);
    const auto agg_packet_collector = _avg_aggregate_final_result(global_packet_collector);
    fprintf(ofp, "[Overall] Flow Metrics: %s.\n", display_string_metrics(agg_flow_collector).c_str());
    fprintf(ofp, "[Overall] Pkt. Metrics: %s.\n", display_string_metrics(agg_packet_collector).c_str());
}


//// Many metrics calculators.

auto safe_div(const size_t a, size_t b) -> double_t {
    if (b == 0) {
        return -1;
    } else {
        return double(a) / double(b);
    }
}

auto get_metrics_F1(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(2 * num_tp, 2 * num_tp + num_fp + num_fn);
}

auto get_metrics_Recall(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(num_tp, num_tp + num_fn);
}

auto get_metrics_Percision(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(num_tp, num_tp + num_fp);
}

auto get_metrics_AUPRC(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    double_t _p = get_metrics_Percision(num_fp, num_tp, num_fn, num_tn);
    double_t _r = get_metrics_Recall(num_fp, num_tp, num_fn, num_tn);
    return (_p * _r) + (((1 - _p) * _r) / 2) + (((1 - _r) * _p) / 2);
}

auto get_metrics_Accuracy(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(num_tp + num_tn, num_fp + num_tp + num_fn + num_tn);
}

auto get_metrics_MCC(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div((num_tp * num_tn) - (num_fp * num_fn), sqrt(1.0 * (num_tp + num_fp) * (num_tp + num_fn) * (num_tn + num_fp) * (num_tn + num_fn)));
}

auto get_metrics_TPR(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(num_tp, num_tp + num_fn);
}

auto get_metrics_FPR(const size_t num_fp, const size_t num_tp, const size_t num_fn, const size_t num_tn) -> double_t {
    return safe_div(num_fp, num_tn + num_fp);
}

auto get_metrics_AUROC(const shared_ptr<vector<double_t> > p_score, const shared_ptr<vector<bool> > p_label) -> double_t {
    size_t n = p_score->size();
    assert(n == p_label->size());

    const auto & score = *p_score;
    const auto & label = *p_label;

    for (int i = 0; i < n; i++)
		if (!std::isfinite(score[i]) || label[i] != 0 && label[i] != 1)
			return std::numeric_limits<double>::signaling_NaN();

    const auto order = new int[n];
	std::iota(order, order + n, 0);
	std::sort(order, order + n, [&] (int a, int b) { return score[a] > score[b]; });
	const auto y = new double[n];
	const auto z = new double[n];
	for (int i = 0; i < n; i++) {
		y[i] = label[order[i]];
		z[i] = score[order[i]];
	}

    const auto tp = y; // Reuse
	std::partial_sum(y, y + n, tp);

	int top = 0; // # diff
	for (int i = 0; i < n - 1; i++)
		if (z[i] != z[i + 1])
			order[top++] = i;
	order[top++] = n - 1;
	n = top; // Size of y/z -> sizeof tps/fps

	const auto fp = z; // Reuse
	for (int i = 0; i < n; i++) {
		tp[i] = tp[order[i]]; // order is mono. inc.
		fp[i] = 1 + order[i] - tp[i]; // Type conversion prevents vectorization
	}
	delete[] order;

	const auto tpn = tp[n - 1], fpn = fp[n - 1];
	for (int i = 0; i < n; i++) { // Vectorization
		tp[i] /= tpn;
		fp[i] /= fpn;
	}

	auto area = tp[0] * fp[0] / 2; // The first triangle from origin;
	double partial = 0; // For Kahan summation
	for (int i = 1; i < n; i++) {
		const auto x = (fp[i] - fp[i - 1]) * (tp[i] + tp[i - 1]) / 2 - partial;
		const auto sum = area + x;
		partial = (sum - area) - x;
		area = sum;
	}

	delete[] tp;
	delete[] fp;

	return area;
}

auto get_metrics_EER(const shared_ptr<vector<double_t> > p_score, const shared_ptr<vector<bool> > p_label) -> double_t {
    size_t n = p_score->size();
    assert(n == p_label->size());

    const auto & score = *p_score;
    const auto & label = *p_label;

    for (int i = 0; i < n; i++)
		if (!std::isfinite(score[i]) || label[i] != 0 && label[i] != 1)
			return std::numeric_limits<double>::signaling_NaN();

    const auto order = new int[n];
	std::iota(order, order + n, 0);
	std::sort(order, order + n, [&] (int a, int b) { return score[a] > score[b]; });
	const auto y = new double[n];
	const auto z = new double[n];
	for (int i = 0; i < n; i++) {
		y[i] = label[order[i]];
		z[i] = score[order[i]];
	}

    const auto tp = y; // Reuse
	std::partial_sum(y, y + n, tp);

	int top = 0; // # diff
	for (int i = 0; i < n - 1; i++)
		if (z[i] != z[i + 1])
			order[top++] = i;
	order[top++] = n - 1;
	n = top; // Size of y/z -> sizeof tps/fps

	const auto fp = z; // Reuse
	for (int i = 0; i < n; i++) {
		tp[i] = tp[order[i]]; // order is mono. inc.
		fp[i] = 1 + order[i] - tp[i]; // Type conversion prevents vectorization
	}
	delete[] order;

	const auto tpn = tp[n - 1], fpn = fp[n - 1];
	for (int i = 0; i < n; i++) { // Vectorization
		tp[i] /= tpn;
		fp[i] /= fpn;
	}

    double_t min_d = 1.0, eer = 1.0;
    for (int i = 0; i < n; i++) {
        const double_t _diff = fabs(fp[i] - (1 - tp[i]));
        if (_diff < min_d) {
            min_d = _diff;
            eer = fp[i];
        }
    }

	delete[] tp;
	delete[] fp;

	return eer;
}

auto generate_metrics_flow(const json_p p_fp_j, const json_p p_tp_j, const json_p p_fn_j, const json_p p_tn_j,
                            const index_list_p p_reduced_fp=nullptr, const index_list_p p_reduced_tp=nullptr) -> metrics_collect_t {
    metrics_collect_t ret;
    size_t num_fp = (*p_fp_j)["result"].size();
    size_t num_tp = (*p_tp_j)["result"].size();
    size_t num_fn = (*p_fn_j)["result"].size();
    size_t num_tn = (*p_tn_j)["result"].size();

    if (p_reduced_fp && p_reduced_tp) {
        num_fp -= p_reduced_fp->size();
        num_tn += p_reduced_fp->size();

        num_tp -= p_reduced_tp->size();
        num_fn += p_reduced_tp->size();
    }

    // cout << num_fp <<' '<< num_tp <<' '<< num_fn <<' '<< num_tn << endl;

    ret.insert(make_pair("F1", get_metrics_F1(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Recall", get_metrics_Recall(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Percision", get_metrics_Percision(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("AUPRC", get_metrics_AUPRC(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Accuracy", get_metrics_Accuracy(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("MCC", get_metrics_MCC(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("TPR", get_metrics_TPR(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("FPR", get_metrics_FPR(num_fp, num_tp, num_fn, num_tn)));

    vector<double_t> save_fp_loss, save_tp_loss;
    if (p_reduced_fp && p_reduced_tp) {
        for (const auto index : *p_reduced_fp) {
            save_fp_loss.push_back((*p_fp_j)["result"][index]["loss"].get<double_t>());
            (*p_fp_j)["result"][index]["loss"] = -1;
        }
        for (const auto index : *p_reduced_tp) {
            save_tp_loss.push_back((*p_tp_j)["result"][index]["loss"].get<double_t>());
            (*p_tp_j)["result"][index]["loss"] = 1000;
        }
    }

    const auto p_score = make_shared<vector<double_t> >();
    const auto p_label = make_shared<vector<bool> >();
    for (const auto & ref: (*p_fp_j)["result"]) {
        p_score->push_back(ref["loss"].get<double>());
        p_label->push_back(false);
    }
    for (const auto & ref: (*p_tp_j)["result"]) {
        p_score->push_back(ref["loss"].get<double>());
        p_label->push_back(true);
    }
    for (const auto & ref: (*p_fn_j)["result"]) {
        p_score->push_back(ref["loss"].get<double>());
        p_label->push_back(true);
    }
    for (const auto & ref: (*p_tn_j)["result"]) {
        p_score->push_back(ref["loss"].get<double>());
        p_label->push_back(false);
    }
    ret.insert(make_pair("AUROC", get_metrics_AUROC(p_score, p_label)));
    ret.insert(make_pair("EER", get_metrics_EER(p_score, p_label)));

    if (p_reduced_fp && p_reduced_tp) {
        for (size_t i = 0; i < p_reduced_fp->size(); ++ i) {
            (*p_fp_j)["result"][p_reduced_fp->at(i)]["loss"] = save_fp_loss[i];
        }
        for (size_t i = 0; i < p_reduced_tp->size(); ++ i) {
            (*p_tp_j)["result"][p_reduced_tp->at(i)]["loss"] = save_tp_loss[i];
        }
    }

    return ret;
}


auto generate_metrics_packet(const json_p p_fp_j, const json_p p_tp_j, const json_p p_fn_j, const json_p p_tn_j,
                                const index_list_p p_reduced_fp=nullptr, const index_list_p p_reduced_tp=nullptr) -> metrics_collect_t {
    metrics_collect_t ret;

    vector<size_t> fp_lens;
    for (const auto & ref: (*p_fp_j)["result"]) {
        fp_lens.push_back(ref["len"].get<size_t>());
    }
    vector<size_t> tp_lens;
    for (const auto & ref: (*p_tp_j)["result"]) {
        tp_lens.push_back(ref["len"].get<size_t>());
    }
    vector<size_t> fn_lens;
    for (const auto & ref: (*p_fn_j)["result"]) {
        fn_lens.push_back(ref["len"].get<size_t>());
    }
    vector<size_t> tn_lens;
    for (const auto & ref: (*p_tn_j)["result"]) {
        tn_lens.push_back(ref["len"].get<size_t>());
    }

    size_t num_fp = accumulate(fp_lens.begin(), fp_lens.end(), 0);
    size_t num_tp = accumulate(tp_lens.begin(), tp_lens.end(), 0);
    size_t num_fn = accumulate(fn_lens.begin(), fn_lens.end(), 0);
    size_t num_tn = accumulate(tn_lens.begin(), tn_lens.end(), 0);

    if (p_reduced_fp && p_reduced_tp) {
        for (const auto index : *p_reduced_fp) {
            num_fp -= fp_lens[index];
            num_tn += fp_lens[index];
        }
        for (const auto index : *p_reduced_tp) {
            num_tp -= tp_lens[index];
            num_fn += tp_lens[index];
        }
    }

    // cout << num_fp <<' '<< num_tp <<' '<< num_fn <<' '<< num_tn << endl;

    ret.insert(make_pair("F1", get_metrics_F1(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Recall", get_metrics_Recall(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Percision", get_metrics_Percision(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("AUPRC", get_metrics_AUPRC(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("Accuracy", get_metrics_Accuracy(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("MCC", get_metrics_MCC(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("TPR", get_metrics_TPR(num_fp, num_tp, num_fn, num_tn)));
    ret.insert(make_pair("FPR", get_metrics_FPR(num_fp, num_tp, num_fn, num_tn)));

    vector<double_t> save_fp_loss, save_tp_loss;
    if (p_reduced_fp && p_reduced_tp) {
        for (const auto index : *p_reduced_fp) {
            save_fp_loss.push_back((*p_fp_j)["result"][index]["loss"].get<double_t>());
            (*p_fp_j)["result"][index]["loss"] = -1;
        }
        for (const auto index : *p_reduced_tp) {
            save_tp_loss.push_back((*p_tp_j)["result"][index]["loss"].get<double_t>());
            (*p_tp_j)["result"][index]["loss"] = 1000;
        }
    }

    const auto p_score = make_shared<vector<double_t> >();
    const auto p_label = make_shared<vector<bool> >();
    for (const auto & ref: (*p_fp_j)["result"]) {
        fill_n(back_inserter(*p_score), ref["len"].get<size_t>(), ref["loss"].get<double>());
        fill_n(back_inserter(*p_label), ref["len"].get<size_t>(), false);
    }
    for (const auto & ref: (*p_tp_j)["result"]) {
        fill_n(back_inserter(*p_score), ref["len"].get<size_t>(), ref["loss"].get<double>());
        fill_n(back_inserter(*p_label), ref["len"].get<size_t>(), true);
    }
    for (const auto & ref: (*p_fn_j)["result"]) {
        fill_n(back_inserter(*p_score), ref["len"].get<size_t>(), ref["loss"].get<double>());
        fill_n(back_inserter(*p_label), ref["len"].get<size_t>(), true);
    }
    for (const auto & ref: (*p_tn_j)["result"]) {
        fill_n(back_inserter(*p_score), ref["len"].get<size_t>(), ref["loss"].get<double>());
        fill_n(back_inserter(*p_label), ref["len"].get<size_t>(), false);
    }
    ret.insert(make_pair("AUROC", get_metrics_AUROC(p_score, p_label)));
    ret.insert(make_pair("EER", get_metrics_EER(p_score, p_label)));

    if (p_reduced_fp && p_reduced_tp) {
        for (size_t i = 0; i < p_reduced_fp->size(); ++ i) {
            (*p_fp_j)["result"][p_reduced_fp->at(i)]["loss"] = save_fp_loss[i];
        }
        for (size_t i = 0; i < p_reduced_tp->size(); ++ i) {
            (*p_tp_j)["result"][p_reduced_tp->at(i)]["loss"] = save_tp_loss[i];
        }
    }

    return ret;
}

