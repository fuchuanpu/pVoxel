#pragma once

#include "reduce_comm.hpp"

#include "./read_matrix.hpp"
#include "./KDTree.hpp"

using namespace std;

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/metrics/lmetric.hpp>


auto reduce_fp_flow(const string tag,
                const feature_matrix_p fp_feature, const feature_matrix_p tp_feature,
                const int32_t max_num_fp=-1, const int32_t max_num_tp=-1,
                const double_t eps=1e-5, const size_t min_point=200,
                const double_t rad=1e-1, const size_t feature_dist=10, const double_t water_line=0.5) 
                -> pair<shared_ptr<vector<size_t> >, shared_ptr<vector<size_t> > > {
    __START_FTIMMER__

    const auto p_feature_vec = make_shared<feature_matrix_t>();
    size_t num_fp = 0, num_tp = 0;
    tie(num_fp, num_tp) = construct_feature_matrix(p_feature_vec, fp_feature, tp_feature, max_num_fp, max_num_tp);
    feature_matrix_t & features_vec = *p_feature_vec;

#ifdef DISP_DATASET_DETAILS
    LOGF("[%-10s] FPScrub processes: %8ld alarms (FP: %7ld, TP: %7ld)", 
        tag.c_str(),features_vec.size(), num_fp, num_tp);
#endif
    assert(num_fp + num_tp == features_vec.size());

    //// Type-I FP
    double_t type_I_start = get_time_spec();
    const auto f_get_sqr_id = [eps] (const vector<double> & in, vector<size_t> & out) -> void {
        out.clear();
        for (const auto v : in) {
            out.push_back(floor(v / eps));
        }
    };
    // create a hash table with square ID as key, point IDs as values
    double_t ___s1 = get_time_spec();
    unordered_map<size_t, vector<size_t> > squares;
    for (size_t i = 0; i < features_vec.size(); ++ i) {
        const auto pf = features_vec[i];
        vector<size_t> sqr_id;
        f_get_sqr_id(*pf, sqr_id);
        size_t hash_code = f_get_vec_hash(sqr_id);
        if (squares.find(hash_code) != squares.end()) {
            squares[hash_code].push_back(i);
        } else {
            squares.insert({hash_code, {i}});
        }
    }
    double_t ___e1 = get_time_spec();

    // Identify the density squares
    double_t ___s2 = get_time_spec();
    size_t cluster_numbering = 0;
    vector<int32_t> clust_id_vec(features_vec.size(), -1);
    vector<size_t> cluster_size_vec;
    size_t num_type_I_FP = 0;
    vector<size_t> cluster_TP_size_vec, cluster_FP_size_vec;
    unordered_map<size_t, size_t> cluid2hashcode, hashcode2cluid;
    for (const auto & ref: squares) {
        const auto & vec_ref = ref.second;
        if (vec_ref.size() > min_point) {
            size_t TP_ctr = 0;
            for (const auto idx: vec_ref) {
                clust_id_vec[idx] = cluster_numbering;
                TP_ctr += (idx >= num_fp);
            }
            cluid2hashcode.insert({cluster_numbering, ref.first});
            hashcode2cluid.insert({ref.first, cluster_numbering});
            ++ cluster_numbering;
            cluster_size_vec.push_back(vec_ref.size());
            cluster_TP_size_vec.push_back(TP_ctr);
            cluster_FP_size_vec.push_back(vec_ref.size() - TP_ctr);
        } else {
            num_type_I_FP += vec_ref.size();
        }
    }
    double_t ___e2 = get_time_spec();

    // adjacency, obtain the square for each density square
    double_t ___s3 = get_time_spec();
    vector<vector<size_t> > vec_id(cluster_numbering);
    for (size_t i = 0; i < cluster_numbering; ++ i) {
        for (size_t j = 0; j < features_vec.size(); ++ j) {
            if (clust_id_vec[j] == i) {
                vector<size_t> _id;
                f_get_sqr_id(*features_vec[j], _id);
                vec_id[i] = _id;
                break;
            }
        }
    }
    double_t ___e3 = get_time_spec();
    // construct the adjacency table
    double_t ___s4 = get_time_spec();
    vector<vector<bool> > adj_mtx(cluster_numbering);
    for (size_t i = 0; i < cluster_numbering; ++ i) {
        for (size_t j = 0; j < cluster_numbering; ++ j) {
            if (i != j) {
                const auto & va = vec_id[i];
                const auto & vb = vec_id[j];

                size_t num_diff = 0;
                for (size_t k = 0; k < va.size(); ++ k) {
                    num_diff += abs(int(vb[k]) - int(va[k]));
                }
                bool is_adj = num_diff < feature_dist * vb.size();
                // bool is_adj = num_diff < 2 * vb.size();
                adj_mtx[i].push_back(is_adj);

            } else {
                adj_mtx[i].push_back(true);
            }
        }
    }
    double_t ___e4 = get_time_spec();
    // Flody Alg. gets the community of adjacent cluster
    double_t ___s5 = get_time_spec();
    double_t BOOST_PP_SEQ_SIZE_51 = get_time_spec();
    vector<vector<bool> > community_vec = adj_mtx;
    for (size_t i = 0; i < cluster_numbering; ++ i) {
        for (size_t j = 0; j < cluster_numbering; ++ j) {
            for (size_t k = 0; k < cluster_numbering; ++ k) {
                if (community_vec[j][i] && community_vec[k][i]) {
                    community_vec[j][k] = true;
                }
            }
        }
    }
    double_t ___e5 = get_time_spec();
    // construct community
    double_t ___s6 = get_time_spec();
    size_t num_comm = 0;
    vector<int32_t> cluster2comm(cluster_numbering, -1);
    vector<size_t> comm_cluster_num(cluster_numbering, 0);
    for (size_t i = 0; i < cluster_numbering; ++ i) {
        if (cluster2comm[i] == -1) {
            for (size_t j = 0; j < cluster_numbering; ++ j) {
                if (community_vec[i][j]) {
                    cluster2comm[j] = num_comm;
                    comm_cluster_num[num_comm] ++;
                }
            }
            num_comm ++;
        }
    }
    double_t ___e6 = get_time_spec();
    // statistical Info. for community
    vector<size_t> comm_size_vec(num_comm, 0);
    vector<int32_t> comm_id_vec(features_vec.size(), -1);
    vector<size_t> comm_TP_size_vec(num_comm, 0), comm_FP_size_vec(num_comm, 0);
    for (size_t i = 0; i < clust_id_vec.size(); ++ i) {
        const auto v = clust_id_vec[i];
        if (v != -1) {
            const auto _comm_id = cluster2comm[v];
            comm_id_vec[i] = _comm_id;
            comm_size_vec[_comm_id] ++;
            (i < num_fp ? comm_FP_size_vec: comm_TP_size_vec)[_comm_id] ++;
        }
    }
    double_t type_I_end = get_time_spec();

    size_t num_reserved_fp = count_if(begin(clust_id_vec), begin(clust_id_vec) + num_fp, [] (const uint32_t x) -> bool { return x != -1; });
    size_t num_reserved_tp = count_if(begin(clust_id_vec) + num_fp, end(clust_id_vec), [] (const uint32_t x) -> bool { return x != -1; });

    //// Type-II FP
    // build KDT
    double_t type_II_start = get_time_spec();
    pointVec points;
    vector<size_t> scatter_id_vec;
    for (size_t i = 0; i < features_vec.size(); ++ i) {
        if (clust_id_vec[i] == -1) {
            points.push_back(*features_vec[i]);
            scatter_id_vec.push_back(i);
        }
    }
    KDTree tree(points);
    vector<int32_t> clust_id_vec2 = clust_id_vec;
    size_t num_type_II_FP = 0;

    // count the scatter for each cluster within the ball
    vector<size_t> comm_scatter(num_comm, 0);
    vector<size_t> comm_scatter2(num_comm, 0);
    vector<size_t> comm_scatter4(num_comm, 0);
    vector<size_t> comm_scatter_TP(num_comm, 0);
    vector<size_t> comm_scatter_FP(num_comm, 0);

    const size_t multiplex = 64;
    mutex wrt_mutex;
    const u_int32_t part_size = ceil(((double) num_comm) / ((double) multiplex));
    vector<pair<size_t, size_t> > _assign2;
    for (size_t core = 0, idx = 0; core < multiplex; ++ core, idx = min(num_comm, idx + part_size)) {
        _assign2.push_back({idx, min(idx + part_size, num_comm)});
    }

    auto __f = [&] (size_t _from, size_t _to) -> void {
        
    for (size_t i = _from; i < _to; ++ i) {
        vector<double_t> _core;
        for (size_t j = 0; j < cluster_numbering; ++ j) {
            if (cluster2comm[j] == i) {
                const auto _clust_id_vec = squares[cluid2hashcode[j]];
                const auto id = _clust_id_vec[0];
                const auto sym_feature = *features_vec[id];
                if (_core.size() == 0) {
                    _core = sym_feature;
                    break;
                } else {
                    for (size_t k = 0; k < sym_feature.size(); ++ k) {
                        _core[k] += sym_feature[k];
                    }
                }
            }
        }
        for (size_t k = 0; k < _core.size(); ++ k) {
            _core[k] /= comm_cluster_num[i];
        }
        const auto res = tree.neighborhood_indices(_core, rad);
        int32_t num_sca = res.size();
        const auto res2 = tree.neighborhood_indices(_core, rad * 2);
        int32_t num_sca2 = res.size();
        const auto res4 = tree.neighborhood_indices(_core, rad * 4);
        int32_t num_sca4 = res.size();

        size_t scatter_FP = 0, scatter_TP = 0;
        for (const auto idx : res) {
            (scatter_id_vec[idx] < num_fp ? scatter_FP : scatter_TP) ++;
        }
        wrt_mutex.lock();
        comm_scatter[i] = num_sca;
        comm_scatter2[i] = num_sca2;
        comm_scatter4[i] = num_sca4;
        comm_scatter_FP[i] = scatter_FP;
        comm_scatter_TP[i] = scatter_TP;
        wrt_mutex.unlock();
    }

    };

    vector<thread> vt2;
    for (size_t core = 0; core < multiplex; ++core) {
        vt2.emplace_back(__f, _assign2[core].first, _assign2[core].second);
    }
    for (auto & t : vt2)
        t.join();

    
    // cluster the community
    vector<vector<double> > community_features(num_comm, {0, 0, 0});
    for (size_t i = 0; i < num_comm; ++ i) {
        community_features[i] = {
            (double_t) comm_size_vec[i],
            (double_t) comm_scatter[i],
            (double_t) comm_scatter2[i],
            (double_t) comm_scatter4[i]
        };
    }

    vector<double_t> comm_dist(num_comm, 0);
    vector<bool> comm_preserve(num_comm, true);
    if (num_comm > 2) {
        auto data_mat = f_trans_armadillo_mat_T(community_features);

        mlpack::data::MinMaxScaler _scale;
        _scale.Fit(data_mat);
        decltype(data_mat) __pre_norm_mat = data_mat;
        _scale.Transform(__pre_norm_mat, data_mat);

        mlpack::pca::PCA<> _pca;
        _pca.Apply(data_mat, 1);

        arma::Row<size_t> assignments;
        arma::mat centroids;
        mlpack::kmeans::KMeans<> k;
        k.Cluster(data_mat, 1, assignments, centroids);

        mlpack::metric::EuclideanDistance euclidean_eval;
        for (size_t i = 0; i < num_comm; ++ i) {
            const auto dist = euclidean_eval.Evaluate(centroids.col(0), data_mat[i]);
            comm_dist[i] = dist;
            if (dist < water_line) {
                comm_preserve[i] = false;
                for (size_t j = 0; j < comm_id_vec.size(); ++ j) {
                    if (comm_id_vec[j] == i) {
                        clust_id_vec2[j] = -1;
                        ++ num_type_II_FP; 
                    }
                }
            }
        }
    }
    double_t type_II_end = get_time_spec();

    size_t num_reserved_fp2 = count_if(begin(clust_id_vec2), begin(clust_id_vec2) + num_fp, [] (const uint32_t x) -> bool { return x != -1; });
    size_t num_reserved_tp2 = count_if(begin(clust_id_vec2) + num_fp, end(clust_id_vec2), [] (const uint32_t x) -> bool { return x != -1; });

    // collect the final results
    const auto p_reduced_fp = make_shared<vector<size_t> >();
    const auto p_reduced_tp = make_shared<vector<size_t> >();

    const auto & res_reduce_vec = clust_id_vec2;

    for (size_t i = 0; i < num_fp; ++ i) {
        if (res_reduce_vec[i] == -1) {
            p_reduced_fp->push_back(i);
        }
    }
    for (size_t i = 0; i < num_tp; ++ i) {
        if (res_reduce_vec[num_fp + i] == -1) {
            p_reduced_tp->push_back(i);
        }
    }

#ifdef DISP_REDUCE_DETAILS
    const auto _f_rep_safe_divide = [] (double_t a, double_t b) -> double_t {
        return b == 0 ? 1.0 : a / b; 
    };

    LOGF("[%-10s] Stage 1: Num. In-Clsuter: [Benign: %6ld / %6ld (%4.2lf)] [Malicious: %6ld / %6ld (%4.2lf)]",
        tag.c_str(),
        num_reserved_fp, num_fp, _f_rep_safe_divide(num_reserved_fp, num_fp),
        num_reserved_tp, num_tp, _f_rep_safe_divide(num_reserved_tp, num_tp));

    #ifdef DISP_SQUARE_DETAILS
    LOGF("Number of cluster: %ld.", cluster_numbering);
    for (size_t i = 0; i < cluster_numbering; ++ i) {
        printf("Cluster Size: %5ld, FP: %5ld, TP: %5ld, ADJ: %2d.\n", 
            cluster_size_vec[i], cluster_FP_size_vec[i], cluster_TP_size_vec[i],
            accumulate(begin(community_vec[i]), end(community_vec[i]), 0));
    }
    #endif

    LOGF("Number of Comm.: %ld.", num_comm);
    for (size_t i = 0; i < num_comm; ++ i) {
        printf("Comm. Size: %5ld, FP: %5ld, TP: %5ld, CLUST: %2ld, SCATTER: %5ld (FP: %5ld / TP: %5ld), DIS:%6.4lf <%s>.\n", 
            comm_size_vec[i], comm_FP_size_vec[i], comm_TP_size_vec[i], comm_cluster_num[i],
            comm_scatter[i], comm_scatter_FP[i], comm_scatter_TP[i], 
            comm_dist[i], comm_preserve[i] ? "Preserved" : "Excluded");
    }

    LOGF("[%-10s] Stage 2: Num. In-Clsuter: [Benign: %6ld / %6ld (%4.2lf)] [Malicious: %6ld / %6ld (%4.2lf)]",
        tag.c_str(),
        num_reserved_fp2, num_fp, _f_rep_safe_divide(num_reserved_fp2, num_fp),
        num_reserved_tp2, num_tp, _f_rep_safe_divide(num_reserved_tp2, num_tp));

    LOGF("[%-10s] Type-[I/II] FP Count: [%6ld / %6ld]",
        tag.c_str(),
        num_type_I_FP, num_type_II_FP);
    LOGF("[%-10s] Type-[I/II] FP Time Cost: [%6.3lf / %6.3lf]",
        tag.c_str(),
        type_I_end - type_I_start, type_II_end - type_II_start);

    LOGF("[%-10s] Overall Time Cost: [%6.3lf]s, Overall Processing Speed: [%6.3lf] p/s.\n",
        tag.c_str(), 
        type_I_end - type_I_start + type_II_end - type_II_start,
        features_vec.size() / (type_I_end - type_I_start + type_II_end - type_II_start));
    // LOGF("[%-10s] Performance Vector: [ %6.3lf / %6.3lf / %6.3lf / %6.3lf/ %6.3lf ]",
    //     tag.c_str(), 
    //     ___e1 - ___s1, 
    //     (type_I_end - type_I_start) - (___e1 - ___s1),
    //     type_II_end - type_II_start,
    //     type_I_end - type_I_start + type_II_end - type_II_start,
    //     features_vec.size() / (type_I_end - type_I_start + type_II_end - type_II_start));
#endif

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return {p_reduced_fp, p_reduced_tp};
}