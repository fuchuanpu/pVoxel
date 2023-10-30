#pragma once

#include "./reduce_comm.hpp"

using namespace std;

#include <boost/filesystem.hpp>

auto test_and_create_dir(const string path) -> bool {
    using namespace boost::filesystem;
    bool ret = exists(path) && is_directory(path);
    if (!ret) {
        if (!create_directory(path)) {
            FATAL_ERROR("Create Dir Failed.");
        }
    }
    return ret;
}


auto test_and_clear_dir(const string path) -> bool {
    using namespace boost::filesystem;
    bool ret = exists(path) && is_directory(path);
    if (!ret) {
        return false;
    } else {
        directory_iterator end_itr;
        for (directory_iterator itr(path); itr != end_itr; ++itr) {
             if (is_regular_file(itr->path())) {
                string current_file = itr->path().string();
                if (!remove(current_file)) {
                    FATAL_ERROR("Clear Dir Failed.");
                }
             }
        }
    }
    return true;
}
