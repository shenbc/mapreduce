//
// Created by FangJ on 2021/11/19.
//

#ifndef PS_MAIN_H
#define PS_MAIN_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/ip.h>
#include <error.h>
#include <inttypes.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <set>
#include "../common/NGAPacket.h"

#define MAX_TENSOR_SIZE 1024000
#define MAX_STORAGE_PER_APP_PER_THREAD 10
#define MAX_WORKER 16


union data_t {
    int32_t *data_int;
    float *data_float;
};

struct tensor_context {
//    bool* isOccupy;
//    bool* isCollision;
//    bool* isFloat;
//    bool isCompleted;
    data_t data;
//    uint32_t len;
//    uint64_t key;
//    uint8_t num_worker;
//    WindowManager* window_manager;
//    std::chrono::time_point<std::chrono::system_clock> start_time;
};

void inline init_tensor(tensor_context* tensor, uint32_t len) {
    tensor->data.data_int = new int32_t[len]();
//    tensor->isCompleted = true;
//    tensor->isOccupy = new bool[MAX_TENSOR_SIZE / MAX_ENTRIES_PER_PACKET + 1]();
//    tensor->isCollision = new bool[MAX_TENSOR_SIZE / MAX_ENTRIES_PER_PACKET + 1]();
//    tensor->isFloat = new bool[MAX_TENSOR_SIZE / MAX_ENTRIES_PER_PACKET + 1]();
//    tensor->len = 0;
//    tensor->num_worker = 0;
//    tensor->key = 0xffffffffffffffff;
//    tensor->window_manager = new WindowManager[MAX_WORKER];
//    for (int i = 0; i < MAX_WORKER; i++) {
//        tensor->window_manager[i].isACKed = new bool[MAX_TENSOR_SIZE / MAX_ENTRIES_PER_PACKET + 1]();
//        tensor->window_manager[i].total_ACK = MAX_TENSOR_SIZE / MAX_ENTRIES_PER_PACKET + 1;
//    }
}

#endif //PS_MAIN_H
