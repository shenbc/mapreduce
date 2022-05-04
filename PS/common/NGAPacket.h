//
// Created by FangJ on 2021/11/19.
//

#ifndef PS_NGAPACKET_H
#define PS_NGAPACKET_H

#include <stdint.h>
#include <net/ethernet.h>
#include <arpa/inet.h>
#include <cstring>
#include <cstdio>
#include <thread>
#include <mutex>
#include <inttypes.h>
#include <iostream>
#include <bitset>
#include <chrono>

#define ETH_TYPE 0x07, 0x00

#define IP_HDRS 0x45, 0x00, 0x00, 0x54, 0x00, 0x00, 0x40, 0x00, 0x40, 0x01, 0xaf, 0xb6

#define SRC_IP 0x0d, 0x07, 0x38, 0x66

#define DST_IP 0x0d, 0x07, 0x38, 0x7f

#define P4ML_PACKET_SIZE 308
#define P4ML_DATA_SIZE 248
#define P4ML_HEADER_SIZE 26
#define P4ML_LAYER_SIZE 274
#define IP_ETH_UDP_HEADER_SIZE 34

#define MAX_ENTRIES_PER_PACKET 62

struct NGAPacket {
    uint32_t workerMap;
    uint8_t degreeOverflowIsAckECN;
    uint32_t switchId;
    uint32_t SequenceId;
    int32_t data[MAX_ENTRIES_PER_PACKET];
};

#endif //PS_NGAPACKET_H
