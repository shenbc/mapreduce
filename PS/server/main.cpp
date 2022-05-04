//
// Created by FangJ on 2021/11/19.
//
#include "main.h"

int receive_packet(int sockfd, char *buf, size_t len, struct sockaddr_in *dst);

void getDataFromNIC(char const *iface, tensor_context* tensors);

tensor_context aggregateTensors(tensor_context* tensors);

void sendDataToWorkers(tensor_context *tensor);

void processPacket(char *buffer, int len);

int main(int argc, char* argv[]){
//    std::string NICName= atoi(argc[0]);

    char const *iface="eno5";

    tensor_context *tensors = new tensor_context[MAX_WORKER*MAX_STORAGE_PER_APP_PER_THREAD];
    for (int i =0 ;i < MAX_WORKER*MAX_STORAGE_PER_APP_PER_THREAD; i++){
        init_tensor(&tensors[i],MAX_TENSOR_SIZE);
    }

    while(1) {
        getDataFromNIC(iface, tensors);

        tensor_context aggregatedTensor=aggregateTensors(tensors);

        sendDataToWorkers(&aggregatedTensor);
    }

    return 0;
}

void sendDataToWorkers(tensor_context *tensor) {

}

tensor_context aggregateTensors(tensor_context* tensors) {

}

void getDataFromNIC(char const *iface, tensor_context* tensors) {
    struct ifreq nif;
    struct sockaddr_in addr;
    int addr_len =sizeof(struct sockaddr_in);
    char buffer[256];
    strcpy(nif.ifr_name,iface);

    printf("SETUP:\n");
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sock == -1) {
        printf("failed.\n");
        perror("ERROR:");
        return;
    }

//    if(inet_aton("172.16.170.3",&addr.sin_addr) != 1){
//        close(sock);
//        return;
//    }
//    addr.sin_family=AF_INET;
//    if (bind(sock, (struct sockaddr *)&raddr, sizeof(raddr))!=0){
//        printf("bind failed\n");
//        close(sock);
//        return;
//    }

    bzero(&addr,sizeof(addr));
    addr.sin_family=AF_INET;
    addr.sin_addr.s_addr= inet_addr("172.16.170.5");

    if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, (char *)&nif, sizeof(nif)) == -1) {
        printf("bind nic %s failed.",(char *)&nif);
        close(sock);
        return;
    }
    printf("Getting packets:\n");
    while(true){
        bzero(buffer,sizeof(buffer));
//        printf("input something...\n");
//        int len= read(STDIN_FILENO,buffer,sizeof(buffer));
//        sendto(sock, buffer, len, 0, (const sockaddr *)(&addr), addr_len);
        int len = recvfrom(sock, buffer, sizeof(buffer), 0,NULL,NULL);
        processPacket(buffer,len);
    }
}

void processPacket(char *buffer, int len) {
    printf("%d\n",len);
    printf("receive: %s\n",buffer);
}
