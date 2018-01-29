#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "mjpegserver.h"
#include "process.h"

//Socket definitions
#define PORT unsigned short
#define SOCKET int
#define HOSTENT struct hostent
#define SOCKADDR struct sockaddr
#define SOCKADDR_IN struct sockaddr_in
#define ADDRPOINTER unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define TIMEOUT_M 200000
#define NUM_CONNECTIONS 10
#define FRAME_UPDATE_DELAY 100 //Minimum time to wait before sending a new frame

//Custom macros
#define LOG(x) printf("%s\n", x)

//Pthread argument payload
struct clientPayload {
    void* context;
    clientFrame cf;
};

//Handle error messages
void handle_error(const char *err) {
	fprintf(stderr, "%s error!\n", err);
}

//Create the new listener inside the current pthread
static void *listen_helper(void* context) {
	((MJPEGServer *)context)->listener();
	return NULL;
}

//Creat the new writer inside the current pthread
static void *writer_helper(void* context) {
	((MJPEGServer *)context)->writer();
	return NULL;
}

MJPEGServer::MJPEGServer(int port) {
	sock = INVALID_SOCKET;
	client = INVALID_SOCKET;
	timeout = TIMEOUT_M;
	fail_c = 0;
	lastFrame = (unsigned char *) malloc(1);
	FD_ZERO(&master);
	if(port) {
		open(port);
	}
}

MJPEGServer::~MJPEGServer() {
	release();
}

void MJPEGServer::listener() {
    fd_set rread;
    SOCKET maxfd;
    while (true) {
        rread = master;
        struct timeval to = { 0, timeout };
        maxfd = sock + 1;

        int sel = select(maxfd, &rread, NULL, NULL, &to);
        if (sel > 0) {
            for (int s = 0; s < maxfd; s++) {
                if (FD_ISSET(s, &rread) && s == sock) {
                    int addrlen = sizeof(SOCKADDR);
                    SOCKADDR_IN address = {0};
                    SOCKET tClient = accept(sock, (SOCKADDR*)&address, (socklen_t*)&addrlen);
                    if (tClient == SOCKET_ERROR) {
						handle_error("socket accept");
                        return;
                    }
                    maxfd = (maxfd>tClient ? maxfd : tClient);
                    pthread_mutex_lock(&mutex_cout);
					LOG("New client");
                    pthread_mutex_unlock(&mutex_cout);
                    pthread_mutex_lock(&mutex_client);
                    http_write(tClient, (char*)"HTTP/1.0 200 OK\r\n", 0);
                    http_write(tClient, (char*)
                        "Server: /2.2\r\n"
                        "Accept-Range: bytes\r\n"
                        "Connection: close\r\n"
                        "Max-Age: 0\r\n"
                        "Expires: 0\r\n"
                        "Cache-Control: no-cache, private\r\n"
                        "Pragma: no-cache\r\n"
                        "Content-Type: multipart/x-mixed-replace; boundary=mjpegbound\r\n"
                        "\r\n", 0);
                    client = tClient;
					pthread_mutex_unlock(&mutex_client);
					pthread_create(&thread_write, NULL, writer_helper, this);
                }
            }
        }
        usleep(10);
    }
}

void MJPEGServer::writer() {
	auto last_time = std::chrono::high_resolution_clock::now();
    while (this->isOpen()) {
        //pthread_t threads[NUM_CONNECTIONS];
        //int count = 0;

		auto current_time = std::chrono::high_resolution_clock::now();

		auto millis_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

		//Only write to the client every X milliseconds
		if(millis_diff < FRAME_UPDATE_DELAY) {
			usleep(10000);
			continue;
		}

        unsigned char *outbuf;
        pthread_mutex_lock(&mutex_writer);
		outbuf = (unsigned char *) malloc(frameSize);
		memcpy(outbuf, lastFrame, frameSize);
        pthread_mutex_unlock(&mutex_writer);

		/*int outlen = outbuf.size();

        pthread_mutex_lock(&mutex_client);
        std::vector<int>::iterator begin = clients.begin();
        std::vector<int>::iterator end = clients.end();
        pthread_mutex_unlock(&mutex_client);

		payload = clientPayload

        std::vector<clientPayload*> payloads;
        for (std::vector<int>::iterator it = begin; it != end; ++it, ++count) {
            if (count > NUM_CONNECTIONS)
                break;
            struct clientPayload *cp = new clientPayload({ (MJPEGServer*)this, { outbuf.data(), outlen, *it } });
            payloads.push_back(cp);
            pthread_create(&threads[count], NULL, client_helper, cp);
        }
        for (; count > 0; count--) {
            pthread_join(threads[count-1], NULL);
            delete payloads.at(count-1);
        }
        usleep(40000);*/

		//client_write_frame(outbuf);
		int bufLen = frameSize + 63 + static_cast<int>(floor(log10(frameSize)));
		char payload[bufLen];
		sprintf(payload, "--mjpegbound\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n",
						frameSize);
		http_write(client, payload, bufLen);
		http_write(client, (char *) outbuf, frameSize);
		//, (char *) outbuf);
		free(outbuf);
		
		last_time = current_time; //Update the last time struct
    }
}

int MJPEGServer::http_write(int tSock, char *s, int len) {
	if (len < 1) { 
		len = strlen(s); 
	}

	if(::write(tSock, s, len) < 0) {
		fail_c += 100;
		if(fail_c > 1000) {
			LOG("Client disconnect!");	
			client = INVALID_SOCKET;
			pthread_exit(NULL);
		}
		//usleep(40000);
	} else {
		fail_c -= 1;
	}

	/*try {
		int retval = ::send(sock, s, len, MSG_NOSIGNAL);
		if
		return retval;
	} catch (int e) {
		handle_error("send");
	}*/
	return 0;
}

bool MJPEGServer::release() {
	if (sock != INVALID_SOCKET) {
		shutdown(sock, 2);
	}
	sock = (INVALID_SOCKET);
	return false;
}

bool MJPEGServer::open(int port) {
	sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	SOCKADDR_IN address;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_family = AF_INET;
	address.sin_port = htons(port);
	
	if (bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR) {
		handle_error("bind");
		return release();
	}

	if (listen(sock, NUM_CONNECTIONS) == SOCKET_ERROR) {
		handle_error("listen");
		return release();
	}

	FD_SET(sock, &master);
	return true;
}

bool MJPEGServer::isOpen() {
	return sock != INVALID_SOCKET;
}

void MJPEGServer::start() {
	pthread_create(&thread_listen, NULL, listen_helper, this);
}

void MJPEGServer::stop() {
	this->release();
	pthread_join(thread_listen, NULL);
	pthread_join(thread_write, NULL);
}

void MJPEGServer::write(unsigned char *frame, size_t size){
	pthread_mutex_lock(&mutex_writer);
	free(lastFrame);
	lastFrame = (unsigned char*) malloc(size);
	memcpy(lastFrame, frame, size);
	frameSize = size;
	pthread_mutex_unlock(&mutex_writer);
}
