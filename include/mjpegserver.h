#include <pthread.h>
#include <vector>

struct clientFrame {
    unsigned char* outbuf;
    int outlen;
    int client;
};

class MJPEGServer{
public:
	MJPEGServer(int);
	~MJPEGServer();
	bool release();
	bool open(int);
	bool isOpen();
	void start();
	void stop();
	void write(unsigned char*, size_t);
    void listener();
    void writer();
	int http_write(int, char*, int);
private:
	int sock;
    fd_set master;
    int client;
    int timeout;
	int fail_c;
	unsigned char *lastFrame;
	size_t frameSize;
    pthread_t thread_listen, thread_write;
    pthread_mutex_t mutex_client = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex_cout = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex_writer = PTHREAD_MUTEX_INITIALIZER;
};
