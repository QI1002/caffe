#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace std;

int main(int argc, char** argv)
{
    const char log_dir[] = "/home/qi/github/caffe/examples/notebook/log";

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    //not use it to avoid disabling log to file
    //FLAGS_alsologtostderr = true
    //FLAGS_logtostderr = true

    //it's ok to set env like as follows:
    //GLOG_log_dir=./log ./glog_test
    //./glog_test --log_dir=./log
    FLAGS_log_dir = log_dir;
    
    //google::SetLogDestination(google::GLOG_INFO, log_dir);
    cout << FLAGS_log_dir << "\n";

    LOG(INFO) << "This is INFO";
    LOG(WARNING) << "This is WARNING";
    LOG(ERROR) << "This is ERROR";

    return 0;
}
