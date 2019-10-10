#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

int getdir (std::string dir, std::vector<std::string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cerr << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

bool isjpeg(const std::string filename) {
    if (filename.size() < 5)
        return false;
    
    std::string ext_jpg_3 = filename.substr(filename.size() - 4, filename.size());
    std::string ext_jpg_4 = filename.substr(filename.size() - 5, filename.size());
    
    std::transform(ext_jpg_3.begin(), ext_jpg_3.end(), ext_jpg_3.begin(), ::tolower);
    std::transform(ext_jpg_4.begin(), ext_jpg_4.end(), ext_jpg_4.begin(), ::tolower);
    
    return ext_jpg_3 == ".jpg" || ext_jpg_4 == ".jpeg";
}
