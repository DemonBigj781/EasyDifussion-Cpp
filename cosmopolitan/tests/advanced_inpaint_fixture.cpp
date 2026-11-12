// Production image utilities only; no model execution or GPU discovery.
#include "image_utils.h"
#include <fstream>
#include <iostream>
#include <iterator>
extern "C" const char* sd_get_last_error() { return ""; }
std::string read(const char* p) { std::ifstream f(p);return std::string(std::istreambuf_iterator<char>(f),{}); }
int main(int argc,char**argv) {
 if(argc==2 && std::string(argv[1])=="--alias") {
   sd_image_t aliased{};
   try { prepareAdvancedInpaint("", "", "", 16,16,.5,0,false,aliased,aliased); }
   catch(const std::invalid_argument&) { return 0; }
   return 3;
 }
 if(argc!=10)return 2;
 sd_image_t image{},mask{};
 try { prepareAdvancedInpaint(read(argv[1]),read(argv[2]),read(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stof(argv[6]),std::stof(argv[7]),std::stoi(argv[8]),image,mask);
 std::ofstream out(std::string(argv[9])+".rgb",std::ios::binary);out.write((char*)image.data,image.width*image.height*image.channel);
 std::ofstream mout(std::string(argv[9])+".mask",std::ios::binary);mout.write((char*)mask.data,mask.width*mask.height);
 freeImage(image);freeImage(mask);return 0;
 }catch(const std::exception&e){
 // Failure must never publish a partial result to caller-owned resources.
 if(image.data || mask.data) { freeImage(image);freeImage(mask);return 3; }
 std::cerr<<e.what()<<"\n";return 1;}
}
