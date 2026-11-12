// Production mask decode/inversion only; no model, provider or GPU execution.
#include "image_utils.h"
#include <fstream>
#include <iterator>
#include <iostream>
extern "C" const char* sd_get_last_error() { return ""; }
int main(int argc,char** argv) {
 if(argc!=4)return 2;
 try {
  std::ifstream in(argv[1]);std::string data(std::istreambuf_iterator<char>(in),{});
  sd_image_t mask=prepareMaskImage(data,16,16,0,std::stoi(argv[2]));
  if(!mask.data || mask.width!=16 || mask.height!=16 || mask.channel!=1){freeImage(mask);return 3;}
  std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<char*>(mask.data),256);freeImage(mask);return out?0:4;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
