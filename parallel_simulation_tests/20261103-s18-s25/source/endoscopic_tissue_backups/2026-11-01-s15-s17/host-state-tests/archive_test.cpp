#include "state_archive.h"
#define CHECK(x) do { if(!(x)) throw std::runtime_error(std::string(#x)+": "+lcs_state::last_error()); } while(false)
#include <cstdlib>
#include <iostream>
#include <map>

namespace {
std::map<void*,size_t> memory;
int invalidFrees=0,failCopyAfter=-1;
bool failHostWrite=false;
int* payload=nullptr;
struct Holder{int* pointer;int size;};
Holder* holder=nullptr;
std::vector<int> hostHistory;
int hostTag=0;
}
cudaError_t cudaMalloc(void** p,size_t n){
    *p=std::malloc(n);if(!*p)return cudaErrorMemoryAllocation;
    memory[*p]=n;return cudaSuccess;
}
cudaError_t cudaFree(void* p){
    if(!p)return cudaSuccess;
    auto it=memory.find(p);
    if(it==memory.end()){++invalidFrees;return cudaErrorInvalidDevicePointer;}
    std::free(p);memory.erase(it);return cudaSuccess;
}
cudaError_t cudaMemcpy(void* a,const void* b,size_t n,cudaMemcpyKind){
    if(failCopyAfter==0){failCopyAfter=-1;return cudaErrorUnknown;}
    if(failCopyAfter>0)--failCopyAfter;
    if(n)std::memcpy(a,b,n);return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize(){return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "injected CUDA copy failure";}

namespace lcs_state {
void collect_recon(Fields&){}
void collect_physics(Fields&){}
void collect_cut(Fields&){}
void collect_haptics(Fields&){}
void invalidate_xpbd_graphs(){}
void collect_xpbd(Fields& f){
    allocation_layout(holder,Layout{sizeof(Holder),{offsetof(Holder,pointer)}});
    f.pod("payload",payload);f.pod("holder",holder);f.vector("history",hostHistory);
    f.pod("tag",hostTag);
    f.values.push_back({"fault",sizeof(hostTag),{},0,
        [](void* p){std::memcpy(p,&hostTag,sizeof(hostTag));},
        [](const void* p,size_t){
            if(failHostWrite)throw std::runtime_error("injected host restore failure");
            std::memcpy(&hostTag,p,sizeof(hostTag));
        }});
}
}

std::vector<unsigned char> save(){
    size_t n=0;CHECK(lcs_state::state_size(&n)==0);
    std::vector<unsigned char> bytes(n);CHECK(lcs_state::save_state(bytes.data(),bytes.size())==0);return bytes;
}
void init(){
    CHECK(memory.empty());hostHistory={3,4,5};hostTag=7;
    CHECK(lcs_state::allocate((void**)&payload,4*sizeof(int),"payload",1)==0);
    CHECK(lcs_state::allocate((void**)&holder,sizeof(Holder),"holder",2)==0);
    for(int i=0;i<4;i++)payload[i]=10+i;
    *holder={payload,4};lcs_state::boundary(true);
}
void replacePayload(){
    CHECK(lcs_state::release(payload)==0);
    CHECK(lcs_state::allocate((void**)&payload,8*sizeof(int),"payload",1)==0);
    for(int i=0;i<8;i++)payload[i]=20+i;
    *holder={payload,8};hostHistory={8,9};hostTag=11;
}
void closeSession(){
    // Same ownership boundary as module shutdown followed by archive cleanup.
    lcs_state::release(payload);payload=nullptr;
    lcs_state::release(holder);holder=nullptr;
#ifdef ARCHIVE_HAS_DISCARD
    lcs_state::discard_session();
#endif
    lcs_state::boundary(false);
}
int runTests(){
    init();auto initial=save();
    payload[1]=99;hostHistory={1};hostTag=55;
    CHECK(lcs_state::load_state(initial.data(),initial.size())==0);
    CHECK(payload[1]==11 && holder->pointer==payload && hostTag==7);
    CHECK(hostHistory==std::vector<int>({3,4,5}));CHECK(save()==initial);
    auto bad=initial;bad.back()^=1;
    CHECK(lcs_state::load_state(bad.data(),bad.size())!=0 && save()==initial);
    replacePayload();CHECK(lcs_state::load_state(initial.data(),initial.size())==0);
    CHECK(holder->pointer==payload && holder->size==4 && save()==initial);
    closeSession();CHECK(memory.empty() && invalidFrees==0);

    init();initial=save();replacePayload();failCopyAfter=0;
    CHECK(lcs_state::load_state(initial.data(),initial.size())!=0);
    size_t invalidSize=0;CHECK(lcs_state::state_size(&invalidSize)!=0);
    closeSession();CHECK(memory.empty() && invalidFrees==0);

    init();initial=save();replacePayload();failHostWrite=true;
    CHECK(lcs_state::load_state(initial.data(),initial.size())!=0);
    CHECK(lcs_state::state_size(&invalidSize)!=0);
    failHostWrite=false;closeSession();
    std::cout<<"{\"remaining_allocations\":"<<memory.size()<<",\"invalid_free_calls\":"<<invalidFrees<<"}"<<std::endl;
    if(!memory.empty() || invalidFrees)return 2;
    std::cout<<"CPU archive ownership, corruption, pointer relocation and failure cleanup checks passed"<<std::endl;
    return 0;
}

int main(){try{return runTests();}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
