#pragma once
// Host-only serialization for the isolated snapshot DLL. No physical kernels.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace lcs_state {
struct Layout {
    size_t stride=0;
    std::vector<uint32_t> pointers;
    bool operator==(const Layout& x)const{return stride==x.stride && pointers==x.pointers;}
};
struct Field {
    std::string name;
    size_t bytes=0;
    Layout layout;
    // A vector permits a different length when restoring another state.
    size_t itemBytes=0;
    std::function<void(void*)> read;
    std::function<void(const void*,size_t)> write;
};
uint64_t digest(const void*,size_t);
void check(cudaError_t);
cudaError_t allocate(void**,size_t,const char*,int);
cudaError_t release(void*);
void allocation_layout(const void*,const Layout&);
void known_layout(const Layout&);
void boundary(bool);
int state_size(size_t*);
int save_state(void*,size_t);
int load_state(const void*,size_t);
const char* last_error();
int allocation_report(char*,size_t);

template<class T> cudaError_t allocate_typed(T** p,size_t n,const char* file,int line){
    return allocate(reinterpret_cast<void**>(p),n,file,line);
}
template<class T> Layout raw_layout(){
    static_assert(std::is_trivially_copyable<T>::value,"Explicit codec required for non-POD state");
    return Layout{};
}

class Fields {
public:
    std::vector<Field> values;
    template<class T> void pod(const std::string& name,T& value,Layout layout={}){
        static_assert(std::is_trivially_copyable<T>::value,"Explicit codec required");
        if constexpr(std::is_pointer<T>::value){layout={sizeof(T),{0}};}
        known_layout(layout);
        T* address=&value;
        values.push_back({name,sizeof(T),layout,0,
            [address](void* out){std::memcpy(out,address,sizeof(T));},
            [address](const void* in,size_t bytes){
                if(bytes!=sizeof(T))throw std::runtime_error("Host field size mismatch");
                std::memcpy(address,in,bytes);
            }});
    }
    template<class T> void vector(const std::string& name,std::vector<T>& value){
        static_assert(std::is_trivially_copyable<T>::value && !std::is_pointer<T>::value,"Explicit vector codec required");
        auto* address=&value;
        values.push_back({name,value.size()*sizeof(T),{},sizeof(T),
            [address](void* out){if(!address->empty())std::memcpy(out,address->data(),address->size()*sizeof(T));},
            [address](const void* in,size_t bytes){
                if(bytes%sizeof(T))throw std::runtime_error("Invalid vector extent");
                address->resize(bytes/sizeof(T));if(bytes)std::memcpy(address->data(),in,bytes);
            }});
    }
    void symbol(const std::string& name,size_t bytes,Layout layout,
                std::function<void(void*)> read,std::function<void(const void*,size_t)> write){
        known_layout(layout);values.push_back({name,bytes,layout,0,read,write});
    }
};
void collect_recon(Fields&);
void collect_physics(Fields&);
void collect_cut(Fields&);
void collect_haptics(Fields&);
void collect_xpbd(Fields&);
void invalidate_xpbd_graphs();
}

// Only generated native translation units opt in to allocation accounting.
// Including CUDA/standard headers before these macros avoids intercepting the
// runtime's own implementation or allocations internal to CUDA graphs.
#ifdef LCS_STATE_TRACK_ALLOCATIONS
#define cudaMalloc(p,n) ::lcs_state::allocate_typed((p),(n),__FILE__,__LINE__)
#define cudaFree(p) ::lcs_state::release((void*)(p))
#endif
