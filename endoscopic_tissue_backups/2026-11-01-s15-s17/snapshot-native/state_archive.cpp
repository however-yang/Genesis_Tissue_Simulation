#include "state_archive.h"
#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace lcs_state {
namespace {
constexpr uint64_t Magic=0x314554415453434cull; // LCSTATE1, little endian
constexpr uint32_t Abi=1;
constexpr uint64_t MaxState=uint64_t(4)*1024*1024*1024;
struct Allocation{uint64_t id=0,key=0;void* address=nullptr;size_t bytes=0;Layout layout;};
std::map<uint64_t,Allocation> allocations;
std::vector<Layout> layouts;
uint64_t nextId=1;
bool atBoundary=false;
thread_local std::string errorText;

struct Header{uint64_t magic,total,checksum,next;uint32_t abi,allocations,fields,reserved;};
struct AllocationHeader{uint64_t id,key,bytes,stride;uint32_t pointers,reserved;};
struct FieldHeader{uint64_t name,bytes,stride;uint32_t pointers,itemBytes;};
static_assert(sizeof(Header)==48 && sizeof(AllocationHeader)==40 && sizeof(FieldHeader)==32,"State layout drift");

struct Buffer {
    uint8_t* write=nullptr;const uint8_t* read=nullptr;size_t capacity=0,offset=0;
    void bounds(size_t n){if(n>capacity-offset)throw std::runtime_error("Truncated state");}
    void put(const void* p,size_t n){bounds(n);if(n)std::memcpy(write+offset,p,n);offset+=n;}
    const uint8_t* take(size_t n){bounds(n);auto p=read+offset;offset+=n;return p;}
    template<class T> void put(const T& value){put(&value,sizeof(T));}
    template<class T> T get(){T value;std::memcpy(&value,take(sizeof(T)),sizeof(T));return value;}
};
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void valid_layout(const Layout& layout,size_t bytes){
    if(layout.pointers.empty()){require(layout.stride==0,"Unexpected pointer-free stride");return;}
    require(layout.stride>0 && bytes%layout.stride==0,"Invalid pointer layout stride");
    require(layout.pointers.size()<512,"Excessive pointer layout");
    uint32_t last=0;bool first=true;
    for(auto p:layout.pointers){
        require(p<=layout.stride && sizeof(uint64_t)<=layout.stride-p,"Pointer offset outside record");
        require(first || p>=last+sizeof(uint64_t),"Overlapping pointer slots");first=false;last=p;
    }
}
Fields fields(){
    Fields f;collect_recon(f);collect_physics(f);collect_cut(f);collect_haptics(f);collect_xpbd(f);
    for(const auto& x:f.values)valid_layout(x.layout,x.bytes);
    return f;
}
size_t total_size(const Fields& f){
    uint64_t n=sizeof(Header);
    for(const auto& pair:allocations){const auto& a=pair.second;n+=sizeof(AllocationHeader)+4*a.layout.pointers.size()+a.bytes;}
    for(const auto& x:f.values)n+=sizeof(FieldHeader)+4*x.layout.pointers.size()+x.bytes;
    require(n<=MaxState,"State exceeds format limit");return (size_t)n;
}
uint64_t encode_pointer(uint64_t ptr){
    if(!ptr)return 0;
    for(const auto& pair:allocations){const auto& a=pair.second;auto base=(uint64_t)a.address;
        if(ptr>=base && ptr-base<a.bytes){
            require(a.id<=UINT32_MAX && ptr-base<=UINT32_MAX,"Pointer token overflow");
            return (a.id<<32)|(ptr-base);
        }
    }
    throw std::runtime_error("A cross-step pointer is not in the allocation inventory");
}
uint64_t decode_pointer(uint64_t token,const std::map<uint64_t,Allocation>& table){
    if(!token)return 0;auto i=table.find(token>>32);
    require(i!=table.end(),"Pointer token names an absent allocation");
    uint64_t offset=token&UINT32_MAX;require(offset<i->second.bytes,"Pointer token exceeds allocation");
    return (uint64_t)i->second.address+offset;
}
template<class Fn> void pointers(uint8_t* data,size_t bytes,const Layout& layout,Fn convert){
    if(layout.pointers.empty())return;
    for(size_t base=0;base<bytes;base+=layout.stride)for(auto offset:layout.pointers){
        uint64_t p;std::memcpy(&p,data+base+offset,sizeof(p));p=convert(p);std::memcpy(data+base+offset,&p,sizeof(p));
    }
}
Layout get_layout(Buffer& b,uint64_t stride,uint32_t count,size_t bytes){
    require(count<512,"Too many pointer offsets");Layout l;l.stride=(size_t)stride;
    for(uint32_t i=0;i<count;i++)l.pointers.push_back(b.get<uint32_t>());
    valid_layout(l,bytes);return l;
}
struct SavedAllocation{Allocation meta;const uint8_t* data;};
struct SavedField{const Field* field;size_t bytes;const uint8_t* data;};
int failure(const std::exception& e){errorText=e.what();return -9001;}
}

void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
// Byte-count plus 64-bit words: detects accidental corruption without a slow
// bytewise hash of hundreds of MB. Python adds a cryptographic SHA-256 binding.
uint64_t digest(const void* data,size_t n){
    const auto* p=(const uint8_t*)data;uint64_t h=1469598103934665603ull^(uint64_t)n;
    size_t i=0;for(;i+8<=n;i+=8){uint64_t w;std::memcpy(&w,p+i,8);h=(h^w)*1099511628211ull;}
    uint64_t tail=0;if(i<n)std::memcpy(&tail,p+i,n-i);return (h^tail)*1099511628211ull;
}
cudaError_t allocate(void** p,size_t n,const char* file,int line){
    cudaError_t e=::cudaMalloc(p,n);if(e!=cudaSuccess || !*p)return e;
    try{
        std::string key=std::string(file)+":"+std::to_string(line);
        auto id=nextId++;allocations.emplace(id,Allocation{id,digest(key.data(),key.size()),*p,n,{}});
    }catch(...){::cudaFree(*p);*p=nullptr;return cudaErrorMemoryAllocation;}
    return e;
}
cudaError_t release(void* p){
    if(!p)return cudaSuccess;
    auto i=std::find_if(allocations.begin(),allocations.end(),[p](const auto& x){return x.second.address==p;});
    cudaError_t e=::cudaFree(p);if(e==cudaSuccess && i!=allocations.end())allocations.erase(i);
    if(allocations.empty()){nextId=1;atBoundary=false;}return e;
}
void known_layout(const Layout& layout){
    if(std::find(layouts.begin(),layouts.end(),layout)==layouts.end())layouts.push_back(layout);
}
void allocation_layout(const void* p,const Layout& layout){
    known_layout(layout);if(!p)return;
    auto i=std::find_if(allocations.begin(),allocations.end(),[p](const auto& x){return x.second.address==p;});
    require(i!=allocations.end(),"Pointer-bearing device object not registered");
    valid_layout(layout,i->second.bytes);i->second.layout=layout;
}
void boundary(bool ready){atBoundary=ready;}
const char* last_error(){return errorText.c_str();}

int state_size(size_t* out){
    try{require(out && atBoundary,"Snapshot requires a completed Finalize");auto f=fields();*out=total_size(f);return 0;}
    catch(const std::exception& e){return failure(e);}
}
int save_state(void* output,size_t capacity){
    try{
        require(output && atBoundary,"Snapshot requires a completed Finalize");check(cudaDeviceSynchronize());
        auto f=fields();size_t n=total_size(f);require(capacity>=n,"Snapshot buffer too small");
        Buffer b{(uint8_t*)output,nullptr,n,0};
        Header h{Magic,n,0,nextId,Abi,(uint32_t)allocations.size(),(uint32_t)f.values.size(),0};b.put(h);
        for(const auto& pair:allocations){const auto& a=pair.second;
            b.put(AllocationHeader{a.id,a.key,a.bytes,a.layout.stride,(uint32_t)a.layout.pointers.size(),0});
            for(auto off:a.layout.pointers)b.put(off);
            b.bounds(a.bytes);auto* data=b.write+b.offset;check(cudaMemcpy(data,a.address,a.bytes,cudaMemcpyDeviceToHost));
            try{pointers(data,a.bytes,a.layout,encode_pointer);}
            catch(const std::exception& e){throw std::runtime_error("Device allocation "+std::to_string(a.id)+": "+e.what());}
            b.offset+=a.bytes;
        }
        for(const auto& x:f.values){
            b.put(FieldHeader{digest(x.name.data(),x.name.size()),x.bytes,x.layout.stride,(uint32_t)x.layout.pointers.size(),(uint32_t)x.itemBytes});
            for(auto off:x.layout.pointers)b.put(off);
            b.bounds(x.bytes);auto* data=b.write+b.offset;x.read(data);
            try{pointers(data,x.bytes,x.layout,encode_pointer);}
            catch(const std::exception& e){throw std::runtime_error(x.name+": "+e.what());}
            b.offset+=x.bytes;
        }
        require(b.offset==n,"State size changed while saving");h.checksum=digest(b.write+sizeof(Header),n-sizeof(Header));
        std::memcpy(output,&h,sizeof(h));errorText.clear();return 0;
    }catch(const std::exception& e){return failure(e);}
}
int load_state(const void* input,size_t bytes){
    std::vector<void*> newMemory;
    try{
        require(input && atBoundary,"Restore requires a completed Finalize");
        require(bytes>=sizeof(Header) && bytes<=MaxState,"Invalid snapshot size");
        Buffer b{nullptr,(const uint8_t*)input,bytes,0};auto h=b.get<Header>();
        require(h.magic==Magic && h.abi==Abi && h.reserved==0 && h.total==bytes,"State ABI/header mismatch");
        require(h.allocations<4096 && h.fields<4096 && h.next<=UINT32_MAX,"Invalid snapshot table sizes");
        require(digest(b.read+sizeof(Header),bytes-sizeof(Header))==h.checksum,"Snapshot checksum mismatch");
        auto f=fields();require(h.fields==f.values.size(),"Host state inventory mismatch");
        std::vector<SavedAllocation> saved;std::map<uint64_t,Allocation> table;
        for(uint32_t i=0;i<h.allocations;i++){
            auto a=b.get<AllocationHeader>();
            require(a.id>0 && a.id<h.next && a.bytes>0 && a.bytes<=UINT32_MAX && a.reserved==0,"Invalid allocation record");
            Layout l=get_layout(b,a.stride,a.pointers,(size_t)a.bytes);
            require(std::find(layouts.begin(),layouts.end(),l)!=layouts.end(),"Unknown device pointer layout");
            Allocation meta{a.id,a.key,nullptr,(size_t)a.bytes,l};
            require(table.emplace(a.id,meta).second,"Duplicate allocation ID");saved.push_back({meta,b.take((size_t)a.bytes)});
        }
        std::vector<SavedField> host;
        for(const auto& x:f.values){
            auto a=b.get<FieldHeader>();
            require(a.name==digest(x.name.data(),x.name.size()) && a.itemBytes==x.itemBytes,"Host state field mismatch");
            require(a.bytes<=MaxState,"Host field too large");
            require(x.itemBytes?(a.bytes%x.itemBytes==0 && a.bytes<=256ull*1024*1024):(a.bytes==x.bytes),"Host field extent mismatch");
            auto l=get_layout(b,a.stride,a.pointers,(size_t)a.bytes);require(l==x.layout,"Host pointer layout mismatch");
            host.push_back({&x,(size_t)a.bytes,b.take((size_t)a.bytes)});
        }
        require(b.offset==bytes,"Trailing snapshot data");
        // Validate every normalized pointer before touching existing state.
        auto validate=[&table](uint64_t token){decode_pointer(token,table);return token;};
        std::vector<uint8_t> scratch;
        for(const auto& a:saved)if(!a.meta.layout.pointers.empty()){
            scratch.assign(a.data,a.data+a.meta.bytes);pointers(scratch.data(),scratch.size(),a.meta.layout,validate);
        }
        for(const auto& x:host)if(!x.field->layout.pointers.empty()){
            scratch.assign(x.data,x.data+x.bytes);pointers(scratch.data(),scratch.size(),x.field->layout,validate);
        }
        check(cudaDeviceSynchronize());
        for(auto& pair:table){auto& a=pair.second;auto old=allocations.find(a.id);
            if(old!=allocations.end() && old->second.bytes==a.bytes && old->second.key==a.key)a.address=old->second.address;
            else{check(::cudaMalloc(&a.address,a.bytes));newMemory.push_back(a.address);}
        }
        // From this point any CUDA failure invalidates the session; the Python
        // adapter closes it. Malformed bytes have already been rejected above.
        invalidate_xpbd_graphs();
        auto decode=[&table](uint64_t token){return decode_pointer(token,table);};
        for(const auto& a:saved){const void* source=a.data;
            if(!a.meta.layout.pointers.empty()){
                scratch.assign(a.data,a.data+a.meta.bytes);pointers(scratch.data(),scratch.size(),a.meta.layout,decode);source=scratch.data();
            }
            check(cudaMemcpy(table.at(a.meta.id).address,source,a.meta.bytes,cudaMemcpyHostToDevice));
        }
        for(const auto& x:host){const void* source=x.data;
            if(!x.field->layout.pointers.empty()){
                scratch.assign(x.data,x.data+x.bytes);pointers(scratch.data(),scratch.size(),x.field->layout,decode);source=scratch.data();
            }
            x.field->write(source,x.bytes);
        }
        for(const auto& old:allocations){auto keep=table.find(old.first);
            if(keep==table.end() || keep->second.address!=old.second.address)check(::cudaFree(old.second.address));
        }
        allocations=std::move(table);nextId=h.next;newMemory.clear();atBoundary=true;errorText.clear();return 0;
    }catch(const std::exception& e){for(auto p:newMemory)::cudaFree(p);return failure(e);}
}
int allocation_report(char* out,size_t capacity){
    try{auto f=fields();std::ostringstream s;s<<"{\"allocations\":[";bool first=true;
        for(const auto& pair:allocations){const auto& a=pair.second;if(!first)s<<",";first=false;
            s<<"{\"id\":"<<a.id<<",\"site_hash\":"<<a.key<<",\"bytes\":"<<a.bytes<<",\"pointer_slots\":"<<a.layout.pointers.size()<<"}";}
        s<<"],\"host_fields\":"<<f.values.size()<<",\"bytes\":"<<total_size(f)<<"}";auto text=s.str();
        require(out && capacity>text.size(),"Inventory report capacity too small");std::memcpy(out,text.c_str(),text.size()+1);return 0;
    }catch(const std::exception& e){return failure(e);}
}
}
