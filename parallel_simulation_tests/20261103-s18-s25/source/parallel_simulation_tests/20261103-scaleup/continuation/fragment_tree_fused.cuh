// Isolated exact-order optimization proposal. This consumes already-computed
// leaves. Every parent uses the identical two children and float operations.
// Only the top levels are fused; __syncthreads supplies the required dependency.
__global__ void k_FragmentTreeTop(int first,FragmentTreeNode* tree,const int* queryCount){
    if(!*queryCount)return; // uniform across this single block
    for(int level=first;level;level/=2){
        for(int index=threadIdx.x;index<level;index+=blockDim.x){
            int i=level+index;FragmentTreeNode a=tree[2*i],b=tree[2*i+1],c;
            c.low=fragmentMin3(a.low,b.low);c.high=fragmentMax3(a.high,b.high);
            c.currentLow=fragmentMin3(a.currentLow,b.currentLow);c.currentHigh=fragmentMax3(a.currentHigh,b.currentHigh);
            c.areaVector=a.areaVector+b.areaVector;c.area=a.area+b.area;c.root=fragmentMergeRoot(a.root,b.root);tree[i]=c;
        }
        __syncthreads();
    }
}
