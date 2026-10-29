"""Halve the union-find paths while the component labels are joined.

Usage: python patch_union_find.py  (from endoscopic_tissue/native/vendor/cuda)

Every union links the larger root under the smaller one, so the final label of a component is its smallest node index
whatever the order and the paths taken: the labels are those of the walk without halving.
"""


def patch(path, edits):
    text = open(path, encoding="utf-8", newline="").read()
    crlf = "\r\n" in text
    for old, new in edits:
        if crlf:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        assert text.count(old) == 1, (path, old[:80], text.count(old))
        text = text.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(text)
    print("ok", path)


patch("solver/material/unified_vertex.cuh", [
("""__device__ inline int unifiedRoot(const int* parent,int i){while(parent[i]!=i)i=parent[i];return i;}""",
"""__device__ inline int unifiedRoot(const int* parent,int i){while(parent[i]!=i)i=parent[i];return i;}
// unifiedRoot that also points each node on the path at its grandparent. A parent only ever moves to an ancestor of
// smaller index, so the concurrent unions of k_JoinComponents still see a valid forest, and later walks are short.
// k_FlattenComponents keeps unifiedRoot: a halving there could replace a node's final root by an older ancestor.
__device__ inline int unifiedRootHalving(int* parent,int i){
    for(int next=parent[i];next!=i;next=parent[i]){int grand=parent[next];if(grand!=next)parent[i]=grand;i=grand;}
    return i;
}"""),
("""        while(true){int x=unifiedRoot(parent,i),y=unifiedRoot(parent,j);if(x==y)break;int hi=max(x,y),lo=min(x,y);if(atomicCAS(parent+hi,hi,lo)==hi)break;}}""",
"""        while(true){int x=unifiedRootHalving(parent,i),y=unifiedRootHalving(parent,j);if(x==y)break;int hi=max(x,y),lo=min(x,y);if(atomicCAS(parent+hi,hi,lo)==hi)break;}}"""),
])
