"""Let an envelope link in a column's own air gap set the column's scissors cut level where an edge sweeps it."""
from pathlib import Path

cut = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda\cut.cu")
s = cut.read_text(encoding="utf8")


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


s = replace_once(s, """// A material link along the column axis, with a node inside the material,
// sets its column's level once its present segment crosses the present shear
// plane at a point an edge has swept, unless the column already has one.
__global__ void k_LevelScissorsColumns(""", """// Whether the column of corner c holds material on both sides of its link from
// c to the next corner along the axis: the link then lies in an air gap of the
// column, such as the one under an overhang, which the lattice fills with
// envelope tets wherever it is narrower than about two cells.
__device__ inline bool ColumnGap(int3 c,int axis,int3 dims,const int* inside){
    int3 dir=make_int3(axis==0,axis==1,axis==2);int top=axis==0?dims.x:axis==1?dims.y:dims.z;
    int k=axis==0?c.x:axis==1?c.y:c.z;bool below=false,above=false;
    int3 p=c;for(int j=k-1;j>=0 && !below;j--){p=p-dir;below=inside[CorIdCuda(p,dims)]!=0;}
    p=c+dir;for(int j=k+2;j<=top && !above;j++){p=p+dir;above=inside[CorIdCuda(p,dims)]!=0;}
    return below && above;
}
// A link along the column axis sets its column's level once its present segment
// crosses the present shear plane at a point an edge has swept, unless the
// column already has one. The link is a material link, with a node inside the
// material, or an envelope link in an air gap of its column (ColumnGap): the
// envelope tets bridging the gap tie the material above to the material below,
// and would hold the parts of a severed column together.
__global__ void k_LevelScissorsColumns(""")
s = replace_once(s, """    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);
    if(nbrIdx[6*a+2*axis+1]<0 || (!inside[a] && !inside[b]))return;
    float da=dot3(pos[a]-plane[1],plane[0]),db=dot3(pos[b]-plane[1],plane[0]);""",
"""    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);
    if(nbrIdx[6*a+2*axis+1]<0)return;
    float da=dot3(pos[a]-plane[1],plane[0]),db=dot3(pos[b]-plane[1],plane[0]);""")
s = replace_once(s, """    if(!scissorsSwept(x,y,frame[4],frame[5],1.f,now,open,reach) && !scissorsSwept(x,y,frame[6],frame[7],-1.f,now,open,reach))return;
    float ya=AlongAxis(rest[a],axis);""", """    if(!scissorsSwept(x,y,frame[4],frame[5],1.f,now,open,reach) && !scissorsSwept(x,y,frame[6],frame[7],-1.f,now,open,reach))return;
    if(!inside[a] && !inside[b] && !ColumnGap(bc,axis,dims,inside))return;
    float ya=AlongAxis(rest[a],axis);""")
cut.write_text(s, encoding="utf8")
print("gap rule written")
