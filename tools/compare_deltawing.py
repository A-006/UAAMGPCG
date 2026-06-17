import glob,os,numpy as np,matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def ea(vol,axis,vmax,lo=0.25,g=1.0,k=6):
    d=np.clip((vol/vmax-lo)/(1-lo),0,1)**g; d=np.moveaxis(d,axis,0)
    al=1-np.exp(-k*d); T=np.cumprod(1-al+1e-6,0); T=np.concatenate([np.ones_like(T[:1]),T[:-1]],0)
    return (lambda im:im/(im.max()+1e-9))(np.sum(al*T,0))
def our(p):
    L=open(p).readlines(); dims=None
    for i,l in enumerate(L):
        t=l.split()
        if t and t[0]=="DIMENSIONS": dims=tuple(int(x) for x in t[1:4])
        if t and t[0]=="SCALARS" and len(t)>1 and t[1]=="vorticity_magnitude":
            n=dims[0]*dims[1]*dims[2]; v=np.array(L[i+2:i+2+n],dtype=float).reshape((dims[2],dims[1],dims[0])); return np.transpose(v,(2,1,0))
def auth(fr):
    d="/tmp/lfm_headless/vel"; vx=np.load(f"{d}/vx_{fr:04d}.npy"); vy=np.load(f"{d}/vy_{fr:04d}.npy"); vz=np.load(f"{d}/vz_{fr:04d}.npy")
    wx=np.gradient(vz,axis=1)-np.gradient(vy,axis=2); wy=np.gradient(vx,axis=2)-np.gradient(vz,axis=0); wz=np.gradient(vy,axis=0)-np.gradient(vx,axis=1)
    return np.sqrt(wx*wx+wy*wy+wz*wz)
ofs=sorted(glob.glob("output_dw_sdf/frame_*.vtk")); afs=sorted(glob.glob("/tmp/lfm_headless/vel/vx_*.npy"))
print("our",len(ofs),"author",len(afs))
ov=our(ofs[-1]); af=int(os.path.basename(afs[-1]).split('_')[1].split('.')[0]); av=auth(af)
print("our max|w|=%.1f  author max|w|(curl)=%.3f"%(ov.max(),av.max()))
fig,ax=plt.subplots(2,2,figsize=(12,9),facecolor='k')
for c,(vol,nm) in enumerate([(ov,"OURS"),(av,"AUTHOR")]):
    vmax=float(np.percentile(vol,99.5))
    ax[0,c].imshow(ea(vol,1,vmax).T,origin="lower",cmap="inferno",aspect="auto"); ax[0,c].set_title(f"{nm} top(down y)",color='w')
    ax[1,c].imshow(vol[int(0.75*vol.shape[0])].T,origin="lower",cmap="inferno"); ax[1,c].set_title(f"{nm} y-z wake",color='w')
    for r in (0,1): ax[r,c].set_xticks([]);ax[r,c].set_yticks([])
plt.suptitle("Delta wing: OURS vs AUTHOR (same SDF geometry) — |omega|",color='w')
plt.tight_layout(); plt.savefig("/tmp/dw_compare.png",dpi=72,facecolor='k'); print("wrote /tmp/dw_compare.png")
