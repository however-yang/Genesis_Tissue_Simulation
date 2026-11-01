import sys,json,pickle,struct,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'))
from replay_archive import open_replay
import cv2
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES
@threadpool_limits.wrap(limits=1,user_api='blas')
def review(folder):
 report=json.loads((folder/'report.json').read_text());wanted={s['end']['step'] for s in report['stages']}
 wanted.update(range(210,421,30))
 if report['steps']>1000:wanted.update(range(780,901,15))
 output=folder/'keyframe-review';output.mkdir(exist_ok=True);thumbs=[]
 with open_replay(folder/'replay.bin') as stream:
  settings,scene_name,_=pickle.load(stream);scene=SCISSORS_SCENES[scene_name];view=EndoscopicView(settings,1280,960,supersample=1,kidney=scene.shape,camera=scene.shot)
  try:
   index=0
   while prefix:=stream.read(4):
    packed=stream.read(struct.unpack('<I',prefix)[0])
    if index in wanted:
     step,stage,surface,obs=pickle.loads(zlib.decompress(packed));assert step==index
     view.update(surface,obs);frame=view.render()[0];cv2.imwrite(str(output/(str(step)+'.png')),frame)
     tile=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,480));ImageDraw.Draw(tile).rectangle((0,0,640,25),fill='black');ImageDraw.Draw(tile).text((8,5),str(step)+' '+stage,fill='white');thumbs.append(tile)
    index+=1
  finally:view.close()
 for start in range(0,len(thumbs),6):
  sheet=Image.new('RGB',(1280,1440),'black')
  for i,tile in enumerate(thumbs[start:start+6]):sheet.paste(tile,((i%2)*640,(i//2)*480))
  sheet.save(output/('sheet-'+str(start//6+1)+'.jpg'),quality=93)
 print(folder.name,'reviewed states saved',len(thumbs),flush=True)
if __name__=='__main__':
 for p in map(Path,sys.argv[1:]):review(p)
