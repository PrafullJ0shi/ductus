"""Zoomed sheet of 20 r-words: RNN r (left) against the r redrawn plain (right).
  python scripts/r_sheet.py out.png"""
import cv2, numpy as np, sys
from ductus.engines.rnn import RNNEngine, render_line
L=["range variance root real","from are first process","error mirror rare river","square average before after","for or per word"]
e=RNNEngine(style=1,bias=1.5)
from PIL import Image, ImageDraw, ImageFont
F=ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf",20)
rows=[]
for i,t in enumerate(L):
    pair=[]
    for on in (False,True):
        e.plain_r=on
        s,_=e.strokes(t,seed=40+i); im=render_line(s,len(t),pen_px=2.6)
        ys,xs=np.where(im<160); im=im[max(0,ys.min()-8):ys.max()+8, max(0,xs.min()-8):xs.max()+8]
        im=cv2.resize(im,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
        pair.append(im)
    h=max(p.shape[0] for p in pair); pair=[cv2.copyMakeBorder(p,0,h-p.shape[0],0,0,cv2.BORDER_CONSTANT,value=255) for p in pair]
    pair=[cv2.copyMakeBorder(p,0,0,0,max(0,1250-p.shape[1]),cv2.BORDER_CONSTANT,value=255)[:, :1250] for p in pair]
    sep=np.full((h,30),255,np.uint8); sep[:,14:16]=180
    rows.append(np.hstack([pair[0],sep,pair[1]])); rows.append(np.full((14,rows[-1].shape[1]),255,np.uint8))
e.close()
body=np.vstack(rows); head=Image.new("L",(body.shape[1],40),255); d=ImageDraw.Draw(head)
d.text((8,8),"before: RNN r (style 1, bias 1.5)",fill=60,font=F); d.text((1290,8),"after: r redrawn plain (stem + one shoulder)",fill=60,font=F)
cv2.imwrite(sys.argv[1], np.vstack([np.array(head),body]))
