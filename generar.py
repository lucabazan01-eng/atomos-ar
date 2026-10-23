"""Átomos AR: franjas de color por nivel de energía con la forma de sus orbitales + electrones como puntos blancos.
Genera GLB (Android), USDZ (iPhone), marcadores impresos y patrones AR.js.
Requiere: pip install numpy pillow
Uso:  python generar.py          (prototipo: H, Fe, Au)
      python generar.py todos    (los 118)"""
import numpy as np, struct, json, zipfile, os
RAW="H Hidrógeno|He Helio|Li Litio|Be Berilio|B Boro|C Carbono|N Nitrógeno|O Oxígeno|F Flúor|Ne Neón|Na Sodio|Mg Magnesio|Al Aluminio|Si Silicio|P Fósforo|S Azufre|Cl Cloro|Ar Argón|K Potasio|Ca Calcio|Sc Escandio|Ti Titanio|V Vanadio|Cr Cromo|Mn Manganeso|Fe Hierro|Co Cobalto|Ni Níquel|Cu Cobre|Zn Zinc|Ga Galio|Ge Germanio|As Arsénico|Se Selenio|Br Bromo|Kr Kriptón|Rb Rubidio|Sr Estroncio|Y Itrio|Zr Circonio|Nb Niobio|Mo Molibdeno|Tc Tecnecio|Ru Rutenio|Rh Rodio|Pd Paladio|Ag Plata|Cd Cadmio|In Indio|Sn Estaño|Sb Antimonio|Te Telurio|I Yodo|Xe Xenón|Cs Cesio|Ba Bario|La Lantano|Ce Cerio|Pr Praseodimio|Nd Neodimio|Pm Prometio|Sm Samario|Eu Europio|Gd Gadolinio|Tb Terbio|Dy Disprosio|Ho Holmio|Er Erbio|Tm Tulio|Yb Iterbio|Lu Lutecio|Hf Hafnio|Ta Tántalo|W Wolframio|Re Renio|Os Osmio|Ir Iridio|Pt Platino|Au Oro|Hg Mercurio|Tl Talio|Pb Plomo|Bi Bismuto|Po Polonio|At Astato|Rn Radón|Fr Francio|Ra Radio|Ac Actinio|Th Torio|Pa Protactinio|U Uranio|Np Neptunio|Pu Plutonio|Am Americio|Cm Curio|Bk Berkelio|Cf Californio|Es Einstenio|Fm Fermio|Md Mendelevio|No Nobelio|Lr Lawrencio|Rf Rutherfordio|Db Dubnio|Sg Seaborgio|Bh Bohrio|Hs Hasio|Mt Meitnerio|Ds Darmstatio|Rg Roentgenio|Cn Copernicio|Nh Nihonio|Fl Flerovio|Mc Moscovio|Lv Livermorio|Ts Teneso|Og Oganesón"
EL=[t.split(" ",1) for t in RAW.split("|")]
ORDER="1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p".split()
EX={24:{"4s":1,"3d":5},29:{"4s":1,"3d":10},41:{"5s":1,"4d":4},42:{"5s":1,"4d":5},44:{"5s":1,"4d":7},45:{"5s":1,"4d":8},46:{"5s":0,"4d":10},47:{"5s":1,"4d":10},57:{"4f":0,"5d":1},58:{"4f":1,"5d":1},64:{"4f":7,"5d":1},78:{"6s":1,"5d":9},79:{"6s":1,"5d":10},89:{"5f":0,"6d":1},90:{"5f":0,"6d":2},91:{"5f":2,"6d":1},92:{"5f":3,"6d":1},93:{"5f":4,"6d":1},96:{"5f":7,"6d":1},103:{"6d":0,"7p":1}}
def config(Z):
    c,left={},Z
    for k in ORDER:
        e=min(left,2*(2*"spdf".index(k[1])+1));c[k]=e;left-=e
    c.update(EX.get(Z,{}))
    return sorted([(int(k[0]),"spdf".index(k[1]),e) for k,e in c.items() if e>0])
AF=[[lambda x,y,z:np.ones_like(x)],
 [lambda x,y,z:x,lambda x,y,z:y,lambda x,y,z:z],
 [lambda x,y,z:x*y,lambda x,y,z:y*z,lambda x,y,z:3*z*z-1,lambda x,y,z:x*z,lambda x,y,z:x*x-y*y],
 [lambda x,y,z:z*(5*z*z-3),lambda x,y,z:x*(5*z*z-1),lambda x,y,z:y*(5*z*z-1),lambda x,y,z:z*(x*x-y*y),lambda x,y,z:x*y*z,lambda x,y,z:x*(x*x-3*y*y),lambda x,y,z:y*(3*x*x-y*y)]]
COLL={0:(.92,.25,.25),1:(.97,.85,.20),2:(.25,.72,.36),3:(.76,.42,.92)}   # s rojo, p amarillo, d verde, f morado
ALFA={0:.45,1:.55,2:.55,3:.55}
MASA=[int(x) for x in "1 4 7 9 11 12 14 16 19 20 23 24 27 28 31 32 35 40 39 40 45 48 51 52 55 56 59 58 63 64 69 74 75 80 79 84 85 88 89 90 93 98 98 102 103 106 107 114 115 120 121 130 127 132 133 138 139 140 141 142 145 152 153 158 159 164 165 166 169 174 175 180 181 184 187 192 193 195 197 202 205 208 209 209 210 222 223 226 227 232 231 238 237 244 243 247 247 251 252 257 258 259 266 267 268 269 270 277 278 281 282 285 286 289 290 293 294 294".split()]
assert len(MASA)==118
rng=np.random.default_rng(3)
_k=np.arange(900)+.5;_p=np.arccos(1-2*_k/900);_t=np.pi*(1+5**.5)*_k
FIB=np.stack([np.cos(_t)*np.sin(_p),np.sin(_t)*np.sin(_p),np.cos(_p)],1)
SUP="⁰¹²³⁴⁵⁶⁷⁸⁹"
def sup(n):return "".join(SUP[int(c)] for c in str(n))
def superficie(f,R,nt=36,npp=72):
    fm=np.abs(f(FIB[:,0],FIB[:,1],FIB[:,2])).max()
    T,P=np.meshgrid(np.linspace(0,np.pi,nt+1),np.linspace(0,2*np.pi,npp+1),indexing="ij")
    x,y,z=np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T)
    r=R*np.abs(f(x,y,z))/fm
    V=np.stack([x*r,y*r,z*r],-1).reshape(-1,3)
    i=np.arange((nt+1)*(npp+1)).reshape(nt+1,npp+1)
    a,b,c,d=i[:-1,:-1].ravel(),i[:-1,1:].ravel(),i[1:,:-1].ravel(),i[1:,1:].ravel()
    return V,np.vstack([np.stack([a,c,b],1),np.stack([b,d,c],1)])
def unir(l):
    Vs,Fs,o=[],[],0
    for V,F in l:Vs.append(V);Fs.append(F+o);o+=len(V)
    return np.vstack(Vs),np.vstack(Fs)
def normales(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for k in range(3):np.add.at(N,F[:,k],n)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
UNO=lambda x,y,z:np.ones_like(x)
RE=0.0055; SP=0.02; R_INI=0.04; ALFA_S=0.08
PALN=[(.55,.70,1.0),(.60,.90,.70),(1.0,.85,.50),(1.0,.65,.65),(.80,.70,1.0),(.60,.95,.95),(1.0,.75,.90)]   # borde de cada nivel
def anillo(R,r=.0015,nu=120,nv=8):
    u=np.linspace(0,2*np.pi,nu+1);v=np.linspace(0,2*np.pi,nv+1);U,W=np.meshgrid(u,v,indexing="ij")
    V=np.stack([(R+r*np.cos(W))*np.cos(U),(R+r*np.cos(W))*np.sin(U),r*np.sin(W)],-1).reshape(-1,3)
    i=np.arange((nu+1)*(nv+1)).reshape(nu+1,nv+1)
    a,b,c,d=i[:-1,:-1].ravel(),i[:-1,1:].ravel(),i[1:,:-1].ravel(),i[1:,1:].ravel()
    return V,np.vstack([np.stack([a,c,b],1),np.stack([b,d,c],1)])
def nucleo(Z,A):
    """Racimo de A bolitas: Z protones (rojo) y A-Z neutrones (celeste)."""
    Rc=.006+.003*A**(1/3);rn=Rc/(A/.6)**(1/3)
    p=rng.normal(size=(A,3));p/=np.linalg.norm(p,axis=1,keepdims=True);p*=(rng.random((A,1))**(1/3))*(Rc-rn)
    for _ in range(80 if A>1 else 0):
        d=p[:,None,:]-p[None,:,:];dist=np.linalg.norm(d,axis=2)+np.eye(A)
        p+=(d/dist[...,None]*((2*rn-dist)*(dist<2*rn))[...,None]).sum(1)*.5
        r=np.linalg.norm(p,axis=1,keepdims=True);p=np.where(r>Rc-rn,p/r*(Rc-rn),p)
    idx=rng.permutation(A);return p[idx[:Z]],p[idx[Z:]],rn
def capas_radios(Z):
    """Radio de cada subnivel (cascarón hueco) y del borde que cierra cada nivel de energía."""
    niv={}
    for n,l,e in config(Z):niv.setdefault(n,[]).append((l,e))
    r=R_INI;shells=[];bordes=[]
    for n in sorted(niv):
        for l,e in niv[n]:r+=SP;shells.append((n,l,e,r))
        bordes.append((n,r+SP/2))
    return shells,bordes
def radio(Z):return round(capas_radios(Z)[1][-1][1]+.01,4)
def escala(Z):return round(min(17,4.2/(2*radio(Z))),2)
W_CINT=0.3   # cintura: fracción del radio que queda en la parte "pellizcada"
FORMA={0:lambda x,y,z:np.ones_like(x),                                   # s: esfera
       1:lambda x,y,z:W_CINT+(1-W_CINT)*np.abs(x)**1.5,                   # p: 2 partes (como una célula dividiéndose)
       2:lambda x,y,z:W_CINT+(1-W_CINT)*np.abs(x*x-y*y)**.7,              # d: 4 partes (trébol)
       3:lambda x,y,z:W_CINT+(1-W_CINT)*np.minimum(1,np.abs(x*y*z)*5.196)**.5}   # f: 8 partes (esquinas de un cubo)
_s3=1/np.sqrt(3)
LOBOS={0:None,1:[(1,0,0),(-1,0,0)],2:[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)],
       3:[(a*_s3,b*_s3,c*_s3) for a in(1,-1) for b in(1,-1) for c in(1,-1)]}
def superficie_g(g,R,nt=40,npp=80):
    T,P=np.meshgrid(np.linspace(0,np.pi,nt+1),np.linspace(0,2*np.pi,npp+1),indexing="ij")
    x,y,z=np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T)
    r=R*g(x,y,z)
    V=np.stack([x*r,y*r,z*r],-1).reshape(-1,3)
    i=np.arange((nt+1)*(npp+1)).reshape(nt+1,npp+1)
    a,b,c,d=i[:-1,:-1].ravel(),i[:-1,1:].ravel(),i[1:,:-1].ravel(),i[1:,1:].ravel()
    return V,np.vstack([np.stack([a,c,b],1),np.stack([b,d,c],1)])
def puntos_en(l,e,R):
    """e electrones repartidos entre las partes de la nube (lo más separados posible), sobre la superficie de la nube."""
    C=FIB;g=FORMA[l];P=C*(R*g(C[:,0],C[:,1],C[:,2]))[:,None];lob=LOBOS[l];grupos=[]
    if lob is None:grupos=[(np.arange(len(C)),e,np.array([0,0,1.]))]
    else:
        L=np.array(lob,float);dots=C@L.T;asig=dots.argmax(1);ok=dots.max(1)>=.8;base,extra=divmod(e,len(L))
        for k in range(len(L)):
            m=base+(1 if k<extra else 0)
            if m:grupos.append((np.where((asig==k)&ok)[0],m,L[k]))
    out=[]
    for idx,m,c0 in grupos:
        pts=P[idx];s=int(np.argmax(C[idx]@c0));ch=[s];d=np.linalg.norm(pts-pts[s],axis=1)
        for _ in range(m-1):
            n=int(np.argmax(d));ch.append(n);d=np.minimum(d,np.linalg.norm(pts-pts[n],axis=1))
        out+=list(pts[ch])
    return out
def modelo(Z):
    """Núcleo en el centro; cada subnivel es una nube hueca muy translúcida con su forma (s esfera, p dos partes, d cuatro, f ocho)
    y sus electrones (puntos azules); cada nivel termina en un borde (dos anillos)."""
    A=MASA[Z-1];shells,bordes=capas_radios(Z);prims=[];elec=[]
    for n,l,e,r in shells:
        V,F=superficie_g(FORMA[l],r);prims.append((V,normales(V,F),F,(COLL[l],ALFA_S,False)))
        elec+=puntos_en(l,e,r)
    assert len(elec)==Z,(len(elec),Z)
    E=np.array(elec);D=np.linalg.norm(E[:,None]-E[None],axis=2)+np.eye(len(E))*9
    if D.min()<2*RE:print("   aviso: dos electrones se tocan (mm):",round(D.min()*1000,1))
    for n,rb in bordes:
        V0,F0=anillo(rb);V,F=unir([(V0,F0),(V0[:,[0,2,1]],F0)])
        prims.append((V,normales(V,F),F,(PALN[(n-1)%7],1.0,True)))
    Ve,Fe=superficie(UNO,RE,12,20)
    V,F=unir([(Ve+c,Fe) for c in elec]);prims.append((V,normales(V,F),F,((.1,.5,1.0),1.0,True)))
    prot,neu,rn=nucleo(Z,A);Vn,Fn=superficie(UNO,rn,8,14)
    for pts,col in((prot,(.85,.12,.18)),(neu,(.55,.75,.92))):
        if len(pts):
            V,F=unir([(Vn+c,Fn) for c in pts]);prims.append((V,normales(V,F),F,(col,1.0,False)))
    return prims
def glb(prims,path):
    blob=bytearray();bv=[];ac=[];mp=[];mt=[]
    def put(a,t):
        blob.extend(b"\0"*(-len(blob)%4));bv.append({"buffer":0,"byteOffset":len(blob),"byteLength":a.nbytes,"target":t});blob.extend(a.tobytes());return len(bv)-1
    for V,N,F,(c,a,em) in prims:
        V=V.astype("<f4");N=N.astype("<f4");big=len(V)>65535;F=F.astype("<u4" if big else "<u2").ravel()
        a0=len(ac)
        ac.append({"bufferView":put(V,34962),"componentType":5126,"count":len(V),"type":"VEC3","min":V.min(0).tolist(),"max":V.max(0).tolist()})
        ac.append({"bufferView":put(N,34962),"componentType":5126,"count":len(V),"type":"VEC3"})
        ac.append({"bufferView":put(F,34963),"componentType":5125 if big else 5123,"count":len(F),"type":"SCALAR"})
        mp.append({"attributes":{"POSITION":a0,"NORMAL":a0+1},"indices":a0+2,"material":len(mt)})
        m={"doubleSided":True,"pbrMetallicRoughness":{"baseColorFactor":[*c,a],"metallicFactor":0,"roughnessFactor":.6}}
        if a<1:m["alphaMode"]="BLEND"
        if em:m["emissiveFactor"]=list(c)
        mt.append(m)
    j={"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":mp}],"materials":mt,
       "buffers":[{"byteLength":len(blob)}],"bufferViews":bv,"accessors":ac}
    js=json.dumps(j,separators=(",",":")).encode();js+=b" "*(-len(js)%4);blob.extend(b"\0"*(-len(blob)%4))
    open(path,"wb").write(struct.pack("<III",0x46546C67,2,28+len(js)+len(blob))+struct.pack("<II",len(js),0x4E4F534A)+js+struct.pack("<II",len(blob),0x004E4942)+bytes(blob))
def usdz(prims,path):
    L=['#usda 1.0','(','    defaultPrim = "Atom"','    metersPerUnit = 0.01','    upAxis = "Y"',')','def Xform "Atom"','{']
    for k,(V,N,F,(c,a,em)) in enumerate(prims):
        L+=['    def Mesh "M%d"'%k,'    {',
         '        int[] faceVertexCounts = [%s]'%", ".join(["3"]*len(F)),
         '        int[] faceVertexIndices = [%s]'%", ".join(map(str,F.ravel())),
         '        point3f[] points = [%s]'%", ".join("(%.3f, %.3f, %.3f)"%tuple(p) for p in V*100),
         '        normal3f[] normals = [%s] ('%", ".join("(%.2f, %.2f, %.2f)"%tuple(p) for p in N),
         '            interpolation = "vertex"','        )','        uniform token subdivisionScheme = "none"','        uniform bool doubleSided = 1',
         '        rel material:binding = </Atom/Mat%d>'%k,'    }',
         '    def Material "Mat%d"'%k,'    {','        token outputs:surface.connect = </Atom/Mat%d/S.outputs:surface>'%k,
         '        def Shader "S"','        {','            uniform token info:id = "UsdPreviewSurface"',
         '            color3f inputs:diffuseColor = (%.2f, %.2f, %.2f)'%tuple(c),
         '            color3f inputs:emissiveColor = (%.2f, %.2f, %.2f)'%(tuple(c) if em else (0,0,0)),
         '            float inputs:opacity = %.2f'%a,'            float inputs:roughness = 0.6','            token outputs:surface','        }','    }']
    L.append('}')
    t=("\n".join(L)+"\n").encode();name="atom.usda";pad=(64-(30+len(name))%64)%64
    if pad<4:pad+=64
    zi=zipfile.ZipInfo(name,(2024,1,1,0,0,0));zi.compress_type=zipfile.ZIP_STORED;zi.extra=struct.pack("<HH",0x4D41,pad-4)+b"\0"*(pad-4)
    with zipfile.ZipFile(path,"w") as z:z.writestr(zi,t)
# ---------- Marcadores (patrón 6x6 único por elemento) ----------
def patrones():
    P=[];r=np.random.default_rng(11)
    while len(P)<118:
        g=r.integers(0,2,(6,6))
        if not 12<=g.sum()<=24:continue
        if any((g!=np.rot90(g,k)).sum()<10 for k in(1,2,3)):continue
        if any((g!=np.rot90(q,k)).sum()<10 for q in P for k in range(4)):continue
        P.append(g)
    return P
def marcador(Z,sym,g):
    from PIL import Image,ImageDraw,ImageFont
    inner=(np.kron(1-g,np.ones((16,16)))*255).astype("u1")
    p16=inner.reshape(16,6,16,6).mean(axis=(1,3)).round().astype(int)
    txt=""
    for k in range(4):
        m=np.rot90(p16,k)
        if k:txt+="\n"
        for _ in range(3):
            for row in m:txt+=" ".join("%3d"%v for v in row)+"\n"
    open(f"patt/{Z}.patt","w").write(txt)
    S=480;img=Image.new("L",(S+160,S+160+110),255);d=ImageDraw.Draw(img)
    d.rectangle([80,80,80+S-1,80+S-1],fill=0)
    img.paste(Image.fromarray(inner).resize((S//2,S//2),Image.NEAREST),(80+S//4,80+S//4))
    f=None
    for nm in("arial.ttf","DejaVuSans-Bold.ttf","DejaVuSans.ttf"):
        try:f=ImageFont.truetype(nm,64);break
        except Exception:pass
    if f is None:f=ImageFont.load_default(size=64)
    d.text((img.width//2,S+160),f"{sym} · {Z}",anchor="mt",font=f,fill=0)
    img.save(f"markers/{Z:03d}-{sym}.png")
LECTOR='''<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Lector de átomos</title>
<script src="https://aframe.io/releases/1.3.0/aframe.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/AR-js-org/AR.js@3.4.5/aframe/build/aframe-ar.js"></script>
<style>body{margin:0;overflow:hidden;background:#000;font-family:system-ui,sans-serif}
#info{position:fixed;left:0;right:0;bottom:0;padding:14px 16px calc(14px + env(safe-area-inset-bottom,0px));background:rgba(5,7,13,.8);color:#fff;text-align:center;z-index:10;font-size:18px}
#info small{display:block;color:#aab;font-size:12.5px;margin-top:4px}</style></head>
<body><div id="info"><b id="t">Apunta la cámara a un marcador</b><small id="s">Si el video se ve con mucho zoom, prueba agregar ?c=1 o ?c=2 a la dirección</small></div>
<script>
window.addEventListener("error",e=>{window.__err=(e.message||"")+" "+String(e.filename||"").split("/").pop()});
{const c=new URLSearchParams(location.search).get("c")||"3",p=innerWidth<innerHeight;
const b="sourceType:webcam;debugUIEnabled:false;detectionMode:mono;patternRatio:0.5;trackingMethod:best;";
const A={"1":b+"sourceWidth:"+(p?720:1280)+";sourceHeight:"+(p?1280:720)+";displayWidth:"+innerWidth+";displayHeight:"+innerHeight,"2":b+"sourceWidth:1280;sourceHeight:960;displayWidth:1280;displayHeight:960","3":b,"4":b+"sourceWidth:1280;sourceHeight:720","5":b+"sourceWidth:"+(p?480:640)+";sourceHeight:"+(p?640:480),"6":b+"sourceWidth:"+(p?480:640)+";sourceHeight:"+(p?640:480)+";cameraParametersUrl:"+(p?"camera_p.dat":"camera_l.dat")};
document.write("<a-scene "+(c>="3"?"":"embedded ")+'vr-mode-ui="enabled:false" renderer="logarithmicDepthBuffer:true;alpha:true;antialias:true" arjs="'+(A[c]||A["3"])+'">')}
</script>
%%M%%
<a-entity camera></a-entity></a-scene>
<script>
const D=%%D%%;
const Q=new URLSearchParams(location.search),quiz=Q.get("quiz");
for(const z in D){const e=document.querySelector("#m"+z+" a-entity"),k=+(Q.get("s")||D[z][3]);
 e.setAttribute("scale",k+" "+k+" "+k);e.setAttribute("position","0 "+(D[z][5]*k+0.1)+" 0")}
if(!Q.get("info"))document.getElementById("info").style.display="none";
(function f(){const v=document.querySelector("video"),st=v&&v.srcObject;if(!st)return setTimeout(f,500);
const k=st.getVideoTracks()[0],cp=k.getCapabilities?k.getCapabilities():{};
if(cp.focusMode&&cp.focusMode.includes("continuous"))k.applyConstraints({advanced:[{focusMode:"continuous"}]}).catch(()=>{})})();
(function(){const sc=document.querySelector("a-scene"),mode=new URLSearchParams(location.search).get("v")||"cover";
const S=(e,o)=>{for(const k in o)e.style.setProperty(k,o[k],"important")};
function fix(){requestAnimationFrame(fix);
 const v=document.querySelector("video"),c=document.querySelector("canvas.a-canvas")||document.querySelector("a-scene canvas"),r=sc&&sc.renderer;
 if(!v||!c||!r||!v.videoWidth)return;
 let L,T,W,H;
 if(mode=="contain"){const k=Math.min(innerWidth/v.videoWidth,innerHeight/v.videoHeight);W=v.videoWidth*k;H=v.videoHeight*k;L=(innerWidth-W)/2;T=(innerHeight-H)/2;
  S(v,{position:"fixed",left:L+"px",top:T+"px",width:W+"px",height:H+"px",margin:"0"})}
 else{const b=v.getBoundingClientRect();L=b.left;T=b.top;W=b.width;H=b.height}
 S(c,{position:"fixed",left:L+"px",top:T+"px",width:W+"px",height:H+"px",margin:"0"});
 const pr=Math.min(devicePixelRatio||1,2,2200/Math.max(W,H)),bw=Math.round(W*pr),bh=Math.round(H*pr);
 if(c.width!=bw||c.height!=bh){r.setPixelRatio(1);r.setSize(bw,bh,false)}}
fix()})();
if(new URLSearchParams(location.search).get("d")){const d=document.createElement("div");d.style.cssText="position:fixed;top:0;left:0;z-index:20;background:rgba(0,0,0,.7);color:#0f0;font:12px monospace;padding:4px";document.body.appendChild(d);
setInterval(()=>{const v=document.querySelector("video"),c=document.querySelector("a-scene canvas");d.textContent=(v?"video "+v.videoWidth+"x"+v.videoHeight:"sin video")+" | canvas "+(c?c.width+"x"+c.height:"-")+" | ventana "+innerWidth+"x"+innerHeight+" | AF:"+(typeof AFRAME)+" THREEx:"+(typeof THREEx)+" err:"+(window.__err||"-")+" | modo "+(new URLSearchParams(location.search).get("c")||"3")},700)}
const t=document.getElementById("t"),s=document.getElementById("s");let cur=0;
for(const z in D){const m=document.getElementById("m"+z);
 m.addEventListener("markerFound",()=>{cur=z;if(quiz){t.textContent=D[z][0]+" ("+D[z][1]+")";s.textContent="¿Cuántos electrones, protones y neutrones ves?"}else{t.textContent=D[z][0]+" ("+D[z][1]+") · "+z+" electrones";s.textContent=D[z][2]}});
 m.addEventListener("markerLost",()=>{if(cur==z){t.textContent="Apunta la cámara a un marcador";s.textContent="azul = electrones · núcleo: rojo = protones, celeste = neutrones"}})}
</script></body></html>'''
def camara(path,W,H):
    f=.78*max(W,H)
    open(path,"wb").write(struct.pack(">ii12d4d",W,H,f,0,W/2,0,0,f,H/2,0,0,0,1,0,W/2,H/2,0.0,1.0))
def escribir_lector(zs):
    camara("camera_p.dat",480,640);camara("camera_l.dat",640,480)
    M="\n".join(f'<a-marker id="m{Z}" type="pattern" url="patt/{Z}.patt" smooth="true" smoothCount="8" smoothTolerance="0.01" smoothThreshold="4"><a-entity id="w{Z}"><a-entity gltf-model="glb/{Z}.glb"></a-entity></a-entity></a-marker>' for Z in zs)
    D={Z:[EL[Z-1][1],EL[Z-1][0]," ".join(f"{n}{'spdf'[l]}{sup(e)}" for n,l,e in config(Z)),escala(Z),[],radio(Z)] for Z in zs}
    open("lector.html","w",encoding="utf8").write(LECTOR.replace("%%M%%",M).replace("%%D%%",json.dumps(D,ensure_ascii=False)))
if __name__=="__main__":
    import sys
    zs=list(range(1,119)) if "todos" in sys.argv else [1,26,79]
    for d in("glb","usdz","patt","markers"):os.makedirs(d,exist_ok=True)
    PAT=patrones()
    for Z in zs:
        pr=modelo(Z);glb(pr,f"glb/{Z}.glb");usdz(pr,f"usdz/{Z}.usdz");marcador(Z,EL[Z-1][0],PAT[Z-1])
        print(Z,EL[Z-1][0],"electrones:",Z,"vértices:",sum(len(p[0]) for p in pr))
    escribir_lector(zs)
