"""Genera un QR por elemento.  pip install qrcode[pil]
Uso:  python make_qr.py https://TU-USUARIO.github.io/atomos-ar/"""
import sys,os,qrcode
from generar import EL
base=sys.argv[1].rstrip("/")+"/"
os.makedirs("qr",exist_ok=True)
for Z,(s,n) in enumerate(EL,1):
    if not os.path.exists(f"glb/{Z}.glb"):continue
    qrcode.make(f"{base}?e={Z}",error_correction=qrcode.constants.ERROR_CORRECT_M,box_size=12,border=2).save(f"qr/{Z:03d}-{s}.png")
qrcode.make(base+"lector.html",error_correction=qrcode.constants.ERROR_CORRECT_M,box_size=12,border=2).save("qr/lector.png")
print("Listo: carpeta qr/ con QR de los elementos generados + qr/lector.png")
