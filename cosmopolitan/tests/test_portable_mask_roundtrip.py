"""Real portable mask pixel helper -> PNG -> native production mask preparation.

Canvas/DOM/browser decoding are NOT exercised. Pillow encodes the production JS
helper's actual RGBA bytes; native image_utils decodes them. No inference/GPU.
"""
import base64,io,json,os,shlex,shutil,subprocess,tempfile,unittest
from pathlib import Path
try:
 from PIL import Image
except ImportError:Image=None
ROOT=Path(__file__).resolve().parents[2];SDKIT=ROOT/'source/sdkit3-port-source'
@unittest.skipUnless(Image and shutil.which('c++') and shutil.which('node'),'Pillow, node and host C++ required')
class PortableMaskRoundTripTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='portable-mask-roundtrip-');cls.addClassCleanup(cls.temp.cleanup);cls.path=Path(cls.temp.name);cls.exe=cls.path/'mask'
  command=['c++','-std=c++17','-O1','-pthread']+shlex.split(os.environ.get('MASK_CXXFLAGS',''))
  for inc in ['include','third_party','stable-diffusion.cpp/include']:command+=['-I'+str(SDKIT/inc)]
  command += [str(Path(__file__).with_name('portable_mask_fixture.cpp')),str(SDKIT/'src/image_utils.cpp'),str(SDKIT/'src/logging.cpp'),'-o',str(cls.exe)]
  subprocess.run(command,check=True,capture_output=True,text=True)
 def make(self,mode):
  script=r'''const fs=require('fs'),vm=require('vm');const source=fs.readFileSync(process.argv[1],'utf8');
const begin='// BEGIN PORTABLE ORDINARY MASK PIXELS',end='// END PORTABLE ORDINARY MASK PIXELS';
const start=source.indexOf(begin),stop=source.indexOf(end);if(start<0||stop<start)throw Error('Production mask helper missing');
const c={window:{}};vm.runInNewContext(source.slice(start,stop),c);const h=c.window.PortableMaskPixels;
const mode=process.argv[2],mask=new Uint8ClampedArray(256);let inputRgba=null;
if(mode==='color-import'){inputRgba=Uint8ClampedArray.from({length:1024},(_,p)=>{const i=Math.floor(p/4);return [i*3,255-i*2,i*4,255][p%4];});mask.set(h.fromRgba(inputRgba,16,16));}
if(mode==='white')mask.fill(255);
if(mode==='gray')for(let i=0;i<256;i++)mask[i]=i;
if(mode==='stroke'){h.stroke(mask,16,16,[1,1],[6,6],3,255,0.5);const selected=new Uint8Array(256);for(let i=0;i<8;i++)selected[i]=1;h.select(mask,selected,16,16,255,1);h.stroke(mask,16,16,[2,2],[3,3],1,0,1);}
if(mode==='invert'||mode==='invert-twice'){for(let i=0;i<256;i++)mask[i]=i;h.invert(mask,16,16);if(mode==='invert-twice')h.invert(mask,16,16);}
if(mode==='fill-white'||mode==='fill-black'){mask.fill(127);h.fill(mask,16,16,mode==='fill-white'?255:0);}
const rgba=h.toRgba(mask,16,16);console.log(JSON.stringify({mask:Array.from(mask),rgba:Array.from(rgba),inputRgba:inputRgba?Array.from(inputRgba):null}));'''
  value=json.loads(subprocess.check_output(['node','-e',script,str(ROOT/'cosmopolitan/resources/cpp-ui/scripts/generate.js'),mode],text=True))
  self.assertEqual(value['rgba'][3::4],[255]*256)
  for c in range(3):self.assertEqual(value['rgba'][c::4],value['mask'])
  out=io.BytesIO();Image.frombytes('RGBA',(16,16),bytes(value['rgba'])).save(out,format='PNG');(self.path/'mask.b64').write_text(base64.b64encode(out.getvalue()).decode());
  if value['inputRgba']:
   original=io.BytesIO();Image.frombytes('RGBA',(16,16),bytes(value['inputRgba'])).save(original,format='PNG');(self.path/'original-color.b64').write_text(base64.b64encode(original.getvalue()).decode())
  if mode=='invert':self.assertEqual(value['mask'],list(range(255,-1,-1)))
  if mode=='invert-twice':self.assertEqual(value['mask'],list(range(256)))
  if mode=='fill-white':self.assertEqual(value['mask'],[255]*256)
  if mode=='fill-black':self.assertEqual(value['mask'],[0]*256)
  return bytes(value['mask'])
 def test_exact_native_decode_and_request_inversion(self):
  for mode in ['blank','white','gray','stroke','color-import','invert','invert-twice','fill-white','fill-black']:
   original=self.make(mode)
   for invert in [0,1]:
    with self.subTest(mode=mode,invert=invert):
     subprocess.run([str(self.exe),str(self.path/'mask.b64'),str(invert),str(self.path/'mask.bin')],check=True,capture_output=True,text=True)
     expected=bytes(255-x for x in original) if invert else original
     self.assertEqual((self.path/'mask.bin').read_bytes(),expected)
     if mode=='color-import':
      subprocess.run([str(self.exe),str(self.path/'original-color.b64'),str(invert),str(self.path/'original.bin')],check=True,capture_output=True,text=True)
      self.assertEqual((self.path/'original.bin').read_bytes(),expected,'Editor import must match native RGB luminance')
if __name__=='__main__':unittest.main()
