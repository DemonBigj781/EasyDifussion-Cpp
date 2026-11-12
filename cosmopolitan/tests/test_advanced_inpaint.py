"""Differential native conditioning against the preserved Python production helper.

No model, GUI, provider or GPU use. Requires host C++ and Pillow; sanitizer flags
can be supplied with ADVANCED_CXXFLAGS. Checks actual image_utils.cpp, not a copy.
"""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import tempfile
import unittest
try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None
ROOT = Path(__file__).resolve().parents[2]
SDKIT = ROOT / 'source/sdkit3-port-source'

@unittest.skipUnless(Image is not None and shutil.which('c++'), 'Pillow and host C++ required')
class AdvancedInpaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='native-advanced-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name)
        cls.exe = cls.path/'conditioning'
        command = ['c++','-std=c++17','-O1','-pthread'] + shlex.split(os.environ.get('ADVANCED_CXXFLAGS',''))
        for inc in ['include','third_party','stable-diffusion.cpp/include']:
            command += ['-I'+str(SDKIT/inc)]
        command += [str(Path(__file__).with_name('advanced_inpaint_fixture.cpp')),str(SDKIT/'src/image_utils.cpp'),str(SDKIT/'src/logging.cpp'),'-o',str(cls.exe)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        spec = importlib.util.spec_from_file_location('preserved_color_inpaint',ROOT/'ui/easydiffusion/color_inpaint.py')
        cls.python = importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.python)

    def invoke(self, original, guide, mask, size, strength=.75, blur=0, invert=False):
        for item,name in [(original,'source'),(guide,'guide'),(mask,'mask')]:
            if isinstance(item,str): encoded=item
            elif item is None: encoded=''
            else:
                stream=io.BytesIO();item.save(stream,format='PNG');encoded=base64.b64encode(stream.getvalue()).decode()
            (self.path/(name+'.b64')).write_text(encoded)
        args=[str(self.exe),*[str(self.path/(n+'.b64')) for n in ['source','guide','mask']],str(size[0]),str(size[1]),str(strength),str(blur),str(int(invert)),str(self.path/'result')]
        result=subprocess.run(args,capture_output=True,text=True)
        if result.returncode==0:
            return result,Image.frombytes('RGB',size,(self.path/'result.rgb').read_bytes()),Image.frombytes('L',size,(self.path/'result.mask').read_bytes())
        return result,None,None

    def test_differential_exact_twenty_cases(self):
        rng=random.Random(42)
        def random_image(mode,size):return Image.frombytes(mode,size,bytes(rng.randrange(256) for _ in range(size[0]*size[1]*len(mode))))
        for srcsize,dstsize in [((32,24),(32,24)),((32,24),(64,48)),((64,48),(32,24)),((31,29),(64,64)),((64,64),(19,23))]:
            for mode in ['combined','guide-only','black-guide','gray-alpha']:
                with self.subTest(source=srcsize,output=dstsize,mode=mode):
                    original=random_image('RGB',srcsize);guide=random_image('RGBA',srcsize)
                    guide.paste((0,0,0,0),(0,0,srcsize[0],srcsize[1]//3))
                    if mode=='black-guide':guide=Image.merge('RGBA',(*[Image.new('L',srcsize)]*3,guide.getchannel('A')))
                    if mode=='gray-alpha':guide=guide.convert('LA')
                    mask=None if mode=='guide-only' else random_image('RGB',srcsize)
                    snapshots=[x.tobytes() if x else None for x in (original,guide,mask)]
                    expected,em=self.python.prepare_color_weighted_inpaint(original,guide,mask,dstsize,.75)
                    result,actual,am=self.invoke(original,guide,mask,dstsize)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(actual.tobytes(),expected.tobytes())
                    self.assertEqual(am.tobytes(),em.convert('L').tobytes())
                    self.assertEqual(snapshots,[x.tobytes() if x else None for x in (original,guide,mask)])

    def test_invalid_inputs_publish_no_partial_outputs(self):
        source=Image.new('RGB',(16,16),(100,50,30));guide=Image.new('RGBA',(16,16),(0,0,0,128))
        cases=[(None,guide,None,(16,16),.5,0),(source,'garbage',None,(16,16),.5,0),
               (source,guide.convert('RGB'),None,(16,16),.5,0),(source,guide.resize((8,8)),None,(16,16),.5,0),
               (source,Image.new('RGBA',(16,16)),None,(16,16),.5,0),(source,guide,'garbage',(16,16),.5,0),
               (source,guide,None,(0,16),.5,0),(source,guide,None,(16385,16),.5,0),
               (source,guide,None,(16,16),0,0),(source,guide,None,(16,16),1,0),
               (source,guide,None,(16,16),float('nan'),0),(source,guide,None,(16,16),.5,65)]
        for case in cases:
            result,_,_=self.invoke(*case)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertTrue(result.stderr)

    def test_alias_outputs_rejected(self):
        result=subprocess.run([str(self.exe),"--alias"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_data_urls_and_header_bounds(self):
        import struct, zlib
        source=Image.new('RGB',(16,16),(100,40,20));guide=Image.new('RGBA',(16,16),(0,0,0,128))
        def data(im):
            b=io.BytesIO();im.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        plain,expected,mask=self.invoke(source,guide,None,(16,16))
        result,actual,am=self.invoke(data(source),data(guide),None,(16,16))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(actual.tobytes(),expected.tobytes());self.assertEqual(am.tobytes(),mask.tobytes())
        # Valid PNG header with impossible dimensions; no pixel payload needed.
        header=struct.pack('>IIBBBBB',20000,20000,8,6,0,0,0)
        png=b'\x89PNG\r\n\x1a\n'+struct.pack('>I',len(header))+b'IHDR'+header+struct.pack('>I',zlib.crc32(b'IHDR'+header))
        huge=base64.b64encode(png).decode()
        for original,g,m in [(huge,guide,None),(source,huge,None),(source,guide,huge),
                             (source,'A'*(24*1024*1024+1),None)]:
            result,_,_=self.invoke(original,g,m,(16,16))
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('exceeds',result.stderr)

    def test_union_then_inversion_then_blur(self):
        source=Image.new('RGB',(32,32),(100,40,20));guide=Image.new('RGBA',(32,32))
        guide.paste((0,0,0,128),(8,8,16,24));ordinary=Image.new('L',(32,32))
        ordinary.paste(128,(16,8,24,24))
        a,image,mask=self.invoke(source,guide,ordinary,(32,32))
        b,ii,inverted=self.invoke(source,guide,ordinary,(32,32),invert=True)
        c,bi,blurred=self.invoke(source,guide,ordinary,(32,32),blur=2,invert=True)
        self.assertEqual([a.returncode,b.returncode,c.returncode],[0,0,0])
        self.assertEqual(image.tobytes(),ii.tobytes());self.assertEqual(image.tobytes(),bi.tobytes())
        self.assertEqual(inverted.tobytes(),ImageChops.invert(mask).tobytes())
        self.assertNotEqual(inverted.tobytes(),blurred.tobytes())
        self.assertTrue(0<blurred.getpixel((8,8))<255)

if __name__=='__main__':unittest.main()
