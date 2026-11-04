"""Opt-in GPU endoscope postprocessing, with unchanged CPU-returned RGB-D.

Only EndoscopicView.render activates the hook. Direct camera calls, segmentation,
and unsupported modes use the original renderer. No CUDA/GL host round trip is
introduced: the multisample framebuffer resolves into resident textures, followed
by one full-screen shader. OpenCV's 1/32-pixel interpolation and uint8 rounding are
reproduced explicitly rather than approximated with hardware bilinear filtering.
"""
from contextlib import contextmanager
import numpy as np

VERTEX = '''#version 410 core
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2-1,0,1);}
'''
FRAGMENT = '''#version 410 core
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform isampler2D map_tex;
uniform sampler2D vignette_tex;
uniform float znear;
uniform float zfar;
uniform bool emit_aligned;
layout(location=0) out uvec4 color;
layout(location=1) out float depth;
layout(location=2) out float aligned_depth;
layout(location=3) out uint aligned_valid;
ivec3 pixel(ivec2 p){
 ivec2 size=textureSize(color_tex,0);
 if(any(lessThan(p,ivec2(0)))||any(greaterThanEqual(p,size)))return ivec3(0);
 return ivec3(roundEven(texelFetch(color_tex,p,0).rgb*255.0));
}
float linear_depth(float z){
 precise float ndc=z*2.0-1.0;
 precise float numerator=znear*zfar;
 precise float difference=zfar-znear;
 precise float term=ndc*difference;
 precise float denominator=(zfar+znear)-term;
 precise float value=numerator/denominator;
 if(zfar<0)value=znear/(1.0-ndc);
 return value*2.0;
}
void main(){
 ivec2 p=ivec2(gl_FragCoord.xy);ivec4 m=texelFetch(map_tex,p,0);
 int fx=m.z&31,fy=m.z>>5;ivec2 q=m.xy;
 ivec3 sum=pixel(q)*(32-fx)*(32-fy)+pixel(q+ivec2(1,0))*fx*(32-fy)
           +pixel(q+ivec2(0,1))*(32-fx)*fy+pixel(q+ivec2(1,1))*fx*fy;
 ivec3 interpolated=(sum+512)>>10;
 precise vec3 shaded=vec3(interpolated)*texelFetch(vignette_tex,p,0).r;
 color=uvec4(uvec3(clamp(roundEven(shaded.bgr),0,255)),255);
 depth=linear_depth(texelFetch(depth_tex,ivec2(p.x,textureSize(depth_tex,0).y-1-p.y),0).r);
 if(emit_aligned){
  // CV_32FC2 conversion followed by INTER_NEAREST uses ties-to-even.
  ivec2 nearest=ivec2(roundEven(vec2(q)+vec2(fx,fy)/32.0));
  ivec2 size=textureSize(depth_tex,0);
  float value=0.0;
  if(all(greaterThanEqual(nearest,ivec2(0)))&&all(lessThan(nearest,size)))
   value=linear_depth(texelFetch(depth_tex,nearest,0).r);
  bool valid=value>znear && value<zfar*(1.0-1e-3) && !isnan(value) && !isinf(value)
             && texelFetch(vignette_tex,p,0).r>0.0;
  aligned_depth=valid?value:0.0;
  aligned_valid=valid?1u:0u;
 }
}
'''

class GPUPostprocess:
    def __init__(self, view, *, aligned=False):
        from OpenGL import GL as gl
        from OpenGL.GL.shaders import compileShader, compileProgram
        if view.supersample != 1:
            raise ValueError('GPU postprocess currently requires supersample=1; never silently reduces supersampling')
        self.view, self.gl = view, gl
        self.active = False
        self.aligned_enabled = bool(aligned)
        self.aligned_depth = self.aligned_valid = None
        raster = view.camera._rasterizer
        if not raster._offscreen:
            raise ValueError('GPU postprocess requires the offscreen endoscope renderer')
        self.context = raster._renderer
        self.target = raster._camera_targets[view.camera.uid]
        self.original = self.target._read_main_framebuffer
        self.textures = []
        self.context.make_current()
        try:
            with self.state():
                self.program=compileProgram(compileShader(VERTEX,gl.GL_VERTEX_SHADER),compileShader(FRAGMENT,gl.GL_FRAGMENT_SHADER))
                self.vao=int(gl.glGenVertexArrays(1))
                self.raw_fbo=int(gl.glGenFramebuffers(1));self.out_fbo=int(gl.glGenFramebuffers(1))
                self.color=self.texture(gl.GL_RGBA8,gl.GL_RGBA,gl.GL_UNSIGNED_BYTE)
                self.depth=self.texture(gl.GL_DEPTH_COMPONENT24,gl.GL_DEPTH_COMPONENT,gl.GL_UNSIGNED_INT)
                mapping=np.zeros((view.height,view.width,4),np.int32)
                mapping[...,:2]=view.lens_map[0];mapping[...,2]=view.lens_map[1]
                self.mapping=self.texture(gl.GL_RGBA32I,gl.GL_RGBA_INTEGER,gl.GL_INT,mapping)
                self.vignette=self.texture(gl.GL_R32F,gl.GL_RED,gl.GL_FLOAT,np.ascontiguousarray(view.vignette[...,0]))
                self.output_color=self.texture(gl.GL_RGBA8UI,gl.GL_RGBA_INTEGER,gl.GL_UNSIGNED_BYTE)
                self.output_depth=self.texture(gl.GL_R32F,gl.GL_RED,gl.GL_FLOAT)
                extra=[]
                if aligned:
                    self.output_aligned=self.texture(gl.GL_R32F,gl.GL_RED,gl.GL_FLOAT)
                    self.output_valid=self.texture(gl.GL_R8UI,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_BYTE)
                    extra=[(gl.GL_COLOR_ATTACHMENT2,self.output_aligned),(gl.GL_COLOR_ATTACHMENT3,self.output_valid)]
                for fbo,attachments in [(self.raw_fbo,[(gl.GL_COLOR_ATTACHMENT0,self.color),(gl.GL_DEPTH_ATTACHMENT,self.depth)]),
                                        (self.out_fbo,[(gl.GL_COLOR_ATTACHMENT0,self.output_color),(gl.GL_COLOR_ATTACHMENT1,self.output_depth)]+extra)]:
                    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,fbo)
                    for attachment,texture in attachments:
                        gl.glFramebufferTexture2D(gl.GL_DRAW_FRAMEBUFFER,attachment,gl.GL_TEXTURE_2D,texture,0)
                    gl.glDrawBuffers(1 if fbo==self.raw_fbo else 2,[gl.GL_COLOR_ATTACHMENT0] if fbo==self.raw_fbo else [gl.GL_COLOR_ATTACHMENT0,gl.GL_COLOR_ATTACHMENT1])
                    if gl.glCheckFramebufferStatus(gl.GL_DRAW_FRAMEBUFFER)!=gl.GL_FRAMEBUFFER_COMPLETE:
                        raise RuntimeError('GPU postprocess framebuffer incomplete')
                gl.glUseProgram(self.program)
                for i,name in enumerate(('color_tex','depth_tex','map_tex','vignette_tex')):
                    gl.glUniform1i(gl.glGetUniformLocation(self.program,name),i)
                self.near_location=gl.glGetUniformLocation(self.program,'znear');self.far_location=gl.glGetUniformLocation(self.program,'zfar');self.aligned_location=gl.glGetUniformLocation(self.program,'emit_aligned')
        finally:
            self.context.make_uncurrent()
        # Keep this closure dynamic so a profiler can wrap self.read per instance.
        self.target._read_main_framebuffer=lambda scene,flags:self.read(scene,flags)

    def texture(self, internal, format_, type_, data=None):
        gl=self.gl;texture=int(gl.glGenTextures(1));self.textures.append(texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER,gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D,0,internal,self.view.width,self.view.height,0,format_,type_,data)
        return texture

    @contextmanager
    def state(self):
        gl=self.gl
        integers={name:int(gl.glGetIntegerv(enum)) for name,enum in [('draw',gl.GL_DRAW_FRAMEBUFFER_BINDING),('read',gl.GL_READ_FRAMEBUFFER_BINDING),
            ('program',gl.GL_CURRENT_PROGRAM),('vao',gl.GL_VERTEX_ARRAY_BINDING),('active',gl.GL_ACTIVE_TEXTURE),('pack_buffer',gl.GL_PIXEL_PACK_BUFFER_BINDING),
            ('pack',gl.GL_PACK_ALIGNMENT),('unpack',gl.GL_UNPACK_ALIGNMENT),('read_buffer',gl.GL_READ_BUFFER)]}
        viewport=gl.glGetIntegerv(gl.GL_VIEWPORT);color_mask=gl.glGetBooleanv(gl.GL_COLOR_WRITEMASK)
        caps=(gl.GL_DEPTH_TEST,gl.GL_CULL_FACE,gl.GL_BLEND,gl.GL_MULTISAMPLE,gl.GL_SCISSOR_TEST,gl.GL_FRAMEBUFFER_SRGB)
        enabled={cap:bool(gl.glIsEnabled(cap)) for cap in caps};textures=[]
        for i in range(4):
            gl.glActiveTexture(gl.GL_TEXTURE0+i);textures.append(int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        try:yield
        finally:
            for i,texture in enumerate(textures):
                gl.glActiveTexture(gl.GL_TEXTURE0+i);gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
            gl.glActiveTexture(integers['active'])
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,integers['draw']);gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,integers['read'])
            gl.glReadBuffer(integers['read_buffer']);gl.glUseProgram(integers['program']);gl.glBindVertexArray(integers['vao'])
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER,integers['pack_buffer']);gl.glPixelStorei(gl.GL_PACK_ALIGNMENT,integers['pack']);gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,integers['unpack'])
            gl.glViewport(*viewport);gl.glColorMask(*color_mask)
            for cap,enabled_ in enabled.items():(gl.glEnable if enabled_ else gl.glDisable)(cap)

    def read(self,scene,flags):
        from genesis.ext.pyrender.constants import RenderFlags
        if not self.active or flags & (RenderFlags.SEG|RenderFlags.DEPTH_ONLY|RenderFlags.RGBA):
            return self.original(scene,flags)
        gl=self.gl;w,h=self.view.width,self.view.height
        with self.state():
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,self.target._main_fb_ms)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,self.raw_fbo)
            bits=gl.GL_COLOR_BUFFER_BIT | (gl.GL_DEPTH_BUFFER_BIT if flags & RenderFlags.RET_DEPTH else 0)
            gl.glBlitFramebuffer(0,0,w,h,0,0,w,h,bits,gl.GL_NEAREST)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,self.out_fbo);gl.glViewport(0,0,w,h)
            for cap in (gl.GL_DEPTH_TEST,gl.GL_CULL_FACE,gl.GL_BLEND,gl.GL_MULTISAMPLE,gl.GL_SCISSOR_TEST,gl.GL_FRAMEBUFFER_SRGB):gl.glDisable(cap)
            gl.glColorMask(True,True,True,True);gl.glUseProgram(self.program);gl.glBindVertexArray(self.vao)
            emit=self.aligned_enabled and bool(flags & RenderFlags.RET_DEPTH)
            gl.glUniform1i(self.aligned_location,int(emit));gl.glDrawBuffers(4 if emit else 2,[gl.GL_COLOR_ATTACHMENT0+i for i in range(4 if emit else 2)])
            camera=scene.main_camera_node.camera
            gl.glUniform1f(self.near_location,camera.znear);gl.glUniform1f(self.far_location,camera.zfar if camera.zfar is not None else -1)
            for i,texture in enumerate((self.color,self.depth,self.mapping,self.vignette)):
                gl.glActiveTexture(gl.GL_TEXTURE0+i);gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,3)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER,0);gl.glPixelStorei(gl.GL_PACK_ALIGNMENT,1)
            color=np.empty((h,w,4),np.uint8);gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glReadPixels(0,0,w,h,gl.GL_RGBA_INTEGER,gl.GL_UNSIGNED_BYTE,color)
            color=color[...,:3]
            if flags & RenderFlags.RET_DEPTH:
                depth=np.empty((h,w),np.float32);gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
                gl.glReadPixels(0,0,w,h,gl.GL_RED,gl.GL_FLOAT,depth)
                if emit:
                    aligned=np.empty((h,w),np.float32);valid=np.empty((h,w),np.uint8)
                    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT2);gl.glReadPixels(0,0,w,h,gl.GL_RED,gl.GL_FLOAT,aligned)
                    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT3);gl.glReadPixels(0,0,w,h,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_BYTE,valid)
                    self.aligned_depth,self.aligned_valid=aligned,valid.view(np.bool_)
                else:self.aligned_depth=self.aligned_valid=None
                return color,depth
            return (color,)

    def close(self):
        self.target._read_main_framebuffer=self.original
        gl=self.gl;self.context.make_current()
        try:
            gl.glDeleteProgram(self.program);gl.glDeleteVertexArrays(1,[self.vao]);gl.glDeleteFramebuffers(2,[self.raw_fbo,self.out_fbo]);gl.glDeleteTextures(self.textures)
        finally:self.context.make_uncurrent()
