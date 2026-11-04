"""Opt-in float OpenGL visibility pass for privileged material support labels.

Uses the displayed, unrefined tissue triangles and all opaque occluders. The
geometry shader supplies perspective-correct barycentrics; IDs are exact float
integers (limited to 2**24). No renderer shaders or physical buffers are changed.
"""
import ctypes
import numpy as np
import cv2

VERTEX = '''#version 330 core
layout(location=0) in vec4 position_label;
uniform mat4 mvp;
flat out int vertex_label;
void main(){gl_Position=mvp*vec4(position_label.xyz,1);vertex_label=int(position_label.w);}
'''
GEOMETRY = '''#version 330 core
layout(triangles) in;
layout(triangle_strip,max_vertices=3) out;
flat in int vertex_label[];
flat out int triangle_label;
out vec3 barycentric;
void main(){
 for(int i=0;i<3;i++){
  gl_Position=gl_in[i].gl_Position;triangle_label=vertex_label[0];
  barycentric=vec3(0);barycentric[i]=1;EmitVertex();
 }
 EndPrimitive();
}
'''
FRAGMENT = '''#version 330 core
flat in int triangle_label;
in vec3 barycentric;
layout(location=0) out vec4 label;
void main(){label=vec4(barycentric,float(triangle_label+1));}
'''


def displayed_primitives(view, surface):
    """Return local triangle soups with native triangle IDs or -1 for occluders."""
    from ..grid import to_genesis
    if view.supersample != 1:
        raise ValueError('Pixel truth currently requires supersample=1, with no color downsampling')
    if len(surface.triangles) >= 2**24-1:
        raise ValueError('Native triangle IDs exceed exact float32 integer range')
    context=view.scene.visualizer.context
    mapping={}
    wall=surface.anchors[surface.triangles[:,0],3]>.5
    for part,mesh in enumerate(view.tissue_meshes):
        if mesh is not None:
            mapping[context.external_nodes[mesh.name]]=np.flatnonzero(wall==bool(part))
    result=[]
    for node in context._scene.mesh_nodes:
        if not node.mesh.is_visible:continue
        for primitive in node.mesh.primitives:
            if primitive.mode != 4:raise ValueError('Pixel truth requires opaque triangle meshes')
            material=primitive.material
            if material is not None and material.alphaMode != 'OPAQUE':
                raise ValueError('Pixel truth does not approximate transparent occlusion')
            positions=np.asarray(primitive.positions)
            indices=primitive.indices
            indices=np.arange(len(positions)).reshape(-1,3) if indices is None else np.asarray(indices).reshape(-1,3)
            soup=positions[indices]
            ids=mapping.get(node)
            if ids is None:ids=np.full(len(indices),-1,np.int32)
            else:
                expected=to_genesis(surface.vertices[surface.triangles[ids]].reshape(-1,3)).reshape(-1,3,3)
                if soup.shape!=expected.shape or not np.allclose(soup,expected,atol=1e-8,rtol=1e-6):
                    raise ValueError('Displayed triangles no longer match native surface; update the view first')
            packed=np.concatenate((soup,np.broadcast_to(ids[:,None,None],soup.shape[:2]+(1,))),axis=-1)
            poses=primitive.poses
            if poses is None:poses=np.eye(4)[None]
            for pose in poses:
                result.append((np.ascontiguousarray(packed.reshape(-1,4),dtype=np.float32),
                               context._scene.get_pose(node)@pose,bool(material and material.doubleSided)))
    return result


class PixelLabelPass:
    def __init__(self, view):
        from OpenGL import GL as gl
        from OpenGL.GL.shaders import compileShader,compileProgram
        self.view=view;self.gl=gl
        self.offscreen=view.camera._rasterizer._renderer
        if not view.camera._rasterizer._offscreen:
            raise ValueError('Pixel labels require the existing offscreen renderer')
        self.offscreen.make_current()
        try:
            self.program=compileProgram(compileShader(VERTEX,gl.GL_VERTEX_SHADER),
                compileShader(GEOMETRY,gl.GL_GEOMETRY_SHADER),compileShader(FRAGMENT,gl.GL_FRAGMENT_SHADER))
            self.vao=int(gl.glGenVertexArrays(1));self.buffer=int(gl.glGenBuffers(1))
            self.fbo=int(gl.glGenFramebuffers(1));self.texture=int(gl.glGenTextures(1));self.depth=int(gl.glGenRenderbuffers(1))
            saved=(int(gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)),int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)),
                   int(gl.glGetIntegerv(gl.GL_RENDERBUFFER_BINDING)))
            try:
                gl.glBindTexture(gl.GL_TEXTURE_2D,self.texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA32F,view.width,view.height,0,gl.GL_RGBA,gl.GL_FLOAT,None)
                gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER,gl.GL_NEAREST)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,self.depth)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER,gl.GL_DEPTH_COMPONENT24,view.width,view.height)
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,self.fbo)
                gl.glFramebufferTexture2D(gl.GL_DRAW_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,self.texture,0)
                gl.glFramebufferRenderbuffer(gl.GL_DRAW_FRAMEBUFFER,gl.GL_DEPTH_ATTACHMENT,gl.GL_RENDERBUFFER,self.depth)
                gl.glDrawBuffers(1,[gl.GL_COLOR_ATTACHMENT0])
                if gl.glCheckFramebufferStatus(gl.GL_DRAW_FRAMEBUFFER)!=gl.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError('Float label framebuffer incomplete')
            finally:
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,saved[0]);gl.glBindTexture(gl.GL_TEXTURE_2D,saved[1])
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,saved[2])
        finally:self.offscreen.make_uncurrent()

    def render(self,surface):
        gl=self.gl;view=self.view;camera=view.camera
        primitives=displayed_primitives(view,surface)
        rasterizer=camera._rasterizer;rasterizer.update_camera(camera)
        node=rasterizer._camera_nodes[camera.uid]
        projection=node.camera.get_projection_matrix(width=view.width,height=view.height)
        inverse=np.linalg.inv(view.scene.visualizer.context._scene.get_pose(node))
        self.offscreen.make_current()
        caps=(gl.GL_DEPTH_TEST,gl.GL_CULL_FACE,gl.GL_BLEND,gl.GL_MULTISAMPLE,gl.GL_SCISSOR_TEST,
              gl.GL_FRAMEBUFFER_SRGB,gl.GL_POLYGON_OFFSET_FILL)
        enabled={cap:bool(gl.glIsEnabled(cap)) for cap in caps}
        saved={key:gl.glGetIntegerv(enum) for key,enum in (
            ('draw',gl.GL_DRAW_FRAMEBUFFER_BINDING),('read',gl.GL_READ_FRAMEBUFFER_BINDING),
            ('program',gl.GL_CURRENT_PROGRAM),('vao',gl.GL_VERTEX_ARRAY_BINDING),('buffer',gl.GL_ARRAY_BUFFER_BINDING),
            ('viewport',gl.GL_VIEWPORT),('depth_func',gl.GL_DEPTH_FUNC),('cull',gl.GL_CULL_FACE_MODE),('front',gl.GL_FRONT_FACE))}
        depth_mask=gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK);color_mask=gl.glGetBooleanv(gl.GL_COLOR_WRITEMASK)
        clear_color=gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE);clear_depth=gl.glGetDoublev(gl.GL_DEPTH_CLEAR_VALUE)
        try:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,self.fbo);gl.glViewport(0,0,view.width,view.height)
            for cap in caps:gl.glDisable(cap)
            gl.glEnable(gl.GL_DEPTH_TEST);gl.glDepthFunc(gl.GL_LESS);gl.glDepthMask(True);gl.glColorMask(True,True,True,True)
            gl.glClearColor(0,0,0,0);gl.glClearDepth(1);gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
            gl.glFrontFace(gl.GL_CCW);gl.glCullFace(gl.GL_BACK)
            gl.glUseProgram(self.program);gl.glBindVertexArray(self.vao);gl.glBindBuffer(gl.GL_ARRAY_BUFFER,self.buffer)
            gl.glEnableVertexAttribArray(0);gl.glVertexAttribPointer(0,4,gl.GL_FLOAT,False,16,ctypes.c_void_p(0))
            loc=gl.glGetUniformLocation(self.program,'mvp')
            for vertices,pose,double_sided in primitives:
                (gl.glDisable if double_sided else gl.glEnable)(gl.GL_CULL_FACE)
                gl.glUniformMatrix4fv(loc,1,True,np.ascontiguousarray(projection@inverse@pose,dtype=np.float32))
                gl.glBufferData(gl.GL_ARRAY_BUFFER,vertices.nbytes,vertices,gl.GL_STREAM_DRAW)
                gl.glDrawArrays(gl.GL_TRIANGLES,0,len(vertices))
            raw=np.empty((view.height,view.width,4),np.float32)
            gl.glReadPixels(0,0,view.width,view.height,gl.GL_RGBA,gl.GL_FLOAT,raw)
            z=np.empty((view.height,view.width),np.float32)
            gl.glReadPixels(0,0,view.width,view.height,gl.GL_DEPTH_COMPONENT,gl.GL_FLOAT,z)
            near,far=camera.near,camera.far
            center_depth=2*near*far/(far+near-(2*z-1)*(far-near))
        finally:
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,int(saved['draw']));gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,int(saved['read']))
            gl.glUseProgram(int(saved['program']));gl.glBindVertexArray(int(saved['vao']));gl.glBindBuffer(gl.GL_ARRAY_BUFFER,int(saved['buffer']))
            gl.glViewport(*saved['viewport']);gl.glDepthFunc(int(saved['depth_func']));gl.glDepthMask(depth_mask)
            gl.glColorMask(*color_mask);gl.glClearColor(*clear_color);gl.glClearDepth(float(clear_depth))
            gl.glCullFace(int(saved['cull']));gl.glFrontFace(int(saved['front']))
            for cap,was_enabled in enabled.items():(gl.glEnable if was_enabled else gl.glDisable)(cap)
            self.offscreen.make_uncurrent()
        # raw is bottom-up GL; RGB lens_map samples the same coordinates.
        xy,_=cv2.convertMaps(*view.lens_map,cv2.CV_32FC2)
        warped=cv2.remap(raw,xy,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        triangle=np.rint(warped[...,3]).astype(np.int32)-1
        triangle[view.vignette[...,0]<=0]=-1
        depth=cv2.remap(center_depth,xy,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        depth[triangle<0]=0
        return triangle,warped[...,:3],depth

    def close(self):
        gl=self.gl;self.offscreen.make_current()
        try:
            gl.glDeleteProgram(self.program);gl.glDeleteVertexArrays(1,[self.vao]);gl.glDeleteBuffers(1,[self.buffer])
            gl.glDeleteFramebuffers(1,[self.fbo]);gl.glDeleteTextures([self.texture]);gl.glDeleteRenderbuffers(1,[self.depth])
        finally:self.offscreen.make_uncurrent()
