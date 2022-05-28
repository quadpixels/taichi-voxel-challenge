# -#- coding: utf-8 -*-
# +X：屏幕右方
# +Y：屏幕里面
# +Z：屏幕上方

# Stolen from:
# https://www.shadertoy.com/view/tltGW8

import numpy as np
from time import time
import taichi as ti
from scipy.spatial.transform import Rotation as R

res = (640, 360)
AlmostZero = 0.001

@ti.func
def LocalPointToGlobalPoint(o, p):
  return o*p;
  
@ti.func
def GlobalPointToLocalPoint(o, pos, p_g):
  return o.transpose() @ (p_g - pos)

cam_orientation = R.from_rotvec(rotvec=[0,0,0]).as_matrix()
cam_pos = ti.Vector([0,0,0])

ti.init(ti.cpu)
pixels = ti.Vector.field(3, dtype=float, shape=res)

@ti.func
def Sphere(local, radius):
  return local.norm() - radius
  
@ti.func
def Union(lhs, rhs):
  return ti.min(lhs, rhs)

@ti.func
def Subtract(lhs, rhs):
  return ti.max(lhs, -rhs)

@ti.func
def Gradient(p):
  dist = SceneSDF(p)
  return ti.Vector([
      SceneSDF(p+ti.Vector([AlmostZero, 0, 0])) - dist,
      SceneSDF(p+ti.Vector([0, AlmostZero, 0])) - dist,
      SceneSDF(p+ti.Vector([0, 0, AlmostZero])) - dist]).normalized();

@ti.func
def SceneSDF(p):
  o = ti.Matrix([[1,0,0],[0,1,0],[0,0,1]])
  s = Sphere(GlobalPointToLocalPoint(o, ti.Vector([0,10,0]), p), 1)
  s = Subtract(s, Sphere(GlobalPointToLocalPoint(o, ti.Vector([-1, 9.5, 0]), p), 1))
  s = Union(s, Sphere(GlobalPointToLocalPoint(o, ti.Vector([-2, 10, 0]), p), 1))
  s = Union(s, Sphere(GlobalPointToLocalPoint(o, ti.Vector([2, 10, 0]), p), 1))
  return s

@ti.func
def GetRayDir(xy, fovx):
  aspect = res[1] / res[0]
  ndc = ti.Vector([xy[0]/res[0]*2.0 - 1.0, xy[1]/res[1]*2.0 - 1.0])
  angles = ti.Vector([ndc[0]*0.5*fovx, ndc[1]*0.5*fovx*aspect])
  rd = ti.Vector([0.0, 0.0, 0.0])
  rd.x = ti.sin(angles.x)
  rd.z = ti.sin(angles.y)
  rd.y = ti.sqrt(1-rd.x*rd.x - rd.z*rd.z)
  return rd
  #return ti.Vector([ndc[0], 0, ndc[1]])

@ti.func
def raymarch(rd, travel_start, travel_end, cam_pos):
  traveled = travel_start
  ret = ti.Vector([0.0, 0.0, 0.0, 0.0])
  hit = False
  for i in range(0, 100):
    pos = rd * traveled + cam_pos
    dist = SceneSDF(pos)
    traveled += dist
    #traveled += 0.2
    hit = (dist < AlmostZero)
    if hit or traveled >= travel_end:
      if hit:
        ret[1] = pos[0]
        ret[2] = pos[1]
        ret[3] = pos[2]
      break
  ret[0] = float(hit)
  return ret

@ti.kernel
def paint(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32):
  cam_pos = [cam_x, cam_y, cam_z]
  light_dir = (ti.Vector([1,1,-1])*-1).normalized()
  for i, j in pixels:
    rd = GetRayDir(ti.Vector([i, j]), 45 * 3.14159 / 180)
    pixels[i, j] = [rd.x, 0, rd.z]
    if True:
      ret = raymarch(rd, 0.1, 1000, cam_pos)
      if ret[0] == 1:
        p = ret[1:4]
        #pixels[i, j] = ti.Vector([1,1,0])
        pixels[i, j] = ti.Vector([.5, .5, 0]) + \
                       ti.Vector([1,1,1]) * ti.pow(Gradient(p).dot(light_dir), 1)
      else:
        pixels[i, j] = ti.Vector([0, 0, 0])

if True:
  last_millis = int(time() * 1000)
  gui = ti.GUI('UV', res)
  while not gui.get_event(ti.GUI.ESCAPE):
    ms = int(time() * 1000)
    delta_ms = ms - last_millis
    last_millis = ms
    if gui.is_pressed('w'):
      cam_pos += ti.Vector(cam_orientation[1]) * delta_ms / 1000.0
    if gui.is_pressed('s'):
      cam_pos -= ti.Vector(cam_orientation[1]) * delta_ms / 1000.0
    if gui.is_pressed('a'):
      cam_pos -= ti.Vector(cam_orientation[0]) * delta_ms / 1000.0
    if gui.is_pressed('d'):
      cam_pos += ti.Vector(cam_orientation[0]) * delta_ms / 1000.0
    if gui.is_pressed('q'):
      cam_pos -= ti.Vector(cam_orientation[2]) * delta_ms / 1000.0
    if gui.is_pressed('e'):
      cam_pos += ti.Vector(cam_orientation[2]) * delta_ms / 1000.0
    paint(cam_pos.x, cam_pos.y, cam_pos.z)
    gui.set_image(pixels)
    gui.show()