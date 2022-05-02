# -*- coding: utf-8 -*-
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0.01, exposure=1.4)
scene.set_floor(-0.05, (1, 0.6, 0.4))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((-1, 1, 0.3), 0.0, (1,1,1))

@ti.func
def sphere(o, r1, r2, c, ylim):
    for i, j, k in ti.ndrange((o.x-r1, o.x+r1), (o.y-r1, o.y+r1), (o.z-r1, o.z+r1)):
        ox = vec3(i-o.x, j-o.y, k-o.z).norm()
        if ox >= r2 and ox <= r1 and j < ylim:
            scene.set_voxel(vec3(i, j, k), 1, c);

@ti.func
def line(p0, p1, c):
    if p0.x > p1.x: p0,p1 = p1,p0
    step = (p1-p0).normalized()
    while p0.x < p1.x:
        scene.set_voxel(p0, 1, c); p0 += step
@ti.func
def manhattan_dist(p0, p1):
    return abs(p0.x - p1.x) + abs(p0.y - p1.y) + abs(p0.z - p1.z)
@ti.func
def box(p0, p1, c, rounding, excl_bb_lb=vec3(-999,-999,-999), excl_bb_ub=vec3(-999,-999,-999)):
    for i, j, k, in ti.ndrange((p0.x,p1.x+1), (p0.y,p1.y+1), (p0.z,p1.z+1)):
        if manhattan_dist(vec3(i,j,k), vec3(p0.x,p0.y,p0.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p0.x,p0.y,p1.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p0.x,p1.y,p0.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p0.x,p1.y,p1.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p1.x,p0.y,p0.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p1.x,p0.y,p1.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p1.x,p1.y,p0.z)) >= rounding and \
           manhattan_dist(vec3(i,j,k), vec3(p1.x,p1.y,p1.z)) >= rounding:
            if i >= excl_bb_lb.x and i <= excl_bb_ub.x and \
                 j >= excl_bb_lb.y and j <= excl_bb_ub.y and \
                 k >= excl_bb_lb.z and k <= excl_bb_ub.z: continue
            scene.set_voxel(vec3(i, j, k), 1, c)
@ti.kernel
def initialize_voxels(): # MagicaVoxel：xyz；这里：x,z,-y
    sphere(vec3(40, 16, -27), 15, 13, vec3(0.866, 0.866, 0.866), 16) # 碗
    sphere(vec3(40, 14, -27), 13, 0,  vec3(1, 0.8, 0.6), 14) # 汤
    line(vec3(36, 17, -48), vec3(56, 17, -17), vec3(0.8, 0.4, 0.4)) # 筷子
    line(vec3(38, 17, -48), vec3(59, 17, -19), vec3(0.8, 0.4, 0.4)) # 筷子
    for x in range(-2, 0):  # 面条
        line(vec3(40-x*3, 14, -40-x), vec3(29, 14, -35-x*2), vec3(1, 1, 0.9))
    for layer in range(0, 2):
        for x in range(24-layer*14):
            c = vec3(0, 0.8, 0)
            if ti.randn() < 0.4 + layer*0.8: c = vec3(0.3, 0.1, 0)
            theta = ti.randn() * 2 * 3.14159; r = ti.randn() * (8-layer*4);
            theta2 = ti.randn() * 2 * 3.14159;
            p0 = vec3(40+r*ti.cos(theta), 14+layer, -27+r*ti.sin(theta))
            line(p0, p0+vec3(2*ti.cos(theta2),0,2*ti.sin(theta2)), c)
    box(vec3(37, 10, -17), vec3(46, 17, -17), vec3(0.9, 0.9, 0.9), 2) # 勺子
    box(vec3(38, 10, -16), vec3(45, 16, -16), vec3(0.9, 0.9, 0.9), 2)
    line(vec3(37, 16, -17), vec3(21, 20, -32), vec3(0.9, 0.9, 0.9));
    line(vec3(37, 15, -17), vec3(21, 19, -32), vec3(0.9, 0.9, 0.9));
    box(vec3(27, 6, -62), vec3(52, 6, -48), vec3(0.15, 0.15, 0.15), 2,\
        vec3(29, 6, -60), vec3(50, 6, -50)); # 菜碟
    box(vec3(28, 2, -61), vec3(51, 6, -49), vec3(0.15, 0.15, 0.15), 2,\
        vec3(29, 3, -60), vec3(50, 6, -50));
    box(vec3(39, 2, -60), vec3(39, 6, -50), vec3(0.15, 0.15, 0.15), 0) #
    box(vec3(40, 3, -59), vec3(49, 3, -51), vec3(1, 0.8, 0.6), 1) # 小菜里的汤
    box(vec3(44, 4, -58), vec3(47, 4, -54), vec3(1, 0.8, 0.4), 0) # 菜
    box(vec3(44, 4, -53), vec3(47, 4, -50), vec3(1, 0.8, 0.4), 0) # 菜
    box(vec3(30, 3, -52), vec3(31, 4, -51), vec3(1, 0.4, 0), 0) # 花生米
    box(vec3(35, 3, -55), vec3(36, 4, -54), vec3(1, 0.4, 0), 0)
    box(vec3(32, 3, -53), vec3(34, 4, -52), vec3(1, 0.8, 0.2), 0)
    box(vec3(31, 3, -55), vec3(32, 4, -54), vec3(1, 0.4, 0), 0)
    
    box(vec3(0, -1, -64), vec3(64, 1, 0), vec3(1, 0.6, 0.4), 4)

initialize_voxels()
scene.finish()
