import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
from force_fields import RandomObjectForce
import taichi as ti

########################## init ##########################
gs.init(backend=gs.cpu)
########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 4e-3,
        substeps = 10,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound   = (-0.5, -1.0, 0.0),
        upper_bound   = (0.5, 1.1, 1.1),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary = True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
    ),
    show_viewer = True,
)

########################## entities ##########################

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

box = scene.add_entity(
   #material=gs.materials.MPM.ElastoPlastic(),
   morph=gs.morphs.Box(
       pos  = (0.2, -0.3, 0.2),
       size = (0.2, 0.2, 0.2),
   ),
   surface=gs.surfaces.Default(
       color    = (1.0, 0.4, 0.4),
       vis_mode = 'visual',
   ),
)

car = scene.add_entity(
    material=gs.materials.MPM.Elastic(),
    morph=gs.morphs.Mesh(
        file='meshes/car.obj',
        scale = 1.0,
        pos  = (0.2, 0, 0.06),
        euler=(0.0, 0, 0.0)
    ),
    surface=gs.surfaces.Default(
        color    = (1.0, 0.2, 0.2),
        vis_mode = 'particle',
    ),
)

car_force_b = scene.add_force_field(RandomObjectForce(car, strength=100, force = [0,1,0] ))
car_force_b.activate()
car_force_f = scene.add_force_field(RandomObjectForce(car, strength=100, force = [0,-1,0] ))

########################## build ##########################
scene.build()

horizon = 105
for i in range(horizon):

    if i == 25:
        car_force_b.deactivate()
        car_force_f.activate()

    p = car.get_particles()
    car_force_b.set_particles(p)
    car_force_f.set_particles(p)
    scene.step()