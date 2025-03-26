import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
from force_fields import RandomObjectForce
import taichi as ti
import numpy as np

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

elastic_box = scene.add_entity(
   material=gs.materials.MPM.ElastoPlastic(),
   morph=gs.morphs.Box(
       pos  = (0.0, -0.15, 0.2),
       size = (0.1, 0.1, 0.1),
   ),
   surface=gs.surfaces.Default(
       color    = (0.2, 1.0, 0.2),
       vis_mode = 'particle',
   ),
)

fluid_box = scene.add_entity(
   material=gs.materials.MPM.Liquid(),
   morph=gs.morphs.Box(
       pos  = (0.0, 0.15, 0.2),
       size = (0.1, 0.1, 0.1),
   ),
   surface=gs.surfaces.Default(
       color    = (1.0, 0.2, 0.2),
       vis_mode = 'particle',
   ),
)

# fluid_box_force = scene.add_force_field(RandomObjectForce(fluid_box, strength=100000))
fluid_box_force = scene.add_force_field(RandomObjectForce(fluid_box, strength=2000))
fluid_box_force.activate()

########################## build ##########################
scene.build()

horizon = 105
for i in range(horizon):



    e_particles = elastic_box.get_particles()
    e_particles[:,1] += 0.3
    f_particles = fluid_box.get_particles()

    directions = e_particles - f_particles
    magnitudes = np.linalg.norm(directions, axis=1)

    #normalized_directions = (directions) / np.linalg.norm(directions, axis=1, keepdims=True) 

    #fluid_box_force.set_particles_and_forces(f_particles, directions, magnitudes)
    fluid_box_force.set_particles_and_forces(f_particles, directions)

    scene.step()