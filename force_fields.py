import taichi as ti
import genesis as gs
from genesis.engine.force_fields import ForceField

@ti.data_oriented
class RandomObjectForce(ForceField):
    """
    Per particle force field with acceleration added to an object's particle

    Parameters:
    -----------
    strength: float
        The strength of the wind.
    """

    def __init__(self, entity, strength = 1.0, abs_tol=1e-2, force=[0,0,0]):
        super().__init__()
        self._strength = strength
        self._abs_tol = abs_tol
        self._entity = entity
        self._force = force

        self._particles = ti.Vector.field(3, dtype=ti.float32, shape=(10000))
        self._forces = ti.Vector.field(3, dtype=ti.float32, shape=(10000))
        # self._magnitudes = ti.field(dtype=ti.float32, shape=(10000))
        self._distances = ti.field(ti.float32, shape=(10000, ))

    @ti.kernel
    #def set_particles_and_forces(self,particles: ti.types.ndarray(), forces: ti.types.ndarray(), magnitudes: ti.types.ndarray()):
    def set_particles_and_forces(self,particles: ti.types.ndarray(), forces: ti.types.ndarray()):
        for i in range(particles.shape[0]):
        # for p in particles:
            self._particles[i][0] = particles[i,0]
            self._particles[i][1] = particles[i,1]
            self._particles[i][2] = particles[i,2]

            self._forces[i][0] = forces[i,0]
            self._forces[i][1] = forces[i,1]
            self._forces[i][2] = forces[i,2]

            # self._magnitudes[i] = magnitudes[i]
        
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        acc = ti.Vector.zero(gs.ti_float,3)

        # get minimum distance from pos and entity particles 
        min_distance = 1000000.
        particle_idx = -1
        for j in range(self._particles.shape[0]):
            self._distances[j] = ti.math.distance(self._particles[j],pos)

            if min_distance > self._distances[j]:
                particle_idx = j
                min_distance = self._distances[j]
        
        # get random acceleration
        if min_distance < self._abs_tol:
            noise = (ti.Vector([self._forces[particle_idx][0],self._forces[particle_idx][1],self._forces[particle_idx][2]],dt=gs.ti_float))
            #noise = (ti.Vector([0,0,1],dt=gs.ti_float))
            # noise = (ti.Vector([self._force[0],self._force[1],self._force[2]],dt=gs.ti_float))
            # noise = (
            #     ti.Vector(
            #         [
            #             ti.random(gs.ti_float),
            #             ti.random(gs.ti_float),
            #             ti.random(gs.ti_float),
            #         ],
            #         dt=gs.ti_float,
            #     )
            #     * 2
            #     - 1
            # )

            # strength = 222222.22 * (self._magnitudes[particle_idx])**2
            # acc = noise * strength
            #acc = noise * self._strength
            #acc = noise * self._strength * self._magnitudes[particle_idx]
            # acc = noise * self._strength
            acc = noise

        return acc

    @property
    def strength(self):
        return self._strength