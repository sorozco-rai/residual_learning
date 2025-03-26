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

    def __init__(self, entity, abs_tol=1e-2):
        super().__init__()
        self._abs_tol = abs_tol
        self._entity = entity

        self._particles = ti.Vector.field(3, dtype=ti.float32, shape=(10000))
        self._forces = ti.Vector.field(3, dtype=ti.float32, shape=(10000))
        self._distances = ti.field(ti.float32, shape=(10000, ))

    @ti.kernel
    def set_particles_and_forces(self,particles: ti.types.ndarray(), forces: ti.types.ndarray()):
        for i in range(particles.shape[0]):
            self._particles[i][0] = particles[i,0]
            self._particles[i][1] = particles[i,1]
            self._particles[i][2] = particles[i,2]

            self._forces[i][0] = forces[i,0]
            self._forces[i][1] = forces[i,1]
            self._forces[i][2] = forces[i,2]
        
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
            acc = (ti.Vector([self._forces[particle_idx][0],self._forces[particle_idx][1],self._forces[particle_idx][2]],dt=gs.ti_float))

        return acc

    @property
    def strength(self):
        return self._strength