from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertsy.env.sky import Sky
from invertsy.sim._helpers import create_dra_axis

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

sun_elevation = np.deg2rad(30)
sun_azimuth = -np.pi/2
sky = Sky(sun_elevation, sun_azimuth)

fov = 56
nb_ommatidia = 3
omm_photoreceptor_angle = 1
sensor = MinimalDevicePolarisationSensor(
            field_of_view=fov, nb_lenses=nb_ommatidia,
            omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
            omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
            omm_pol_op=1., noise=0.
        )

print(sensor)
print(sensor.omm_xyz)
r = sensor(sky=sky)
print(r)

plt.figure("env", figsize=(3.33, 3.33))
pol = create_dra_axis(sensor, draw_axis=True, flip=False)
pol.set_array(np.array(r).flatten())
plt.show()
