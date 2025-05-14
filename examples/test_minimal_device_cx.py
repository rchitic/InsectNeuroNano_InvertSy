# # ! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.brain.centralcomplex.minimal_device import MinimalDeviceCX
from invertsy.env.sky import Sky
from invertpy.sense.polarisation import MinimalDevicePolarisationSensor

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def main(*args):
    # create sky instance
    sun_elevation = np.deg2rad(45)
    sun_azimuth = np.deg2rad(0)
    sky = Sky(sun_elevation, sun_azimuth)

    # create polarization sensor
    POL_method = "single_0"
    fov = 60
    nb_ommatidia = 6
    omm_photoreceptor_angle = 2
    sensor = MinimalDevicePolarisationSensor(
        POL_method=POL_method,
        field_of_view=fov, nb_lenses=nb_ommatidia,
        omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
        omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
        omm_pol_op=1., noise=0.
    )

    # create central complex compass + integrator
    memory_update = True
    cx = MinimalDeviceCX(update=memory_update)
    print(cx)

    #_____________________________________________________________________________________________________________________________________________
    # Do full circle on the right side
    pred_angles=[]
    mem_angles=[]
    yaws, steering_responses = [], []
    for i in np.linspace(0, 360, 37)[:-1]:
        for _ in range(10):
            sky = Sky(sun_elevation, sun_azimuth)
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
            vectors1,vectors2=[],[]
            thetas_3 = [-120, 0, 120]
            thetas_3 = [np.deg2rad(el) for el in thetas_3]

            for idx in range(3):
                vectors1.append(POL_direction[idx] * np.exp(-1j * thetas_3[idx]))
                vectors2.append(cx.memory.r_memory[idx] * np.exp(-1j * thetas_3[idx]))
            summed_vector1 = np.array(vectors1).sum()
            predicted_angle1 = np.angle(summed_vector1)
            summed_vector2 = np.array(vectors2).sum()
            predicted_angle2 = np.angle(summed_vector2)
            pred_angles.append(np.rad2deg(predicted_angle1))
            mem_angles.append(np.rad2deg(predicted_angle2))

        sensor.rotate(R.from_euler('ZYX', [10, 0, 0], degrees=True))

    # Do full circle on the left side
    for i in np.linspace(0, 360, 37)[:-1]:
        print(i)
        for _ in range(10):
            sky = Sky(sun_elevation, sun_azimuth)
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
            vectors1,vectors2=[],[]
            thetas_3 = [-120, 0, 120]
            thetas_3 = [np.deg2rad(el) for el in thetas_3]

            for idx in range(3):
                vectors1.append(POL_direction[idx] * np.exp(-1j * thetas_3[idx]))
                vectors2.append(cx.memory.r_memory[idx] * np.exp(-1j * thetas_3[idx]))
            summed_vector1 = np.array(vectors1).sum()
            predicted_angle1 = np.angle(summed_vector1)
            summed_vector2 = np.array(vectors2).sum()
            predicted_angle2 = np.angle(summed_vector2)
            pred_angles.append(np.rad2deg(predicted_angle1))
            mem_angles.append(np.rad2deg(predicted_angle2))

        sensor.rotate(R.from_euler('ZYX', [-10, 0, 0], degrees=True))

    #_____________________________________________________________________________________________________________________________________________
    # Visualize results
    pred_angles=np.array(pred_angles)
    pred_angles = (360 + np.round(pred_angles)) % 360
    mem_angles = np.array(mem_angles) % 360

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
    ax1.scatter(range(len(pred_angles)),pred_angles,c='blue',s=5,label='current direction')
    ax1.plot(range(len(mem_angles)),mem_angles,c='orange',label='memory direction')

    steering_responses = np.array(steering_responses)
    ax2 = ax1.twinx()
    ax2.plot(range(len(pred_angles)),steering_responses[:,1] - steering_responses[:,0],c='green',label='right-left steering response')
    ax2.axhline(0, color='green', linestyle='--', linewidth=1)

    xticks=[90,180,270,360,450,540,630,719]
    labels=[]
    for idx in xticks:
        ax1.scatter(idx,pred_angles[idx],s=15,c='red')
        ax1.scatter(idx,mem_angles[idx],s=15,c='red')
        ax2.scatter(idx,steering_responses[idx,1] - steering_responses[idx,0],s=15,c='red')
        plt.axvline(idx,c='red')
        labels.append(f"h={int(round(pred_angles[idx]))}, m={int(round(mem_angles[idx]))}")

    plt.xticks(xticks,labels,fontsize=5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Angle (degrees)',c='blue',fontsize=10)
    ax2.set_ylabel('Steering magnitude',c='green',fontsize=10)
    plt.title('Comparison of right-left steering response with heading and memory direction angles')
    ax1.legend(loc=(0,0.85))
    ax2.legend(loc=(0,0.75))

    save_folder = f"..//data//results_minimal_device//"
    plt.savefig(save_folder+f"SteeringResponse_CirclePath_MemoryUpdate{memory_update}_Compass{nb_ommatidia}Neurons.png")
    plt.show()

if __name__ == '__main__':
    main(*sys.argv)
