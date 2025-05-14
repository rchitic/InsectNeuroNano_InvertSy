from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertsy.env.sky import Sky, UniformLuminanceSky
from invertsy.sim._helpers import create_dra_axis_minimal_device

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def main(POL_method,POL_method_description,fov,nb_ommatidia,omm_photoreceptor_angle,noise,sun_elevation_degrees):
    # Create the polarization sensor
    sensor = MinimalDevicePolarisationSensor(
                POL_method=POL_method,
                field_of_view=fov, nb_lenses=nb_ommatidia,
                omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
                omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                omm_pol_op=1., noise=noise
            )
    print(sensor)
    print("ommatidia rotations:\n",sensor.omm_ori.as_euler('ZYX', degrees=True))
    print("ommatidia xyz:\n",sensor.omm_xyz)

    # Get polarization sensor responses of all ommatidia
    # for multiple combinations of solar azimuth and elevation
    sun_azimuths = np.arange(-np.pi+np.pi/180, np.pi+np.pi/180, np.pi/180)
    sun_elevation = np.deg2rad(90 - sun_elevation_degrees) #np.arange(0, np.pi/2, np.pi/2/90)

    # Create all_responses array in the following shape:
    # (# sun azimuths, # ommatidia)
    all_responses = []
    for sun_azimuth in sun_azimuths:
        sky = Sky(sun_elevation, sun_azimuth)
        response = sensor(sky=sky)
        all_responses.append(response)
    all_responses = np.squeeze(np.array(all_responses))

    # Plot responses
    show_heatmaps = False
    if show_heatmaps:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(15, 5))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["P"+str(i) for i in range(1, nb_ommatidia+1)]

    # Plot: variation of POL neuron responses with solar azimuth
    for idx in range(3):
        ax1.plot(all_responses[:, idx], sun_azimuths*180/np.pi, label=labels[idx])
    ax1.set_ylabel('Solar azimuth (deg)',fontsize=13)
    ax1.set_xlabel('POL neuron response',fontsize=13)
    ax1.set_title(f'POL neuron response vs. azimuth for {sun_elevation_degrees} deg elevation',fontsize=15)
    ax1.legend(loc='upper left')

    # Visualize the positions of the 3 POL neurons separated by 120 degrees
    create_dra_axis_minimal_device(sensor, draw_axis=True, ax=ax2, flip=False)
    plt.subplots_adjust(wspace=0.2)

    # _______________________________________________________________________________________
    # Predict compass direction / angle based on POL neuron responses
    # _______________________________________________________________________________________
    """
    Set the ommatidia preference angles to correspond to the definition of solar azimuth
    (0 degrees towards left, then positively increase clockwise towards 180 
    and negatively increase counter-clockwise up to -180.
    """

    thetas = [-120, 0, 120]
    thetas = [np.deg2rad(el) for el in thetas]

    # Create complex vectors for the POL-neurons following v=r*e^(-i*theta).
    vectors = []
    for idx in range(3):
       vectors.append(all_responses[:, idx] * np.exp(1j * thetas[idx]))

    # Get vector representing the summed response of the POL neurons
    summed_vector = np.array(vectors).sum(axis=0)
    # Get the direction of the final compass vector
    predicted_angle = np.angle(summed_vector)

    """
    Since +180 and -180 represent the same direction, 
    sometimes the prediction is reversed, which disturbs the graph,
    so make sure that the first prediction is negative (for -180)
    and the last is positive (for +180). 
    """
    if round(predicted_angle[-1]/np.pi*180) == -180:
        predicted_angle[-1] = -predicted_angle[-1]
    if round(predicted_angle[0]/np.pi*180) == 180:
        predicted_angle[0] = -predicted_angle[0]

    # Plot predicted angle vs. real azimuth
    ax3.plot(sun_azimuths/np.pi*180, predicted_angle/np.pi*180, c='black')
    ax3.set_xticks([-179, -120, -60, 0, 60, 120, 180])
    ax3.set_yticks([-179, -120, -60, 0, 60, 120, 180])
    # Mark points where match between real and predicted azimuth is perfect
    # (corresponding to the peaks of the activity bumps - clear direction)
    plt.scatter(0,0,color='red')
    plt.scatter(-60, -60, color='red')
    plt.scatter(60, 60, color='red')
    plt.scatter(-120,-120,color='red')
    plt.scatter(120,120,color='red')
    plt.scatter(180,180,color='red')
    ax3.set_xlabel('Real solar azimuth',fontsize=13)
    ax3.set_ylabel('Predicted solar azimuth',fontsize=13)
    ax3.set_title('Predicted vs. real solar azimuth',fontsize=15)

    save_folder = f"..//data//results_minimal_device//POL_method_variation//complete_skylight//{nb_ommatidia}_ommatidia//POL{POL_method_description[POL_method]}_elevation{sun_elevation_degrees}_noise{noise}.png"
    plt.savefig(save_folder)

if __name__ == '__main__':
    POL_method_description = {"experimental":"exp",
                              "double_multiply":"I0xI90",
                              "single_0": "I0",
                              "single_90": "I90",
                              "double_sum":"I0+I90",
                              "double_subtraction_flipped":"I0-I90",
                              "double_subtraction":"I90-I0",
                              "double_subtraction_abs":"abs(I0-I90)",
                              "double_sqrt": "sqrt(I0^2+I90^2)",
                              "double_normalized_contrast_flipped":"(I0-I90)\u00F7(I90+I0)",
                              "double_normalized_contrast": "(I90-I0)\u00F7(I90+I0)"}
    POL_methods = ["single_0"] #["single_0","single_90","double_sum","double_subtraction","double_subtraction_flipped","double_normalized_contrast","double_normalized_contrast_flipped"]  # choose from POL_method_description keys
    nbs_ommatidia = [3,6]
    fov = 30
    omm_photoreceptor_angle = 2
    noises = [0]
    sun_elevations_degrees = [15,30,45,60,75]

    for nb_ommatidia in nbs_ommatidia:
        for POL_method in POL_methods:
            for noise in noises:
                for sun_elevation_degrees in sun_elevations_degrees:
                    main(POL_method,POL_method_description,fov,nb_ommatidia,omm_photoreceptor_angle,noise,sun_elevation_degrees)


