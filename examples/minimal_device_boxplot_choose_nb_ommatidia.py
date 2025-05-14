from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertsy.env.sky import Sky

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def main(POL_method,fov,nb_ommatidia,omm_photoreceptor_angle,sun_elevation_degrees,axs,elevation_idx):
    # Create the polarization sensor
    sensor = MinimalDevicePolarisationSensor(
                POL_method=POL_method,
                field_of_view=fov, nb_lenses=nb_ommatidia,
                omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
                omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                omm_pol_op=1., noise=0
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

    # _______________________________________________________________________________________f
    # Predict compass direction / angle based on POL neuron responses
    # _______________________________________________________________________________________
    """
    Set the ommatidia preference angles to correspond to the definition of solar azimuth
    (0 degrees towards left, then positively increase clockwise towards 180 
    and negatively increase anti-clockwise up to -180.
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

    errors = np.abs((np.rad2deg(predicted_angle) - np.rad2deg(sun_azimuths) + 180) % 360 - 180)
    return errors

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
                              "double_normalized_contrast_flipped":"(I0-I90)\u00F7(I0+I90)",
                              "double_normalized_contrast": "(I90-I0)\u00F7(I0+I90)"}
    luminance_description = {False: 'L', True: 'ct'}
    n_luminance_options = len(list(luminance_description.keys()))
    POL_methods = ["single_0"]  # choose from POL_method_description keys
    fov = 30
    omm_photoreceptor_angle = 2
    noises = [0]
    sun_elevations_degrees = [15,30,45,60,75]
    nbs_ommatidia = [3,6]
    POL_method = "single_0"
    log = True

    results = []
    fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
    #fig.text(0.04, 0.5, 'Error (Degrees)', va='center', rotation='vertical', fontsize=15)
    fig.suptitle('Compass prediction error for complete skylight',fontsize=20)
    all_errors = []
    for pol_idx, POL_method in enumerate(POL_methods): # each value gives two rows of results
            errors_per_row = []
            for elevation_idx, sun_elevation_degrees in enumerate(sun_elevations_degrees): # each value gives one column of results
                errors_per_elevation = []
                for nb_ommatidia in nbs_ommatidia:
                    errors = main(POL_method,fov,nb_ommatidia,omm_photoreceptor_angle,sun_elevation_degrees,axs,elevation_idx)
                    errors_per_elevation.append(errors)
                errors_per_row.append(errors_per_elevation)
                axs[elevation_idx].boxplot(errors_per_elevation, positions=np.arange(1, len(nbs_ommatidia)+1), widths=0.6, labels=[3,6])
                axs[elevation_idx].set_title(f"Sun elevation {sun_elevations_degrees[elevation_idx]}",fontsize=15)
            all_errors.append(errors_per_row)

    axs[int(len(sun_elevations_degrees)/2)].set_xlabel('Number of ommatidia', fontsize=15)
    axs[0].set_ylabel('Errors (degrees)', fontsize=15)
    plt.subplots_adjust(hspace=1.5)

    save_folder = f"..//data//results_minimal_device//"
    if log:
        plt.yscale('log')
        plt.savefig(save_folder + "boxplots_choose_nb_ommatidia_log_scale.png")
    else:
        plt.savefig(save_folder + "boxplots_choose_nb_ommatidia_linear_scale.png")
    plt.show()

    # save all array numbers for later analysis
    all_errors = np.array(all_errors)
    all_errors = np.median(all_errors, axis=3)
    np.save(save_folder + "median compass prediction errors.npy", all_errors)

