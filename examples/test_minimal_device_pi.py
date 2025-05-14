from invertsy.env.world import Seville2009
from invertsy.agent.agent import MinimalDeviceCentralComplexAgent
from invertsy.sim.minimal_device_simulation import MinimalDevicePathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation
from invertsy.sim.minimal_device_animation import MinimalDevicePathIntegrationAnimation


def main(*args):
    routes = Seville2009.load_routes(args[0], degrees=True)

    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    rt = rt[::-1]
    rt[:, 3] = (rt[:, 3] - 0) % 360 - 180
    use_nanowires = False
    use_dye = False
    sigmoid_bool = True
    agent = MinimalDeviceCentralComplexAgent(cx_params={"use_nanowires":use_nanowires,"sigmoid_bool":sigmoid_bool,"use_dye":use_dye})
    agent.step_size = .01
    sim = MinimalDevicePathIntegrationSimulation(rt, agent=agent, noise=0., name="pi-ant%d-route%d" % (ant_no, rt_no))
    ani = MinimalDevicePathIntegrationAnimation(sim, show_history=True)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import argparse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parser = argparse.ArgumentParser(
            description="Run a path integration test."
        )

        parser.add_argument("-i", dest='input', type=str, required=False, default=Seville2009.ROUTES_FILENAME,
                            help="File with the recorded routes.")

        p_args = parser.parse_args()

        main(p_args.input)
