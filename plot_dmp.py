import matplotlib.pyplot as plt
import numpy as np


def plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd, result_y=np.array([0]), result_yd=np.array([0]), filename="test.pdf", space="cartesian"):
    plt.figure(figsize=(12, 16))
    if space == "cartesian":
        plt.suptitle(r"Cartesian trajectory with $N = 15, \tau = 15$")
        ax1 = plt.subplot(7,2,1)
        ax1.set_xlabel(r"$t [s]$")
        ax1.set_ylabel(r"$x [m]$")

        ax2 = plt.subplot(7,2,3)
        ax2.set_xlabel(r"$t [s]$")
        ax2.set_ylabel(r"$y [m]$")

        ax3 = plt.subplot(7,2,5)
        ax3.set_xlabel(r"$t [s]$")
        ax3.set_ylabel(r"$z [m]$")

        ax4 = plt.subplot(7,2,7)
        ax4.set_xlabel(r"$t [s]$")
        ax4.set_ylabel(r"$q_x$")

        ax5 = plt.subplot(7,2,9)
        ax5.set_xlabel(r"$t [s]$")
        ax5.set_ylabel(r"$q_y$")

        ax6 = plt.subplot(7,2,11)
        ax6.set_xlabel(r"$t [s]$")
        ax6.set_ylabel(r"$q_z$")

        ax7 = plt.subplot(7,2,13)
        ax7.set_xlabel(r"$t [s]$")
        ax7.set_ylabel(r"$q_w$")

        ax8 = plt.subplot(7,2,2)
        ax8.set_xlabel(r"$t [s]$")
        ax8.set_ylabel(r"$\dot{x} [\frac{m}{s}]$")

        ax9 = plt.subplot(7,2,4)
        ax9.set_xlabel(r"$t [s]$")
        ax9.set_ylabel(r"$\dot{y} [\frac{m}{s}]$")

        ax10 = plt.subplot(7,2,6)
        ax10.set_xlabel(r"$t [s]$")
        ax10.set_ylabel(r"$\dot{z} [\frac{m}{s}]$")

        ax11 = plt.subplot(7,2,8)
        ax11.set_xlabel(r"$t [s]$")
        ax11.set_ylabel(r"$\dot{q}_x$")

        ax12 = plt.subplot(7,2,10)
        ax12.set_xlabel(r"$t [s]$")
        ax12.set_ylabel(r"$\dot{q}_y$")

        ax13 = plt.subplot(7,2,12)
        ax13.set_xlabel(r"$t [s]$")
        ax13.set_ylabel(r"$\dot{q}_z$")

        ax14 = plt.subplot(7,2,14)
        ax14.set_xlabel(r"$t [s]$")
        ax14.set_ylabel(r"$\dot{q}_w$")

    if space == "joint":
        plt.suptitle(r"Joint space trajectory with $N = 15, \tau = 25$")
        ax1 = plt.subplot(7,2,1)
        # ax1.set_title("Dimension 1")
        ax1.set_xlabel(r"$t [s]$")
        ax1.set_ylabel(r"$q_1 [rad]$")

        ax2 = plt.subplot(7,2,3)
        # ax2.set_title("Dimension 2")
        ax2.set_xlabel(r"$t [s]$")
        ax2.set_ylabel(r"$q_2 [rad]$")

        ax3 = plt.subplot(7,2,5)
        # ax3.set_title("Dimension 3")
        ax3.set_xlabel(r"$t [s]$")
        ax3.set_ylabel(r"$q_3 [rad]$")

        ax4 = plt.subplot(7,2,7)
        # ax4.set_title("Dimension 4")
        ax4.set_xlabel(r"$t [s]$")
        ax4.set_ylabel(r"$q_4 [rad]$")

        ax5 = plt.subplot(7,2,9)
        # ax5.set_title("Dimension 5")
        ax5.set_xlabel(r"$t [s]$")
        ax5.set_ylabel(r"$q_5 [rad]$")

        ax6 = plt.subplot(7,2,11)
        # ax6.set_title("Dimension 6")
        ax6.set_xlabel(r"$t [s]$")
        ax6.set_ylabel(r"$q_6 [rad]$")

        ax7 = plt.subplot(7,2,13)
        # ax7.set_title("Dimension 7")
        ax7.set_xlabel(r"$t [s]$")
        ax7.set_ylabel(r"$q_7 [rad]$")

        ax8 = plt.subplot(7,2,2)
        ax8.set_xlabel(r"$t [s]$")
        ax8.set_ylabel(r"$\dot{q}_1 [\frac{rad}{s}]$")

        ax9 = plt.subplot(7,2,4)
        ax9.set_xlabel(r"$t [s]$")
        ax9.set_ylabel(r"$\dot{q}_2 [\frac{rad}{s}]$")

        ax10 = plt.subplot(7,2,6)
        ax10.set_xlabel(r"$t [s]$")
        ax10.set_ylabel(r"$\dot{q}_3 [\frac{rad}{s}]$")

        ax11 = plt.subplot(7,2,8)
        ax11.set_xlabel(r"$t [s]$")
        ax11.set_ylabel(r"$\dot{q}_4 [\frac{rad}{s}]$")

        ax12 = plt.subplot(7,2,10)
        ax12.set_xlabel(r"$t [s]$")
        ax12.set_ylabel(r"$\dot{q}_5 [\frac{rad}{s}]$")

        ax13 = plt.subplot(7,2,12)
        ax13.set_xlabel(r"$t [s]$")
        ax13.set_ylabel(r"$\dot{q}_6 [\frac{rad}{s}]$")

        ax14 = plt.subplot(7,2,14)
        ax14.set_xlabel(r"$t [s]$")
        ax14.set_ylabel(r"$\dot{q}_7 [\frac{rad}{s}]$")


    if len(result_y) > 1:
        result_t = np.linspace(0, execution_time, max(result_y.shape))

        ax1.plot(result_t, result_y[:, 0], label="Result")
        ax2.plot(result_t, result_y[:, 1], label="Result")
        ax3.plot(result_t, result_y[:, 2], label="Result")
        ax4.plot(result_t, result_y[:, 3], label="Result")
        ax5.plot(result_t, result_y[:, 4], label="Result")
        ax6.plot(result_t, result_y[:, 5], label="Result")
        ax7.plot(result_t, result_y[:, 6], label="Result")

        result_t = np.linspace(0, execution_time, max(result_yd.shape))

        ax8.plot(result_t, result_yd[:, 0])
        ax9.plot(result_t, result_yd[:, 1])
        ax10.plot(result_t, result_yd[:, 2])
        ax11.plot(result_t, result_yd[:, 3])
        ax12.plot(result_t, result_yd[:, 4])
        ax13.plot(result_t, result_yd[:, 5])
        ax14.plot(result_t, result_yd[:, 6])

    demo_t = np.linspace(0, execution_time, max(demo_y.shape))

    ax1.plot(demo_t, demo_y[:, 0], label="Demo")
    ax2.plot(demo_t, demo_y[:, 1], label="Demo")
    ax3.plot(demo_t, demo_y[:, 2], label="Demo")
    ax4.plot(demo_t, demo_y[:, 3], label="Demo")
    ax5.plot(demo_t, demo_y[:, 4], label="Demo")
    ax6.plot(demo_t, demo_y[:, 5], label="Demo")
    ax7.plot(demo_t, demo_y[:, 6], label="Demo")

    ax8.plot(demo_t, np.gradient(demo_y[:, 0]) / dt)
    ax9.plot(demo_t, np.gradient(demo_y[:, 1]) / dt)
    ax10.plot(demo_t, np.gradient(demo_y[:, 2]) / dt)
    ax11.plot(demo_t, np.gradient(demo_y[:, 3]) / dt)
    ax12.plot(demo_t, np.gradient(demo_y[:, 4]) / dt)
    ax13.plot(demo_t, np.gradient(demo_y[:, 5]) / dt)
    ax14.plot(demo_t, np.gradient(demo_y[:, 6]) / dt)

    dmp_t = np.linspace(0, execution_time, max(dmp_y.shape))

    ax1.plot(dmp_t, dmp_y[:, 0], label="DMP")
    ax2.plot(dmp_t, dmp_y[:, 1], label="DMP")
    ax3.plot(dmp_t, dmp_y[:, 2], label="DMP")
    ax4.plot(dmp_t, dmp_y[:, 3], label="DMP")
    ax5.plot(dmp_t, dmp_y[:, 4], label="DMP")
    ax6.plot(dmp_t, dmp_y[:, 5], label="DMP")
    ax7.plot(dmp_t, dmp_y[:, 6], label="DMP")

    ax8.plot(dmp_t, dmp_yd[:, 0])
    ax9.plot(dmp_t, dmp_yd[:, 1])
    ax10.plot(dmp_t, dmp_yd[:, 2])
    ax11.plot(dmp_t, dmp_yd[:, 3])
    ax12.plot(dmp_t, dmp_yd[:, 4])
    ax13.plot(dmp_t, dmp_yd[:, 5])
    ax14.plot(dmp_t, dmp_yd[:, 6])

    # ax8.scatter([dmp_t[-1]], [dmp_yd[-1, 0]])
    # ax9.scatter([dmp_t[-1]], [dmp_yd[-1, 1]])
    # ax10.scatter([dmp_t[-1]], [dmp_yd[-1, 2]])
    # ax11.scatter([dmp_t[-1]], [dmp_yd[-1, 3]])
    # ax12.scatter([dmp_t[-1]], [dmp_yd[-1, 4]])
    # ax13.scatter([dmp_t[-1]], [dmp_yd[-1, 5]])
    # ax14.scatter([dmp_t[-1]], [dmp_yd[-1, 6]])

    ax1.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
