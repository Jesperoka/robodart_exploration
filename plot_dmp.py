import matplotlib.pyplot as plt
import numpy as np


def plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd, result_y=np.array([0]), result_yd=np.array([0]), filename="test.pdf"):
    plt.figure(figsize=(20, 15))
    ax1 = plt.subplot(2,7,1)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")

    ax2 = plt.subplot(2,7,2)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position")

    ax3 = plt.subplot(2,7,3)
    ax3.set_title("Dimension 3")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Position")

    ax4 = plt.subplot(2,7,4)
    ax4.set_title("Dimension 4")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Position")

    ax5 = plt.subplot(2,7,5)
    ax5.set_title("Dimension 5")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Position")

    ax6 = plt.subplot(2,7,6)
    ax6.set_title("Dimension 6")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Position")

    ax7 = plt.subplot(2,7,7)
    ax7.set_title("Dimension 7")
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Position")

    ax8 = plt.subplot(2,7,8)
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Velocity")

    ax9 = plt.subplot(2,7,9)
    ax9.set_xlabel("Time")
    ax9.set_ylabel("Velocity")

    ax10 = plt.subplot(2,7,10)
    ax10.set_xlabel("Time")
    ax10.set_ylabel("Velocity")

    ax11 = plt.subplot(2,7,11)
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Velocity")

    ax12 = plt.subplot(2,7,12)
    ax12.set_xlabel("Time")
    ax12.set_ylabel("Velocity")

    ax13 = plt.subplot(2,7,13)
    ax13.set_xlabel("Time")
    ax13.set_ylabel("Velocity")

    ax14 = plt.subplot(2,7,14)
    ax14.set_xlabel("Time")
    ax14.set_ylabel("Velocity")

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

    ax8.scatter([dmp_t[-1]], [dmp_yd[-1, 0]])
    ax9.scatter([dmp_t[-1]], [dmp_yd[-1, 1]])
    ax10.scatter([dmp_t[-1]], [dmp_yd[-1, 2]])
    ax11.scatter([dmp_t[-1]], [dmp_yd[-1, 3]])
    ax12.scatter([dmp_t[-1]], [dmp_yd[-1, 4]])
    ax13.scatter([dmp_t[-1]], [dmp_yd[-1, 5]])
    ax14.scatter([dmp_t[-1]], [dmp_yd[-1, 6]])

    ax1.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
