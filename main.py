import matplotlib.pyplot as plt
import numpy as np
import math

TIME_CYCLE = 2 * math.pi
B1 = 1
E1 = 1
m1 = 1
q1 = 1
w = q1 * B1 / m1


class AxesIndexes:
    X = 0
    Y = 1
    Z = 2


r_func = lambda r, v, a, dt: r + v * dt
v_func = lambda r, v, a, dt: v + a * dt
ay_func = lambda vz, dt: -(q1 * B1 / m1) * (vz) + (q1 * E1 / m1)
az_func = lambda vy, dt: (q1 * B1 / m1) * vy


def calc_ex3(r, v, a, iterations=10, total_time=TIME_CYCLE / w, plot_me=False):
    dt = total_time/iterations
    for i in range(iterations):
        a[AxesIndexes.Y].append(ay_func(v[AxesIndexes.Z][-1], dt))
        a[AxesIndexes.Z].append(az_func(v[AxesIndexes.Y][-1], dt))

        v[AxesIndexes.Y].append(v_func(r[AxesIndexes.Y][-1], v[AxesIndexes.Y][-1], a[AxesIndexes.Y][-1], dt))
        v[AxesIndexes.Z].append(v_func(r[AxesIndexes.Z][-1], v[AxesIndexes.Z][-1], a[AxesIndexes.Z][-1], dt))

        r[AxesIndexes.Y].append(r_func(r[AxesIndexes.Y][-1], v[AxesIndexes.Y][-1], a[AxesIndexes.Y][-1], dt))
        r[AxesIndexes.Z].append(r_func(r[AxesIndexes.Z][-1], v[AxesIndexes.Z][-1], a[AxesIndexes.Z][-1], dt))

    if plot_me:
        draw_graph('y [m]', 'z [m]', r[AxesIndexes.Y], r[AxesIndexes.Z], title="z(y) - ex3")
        draw_graph('Vy [m/s]', 'Vz [m/s]', v[AxesIndexes.Y], v[AxesIndexes.Z], title="Vz(Vy) - ex3")


def calc_k1(dt, r, v, a):
    k1_vy = dt*ay_func(v[AxesIndexes.Z][-1], dt)
    k1_vz = dt*az_func(v[AxesIndexes.Y][-1], dt)
    k1_y = dt*(v[AxesIndexes.Y][-1])
    k1_z = dt * (v[AxesIndexes.Z][-1])
    return k1_vy, k1_vz, k1_y, k1_z


def calc_k2(dt, r, v, a, k1_vy, k1_vz, k1_y, k1_z):
    k2_vy = dt*ay_func(v[AxesIndexes.Z][-1] + 0.5*k1_vz, dt)
    k2_vz = dt*az_func(v[AxesIndexes.Y][-1] + 0.5*k1_vy, dt)
    k2_y = dt*(v[AxesIndexes.Y][-1] + 0.5*k1_vy)
    k2_z = dt * (v[AxesIndexes.Z][-1] + 0.5*k1_vz)
    return k2_vy, k2_vz, k2_y, k2_z


def calc_k3(dt, r, v, a, k2_vy, k2_vz, k2_y, k2_z):
    k3_vy = dt*ay_func(v[AxesIndexes.Z][-1] + 0.5*k2_vz, dt)
    k3_vz = dt*az_func(v[AxesIndexes.Y][-1] + 0.5*k2_vy, dt)
    k3_y = dt*(v[AxesIndexes.Y][-1] + 0.5*k2_vy)
    k3_z = dt * (v[AxesIndexes.Z][-1] + 0.5*k2_vz)
    return k3_vy, k3_vz, k3_y, k3_z


def calc_k4(dt, r, v, a, k3_vy, k3_vz, k3_y, k3_z):
    k4_vy = dt*ay_func(v[AxesIndexes.Z][-1] + k3_vz, dt)
    k4_vz = dt*az_func(v[AxesIndexes.Y][-1] + k3_vy, dt)
    k4_y = dt*(v[AxesIndexes.Y][-1] + k3_vy)
    k4_z = dt * (v[AxesIndexes.Z][-1] + k3_vz)
    return k4_vy, k4_vz, k4_y, k4_z


def calc_ex4_midpoint(r, v, a, iterations=10, total_time=TIME_CYCLE / w, plot_me=False):
    dt = total_time/iterations

    for i in range(iterations):
        k1_vy, k1_vz, k1_y, k1_z = calc_k1(dt, r, v, a)
        k2_vy, k2_vz, k2_y, k2_z = calc_k2(dt, r, v, a, k1_vy, k1_vz, k1_y, k1_z)

        v[AxesIndexes.Y].append(v[AxesIndexes.Y][-1] + k2_vy)
        v[AxesIndexes.Z].append(v[AxesIndexes.Z][-1] + k2_vz)

        r[AxesIndexes.Y].append(r[AxesIndexes.Y][-1] + k2_y)
        r[AxesIndexes.Z].append(r[AxesIndexes.Z][-1] + k2_z)

    if plot_me:
        draw_graph('y [m]', 'z [m]', r[AxesIndexes.Y], r[AxesIndexes.Z], "z(y) - ex4 midpoint")
        draw_graph('Vy [m/s]', 'Vz [m/s]', v[AxesIndexes.Y], v[AxesIndexes.Z], "Vz(Vy) - ex4 midpoint")


rk_func = lambda y, k1, k2, k3, k4: y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

weird_times = [[],[]]


def calc_ex4_runge_kutta(r, v, a, iterations=10, total_time=TIME_CYCLE / w, plot_me=True):
    dt = total_time/iterations
    for i in range(iterations):
        a[AxesIndexes.Y].append(ay_func(v[AxesIndexes.Z][-1], dt))
        a[AxesIndexes.Z].append(az_func(v[AxesIndexes.Y][-1], dt))

        k1_vy, k1_vz, k1_y, k1_z = calc_k1(dt, r, v, a)
        k2_vy, k2_vz, k2_y, k2_z = calc_k2(dt, r, v, a, k1_vy, k1_vz, k1_y, k1_z)
        k3_vy, k3_vz, k3_y, k3_z = calc_k3(dt, r, v, a, k2_vy, k2_vz, k2_y, k2_z)
        k4_vy, k4_vz, k4_y, k4_z = calc_k4(dt, r, v, a, k3_vy, k3_vz, k3_y, k3_z)


        v[AxesIndexes.Y].append(rk_func(v[AxesIndexes.Y][-1], k1_vy, k2_vy, k3_vy, k4_vy))
        v[AxesIndexes.Z].append(rk_func(v[AxesIndexes.Z][-1], k1_vz, k2_vz, k3_vz, k4_vz))

        r[AxesIndexes.Y].append(rk_func(r[AxesIndexes.Y][-1], k1_y, k2_y, k3_y, k4_y))
        r[AxesIndexes.Z].append(rk_func(r[AxesIndexes.Z][-1], k1_z, k2_z, k3_z, k4_z))
    if plot_me:
        draw_graph('y [mr]', 'z [mr]', r[AxesIndexes.Y], r[AxesIndexes.Z], "z(y) - ex4 runge-kutta")



def ex4_error():
    error_per_dt_kutte = [[], []]  # [[dt],[oclidic_error]]
    error_per_dt_midpoint = [[], []]  # [[dt],[oclidic_error]]
    error_per_dt_basic = [[], []]  # [[dt],[oclidic_error]]
    total_time = (2 * math.pi) / w
    analytic_solution = [((2 * E1) / (w * B1)) * math.cos(w * total_time) - (2 * E1) / (w * B1),
                         (((-2) * E1) / (w * B1)) * math.sin(w * total_time) + (E1 * total_time) / (w * B1)]

    iterations_start = 10
    iterations_stop = 1000
    iterations_delta = 100
    for (calc_method, error_per_dt) in [(calc_ex4_runge_kutta, error_per_dt_kutte),
                                        (calc_ex4_midpoint, error_per_dt_midpoint),
                                        (calc_ex3, error_per_dt_basic)]:
        for i in range(iterations_start, iterations_stop, iterations_delta):
            r = [[0], [0], [0]]
            v = [[0], [0], [3]]
            a = [[0],[0], [0]]
            calc_method(r, v, a, iterations=i, total_time=total_time, plot_me=False)
            error_per_dt[0].append(math.log(total_time/i))
            error_per_dt[1].append(
                math.log(math.sqrt((r[AxesIndexes.Y][-1] - analytic_solution[0]) ** 2 + (r[AxesIndexes.Z][-1] - analytic_solution[1]) ** 2)))
    draw_graph("a", "b", weird_times[0], weird_times[1], "b")

    draw_graphs_multiple(1, "dt", "error", error_per_dt_kutte[0], error_per_dt_midpoint[0], error_per_dt_basic[0],
                        error_per_dt_kutte[1], error_per_dt_midpoint[1], error_per_dt_basic[1], "error(dt) - ex4")




# PART C
ay_func6 = lambda r, vz, a: -(q6 * B_ex6 / M_p) * (vz) + (q6 * E_ex6 / M_p)
az_func6 = lambda r, vy, a: (q6 * B_ex6 / M_p) * vy
L = 1  # [m]
R = 3 * (10 ** (-3))  # [m]
E0 = 8 * (10 ** (-13))  # [J]
dE = E0 / 20  # [J]
M_p = 1.67 * (10 ** (-27))  # [kg]
R_in = 0
E_in = E0
q6 = 1.6 * (10 ** (-19))
R_out = 0.5 * R
E_out = E0 + 0.4*dE # 0.25 * dE
convert_E_to_V0 = lambda e: math.sqrt(2 * e / M_p)
B_ex6 = 0.5
E_ex6 = B_ex6 * math.sqrt(2 * E0 / M_p)
ITERATION = 1000
got_inside = 0
all_particles = 0

def calc_ex6_runge_kutta(r, v, a, radius,plot_me=True):
    dt = 0.0000000001
    time = 0

    while r[AxesIndexes.Z][-1] < L:
        #print(r[AxesIndexes.Z][-1])
        a[1].append(ay_func6(r[AxesIndexes.Y][-1], v[AxesIndexes.Z][-1], a[AxesIndexes.Y][-1], dt))
        a[2].append(az_func6(r[2][-1], v[AxesIndexes.Y][-1], a[AxesIndexes.Z][-1], dt))

        k1_vy, k2_vy, k3_vy, k4_vy = calc_ks(v, a, dt, AxesIndexes.Y)
        k1_vz, k2_vz, k3_vz, k4_vz = calc_ks(v, a, dt, AxesIndexes.Z)

        v[AxesIndexes.Y].append(rk_func(v[AxesIndexes.Y][-1], k1_vy, k2_vy, k3_vy, k4_vy))
        v[AxesIndexes.Z].append(rk_func(v[AxesIndexes.Z][-1], k1_vz, k2_vz, k3_vz, k4_vz))

        k1_ry, k2_ry, k3_ry, k4_ry = calc_ks(r, v, dt, AxesIndexes.Y)
        k1_rz, k2_rz, k3_rz, k4_rz = calc_ks(r, v, dt, AxesIndexes.Z)

        r[AxesIndexes.Y].append(rk_func(r[AxesIndexes.Y][-1], k1_ry, k2_ry, k3_ry, k4_ry))
        r[AxesIndexes.Z].append(rk_func(r[AxesIndexes.Z][-1], k1_rz, k2_rz, k3_rz, k4_rz))

        if math.sqrt(r[AxesIndexes.Y][-1] ** 2 + r[AxesIndexes.X][-1] ** 2) >= radius:
            if plot_me:
                draw_graph([0, 2 * math.pi], 'y [mr5]', 'z [mr5]', r[AxesIndexes.Y], r[AxesIndexes.Z])
            raise Exception("out of range")
        time += dt
    #print('%%%', a[AxesIndexes.Y][-1], a[AxesIndexes.Z][-1])
    return r, v, a


def calc_ex_6():
    # in
    r_in = [[0], [R_in], [0]]
    v_in = [[0], [0], [convert_E_to_V0(E_in)]]
    a_in = [[0], [(q6 / M_p) * (E_in - v_in[1][0] * B_ex6)], [0]]
    try:
        r, v, a = calc_ex6_runge_kutta(r_in, v_in, a_in, R)
        draw_graph([0, 2 * math.pi], 'y [mr3]', 'z [mr3]', r[AxesIndexes.Y], r[AxesIndexes.Z])
    except:
        pass
    # out
    r_out = [[0], [R_out], [0]]
    v_out = [[0], [0], [convert_E_to_V0(E_out)]]
    dt = 0.0000000001
    a_out = [[0], [ay_func6(r_out[1][0], v_out[2][0], 0, dt, q6)], [az_func6(r_out[2][0], v_out[1][0], 0, dt)]]
    try:
        r, v, a = calc_ex6_runge_kutta(r_out, v_out, a_out, R)
        draw_graph([0, 2 * math.pi], 'y [mr3]', 'z [mr3]', r[AxesIndexes.Y], r[AxesIndexes.Z])
    except:
        pass


def calc_ex7():
    global all_particles
    global got_inside
    valid_points = [[], []]
    for y in np.linspace(-R, R, 100):
        for e in np.linspace(E0 - dE, E0 + dE, 100):
            v0 = convert_E_to_V0(e)
            r_in = [[0], [y], [0]]
            v_in = [[0], [0], [v0]]
            a_in = [[0], [(q6 / M_p) * (e - v_in[1][0] * B_ex6)], [0]]
            try:
                all_particles += 1
                calc_ex6_runge_kutta(r_in, v_in, a_in, R, False)
            except:
                pass
            else:
                got_inside += 1
                valid_points[0].append(y / R)
                valid_points[1].append((v0 - convert_E_to_V0(E0)) / convert_E_to_V0(E0))
    #plt.scatter(valid_points[0], valid_points[1])
    plt.show()


def calc_ex7_random():
    global all_particles
    global got_inside
    valid_points = [[], []]
    for _ in range(100):
        if _%10 == 0:
            print(_)
        y = random.uniform(-R, R)
        for __ in range(100):
            e = random.uniform(E0 - dE, E0 + dE)
            v0 = convert_E_to_V0(e)
            r_in = [[0], [y], [0]]
            v_in = [[0], [0], [v0]]
            a_in = [[0], [(q6 / M_p) * (e - v_in[1][0] * B_ex6)], [0]]
            try:
                all_particles += 1
                calc_ex6_runge_kutta(r_in, v_in, a_in, R, False)
            except:
                pass
            else:
                got_inside += 1
                valid_points[0].append(y / R)
                valid_points[1].append((v0 - convert_E_to_V0(E0)) / convert_E_to_V0(E0))



def calc_ex8():
    end_v = []
    e_options = (np.random.random(10**5) * 2 * dE) + E0 - dE
    y_options = (np.random.random(10**5) * 2 * R) - R
    for e, y in zip(e_options, y_options):
        v0 = convert_E_to_V0(e)
        r_arr = [[0], [y], [0]]
        v_arr = [[0], [0], [v0]]
        a_arr = [[0], [(q6 / M_p)*(e - v0 * B_ex6)], [0]]
        try:
            r, v, a = calc_ex6_runge_kutta(r_arr, v_arr, a_arr, R, False)
        except:
            pass
        else:
            end_v.append(math.sqrt(v[AxesIndexes.Y][-1] ** 2 + v[AxesIndexes.Z][-1] ** 2))
    fig = plt.figure()
    _ = plt.hist(end_v)
    plt.show()


def calc_ex9():
    global all_particles
    global got_inside
    print("%{} of the particles got out successfully.".format((got_inside / all_particles) * 100))


def draw_graph(x_name, y_name, ydata, zdata, title):
    # naming the x axis
    plt.xlabel(x_name)
    # naming the y axis
    plt.ylabel(y_name)

    # giving a title to my graph
    plt.title(title)

    plt.plot(ydata, zdata)

    # function to show the plot
    plt.show()


def draw_graphs_multiple(domain, x_name, y_name, ydata1, ydata2, ydata3, zdata1, zdata2, zdata3, title, plot_log=False):
    # naming the x axis
    plt.xlabel(x_name)
    # naming the y axis
    plt.ylabel(y_name)

    # giving a title to my graph
    plt.title(title)

    plt.plot(ydata1, zdata1, color="green")
    plt.plot(ydata2, zdata2, color="red")
    plt.plot(ydata3, zdata3)

    # function to show the plot
    plt.show()


def main():
    # 100 linearly spaced numbers
    #calc_q3()
    r = [[0], [0], [0]]
    v = [[0], [0], [3]]
    a = [[0], [-3], [0]]
    # calc_ex4_midpoint(r, v, a, plot_me=True)
    #
    # calc_ex3(r, v, a, dt=0.1*(TIME_CYCLE/w), plot_me=True)
    # calc_ex_6()
    # calc_ex7()
    # calc_ex7()
    # calc_ex8()
    # ex4_error()
    # ex4_error()
    # calc_ex3(r, v, a, plot_me=True)
    # calc_ex7()
    # calc_ex9()
    # calc_ex6_runge_kutta(r, v, a, 0.03)
    # calc_ex4_midpoint(r, v, a, plot_me=True)
    # calc_ex_6()
    calc_ex8()

if __name__ == '__main__':
    main()

