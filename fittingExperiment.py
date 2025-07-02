#Analyses of aspiration curves for rectangular constrictions
#Aldo, Blanca, Cristina, María, Gustavo. 2022

import os
from statistics import mean
import numpy as np
from numpy.linalg import inv
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

#plt.style.use(['science','no-latex'])

########################################################
# Function for eQLV simulating the viscous entry process
########################################################
def Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el,
                       P_exp, E0, gA, tau_fit):
    gB = 1. - gA
    tau = 1.
    #Relaxation function: gA + gB*math.exp(-t/tau)
    Pi_max = P_exp/E0 #maximum value (constant during the creep process) of the non-dimensional differential pressure
    if Pi_max >= max_Pi_el:
        print('E0 is too low in the computations, since the non-dimensional aspiration pressure exceeds the maximum value in que quasistatic curve')
        #input("press any key to continue")
        #quit()
    def Relax_f(t):
            return gA+gB*math.exp(-t/tau)
    
    def Creep_f(t):
            return (1.+gB/gA)-(gB/gA)*math.exp(-t/(tau*(1.+gB/gA)))

    T_max = -tau*(1.+gB/gA)*math.log((gA/gB)*((gB/gA+1.)-max_Pi_el/Pi_max))
    nb_points_creep = 20
    T_increment = T_max/(nb_points_creep-1)
    
    lambda_vec = [] #Vector of the non-dimensional advance of the cell lambda=AL/Wc
    Pi_el_vec = [] #Vector of the non-dimensional differential pressure Pi=DP/E0
    time_T_vec = np.array([]) #Vector of non-dimensional time T


    for k in range(nb_points_creep): #k=0,1,...,nb_points_creep-1
        T_value = k*T_increment
        time_T_vec = np.append(time_T_vec, T_value)
        Pi_el_vec.append(Pi_max*Creep_f(T_value))
        lambda_vec.append(lambda_vs_Pi_el(Pi_el_vec[k]))

    time_t_vec = tau_fit*time_T_vec
    t_vs_lambda_teo = interp1d(lambda_vec, time_t_vec, fill_value='extrapolate')
    lambda_vs_t_teo = interp1d(time_t_vec, lambda_vec, fill_value='extrapolate')

    t_max_teo = max(time_t_vec)
    
    return lambda_vec, lambda_vs_t_teo, t_vs_lambda_teo, t_max_teo


########################################################
# Computing mechnical parameters by fitting
# non-linear least-squares method
# based on a script by G Guinea
########################################################

def fitting_nlls(t_exp_vec, lambda_exp_vec, P_exp, E0_ini, gA_ini, tau_ini,
                 lambda_vs_Pi_el, max_Pi_el, max_lambda_el, show_curves=False, save_curves=False):
    y = t_exp_vec
    x = lambda_exp_vec
    numDatos = len(x)
    bestError = 1e10
    numIter = 50  # Número de iteraciones
    numParam = 3  # E0, gA, tau

    # Cota inferior de E0 para que lambda_teo(0) no sea menor que lambda_exp(0)
    percent = 10 
    initialValues = lambda_exp_vec[:len(lambda_exp_vec)//percent] if len(lambda_exp_vec) > percent else lambda_exp_vec[:3]
    median_initial = np.median(initialValues)

    Pi_max_target = Pi_vs_lambda_el(median_initial)
    E0_min = P_exp / Pi_max_target

    def delta(a, b):
        return 1 if a == b else 0

    for gA_index in range(1, 10):
        try:
            gA_ini = gA_index / 10.
            Beta = np.array([E0_ini, gA_ini, tau_ini])  # E0, gA, tau
            epsilonBeta = 1e-5 * Beta
            levenberg = 100.

            # Respetar cota inferior de E0
            if Beta[0] < E0_min:
                Beta[0] = E0_min

            for j in range(numIter):
                lambda_vec, lambda_vs_t_teo, t_vs_lambda_teo, _ = Theoret_aspiration(
                    lambda_vs_Pi_el, max_Pi_el, P_exp, Beta[0], Beta[1], Beta[2])
                
                R = np.zeros(numDatos)
                R2 = 0
                y2 = 0
                for i in range(numDatos):
                    R[i] = t_vs_lambda_teo(x[i]) - y[i]
                    R2 += R[i]**2
                    y2 += y[i]**2
                prevError = 100 * R2 / y2

                # Jacobiano
                J = np.zeros((numDatos, numParam))
                for k in range(numParam):
                    perturbed = Beta.copy()
                    perturbed[k] += epsilonBeta[k]
                    lambda_vec, _, t_vs_lambda_teo_perturbed, _ = Theoret_aspiration(
                        lambda_vs_Pi_el, max_Pi_el, P_exp, perturbed[0], perturbed[1], perturbed[2])
                    for i in range(numDatos):
                        J[i, k] = (t_vs_lambda_teo_perturbed(x[i]) - t_vs_lambda_teo(x[i])) / epsilonBeta[k]

                # Levenberg-Marquardt update
                deltaBeta = -np.linalg.inv(J.T @ J + levenberg * np.eye(numParam)) @ (J.T @ R)
                Beta += deltaBeta

                # Aplicar restricciones
                if Beta[0] < E0_min:
                    Beta[0] = E0_min
                Beta[1] = max(Beta[1], 1e-4)
                Beta[2] = max(Beta[2], 1e-4)

                lambda_vec, lambda_vs_t_teo, t_vs_lambda_teo, _ = Theoret_aspiration(
                    lambda_vs_Pi_el, max_Pi_el, P_exp, Beta[0], Beta[1], Beta[2])

                R = np.zeros(numDatos)
                R2 = 0
                y2 = 0
                for i in range(numDatos):
                    R[i] = t_vs_lambda_teo(x[i]) - y[i]
                    R2 += R[i]**2
                    y2 += y[i]**2
                nowError = 100 * R2 / y2

                if nowError < bestError:
                    bestError = nowError
                    bestBeta = Beta.copy()
                    bestlambda_vs_t_teo = lambda_vs_t_teo

                print(f'Iter {j}, Error: {nowError:.3f} | E0: {Beta[0]:.2f}, gA: {Beta[1]:.3f}, tau: {Beta[2]:.3f}')

                if nowError > prevError:
                    levenberg *= 10
                else:
                    levenberg /= 2
                prevError = nowError

                if show_curves or save_curves:
                    plt.plot(t_exp_vec, lambda_exp_vec, "-b", label="Experimental curve")
                    plt.scatter(t_exp_vec, lambda_exp_vec)
                    plt.plot(t_exp_vec, lambda_vs_t_teo(t_exp_vec), "-r",
                             label=f"Fit: $E_0$={Beta[0]:.1f}, $g$={Beta[1]:.3f}, $\\tau$={Beta[2]:.3f}, Err={nowError:.2f}%")
                    plt.legend()
                    plt.xlabel("Time [s]")
                    plt.ylabel("$A_L/W_{ch}$")
                    plt.grid(True, alpha=0.5)
                    if save_curves:
                        plt.savefig(f'Fitting_progress_{Beta[0]:.1f}_{Beta[1]:.3f}_{Beta[2]:.3f}.png', dpi=300, bbox_inches='tight')
                    if show_curves:
                        plt.pause(0.1)
                        plt.show()
                    plt.clf()

        except Exception as e:
            print(f"Fitting failed: {e}")
            continue

    return bestBeta[0], bestBeta[1], bestBeta[2], bestlambda_vs_t_teo, bestError




########################################################
# global data
########################################################
cwd = os.getcwd()
folder = cwd+"\\"+"Results\\"
folderqs = cwd+"\\"+"sim\\"
outputFolder = cwd+"\\"+"Output\\"
os.makedirs(outputFolder, exist_ok=True)
show_curves = False
save_curves = True


########################################################
#metadata Rc	Wc	BLch

# Read the file Results/Metadata.csv and save its contents in a numpy array
metadata_file = os.path.join(folder, "metadata.txt")
metadata = np.genfromtxt(metadata_file, delimiter=' ', skip_header=1)
print(metadata)
for i,exp in enumerate(metadata):
    num_exp = int(exp[0])
    rce=exp[1]
    wc=exp[2]
    wch=wc/2
    blch=exp[3]
    exp_file = "exp"+str(num_exp).zfill(2)
    wch=wc/2
    
    d = ["2.0","4.0","8.0","16.0"]
    rc = ["1.6", "2.0", "2.4", "3.2"] #List of values of Rc/Wch of the curves obtained by numerical analyses
    rf = ["0.4", "0.8", "1.2", "1.6", "2.4", "4.0"]
    wc = ["0.7", "0.8", "1.0", "1.2"]



    #folder = 'D:/TESIS/TESIS EXPERIMENTOS/MICROF/2022-11-29/VIVAS/res/code_gus/'
    InputQSFilepreName = folderqs+'rc_' #csv format, with ",". First column: lambda=AL/Wc, second column: Pi=DP/E0. No headings
    List_of_QS_curves = ["1.6", "2.0", "2.4", "3.2"] #List of values of Rc/Wch of the curves obtained by numerical analyses
    InputEXPFileName = folder+exp_file+'.txt' #csv format, with ",". First column: time (s), second column: lambda=AL/Wc. No headings
    OutputFileName = folder+'Outf_'+exp_file+'.csv'
    nb_points_QS_curves = 1000 #the numerical curves from Abaqus are discretized in this number of points
    #Relaxation function: gA + gB*math.exp(-t/tau) with gA + gB = 1.



    #Relation of P_exp to blocked channels, using Q=10ml/hr
    #rblqch = [[0,7413.9], [2, 8014.5], [4, 8972.3], [6, 10610.3], [8, 14038.7], [10, 18644.0]]

    def rPbch(x):
        # Relation of Pressure to number of blocked channels (x)
        return 0.5056*x**6-15.468*x**5+184.86*x**4-1076.9*x**3+3160.9*x**2-4062.7*x+3519.2

    Rc_to_Wch = rce/wch #Ratio cell radius Rc to half of the width of the constriction Wch
    P_exp = rPbch(blch) #Actual differential pressure in the experiment (Pa)


    ########################################################
    # Import and compute numerical quasistatic-curve data
    ########################################################
    List_of_QS_curves_float = [float(x) for x in List_of_QS_curves]
    nb_integers = np.array([float(x) for x in range(0, nb_points_QS_curves)])
    lambda_values = np.zeros((nb_points_QS_curves, len(List_of_QS_curves)))
    Pi_values = np.zeros((nb_points_QS_curves, len(List_of_QS_curves)))

    for i in range(len(List_of_QS_curves)): #i=0,1,...,List_of_QS_curves-1
        InputQSFileName = InputQSFilepreName + List_of_QS_curves[i] + '.txt'
        with open(InputQSFileName, 'r') as f:
            l_reading = [[float(num) for num in line.split('\t')] for line in f]
        A = np.array(l_reading)
        
        A_lambda_corr = []
        A_lambda_corr.append(A[0,0])
        A_Pi_corr = []
        A_Pi_corr.append(A[0,0])
        k = int(0)
        for j in range(len(A)):
            if A[j,0] > A_lambda_corr[k] and A[j,1] > A_Pi_corr[k]:
                A_lambda_corr.append(A[j,0])
                A_Pi_corr.append(A[j,1])
                k += 1
        
        Pi_vs_lambda_el = interp1d(A_lambda_corr, A_Pi_corr, fill_value='extrapolate')
        max_lambda_el = max(A_lambda_corr)
        lambda_values[:,i] = nb_integers*max_lambda_el/(nb_points_QS_curves-1)
        Pi_values[:,i] = Pi_vs_lambda_el(lambda_values[:,i])

    lambda_Pi_matrix = np.zeros((nb_points_QS_curves,2))
    for i in range(nb_points_QS_curves):
        interp_lambda = interp1d(List_of_QS_curves_float, lambda_values[i,:], fill_value='extrapolate')
        lambda_Pi_matrix[i,0] = interp_lambda(Rc_to_Wch)
        interp_Pi = interp1d(List_of_QS_curves_float, Pi_values[i,:], fill_value='extrapolate')
        lambda_Pi_matrix[i,1] = interp_Pi(Rc_to_Wch)

    Pi_vs_lambda_el = interp1d(lambda_Pi_matrix[:,0], lambda_Pi_matrix[:,1], fill_value='extrapolate')
    lambda_vs_Pi_el = interp1d(lambda_Pi_matrix[:,1], lambda_Pi_matrix[:,0], fill_value='extrapolate')
    max_Pi_el = max(lambda_Pi_matrix[:,1])
    max_lambda_el = max(lambda_Pi_matrix[:,0])

    #print(List_of_QS_curves)

    plt.figure(1)
    for i in range(len(List_of_QS_curves)):
        plt.plot(lambda_values[:,i], Pi_values[:,i], label='Numerical curve - $R_c^*$: %.2f' % (float(List_of_QS_curves[i])))
    plt.ylabel("$\Delta P/ E_0$")
    plt.xlabel("$A_L$/$W_{ch}$")
    plt.plot(lambda_Pi_matrix[:,0], lambda_Pi_matrix[:,1], '--', label='Extrapolated curve - $R_c^*$: %.2f' % (Rc_to_Wch))
    plt.legend(loc="upper left")
    plt.xlim([0, 3.0])
    plt.ylim([0, 0.8])
    plt.grid(True, alpha=0.5)
    plt.legend(framealpha=1, frameon=True);
    if save_curves:
        plt.savefig(outputFolder+'QS_curves_'+exp_file+'.png', dpi=300, bbox_inches='tight')
    if show_curves:
        plt.show()
    plt.close()



    ########################################################
    # Import experiment data
    ########################################################
    with open(InputEXPFileName, 'r') as f:
        l_reading = [[float(num) for num in line.split(' ')] for line in f]
    B = np.array(l_reading)
    #print(B[:,1])
    #plt.plot(B[:,0], B[:,1])
    lambda_exp_vec = B[:,1]/wch
    t_exp_vec = B[:,0]
    max_lambda_exp = max(B[:,1])
    max_t_exp = max(B[:,0])




    ########################################################
    # Run of the fitting process
    ########################################################
    E0_ini = P_exp/(Pi_vs_lambda_el(mean(lambda_exp_vec[0:4]))) #We consider instant deformation equal to the average of the ___just the first measurement
    
    ###
    percentaje = 10 #Percentage of the first values to consider for E0_ini
    inital_values = lambda_exp_vec[:len(lambda_exp_vec)//percentaje] if len(lambda_exp_vec) > percentaje else lambda_exp_vec[:3]
    median_initial = np.median(inital_values)
    Pi_max_target = Pi_vs_lambda_el(median_initial) #Maximum value of Pi_el in the quasistatic curve for the median of the first values
    E0_ini = P_exp / Pi_max_target
    ###

    gA_ini = 0.05
    tau_ini = max_t_exp*1
    ####
    # plt.figure(2)
    # plt.plot(t_exp_vec, lambda_exp_vec)
    # plt.show()

    ####
    E0, gA, tau, lambda_vs_t_fit, bestError = fitting_nlls(t_exp_vec, lambda_exp_vec, P_exp,
                                                E0_ini, gA_ini, tau_ini,lambda_vs_Pi_el,
                                                max_Pi_el, max_lambda_el)

    ########################################################
    # Export the results
    ########################################################
    OutputFile = open(OutputFileName, 'w')
    FirstLine = 'E0 = '+str(E0)+'; gA = '+str(gA)+'; tau = '+str(tau)+'\n'
    OutputFile.write(FirstLine)
    FirstLine = 'time (s)'+';'+'AL/Wc exp'+';'+'AL/Wc fit'+'\n'
    OutputFile.write(FirstLine)
    for i in range(len(lambda_exp_vec)):
        NewLine = str(t_exp_vec[i])+';'+str(lambda_exp_vec[i])+';'+str(lambda_vs_t_fit(t_exp_vec[i]))+'\n' #We assume Pi_max constant
        #print(NewLine)
        OutputFile.write(NewLine)
    OutputFile.close()

    print(E0, gA, tau, bestError)

    with open(outputFolder+"\\Visco_properties"+'.txt', 'a') as f:
        #save.append(pres_e[jj], item)
        f.write("%i %s %s %s %s\n" % (num_exp, E0, gA, tau, bestError))

    plt.figure(3)

    plt.plot(t_exp_vec, lambda_exp_vec, "-b", alpha=0.5, label="Experimental curve")
    plt.scatter(t_exp_vec, lambda_exp_vec, alpha=0.5)
    plt.plot(t_exp_vec, lambda_vs_t_fit(t_exp_vec), "-r", label="Fitted curve, $E_0$: %.2f [Pa], $g_\infty$: %.3f, $τ_C$: %.3f [s], Error: %.3f %%" % (E0, gA, tau, bestError))
    #plt.legend(loc="upper left")
    plt.xlabel("Time [s]")
    plt.ylabel("$A_L$/$W_{ch}$")
    plt.grid(True, alpha=0.5)

    plt.xlim([0, max(t_exp_vec)*1.2])
    plt.ylim([0, max(lambda_exp_vec)*1.2])


    plt.legend(framealpha=1, frameon=True)
    plt.xlabel("Time [s]")
    plt.ylabel("$A_L$/$W_{ch}$")
    if save_curves:
        plt.savefig(outputFolder+'Fitted_curve_'+exp_file+'.png', dpi=300, bbox_inches='tight')
    if show_curves:
        plt.show()

    plt.close()

