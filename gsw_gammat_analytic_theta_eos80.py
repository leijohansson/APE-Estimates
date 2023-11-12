
# Import modules
import numexpr as ne
import numpy as np

def gsw_gammat_analytic_theta_eos80(s, th):
    # gsw_gammat_analytic_theta_eos80: Compute thermodynamic neutral density based on an
    # analytical expression of Lorenz reference density
    #
    # INPUT:
    #   s           : practical salinity 
    #   th          : potential Temperature (deg C)
    #
    # OUTPUT: 
    #   zref        : Reference position
    #   pref        : Reference pressure
    #   sigref      : Reference density 
    #   gammat      : Thermodynamic neutral density 
    #
    # DEPENDENCIES: 
    #
    # AUTHOR OF ORIGINAL MATLAB CODE:
    #   Remi Tailleux, University of Reading, 8 July 2020
    #
    # CHANGES TO ORIGINAL CODE:
    #   25.Apr.2022: Converted into Python (Gabriel Wolf)
    #==========================================================================

    # check input format
    # ------------------
    if np.isscalar(th):
        output_c = 'scalar'
    elif isinstance(th, list):
        output_c = 'list'
        s  = np.asarray(s)
        th = np.asarray(th)
    else:
        output_c = 'array'

    # Coefficients for numerator of equation of state for density
    #------------------------------------------------------------
    # https://doi.org/10.1175/JTECH1946.1 ; equation (A2) and table A2 therein
    # numerator of equ.(A2)
    c_0      = 9.9984085444849347e02
    c_th1    = 7.3471625860981584e00
    c_th2    = -5.3211231792841769e-02
    c_th3    = 3.6492439109814549e-04
    c_s1     = 2.5880571023991390e+00
    c_s1th1  = -6.7168282786692355e-03
    c_s2     = 1.9203202055760151e-03
    c_p1     = 1.1798263740430364e-02
    c_p1th2  = 9.8920219266399117e-08
    c_p1s1   = 4.6996642771754730e-06
    c_p2     = -2.5862187075154352e-08
    c_p2th2  = -3.2921414007960662e-12
    # denominator of equ.(A2)
    d_0       = 1e00
    d_th1     = 7.2815210113327091e-03
    d_th2     = -4.4787265461983921e-05
    d_th3     = 3.3851002965802430e-07
    d_th4     = 1.3651202389758572e-10
    d_s1      = 1.7632126669040377e-03
    d_s1th1   = -8.8066583251206474e-06
    d_s1th3   = -1.8832689434804897e-10
    d_s3d2    = 5.7463776745432097e-06
    d_s3d2th2 = 1.4716275472242334e-09
    d_p1      = 6.7103246285651894e-06
    d_p2th3   = -2.4461698007024582e-17
    d_p3th1   = -9.1534417604289062e-18
    # equation (B5) for the equation of state


    # Set values of coefficients
    # --------------------------
    # parameter sets based on ID 8 and 9
    a = 4.56016575
    b = -1.24898501
    c = 0.00439778209
    d = 1030.99373
    e = 8.32218903

    # Set polynomial corrections
    #     Linear model Poly8:
    #      f(x) = p1*x^8 + p2*x^7 + p3*x^6 + p4*x^5 + 
    #                     p5*x^4 + p6*x^3 + p7*x^2 + p8*x + p9
    #        where x is normalized by mean 1440 and std 1470
    #      Coefficients (with 95% confidence bounds):
    p1 =   0.0007824 # (0.0007792, 0.0007855)
    p2 =   -0.008056 # (-0.008082, -0.008031)
    p3 =     0.03216 # (0.03209, 0.03223)
    p4 =    -0.06387 # (-0.06393, -0.06381)
    p5 =     0.06807 # (0.06799, 0.06816)
    p6 =    -0.03696 # (-0.03706, -0.03687)
    p7 =    -0.08414 # (-0.08419, -0.0841)
    p8 =       6.677 # (6.677, 6.677)
    p9 =       6.431 # (6.431, 6.431)

    # Set value of gravity 
    # --------------------
    grav = 9.81

    # Define the different analytical functions
    # -----------------------------------------
    # drhordz = @(z) a.*(z+e).^b + c;
    rhor = 'a/(b+1)*(zref+e)**(b+1) + c*zref + d'
    pr   = 'grav * (a/((b+1)*(b+2))*((zref+e)**(b+2)) + c/2.*zref**2 + d*zref - a/((b+1)*(b+2))*e**(b+2))/1e4'

    # Polynomial correction
    # --------------------
    f = 'p9 + x*( p8 + x*( p7 + x*( p6 + x*( p5 + x*( p4 + x*( p3 + x*( p2 + x*p1)))))))'

    # Define equation of state for density 
    # -------------------------------------
    # numerator of equ.(A2)
    eqA2_num = 'c_0+th*(c_th1+th*(c_th2+th*c_th3))+s*(c_s1+th*c_s1th1+s*c_s2)+p*(c_p1+th2*c_p1th2+s*c_p1s1+p*(c_p2+th2*c_p2th2))'
    # denominator of equ.(A2)
    eqA2_den  = 'd_0+th*(d_th1+th*(d_th2+th*(d_th3+th*d_th4)))+s*(d_s1+th*(d_s1th1+th2*d_s1th3)+sqrts*(d_s3d2+th2*d_s3d2th2))+p*(d_p1+pth*(th2*d_p2th3+p*d_p3th1))'

    # Compute the reference positions
    # -------------------------------
    zmin = 0.; zmax = 6000.
    zref = np.ones_like(s) * 2000.
    ztop = np.ones_like(s) * zmin
    zbot = np.ones_like(s) * zmax

    # Valid points
    #-------------
    zref_new_ev = '0.25*(1.+ss)*(ztop+zref) + 0.25*(1.-ss)*(zbot+zref)'
    ztop_ev     = '0.5*(1+ss)*ztop + 0.5*(1-ss)*zref'
    zbot_ev     = '0.5*(1-ss)*zbot + 0.5*(1+ss)*zref'

    th2   = th*th 
    sqrts = np.sqrt(s)
    n_c = 30
    for i_c in range(0,n_c):
        # Compute density at reference pressure pr(zref)
        p = ne.evaluate(pr)
        pth = p*th 
        rho_ref = ne.evaluate(eqA2_num) / ne.evaluate(eqA2_den)

        # Compute buoyancy 
        buoyancy = ne.evaluate(rhor) - rho_ref
        # Compute sign of buoyancy
        ss = np.sign(buoyancy)
        # Redefine zref depending on sign of buoyancy
        zref_new = ne.evaluate(zref_new_ev)
        ztop     = ne.evaluate(ztop_ev)
        zbot     = ne.evaluate(zbot_ev)
        zref     = zref_new

    # Compute analytic gammat
    # -----------------------
    pmean = 1440.
    pstd  = 1470.

    pref = ne.evaluate(pr)
    x = (pref-pmean)/pstd
    pth = p*th 
    rho_ref = ne.evaluate(eqA2_num) / ne.evaluate(eqA2_den)
    sigref = rho_ref - 1000.
    gammat = sigref - ne.evaluate(f)

    # Return values
    # -----------------------
    if output_c=='scalar':
        return gammat, np.float(zref), np.float(pref), sigref
    elif output_c=='list':
        return gammat.tolist(), zref.tolist(), pref.tolist(), sigref.tolist()
    else:
        return gammat, zref, pref, sigref

