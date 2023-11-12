
# Import modules
import numexpr as ne
import numpy as np

def gsw_gammat_analytic_CT(sr, ct):
    # gsw_gammat_analytic_CT: Compute thermodynamic neutral density based on an
    # analytical expression of Lorenz reference density
    #
    # INPUT:
    #   
    #   sr          : reference composition salinity (g/kg) 
    #   ct          : Conservative Temperature (deg C)
    #
    # OUTPUT: 
    #   zref        : Reference position
    #   pref        : Reference pressure
    #   sigref      : Reference density 
    #   gammat      : Thermodynamic neutral density 
    #
    # DEPENDENCIES: none. The routine uses the polynomial approximation of the
    # equatin of state
    #
    # AUTHOR OF ORIGINAL MATLAB CODE:
    #   Remi Tailleux, University of Reading, 8 July 2020
    #
    # CHANGES TO ORIGINAL CODE:
    #   25.Apr.2022: Converted into Python (Gabriel Wolf)
    #==========================================================================

    # check input format
    # ------------------
    if np.isscalar(sr):
        output_c = 'scalar'
    elif isinstance(sr, list):
        output_c = 'list'
        sr = np.asarray(sr)
        ct = np.asarray(ct)
    else:
        output_c = 'array'

    # Set values for TEOS10 simplified equation of state
    # --------------------------------------------------
    v000 =  1.0769995862e-3
    v001 = -6.0799143809e-5 
    v002 =  9.9856169219e-6 
    v003 = -1.1309361437e-6 
    v004 =  1.0531153080e-7 
    v005 = -1.2647261286e-8 
    v006 =  1.9613503930e-9 
    v010 = -1.5649734675e-5 
    v011 =  1.8505765429e-5 
    v012 = -1.1736386731e-6 
    v013 = -3.6527006553e-7 
    v014 =  3.1454099902e-7 
    v020 =  2.7762106484e-5 
    v021 = -1.1716606853e-5 
    v022 =  2.1305028740e-6 
    v023 =  2.8695905159e-7 
    v030 = -1.6521159259e-5 
    v031 =  7.9279656173e-6 
    v032 = -4.6132540037e-7 
    v040 =  6.9111322702e-6 
    v041 = -3.4102187482e-6 
    v042 = -6.3352916514e-8 
    v050 = -8.0539615540e-7 
    v051 =  5.0736766814e-7 
    v060 =  2.0543094268e-7
    v100 = -3.1038981976e-4 
    v101 =  2.4262468747e-5 
    v102 = -5.8484432984e-7 
    v103 =  3.6310188515e-7 
    v104 = -1.1147125423e-7
    v110 =  3.5009599764e-5 
    v111 = -9.5677088156e-6 
    v112 = -5.5699154557e-6 
    v113 = -2.7295696237e-7 
    v120 = -3.7435842344e-5 
    v121 = -2.3678308361e-7 
    v122 =  3.9137387080e-7 
    v130 =  2.4141479483e-5 
    v131 = -3.4558773655e-6 
    v132 =  7.7618888092e-9 
    v140 = -8.7595873154e-6 
    v141 =  1.2956717783e-6 
    v150 = -3.3052758900e-7 
    v200 =  6.6928067038e-4 
    v201 = -3.4792460974e-5 
    v202 = -4.8122251597e-6 
    v203 =  1.6746303780e-8 
    v210 = -4.3592678561e-5 
    v211 =  1.1100834765e-5 
    v212 =  5.4620748834e-6 
    v220 =  3.5907822760e-5 
    v221 =  2.9283346295e-6 
    v222 = -6.5731104067e-7 
    v230 = -1.4353633048e-5 
    v231 =  3.1655306078e-7 
    v240 =  4.3703680598e-6 
    v300 = -8.5047933937e-4 
    v301 =  3.7470777305e-5 
    v302 =  4.9263106998e-6 
    v310 =  3.4532461828e-5 
    v311 = -9.8447117844e-6 
    v312 = -1.3544185627e-6 
    v320 = -1.8698584187e-5 
    v321 = -4.8826139200e-7 
    v330 =  2.2863324556e-6
    v400 =  5.8086069943e-4 
    v401 = -1.7322218612e-5 
    v402 = -1.7811974727e-6 
    v410 = -1.1959409788e-5 
    v411 =  2.5909225260e-6 
    v420 =  3.8595339244e-6 
    v500 = -2.1092370507e-4 
    v501 =  3.0927427253e-6 
    v510 =  1.3864594581e-6
    v600 =  3.1932457305e-5 
    
    # Set values of coefficients
    # --------------------------
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

    # Equation of state
    # -----------------
    sfac   = 0.0248826675584615   # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1 # offset = deltaS*sfac.

    rho = lambda x2, xs, ys, z : 1. / (v000 + xs*(v100 + xs*(v200 + xs*(v300 + xs*(v400 + xs*(v500 \
        + v600*xs))))) + ys*(v010 + xs*(v110 + xs*(v210 + xs*(v310 + xs*(v410 \
        + v510*xs)))) + ys*(v020 + xs*(v120 + xs*(v220 + xs*(v320 + v420*xs))) \
        + ys*(v030 + xs*(v130 + xs*(v230 + v330*xs)) + ys*(v040 + xs*(v140 \
        + v240*xs) + ys*(v050 + v150*xs + v060*ys))))) + z*(v001 + xs*(v101 \
        + xs*(v201 + xs*(v301 + xs*(v401 + v501*xs)))) + ys*(v011 + xs*(v111 \
        + xs*(v211 + xs*(v311 + v411*xs))) + ys*(v021 + xs*(v121 + xs*(v221 \
        + v321*xs)) + ys*(v031 + xs*(v131 + v231*xs) + ys*(v041 + v141*xs \
        + v051*ys)))) + z*(v002 + xs*(v102 + xs*(v202 + xs*(v302 + v402*xs))) \
        + ys*(v012 + xs*(v112 + xs*(v212 + v312*xs)) + ys*(v022 + xs*(v122 \
        + v222*xs) + ys*(v032 + v132*xs + v042*ys))) + z*(v003 + xs*(v103 \
        + v203*xs) + ys*(v013 + v113*xs + v023*ys) + z*(v004 + v104*xs + v014*ys \
        + z*(v005 + v006*z))))))

    # Compute the reference positions
    # -------------------------------
    zmin = 0.; zmax = 6000.
    zref = np.ones_like(sr) * 2000.
    ztop = np.ones_like(sr) * zmin
    zbot = np.ones_like(sr) * zmax

    # Valid points
    #-------------
    zref_new_ev = '0.25*(1.+ss)*(ztop+zref) + 0.25*(1.-ss)*(zbot+zref)'
    ztop_ev     = '0.5*(1+ss)*ztop + 0.5*(1-ss)*zref'
    zbot_ev     = '0.5*(1-ss)*zbot + 0.5*(1+ss)*zref'

    x2 = sfac*sr
    xs = np.sqrt(x2 + offset)
    ys = ct*0.025

    n_c = 30
    for i_c in range(0,n_c):
        # Compute buoyancy 
        buoyancy = ne.evaluate(rhor) - rho(x2,xs,ys,ne.evaluate(pr)*1e-4)
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
    sigref = rho(x2,xs,ys,ne.evaluate(pr)*1e-4) - 1000.
    gammat = sigref - ne.evaluate(f)

    # Return values
    # -----------------------
    if output_c=='scalar':
        return gammat, np.float(zref), np.float(pref), sigref
    elif output_c=='list':
        return gammat.tolist(), zref.tolist(), pref.tolist(), sigref.tolist()
    else:
        return gammat, zref, pref, sigref
