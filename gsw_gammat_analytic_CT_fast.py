# Import modules
import gsw
import numexpr as ne
import numpy as np

def gsw_gammat_analytic_CT_fast(sr, ct):
    # gsw_gammat_analytic_CT_fast: Compute thermodynamic neutral density based on an
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
    # DEPENDENCIES: 
    #   - gsw_rho_CT_exact(sr,ct,p): equation of state for seawater as a function
    #     of reference composition salinity (sr), conservative temperature (ct)
    #     and pressure (p, in dbars) 
    #   - gsw
    #     git repository cloned from https://github.com/TEOS-10/GSW-Python (29.Oct.2020)
    #     function rho_CT_exact not available in this repository, included here as function
    #     gsw_rho_CT_exact from original python-gsw (https://github.com/TEOS-10/python-gsw/) which will be
    #     replaced by GSW-Python after a brief overlap period (https://teos-10.github.io/GSW-Python/)
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
    else:
        output_c = 'array'

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
    n_c = 30
    for i_c in range(0,n_c):
        # Compute buoyancy 
        buoyancy = ne.evaluate(rhor) - gsw.density.rho(sr,ct,ne.evaluate(pr))
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
    sigref = gsw.density.rho(sr,ct,pref) - 1000.
    gammat = sigref - ne.evaluate(f)

    # Return values
    # -----------------------
    if output_c=='scalar':
        return gammat, np.float(zref), np.float(pref), sigref
    elif output_c=='list':
        return gammat.tolist(), zref.tolist(), pref.tolist(), sigref.tolist()
    else:
        return gammat, zref, pref, sigref

