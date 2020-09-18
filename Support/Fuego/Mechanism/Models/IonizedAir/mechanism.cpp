#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[0], fwd_beta[0], fwd_Ea[0];
    double low_A[0], low_beta[0], low_Ea[0];
    double rev_A[0], rev_beta[0], rev_Ea[0];
    double troe_a[0],troe_Ts[0], troe_Tss[0], troe_Tsss[0];
    double sri_a[0], sri_b[0], sri_c[0], sri_d[0], sri_e[0];
    double activation_units[0], prefactor_units[0], phase_units[0];
    int is_PD[0], troe_len[0], sri_len[0], nTB[0], *TBid[0];
    double *TB[0];
    std::vector<std::vector<double>> kiv(0); 
    std::vector<std::vector<double>> nuv(0); 

    double fwd_A_DEF[0], fwd_beta_DEF[0], fwd_Ea_DEF[0];
    double low_A_DEF[0], low_beta_DEF[0], low_Ea_DEF[0];
    double rev_A_DEF[0], rev_beta_DEF[0], rev_Ea_DEF[0];
    double troe_a_DEF[0],troe_Ts_DEF[0], troe_Tss_DEF[0], troe_Tsss_DEF[0];
    double sri_a_DEF[0], sri_b_DEF[0], sri_c_DEF[0], sri_d_DEF[0], sri_e_DEF[0];
    double activation_units_DEF[0], prefactor_units_DEF[0], phase_units_DEF[0];
    int is_PD_DEF[0], troe_len_DEF[0], sri_len_DEF[0], nTB_DEF[0], *TBid_DEF[0];
    double *TB_DEF[0];
    std::vector<int> rxn_map;
};

using namespace thermo;
#endif

/* Inverse molecular weights */
/* TODO: check necessity on CPU */
static AMREX_GPU_DEVICE_MANAGED double imw[5] = {
    1.0 / 31.998800,  /*O2 */
    1.0 / 28.013400,  /*N2 */
    1.0 / 0.000549,  /*E */
    1.0 / 15.999949,  /*On */
    1.0 / 31.999349};  /*O2n */

/* Inverse molecular weights */
/* TODO: check necessity because redundant with molecularWeight */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[5] = {
    31.998800,  /*O2 */
    28.013400,  /*N2 */
    0.000549,  /*E */
    15.999949,  /*On */
    31.999349};  /*O2n */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<5; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<5; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    rxn_map = {};

    SetAllDefaults();
}

void GET_REACTION_MAP(int *rmap)
{
    for (int i=0; i<0; ++i) {
        rmap[i] = rxn_map[i] + 1;
    }
}

#include <ReactionData.H>
double* GetParamPtr(int                reaction_id,
                    REACTION_PARAMETER param_id,
                    int                species_id,
                    int                get_default)
{
  printf("No reactions in this model");
  abort();
  return 0;
}

void ResetAllParametersToDefault()
{
    for (int i=0; i<0; i++) {
        if (nTB[i] != 0) {
            nTB[i] = 0;
            free(TB[i]);
            free(TBid[i]);
        }

        fwd_A[i]    = fwd_A_DEF[i];
        fwd_beta[i] = fwd_beta_DEF[i];
        fwd_Ea[i]   = fwd_Ea_DEF[i];

        low_A[i]    = low_A_DEF[i];
        low_beta[i] = low_beta_DEF[i];
        low_Ea[i]   = low_Ea_DEF[i];

        rev_A[i]    = rev_A_DEF[i];
        rev_beta[i] = rev_beta_DEF[i];
        rev_Ea[i]   = rev_Ea_DEF[i];

        troe_a[i]    = troe_a_DEF[i];
        troe_Ts[i]   = troe_Ts_DEF[i];
        troe_Tss[i]  = troe_Tss_DEF[i];
        troe_Tsss[i] = troe_Tsss_DEF[i];

        sri_a[i] = sri_a_DEF[i];
        sri_b[i] = sri_b_DEF[i];
        sri_c[i] = sri_c_DEF[i];
        sri_d[i] = sri_d_DEF[i];
        sri_e[i] = sri_e_DEF[i];

        is_PD[i]    = is_PD_DEF[i];
        troe_len[i] = troe_len_DEF[i];
        sri_len[i]  = sri_len_DEF[i];

        activation_units[i] = activation_units_DEF[i];
        prefactor_units[i]  = prefactor_units_DEF[i];
        phase_units[i]      = phase_units_DEF[i];

        nTB[i]  = nTB_DEF[i];
        if (nTB[i] != 0) {
           TB[i] = (double *) malloc(sizeof(double) * nTB[i]);
           TBid[i] = (int *) malloc(sizeof(int) * nTB[i]);
           for (int j=0; j<nTB[i]; j++) {
             TB[i][j] = TB_DEF[i][j];
             TBid[i][j] = TBid_DEF[i][j];
           }
        }
    }
}

void SetAllDefaults()
{
    for (int i=0; i<0; i++) {
        if (nTB_DEF[i] != 0) {
            nTB_DEF[i] = 0;
            free(TB_DEF[i]);
            free(TBid_DEF[i]);
        }

        fwd_A_DEF[i]    = fwd_A[i];
        fwd_beta_DEF[i] = fwd_beta[i];
        fwd_Ea_DEF[i]   = fwd_Ea[i];

        low_A_DEF[i]    = low_A[i];
        low_beta_DEF[i] = low_beta[i];
        low_Ea_DEF[i]   = low_Ea[i];

        rev_A_DEF[i]    = rev_A[i];
        rev_beta_DEF[i] = rev_beta[i];
        rev_Ea_DEF[i]   = rev_Ea[i];

        troe_a_DEF[i]    = troe_a[i];
        troe_Ts_DEF[i]   = troe_Ts[i];
        troe_Tss_DEF[i]  = troe_Tss[i];
        troe_Tsss_DEF[i] = troe_Tsss[i];

        sri_a_DEF[i] = sri_a[i];
        sri_b_DEF[i] = sri_b[i];
        sri_c_DEF[i] = sri_c[i];
        sri_d_DEF[i] = sri_d[i];
        sri_e_DEF[i] = sri_e[i];

        is_PD_DEF[i]    = is_PD[i];
        troe_len_DEF[i] = troe_len[i];
        sri_len_DEF[i]  = sri_len[i];

        activation_units_DEF[i] = activation_units[i];
        prefactor_units_DEF[i]  = prefactor_units[i];
        phase_units_DEF[i]      = phase_units[i];

        nTB_DEF[i]  = nTB[i];
        if (nTB_DEF[i] != 0) {
           TB_DEF[i] = (double *) malloc(sizeof(double) * nTB_DEF[i]);
           TBid_DEF[i] = (int *) malloc(sizeof(int) * nTB_DEF[i]);
           for (int j=0; j<nTB_DEF[i]; j++) {
             TB_DEF[i][j] = TB[i][j];
             TBid_DEF[i][j] = TBid[i][j];
           }
        }
    }
}

/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<0; ++i) {
    free(TB[i]); TB[i] = 0; 
    free(TBid[i]); TBid[i] = 0;
    nTB[i] = 0;

    free(TB_DEF[i]); TB_DEF[i] = 0; 
    free(TBid_DEF[i]); TBid_DEF[i] = 0;
    nTB_DEF[i] = 0;
  }
}

#else
/* TODO: Remove on GPU, right now needed by chemistry_module on FORTRAN */
AMREX_GPU_HOST_DEVICE void CKINIT()
{
}

AMREX_GPU_HOST_DEVICE void CKFINALIZE()
{
}

#endif


/*A few mechanism parameters */
void CKINDX(int * mm, int * kk, int * ii, int * nfit)
{
    *mm = 3;
    *kk = 5;
    *ii = 0;
    *nfit = -1; /*Why do you need this anyway ?  */
}



/* ckxnum... for parsing strings  */
void CKXNUM(char * line, int * nexp, int * lout, int * nval, double *  rval, int * kerr, int lenline )
{
    int n,i; /*Loop Counters */
    char cstr[1000];
    char *saveptr;
    char *p; /*String Tokens */
    /* Strip Comments  */
    for (i=0; i<lenline; ++i) {
        if (line[i]=='!') {
            break;
        }
        cstr[i] = line[i];
    }
    cstr[i] = '\0';

    p = strtok_r(cstr," ", &saveptr);
    if (!p) {
        *nval = 0;
        *kerr = 1;
        return;
    }
    for (n=0; n<*nexp; ++n) {
        rval[n] = atof(p);
        p = strtok_r(NULL, " ", &saveptr);
        if (!p) break;
    }
    *nval = n+1;
    if (*nval < *nexp) *kerr = 1;
    return;
}


/* cksnum... for parsing strings  */
void CKSNUM(char * line, int * nexp, int * lout, char * kray, int * nn, int * knum, int * nval, double *  rval, int * kerr, int lenline, int lenkray)
{
    /*Not done yet ... */
}


/* Returns the vector of strings of element names */
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(3);
    ename[0] = "O";
    ename[1] = "N";
    ename[2] = "E";
}


/* Returns the char strings of element names */
void CKSYME(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*3; i++) {
        kname[i] = ' ';
    }

    /* O  */
    kname[ 0*lenkname + 0 ] = 'O';
    kname[ 0*lenkname + 1 ] = ' ';

    /* N  */
    kname[ 1*lenkname + 0 ] = 'N';
    kname[ 1*lenkname + 1 ] = ' ';

    /* E  */
    kname[ 2*lenkname + 0 ] = 'E';
    kname[ 2*lenkname + 1 ] = ' ';

}


/* Returns the vector of strings of species names */
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(5);
    kname[0] = "O2";
    kname[1] = "N2";
    kname[2] = "E";
    kname[3] = "On";
    kname[4] = "O2n";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*5; i++) {
        kname[i] = ' ';
    }

    /* O2  */
    kname[ 0*lenkname + 0 ] = 'O';
    kname[ 0*lenkname + 1 ] = '2';
    kname[ 0*lenkname + 2 ] = ' ';

    /* N2  */
    kname[ 1*lenkname + 0 ] = 'N';
    kname[ 1*lenkname + 1 ] = '2';
    kname[ 1*lenkname + 2 ] = ' ';

    /* E  */
    kname[ 2*lenkname + 0 ] = 'E';
    kname[ 2*lenkname + 1 ] = ' ';

    /* On  */
    kname[ 3*lenkname + 0 ] = 'O';
    kname[ 3*lenkname + 1 ] = 'N';
    kname[ 3*lenkname + 2 ] = ' ';

    /* O2n  */
    kname[ 4*lenkname + 0 ] = 'O';
    kname[ 4*lenkname + 1 ] = '2';
    kname[ 4*lenkname + 2 ] = 'N';
    kname[ 4*lenkname + 3 ] = ' ';

}


/* Returns R, Rc, Patm */
void CKRP(double *  ru, double *  ruc, double *  pa)
{
     *ru  = 8.31446261815324e+07; 
     *ruc = 1.98721558317399615845; 
     *pa  = 1.01325e+06; 
}


/*Compute P = rhoRT/W(x) */
void CKPX(double *  rho, double *  T, double *  x, double *  P)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    *P = *rho * 8.31446261815324e+07 * (*T) / XW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(y) */
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW = 0;/* for computing mean MW */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    *P = *rho * 8.31446261815324e+07 * (*T) * YOW; /*P = rho*R*T/W */

    return;
}


#ifndef AMREX_USE_CUDA
/*Compute P = rhoRT/W(y) */
void VCKPY(int *  np, double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<5; n++) {
        for (int i=0; i<(*np); i++) {
            YOW[i] += y[n*(*np)+i] * imw[n];
        }
    }

    for (int i=0; i<(*np); i++) {
        P[i] = rho[i] * 8.31446261815324e+07 * T[i] * YOW[i]; /*P = rho*R*T/W */
    }

    return;
}
#endif


/*Compute P = rhoRT/W(c) */
void CKPC(double *  rho, double *  T, double *  c,  double *  P)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*31.998800; /*O2 */
    W += c[1]*28.013400; /*N2 */
    W += c[2]*0.000549; /*E */
    W += c[3]*15.999949; /*On */
    W += c[4]*31.999349; /*O2n */

    for (id = 0; id < 5; ++id) {
        sumC += c[id];
    }
    *P = *rho * 8.31446261815324e+07 * (*T) * sumC / W; /*P = rho*R*T/W */

    return;
}


/*Compute rho = PW(x)/RT */
void CKRHOX(double *  P, double *  T, double *  x,  double *  rho)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    *rho = *P * XW / (8.31446261815324e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[5];

    for (int i = 0; i < 5; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 5; i++)
    {
        YOW += tmp[i];
    }

    *rho = *P / (8.31446261815324e+07 * (*T) * YOW);/*rho = P*W/(R*T) */
    return;
}


/*Compute rho = P*W(c)/(R*T) */
void CKRHOC(double *  P, double *  T, double *  c,  double *  rho)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*31.998800; /*O2 */
    W += c[1]*28.013400; /*N2 */
    W += c[2]*0.000549; /*E */
    W += c[3]*15.999949; /*On */
    W += c[4]*31.999349; /*O2n */

    for (id = 0; id < 5; ++id) {
        sumC += c[id];
    }
    *rho = *P * W / (sumC * (*T) * 8.31446261815324e+07); /*rho = PW/(R*T) */

    return;
}


/*get molecular weight for all species */
void CKWT( double *  wt)
{
    get_mw(wt);
}


/*get atomic weight for all elements */
void CKAWT( double *  awt)
{
    atomicWeight(awt);
}


/*given y[species]: mass fractions */
/*returns mean molecular weight (gm/mole) */
AMREX_GPU_HOST_DEVICE void CKMMWY(double *  y,  double *  wtm)
{
    double YOW = 0;
    double tmp[5];

    for (int i = 0; i < 5; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 5; i++)
    {
        YOW += tmp[i];
    }

    *wtm = 1.0 / YOW;
    return;
}


/*given x[species]: mole fractions */
/*returns mean molecular weight (gm/mole) */
void CKMMWX(double *  x,  double *  wtm)
{
    double XW = 0;/* see Eq 4 in CK Manual */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    *wtm = XW;

    return;
}


/*given c[species]: molar concentration */
/*returns mean molecular weight (gm/mole) */
void CKMMWC(double *  c,  double *  wtm)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*31.998800; /*O2 */
    W += c[1]*28.013400; /*N2 */
    W += c[2]*0.000549; /*E */
    W += c[3]*15.999949; /*On */
    W += c[4]*31.999349; /*O2n */

    for (id = 0; id < 5; ++id) {
        sumC += c[id];
    }
    /* CK provides no guard against divison by zero */
    *wtm = W/sumC;

    return;
}


/*convert y[species] (mass fracs) to x[species] (mole fracs) */
AMREX_GPU_HOST_DEVICE void CKYTX(double *  y,  double *  x)
{
    double YOW = 0;
    double tmp[5];

    for (int i = 0; i < 5; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 5; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 5; i++)
    {
        x[i] = y[i]*imw[i]*YOWINV;
    }
    return;
}


#ifndef AMREX_USE_CUDA
/*convert y[npoints*species] (mass fracs) to x[npoints*species] (mole fracs) */
void VCKYTX(int *  np, double *  y,  double *  x)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<5; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] = y[n*(*np)+i] * imw[n];
            YOW[i] += x[n*(*np)+i];
        }
    }

    for (int i=0; i<(*np); i++) {
        YOW[i] = 1.0/YOW[i];
    }

    for (int n=0; n<5; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] *=  YOW[i];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKYTX(int *  np, double *  y,  double *  x)
{
}
#endif


/*convert y[species] (mass fracs) to c[species] (molar conc) */
void CKYTCP(double *  P, double *  T, double *  y,  double *  c)
{
    double YOW = 0;
    double PWORT;

    /*Compute inverse of mean molecular wt first */
    for (int i = 0; i < 5; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 5; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 5; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 5; i++)
    {
        c[i] = (*rho)  * y[i] * imw[i];
    }
}


/*convert x[species] (mole fracs) to y[species] (mass fracs) */
AMREX_GPU_HOST_DEVICE void CKXTY(double *  x,  double *  y)
{
    double XW = 0; /*See Eq 4, 9 in CK Manual */
    /*Compute mean molecular wt first */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*31.998800*XWinv; 
    y[1] = x[1]*28.013400*XWinv; 
    y[2] = x[2]*0.000549*XWinv; 
    y[3] = x[3]*15.999949*XWinv; 
    y[4] = x[4]*31.999349*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446261815324e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*PORT;
    }

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCR(double *  rho, double *  T, double *  x, double *  c)
{
    int id; /*loop counter */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*ROW;
    }

    return;
}


/*convert c[species] (molar conc) to x[species] (mole fracs) */
void CKCTX(double *  c, double *  x)
{
    int id; /*loop counter */
    double sumC = 0; 

    /*compute sum of c  */
    for (id = 0; id < 5; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 5; ++id) {
        x[id] = c[id]*sumCinv;
    }

    return;
}


/*convert c[species] (molar conc) to y[species] (mass fracs) */
void CKCTY(double *  c, double *  y)
{
    double CW = 0; /*See Eq 12 in CK Manual */
    /*compute denominator in eq 12 first */
    CW += c[0]*31.998800; /*O2 */
    CW += c[1]*28.013400; /*N2 */
    CW += c[2]*0.000549; /*E */
    CW += c[3]*15.999949; /*On */
    CW += c[4]*31.999349; /*O2n */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*31.998800*CWinv; 
    y[1] = c[1]*28.013400*CWinv; 
    y[2] = c[2]*0.000549*CWinv; 
    y[3] = c[3]*15.999949*CWinv; 
    y[4] = c[4]*31.999349*CWinv; 

    return;
}


/*get Cp/R as a function of T  */
/*for all species (Eq 19) */
void CKCPOR(double *  T, double *  cpor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpor, tc);
}


/*get H/RT as a function of T  */
/*for all species (Eq 20) */
void CKHORT(double *  T, double *  hort)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEnthalpy(hort, tc);
}


/*get S/R as a function of T  */
/*for all species (Eq 21) */
void CKSOR(double *  T, double *  sor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sor, tc);
}


/*get specific heat at constant volume as a function  */
/*of T for all species (molar units) */
void CKCVML(double *  T,  double *  cvml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        cvml[id] *= 8.31446261815324e+07;
    }
}


/*get specific heat at constant pressure as a  */
/*function of T for all species (molar units) */
void CKCPML(double *  T,  double *  cpml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        cpml[id] *= 8.31446261815324e+07;
    }
}


/*get internal energy as a function  */
/*of T for all species (molar units) */
void CKUML(double *  T,  double *  uml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        uml[id] *= RT;
    }
}


/*get enthalpy as a function  */
/*of T for all species (molar units) */
void CKHML(double *  T,  double *  hml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        hml[id] *= RT;
    }
}


/*get standard-state Gibbs energy as a function  */
/*of T for all species (molar units) */
void CKGML(double *  T,  double *  gml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    gibbs(gml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        gml[id] *= RT;
    }
}


/*get standard-state Helmholtz free energy as a  */
/*function of T for all species (molar units) */
void CKAML(double *  T,  double *  aml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    helmholtz(aml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        aml[id] *= RT;
    }
}


/*Returns the standard-state entropies in molar units */
void CKSML(double *  T,  double *  sml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        sml[id] *= 8.31446261815324e+07;
    }
}


/*Returns the specific heats at constant volume */
/*in mass units (Eq. 29) */
AMREX_GPU_HOST_DEVICE void CKCVMS(double *  T,  double *  cvms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvms, tc);
    /*multiply by R/molecularweight */
    cvms[0] *= 2.598367006935648e+06; /*O2 */
    cvms[1] *= 2.968030520448514e+06; /*N2 */
    cvms[2] *= 1.515633290043376e+11; /*E */
    cvms[3] *= 5.196555836744617e+06; /*On */
    cvms[4] *= 2.598322461890334e+06; /*O2n */
}


/*Returns the specific heats at constant pressure */
/*in mass units (Eq. 26) */
AMREX_GPU_HOST_DEVICE void CKCPMS(double *  T,  double *  cpms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpms, tc);
    /*multiply by R/molecularweight */
    cpms[0] *= 2.598367006935648e+06; /*O2 */
    cpms[1] *= 2.968030520448514e+06; /*N2 */
    cpms[2] *= 1.515633290043376e+11; /*E */
    cpms[3] *= 5.196555836744617e+06; /*On */
    cpms[4] *= 2.598322461890334e+06; /*O2n */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 5; i++)
    {
        ums[i] *= RT*imw[i];
    }
}


/*Returns enthalpy in mass units (Eq 27.) */
AMREX_GPU_HOST_DEVICE void CKHMS(double *  T,  double *  hms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hms, tc);
    for (int i = 0; i < 5; i++)
    {
        hms[i] *= RT*imw[i];
    }
}


#ifndef AMREX_USE_CUDA
/*Returns enthalpy in mass units (Eq 27.) */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
    double tc[5], h[5];

    for (int i=0; i<(*np); i++) {
        tc[0] = 0.0;
        tc[1] = T[i];
        tc[2] = T[i]*T[i];
        tc[3] = T[i]*T[i]*T[i];
        tc[4] = T[i]*T[i]*T[i]*T[i];

        speciesEnthalpy(h, tc);

        hms[0*(*np)+i] = h[0];
        hms[1*(*np)+i] = h[1];
        hms[2*(*np)+i] = h[2];
        hms[3*(*np)+i] = h[3];
        hms[4*(*np)+i] = h[4];
    }

    for (int n=0; n<5; n++) {
        for (int i=0; i<(*np); i++) {
            hms[n*(*np)+i] *= 8.31446261815324e+07 * T[i] * imw[n];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
}
#endif


/*Returns gibbs in mass units (Eq 31.) */
void CKGMS(double *  T,  double *  gms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    gibbs(gms, tc);
    for (int i = 0; i < 5; i++)
    {
        gms[i] *= RT*imw[i];
    }
}


/*Returns helmholtz in mass units (Eq 32.) */
void CKAMS(double *  T,  double *  ams)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    helmholtz(ams, tc);
    for (int i = 0; i < 5; i++)
    {
        ams[i] *= RT*imw[i];
    }
}


/*Returns the entropies in mass units (Eq 28.) */
void CKSMS(double *  T,  double *  sms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sms, tc);
    /*multiply by R/molecularweight */
    sms[0] *= 2.598367006935648e+06; /*O2 */
    sms[1] *= 2.968030520448514e+06; /*N2 */
    sms[2] *= 1.515633290043376e+11; /*E */
    sms[3] *= 5.196555836744617e+06; /*On */
    sms[4] *= 2.598322461890334e+06; /*O2n */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[5]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 5; ++id) {
        result += x[id]*cpor[id];
    }

    *cpbl = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CP (Eq. 34) */
AMREX_GPU_HOST_DEVICE void CKCPBS(double *  T, double *  y,  double *  cpbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[5], tresult[5]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 5; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 5; i++)
    {
        result += tresult[i];
    }

    *cpbs = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CV (Eq. 35) */
void CKCVBL(double *  T, double *  x,  double *  cvbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[5]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 5; ++id) {
        result += x[id]*cvor[id];
    }

    *cvbl = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CV (Eq. 36) */
AMREX_GPU_HOST_DEVICE void CKCVBS(double *  T, double *  y,  double *  cvbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[5]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*O2 */
    result += cvor[1]*y[1]*imw[1]; /*N2 */
    result += cvor[2]*y[2]*imw[2]; /*E */
    result += cvor[3]*y[3]*imw[3]; /*On */
    result += cvor[4]*y[4]*imw[4]; /*O2n */

    *cvbs = result * 8.31446261815324e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[5]; /* temporary storage */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 5; ++id) {
        result += x[id]*hml[id];
    }

    *hbml = result * RT;
}


/*Returns mean enthalpy of mixture in mass units */
AMREX_GPU_HOST_DEVICE void CKHBMS(double *  T, double *  y,  double *  hbms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[5], tmp[5]; /* temporary storage */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 5; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 5; ++id) {
        result += tmp[id];
    }

    *hbms = result * RT;
}


/*get mean internal energy in molar units */
void CKUBML(double *  T, double *  x,  double *  ubml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double uml[5]; /* temporary energy array */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 5; ++id) {
        result += x[id]*uml[id];
    }

    *ubml = result * RT;
}


/*get mean internal energy in mass units */
AMREX_GPU_HOST_DEVICE void CKUBMS(double *  T, double *  y,  double *  ubms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double ums[5]; /* temporary energy array */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*O2 */
    result += y[1]*ums[1]*imw[1]; /*N2 */
    result += y[2]*ums[2]*imw[2]; /*E */
    result += y[3]*ums[3]*imw[3]; /*On */
    result += y[4]*ums[4]*imw[4]; /*O2n */

    *ubms = result * RT;
}


/*get mixture entropy in molar units */
void CKSBML(double *  P, double *  T, double *  x,  double *  sbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[5]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 5; ++id) {
        result += x[id]*(sor[id]-log((x[id]+1e-100))-logPratio);
    }

    *sbml = result * 8.31446261815324e+07;
}


/*get mixture entropy in mass units */
void CKSBMS(double *  P, double *  T, double *  y,  double *  sbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[5]; /* temporary storage */
    double x[5]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    /*Now compute y to x conversion */
    x[0] = y[0]/(31.998800*YOW); 
    x[1] = y[1]/(28.013400*YOW); 
    x[2] = y[2]/(0.000549*YOW); 
    x[3] = y[3]/(15.999949*YOW); 
    x[4] = y[4]/(31.999349*YOW); 
    speciesEntropy(sor, tc);
    /*Perform computation in Eq 42 and 43 */
    result += x[0]*(sor[0]-log((x[0]+1e-100))-logPratio);
    result += x[1]*(sor[1]-log((x[1]+1e-100))-logPratio);
    result += x[2]*(sor[2]-log((x[2]+1e-100))-logPratio);
    result += x[3]*(sor[3]-log((x[3]+1e-100))-logPratio);
    result += x[4]*(sor[4]-log((x[4]+1e-100))-logPratio);
    /*Scale by R/W */
    *sbms = result * 8.31446261815324e+07 * YOW;
}


/*Returns mean gibbs free energy in molar units */
void CKGBML(double *  P, double *  T, double *  x,  double *  gbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double gort[5]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 5; ++id) {
        result += x[id]*(gort[id]+log((x[id]+1e-100))+logPratio);
    }

    *gbml = result * RT;
}


/*Returns mixture gibbs free energy in mass units */
void CKGBMS(double *  P, double *  T, double *  y,  double *  gbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double gort[5]; /* temporary storage */
    double x[5]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    /*Now compute y to x conversion */
    x[0] = y[0]/(31.998800*YOW); 
    x[1] = y[1]/(28.013400*YOW); 
    x[2] = y[2]/(0.000549*YOW); 
    x[3] = y[3]/(15.999949*YOW); 
    x[4] = y[4]/(31.999349*YOW); 
    gibbs(gort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(gort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(gort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(gort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(gort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(gort[4]+log((x[4]+1e-100))+logPratio);
    /*Scale by RT/W */
    *gbms = result * RT * YOW;
}


/*Returns mean helmholtz free energy in molar units */
void CKABML(double *  P, double *  T, double *  x,  double *  abml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double aort[5]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 5; ++id) {
        result += x[id]*(aort[id]+log((x[id]+1e-100))+logPratio);
    }

    *abml = result * RT;
}


/*Returns mixture helmholtz free energy in mass units */
void CKABMS(double *  P, double *  T, double *  y,  double *  abms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double aort[5]; /* temporary storage */
    double x[5]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    /*Now compute y to x conversion */
    x[0] = y[0]/(31.998800*YOW); 
    x[1] = y[1]/(28.013400*YOW); 
    x[2] = y[2]/(0.000549*YOW); 
    x[3] = y[3]/(15.999949*YOW); 
    x[4] = y[4]/(31.999349*YOW); 
    helmholtz(aort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(aort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(aort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(aort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(aort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(aort[4]+log((x[4]+1e-100))+logPratio);
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 5; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*multiply by 1e6 so c goes to SI */
    PWORT *= 1e6; 
    /*Now compute conversion (and go to SI) */
    c[0] = PWORT * y[0]*imw[0]; 
    c[1] = PWORT * y[1]*imw[1]; 
    c[2] = PWORT * y[2]*imw[2]; 
    c[3] = PWORT * y[3]*imw[3]; 
    c[4] = PWORT * y[4]*imw[4]; 

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
void VCKWYR(int *  np, double *  rho, double *  T,
	    double *  y,
	    double *  wdot)
{
#ifndef AMREX_USE_CUDA
    double c[5*(*np)]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    for (int n=0; n<5; n++) {
        for (int i=0; i<(*np); i++) {
            c[n*(*np)+i] = 1.0e6 * rho[i] * y[n*(*np)+i] * imw[n];
        }
    }

    /*call productionRate */
    vproductionRate(*np, wdot, c, T);

    /*convert to chemkin units */
    for (int i=0; i<5*(*np); i++) {
        wdot[i] *= 1.0e-6;
    }
#endif
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the rate of progress for each reaction */
void CKQC(double *  T, double *  C, double *  qdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 5; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    progressRate(qdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 5; ++id) {
        C[id] *= 1.0e-6;
    }

    for (id = 0; id < 0; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKKFKR(double *  P, double *  T, double *  x, double *  q_f, double *  q_r)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRateFR(q_f, q_r, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 0; ++id) {
        q_f[id] *= 1.0e-6;
        q_r[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mass fractions */
void CKQYP(double *  P, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*O2 */
    YOW += y[1]*imw[1]; /*N2 */
    YOW += y[2]*imw[2]; /*E */
    YOW += y[3]*imw[3]; /*On */
    YOW += y[4]*imw[4]; /*O2n */
    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*multiply by 1e6 so c goes to SI */
    PWORT *= 1e6; 
    /*Now compute conversion (and go to SI) */
    c[0] = PWORT * y[0]*imw[0]; 
    c[1] = PWORT * y[1]*imw[1]; 
    c[2] = PWORT * y[2]*imw[2]; 
    c[3] = PWORT * y[3]*imw[3]; 
    c[4] = PWORT * y[4]*imw[4]; 

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 0; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKQXP(double *  P, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 0; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mass fractions */
void CKQYR(double *  rho, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 

    /*call progressRate */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 0; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mole fractions */
void CKQXR(double *  rho, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[5]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*31.998800; /*O2 */
    XW += x[1]*28.013400; /*N2 */
    XW += x[2]*0.000549; /*E */
    XW += x[3]*15.999949; /*On */
    XW += x[4]*31.999349; /*O2n */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 5; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 0; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the stoichiometric coefficients */
/*of the reaction mechanism. (Eq 50) */
void CKNU(int * kdim,  int * nuki)
{
    int id; /*loop counter */
    int kd = (*kdim); 
    /*Zero nuki */
    for (id = 0; id < 5 * kd; ++ id) {
         nuki[id] = 0; 
    }
}


#ifndef AMREX_USE_CUDA
/*Returns a count of species in a reaction, and their indices */
/*and stoichiometric coefficients. (Eq 50) */
void CKINU(int * i, int * nspec, int * ki, int * nu)
{
    if (*i < 1) {
        /*Return max num species per reaction */
        *nspec = 0;
    } else {
        if (*i > 0) {
            *nspec = -1;
        } else {
            *nspec = kiv[*i-1].size();
            for (int j=0; j<*nspec; ++j) {
                ki[j] = kiv[*i-1][j] + 1;
                nu[j] = nuv[*i-1][j];
            }
        }
    }
}
#endif


/*Returns the elemental composition  */
/*of the speciesi (mdim is num of elements) */
void CKNCF(int * ncf)
{
    int id; /*loop counter */
    int kd = 3; 
    /*Zero ncf */
    for (id = 0; id < kd * 5; ++ id) {
         ncf[id] = 0; 
    }

    /*O2 */
    ncf[ 0 * kd + 0 ] = 2; /*O */

    /*N2 */
    ncf[ 1 * kd + 1 ] = 2; /*N */

    /*E */
    ncf[ 2 * kd + 2 ] = 1; /*E */

    /*On */
    ncf[ 3 * kd + 0 ] = 1; /*O */
    ncf[ 3 * kd + 2 ] = 1; /*E */

    /*O2n */
    ncf[ 4 * kd + 0 ] = 2; /*O */
    ncf[ 4 * kd + 2 ] = 1; /*E */

}


/*Returns the arrehenius coefficients  */
/*for all reactions */
void CKABE( double *  a, double *  b, double *  e)
{

    return;
}


/*Returns the equil constants for each reaction */
void CKEQC(double *  T, double *  C, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[5]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mass fractions */
void CKEQYP(double *  P, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[5]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mole fractions */
void CKEQXP(double *  P, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[5]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mass fractions */
void CKEQYR(double *  rho, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[5]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mole fractions */
void CKEQXR(double *  rho, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[5]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);
}


/*Returns the electronic charges of the species */
void CKCHRG(int * kcharge)
{
    kcharge[0] = 0; /* O2 */
    kcharge[1] = 0; /* N2 */
    kcharge[2] = -1; /* E */
    kcharge[3] = -1; /* On */
    kcharge[4] = -1; /* O2n */
}

#ifdef AMREX_USE_CUDA
/*GPU version of productionRate: no more use of thermo namespace vectors */
/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE inline void  productionRate(double * wdot, double * sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];


    for (int i = 0; i < 5; ++i) {
        wdot[i] = 0.0;
    }

    return;
}

AMREX_GPU_HOST_DEVICE inline void comp_qfqr(double *  qf, double * qr, double * sc, double * tc, double invT)
{

    return;
}
#endif


#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[0];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[0];
#ifdef _OPENMP
#pragma omp threadprivate(Kc_save)
#endif


/*compute the production rate for each species pointwise on CPU */
void productionRate(double *  wdot, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }

    double qdot, q_f[0], q_r[0];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 5; ++i) {
        wdot[i] = 0.0;
    }

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
    for (int i=0; i<0; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[5];
    gibbs(g_RT, tc);


    for (int i=0; i<0; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;


    return;
}

void comp_qfqr(double *  qf, double *  qr, double *  sc, double *  tc, double invT)
{

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 5; ++i) {
        mixture += sc[i];
    }

    double Corr[0];
    for (int i = 0; i < 0; ++i) {
        Corr[i] = 1.0;
    }

    for (int i=0; i<0; i++)
    {
        qf[i] *= Corr[i] * k_f_save[i];
        qr[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}
#endif


#ifndef AMREX_USE_CUDA
/*compute the production rate for each species */
void vproductionRate(int npt, double *  wdot, double *  sc, double *  T)
{
    double k_f_s[0*npt], Kc_s[0*npt], mixture[npt], g_RT[5*npt];
    double tc[5*npt], invT[npt];

    for (int i=0; i<npt; i++) {
        tc[0*npt+i] = log(T[i]);
        tc[1*npt+i] = T[i];
        tc[2*npt+i] = T[i]*T[i];
        tc[3*npt+i] = T[i]*T[i]*T[i];
        tc[4*npt+i] = T[i]*T[i]*T[i]*T[i];
        invT[i] = 1.0 / T[i];
    }

    for (int i=0; i<npt; i++) {
        mixture[i] = 0.0;
    }

    for (int n=0; n<5; n++) {
        for (int i=0; i<npt; i++) {
            mixture[i] += sc[n*npt+i];
            wdot[n*npt+i] = 0.0;
        }
    }

    vcomp_k_f(npt, k_f_s, tc, invT);

    vcomp_gibbs(npt, g_RT, tc);

    vcomp_Kc(npt, Kc_s, g_RT, invT);

    vcomp_wdot(npt, wdot, mixture, sc, k_f_s, Kc_s, tc, invT, T);
}

void vcomp_k_f(int npt, double *  k_f_s, double *  tc, double *  invT)
{
    for (int i=0; i<npt; i++) {
    }
}

void vcomp_gibbs(int npt, double *  g_RT, double *  tc)
{
    /*compute the Gibbs free energy */
    for (int i=0; i<npt; i++) {
        double tg[5], g[5];
        tg[0] = tc[0*npt+i];
        tg[1] = tc[1*npt+i];
        tg[2] = tc[2*npt+i];
        tg[3] = tc[3*npt+i];
        tg[4] = tc[4*npt+i];

        gibbs(g, tg);

        g_RT[0*npt+i] = g[0];
        g_RT[1*npt+i] = g[1];
        g_RT[2*npt+i] = g[2];
        g_RT[3*npt+i] = g[3];
        g_RT[4*npt+i] = g[4];
    }
}

void vcomp_Kc(int npt, double *  Kc_s, double *  g_RT, double *  invT)
{
    for (int i=0; i<npt; i++) {
        /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
        double refC = (101325. / 8.31451) * invT[i];
        double refCinv = 1.0 / refC;

    }
}

void vcomp_wdot(int npt, double *  wdot, double *  mixture, double *  sc,
		double *  k_f_s, double *  Kc_s,
		double *  tc, double *  invT, double *  T)
{
    for (int i=0; i<npt; i++) {
        double qdot, q_f, q_r, phi_f, phi_r, k_f, k_r, Kc;
        double alpha;
    }
}
#endif

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[5];

    for (int k=0; k<5; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<5; k++) {
        J[30+k] *= 1.e-6;
        J[k*6+5] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[5];

    for (int k=0; k<5; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<5; k++) {
        J[30+k] *= 1.e-6;
        J[k*6+5] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[5];
    double J[36];

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<6; k++) {
        for (int l=0; l<6; l++) {
            if(J[ 6 * k + l] != 0.0){
                nJdata_tmp = nJdata_tmp + 1;
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST( int * nJdata, int * consP, int NCELLS)
{
    double c[5];
    double J[36];

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<6; k++) {
        for (int l=0; l<6; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 6 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the simplified (for preconditioning) system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, int * consP)
{
    double c[5];
    double J[36];

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<6; k++) {
        for (int l=0; l<6; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 6 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    nJdata[0] = nJdata_tmp;

    return;
}


/*compute the sparsity pattern of the chemistry Jacobian in CSC format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, int * consP, int NCELLS)
{
    double c[5];
    double J[36];
    int offset_row;
    int offset_col;

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 6;
        offset_col = nc * 6;
        for (int k=0; k<6; k++) {
            for (int l=0; l<6; l++) {
                if(J[6*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l + offset_row; 
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
            colPtrs[offset_col + (k + 1)] = nJdata_tmp;
        }
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian in CSR format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base)
{
    double c[5];
    double J[36];
    int offset;

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 6;
            for (int l=0; l<6; l++) {
                for (int k=0; k<6; k++) {
                    if(J[6*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtrs[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 6;
            for (int l=0; l<6; l++) {
                for (int k=0; k<6; k++) {
                    if(J[6*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the system Jacobian */
/*CSR format BASE is user choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, int * consP, int NCELLS, int base)
{
    double c[5];
    double J[36];
    int offset;

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 6;
            for (int l=0; l<6; l++) {
                for (int k=0; k<6; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[6*k + l] != 0.0) {
                            colVals[nJdata_tmp-1] = k+1 + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 6;
            for (int l=0; l<6; l++) {
                for (int k=0; k<6; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[6*k + l] != 0.0) {
                            colVals[nJdata_tmp] = k + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian on CPU */
/*BASE 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP)
{
    double c[5];
    double J[36];

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<6; k++) {
        for (int l=0; l<6; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 6*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[6*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 6*k + l;
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
        }
        colPtrs[k+1] = nJdata_tmp;
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian */
/*CSR format BASE is under choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base)
{
    double c[5];
    double J[36];

    for (int k=0; k<5; k++) {
        c[k] = 1.0/ 5.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<6; l++) {
            for (int k=0; k<6; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[6*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int l=0; l<6; l++) {
            for (int k=0; k<6; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[6*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    }

    return;
}


#ifdef AMREX_USE_CUDA
/*compute the reaction Jacobian on GPU */
AMREX_GPU_HOST_DEVICE
void aJacobian(double * J, double * sc, double T, int consP)
{


    for (int i=0; i<36; i++) {
        J[i] = 0.0;
    }

    double wdot[5];
    for (int k=0; k<5; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 5; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[5];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[5];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[5];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    double c_R[5], dcRdT[5], e_RT[5];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 5; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[30+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 5; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 5; ++m) {
            dehmixdc += eh_RT[m]*J[k*6+m];
        }
        J[k*6+5] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[35] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;

return;
}
#endif


#ifndef AMREX_USE_CUDA
/*compute the reaction Jacobian on CPU */
void aJacobian(double *  J, double *  sc, double T, int consP)
{
    for (int i=0; i<36; i++) {
        J[i] = 0.0;
    }

    double wdot[5];
    for (int k=0; k<5; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 5; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[5];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[5];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[5];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    double c_R[5], dcRdT[5], e_RT[5];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 5; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[30+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 5; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 5; ++m) {
            dehmixdc += eh_RT[m]*J[k*6+m];
        }
        J[k*6+5] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[35] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<36; i++) {
        J[i] = 0.0;
    }

    double wdot[5];
    for (int k=0; k<5; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 5; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[5];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[5];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[5];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    double c_R[5], dcRdT[5], e_RT[5];
    double * eh_RT;
    if (HP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 5; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[30+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 5; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 5; ++m) {
            dehmixdc += eh_RT[m]*J[k*6+m];
        }
        J[k*6+5] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[35] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
}


/*compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void dcvpRdT(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +1.12748635e-03
            -1.15123009e-06 * tc[1]
            +3.94163169e-09 * tc[2]
            -3.50742157e-12 * tc[3];
        /*species 1: N2 */
        species[1] =
            +1.40824000e-03
            -7.92644400e-06 * tc[1]
            +1.69245450e-08 * tc[2]
            -9.77942000e-12 * tc[3];
        /*species 2: E */
        species[2] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 3: On */
        species[3] =
            -1.69804907e-03
            +5.96139910e-06 * tc[1]
            -7.31505381e-09 * tc[2]
            +3.04491724e-12 * tc[3];
        /*species 4: O2n */
        species[4] =
            -9.28741138e-04
            +1.29095416e-05 * tc[1]
            -2.32411014e-08 * tc[2]
            +1.17333065e-11 * tc[3];
    } else {
        /*species 0: O2 */
        species[0] =
            +6.13519689e-04
            -2.51768398e-07 * tc[1]
            +5.32584444e-11 * tc[2]
            -4.54574124e-15 * tc[3];
        /*species 1: N2 */
        species[1] =
            +1.48797700e-03
            -1.13695220e-06 * tc[1]
            +3.02911200e-10 * tc[2]
            -2.70134040e-14 * tc[3];
        /*species 2: E */
        species[2] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 3: On */
        species[3] =
            -4.66695513e-05
            +3.69824714e-08 * tc[1]
            -9.54477669e-12 * tc[2]
            +7.95851824e-16 * tc[3];
        /*species 4: O2n */
        species[4] =
            +5.98141823e-04
            -4.24267810e-07 * tc[1]
            +1.08980274e-10 * tc[2]
            -8.99956912e-15 * tc[3];
    }
    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRate(double *  qdot, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

#ifndef AMREX_USE_CUDA
    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }
#endif


    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRateFR(double *  q_f, double *  q_r, double *  sc, double T)
{
    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;

    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            -1.005249020000000e+03 * invT
            -2.821801190000000e+00
            -3.212936400000000e+00 * tc[0]
            -5.637431750000000e-04 * tc[1]
            +9.593584116666666e-08 * tc[2]
            -1.094897691666667e-10 * tc[3]
            +4.384276960000000e-14 * tc[4];
        /*species 1: N2 */
        species[1] =
            -1.020900000000000e+03 * invT
            -6.516950000000001e-01
            -3.298677000000000e+00 * tc[0]
            -7.041200000000000e-04 * tc[1]
            +6.605369999999999e-07 * tc[2]
            -4.701262500000001e-10 * tc[3]
            +1.222427500000000e-13 * tc[4];
        /*species 2: E */
        species[2] =
            -7.453750000000000e+02 * invT
            +1.422081220000000e+01
            -2.500000000000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.143577170000000e+04 * invT
            +1.046682399999996e-01
            -2.908059210000000e+00 * tc[0]
            +8.490245350000000e-04 * tc[1]
            -4.967832583333333e-07 * tc[2]
            +2.031959391666667e-10 * tc[3]
            -3.806146555000000e-14 * tc[4];
        /*species 4: O2n */
        species[4] =
            -6.870769830000000e+03 * invT
            -6.869815900000003e-01
            -3.664425220000000e+00 * tc[0]
            +4.643705690000000e-04 * tc[1]
            -1.075795136666667e-06 * tc[2]
            +6.455861500000001e-10 * tc[3]
            -1.466663310000000e-13 * tc[4];
    } else {
        /*species 0: O2 */
        species[0] =
            -1.233930180000000e+03 * invT
            +5.084126000000002e-01
            -3.697578190000000e+00 * tc[0]
            -3.067598445000000e-04 * tc[1]
            +2.098069983333333e-08 * tc[2]
            -1.479401233333333e-12 * tc[3]
            +5.682176550000000e-17 * tc[4];
        /*species 1: N2 */
        species[1] =
            -9.227977000000000e+02 * invT
            -3.053888000000000e+00
            -2.926640000000000e+00 * tc[0]
            -7.439885000000000e-04 * tc[1]
            +9.474601666666666e-08 * tc[2]
            -8.414199999999999e-12 * tc[3]
            +3.376675500000000e-16 * tc[4];
        /*species 2: E */
        species[2] =
            -7.453750000000000e+02 * invT
            +1.422081220000000e+01
            -2.500000000000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.150420890000000e+04 * invT
            -1.976561460000000e+00
            -2.544748690000000e+00 * tc[0]
            +2.333477565000000e-05 * tc[1]
            -3.081872616666667e-09 * tc[2]
            +2.651326858333333e-13 * tc[3]
            -9.948147799999999e-18 * tc[4];
        /*species 4: O2n */
        species[4] =
            -7.062872290000000e+03 * invT
            +1.677952770000000e+00
            -3.956662940000000e+00 * tc[0]
            -2.990709115000000e-04 * tc[1]
            +3.535565083333334e-08 * tc[2]
            -3.027229841666666e-12 * tc[3]
            +1.124946140000000e-16 * tc[4];
    }
    return;
}


/*compute the a/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void helmholtz(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            -1.00524902e+03 * invT
            -3.82180119e+00
            -3.21293640e+00 * tc[0]
            -5.63743175e-04 * tc[1]
            +9.59358412e-08 * tc[2]
            -1.09489769e-10 * tc[3]
            +4.38427696e-14 * tc[4];
        /*species 1: N2 */
        species[1] =
            -1.02090000e+03 * invT
            -1.65169500e+00
            -3.29867700e+00 * tc[0]
            -7.04120000e-04 * tc[1]
            +6.60537000e-07 * tc[2]
            -4.70126250e-10 * tc[3]
            +1.22242750e-13 * tc[4];
        /*species 2: E */
        species[2] =
            -7.45375000e+02 * invT
            +1.32208122e+01
            -2.50000000e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.14357717e+04 * invT
            -8.95331760e-01
            -2.90805921e+00 * tc[0]
            +8.49024535e-04 * tc[1]
            -4.96783258e-07 * tc[2]
            +2.03195939e-10 * tc[3]
            -3.80614656e-14 * tc[4];
        /*species 4: O2n */
        species[4] =
            -6.87076983e+03 * invT
            -1.68698159e+00
            -3.66442522e+00 * tc[0]
            +4.64370569e-04 * tc[1]
            -1.07579514e-06 * tc[2]
            +6.45586150e-10 * tc[3]
            -1.46666331e-13 * tc[4];
    } else {
        /*species 0: O2 */
        species[0] =
            -1.23393018e+03 * invT
            -4.91587400e-01
            -3.69757819e+00 * tc[0]
            -3.06759845e-04 * tc[1]
            +2.09806998e-08 * tc[2]
            -1.47940123e-12 * tc[3]
            +5.68217655e-17 * tc[4];
        /*species 1: N2 */
        species[1] =
            -9.22797700e+02 * invT
            -4.05388800e+00
            -2.92664000e+00 * tc[0]
            -7.43988500e-04 * tc[1]
            +9.47460167e-08 * tc[2]
            -8.41420000e-12 * tc[3]
            +3.37667550e-16 * tc[4];
        /*species 2: E */
        species[2] =
            -7.45375000e+02 * invT
            +1.32208122e+01
            -2.50000000e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.15042089e+04 * invT
            -2.97656146e+00
            -2.54474869e+00 * tc[0]
            +2.33347756e-05 * tc[1]
            -3.08187262e-09 * tc[2]
            +2.65132686e-13 * tc[3]
            -9.94814780e-18 * tc[4];
        /*species 4: O2n */
        species[4] =
            -7.06287229e+03 * invT
            +6.77952770e-01
            -3.95666294e+00 * tc[0]
            -2.99070912e-04 * tc[1]
            +3.53556508e-08 * tc[2]
            -3.02722984e-12 * tc[3]
            +1.12494614e-16 * tc[4];
    }
    return;
}


/*compute Cv/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cv_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +2.21293640e+00
            +1.12748635e-03 * tc[1]
            -5.75615047e-07 * tc[2]
            +1.31387723e-09 * tc[3]
            -8.76855392e-13 * tc[4];
        /*species 1: N2 */
        species[1] =
            +2.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 2: E */
        species[2] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.90805921e+00
            -1.69804907e-03 * tc[1]
            +2.98069955e-06 * tc[2]
            -2.43835127e-09 * tc[3]
            +7.61229311e-13 * tc[4];
        /*species 4: O2n */
        species[4] =
            +2.66442522e+00
            -9.28741138e-04 * tc[1]
            +6.45477082e-06 * tc[2]
            -7.74703380e-09 * tc[3]
            +2.93332662e-12 * tc[4];
    } else {
        /*species 0: O2 */
        species[0] =
            +2.69757819e+00
            +6.13519689e-04 * tc[1]
            -1.25884199e-07 * tc[2]
            +1.77528148e-11 * tc[3]
            -1.13643531e-15 * tc[4];
        /*species 1: N2 */
        species[1] =
            +1.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 2: E */
        species[2] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +1.54474869e+00
            -4.66695513e-05 * tc[1]
            +1.84912357e-08 * tc[2]
            -3.18159223e-12 * tc[3]
            +1.98962956e-16 * tc[4];
        /*species 4: O2n */
        species[4] =
            +2.95666294e+00
            +5.98141823e-04 * tc[1]
            -2.12133905e-07 * tc[2]
            +3.63267581e-11 * tc[3]
            -2.24989228e-15 * tc[4];
    }
    return;
}


/*compute Cp/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cp_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +3.21293640e+00
            +1.12748635e-03 * tc[1]
            -5.75615047e-07 * tc[2]
            +1.31387723e-09 * tc[3]
            -8.76855392e-13 * tc[4];
        /*species 1: N2 */
        species[1] =
            +3.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 2: E */
        species[2] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +2.90805921e+00
            -1.69804907e-03 * tc[1]
            +2.98069955e-06 * tc[2]
            -2.43835127e-09 * tc[3]
            +7.61229311e-13 * tc[4];
        /*species 4: O2n */
        species[4] =
            +3.66442522e+00
            -9.28741138e-04 * tc[1]
            +6.45477082e-06 * tc[2]
            -7.74703380e-09 * tc[3]
            +2.93332662e-12 * tc[4];
    } else {
        /*species 0: O2 */
        species[0] =
            +3.69757819e+00
            +6.13519689e-04 * tc[1]
            -1.25884199e-07 * tc[2]
            +1.77528148e-11 * tc[3]
            -1.13643531e-15 * tc[4];
        /*species 1: N2 */
        species[1] =
            +2.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 2: E */
        species[2] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 3: On */
        species[3] =
            +2.54474869e+00
            -4.66695513e-05 * tc[1]
            +1.84912357e-08 * tc[2]
            -3.18159223e-12 * tc[3]
            +1.98962956e-16 * tc[4];
        /*species 4: O2n */
        species[4] =
            +3.95666294e+00
            +5.98141823e-04 * tc[1]
            -2.12133905e-07 * tc[2]
            +3.63267581e-11 * tc[3]
            -2.24989228e-15 * tc[4];
    }
    return;
}


/*compute the e/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +2.21293640e+00
            +5.63743175e-04 * tc[1]
            -1.91871682e-07 * tc[2]
            +3.28469308e-10 * tc[3]
            -1.75371078e-13 * tc[4]
            -1.00524902e+03 * invT;
        /*species 1: N2 */
        species[1] =
            +2.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 2: E */
        species[2] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 3: On */
        species[3] =
            +1.90805921e+00
            -8.49024535e-04 * tc[1]
            +9.93566517e-07 * tc[2]
            -6.09587818e-10 * tc[3]
            +1.52245862e-13 * tc[4]
            +1.14357717e+04 * invT;
        /*species 4: O2n */
        species[4] =
            +2.66442522e+00
            -4.64370569e-04 * tc[1]
            +2.15159027e-06 * tc[2]
            -1.93675845e-09 * tc[3]
            +5.86665324e-13 * tc[4]
            -6.87076983e+03 * invT;
    } else {
        /*species 0: O2 */
        species[0] =
            +2.69757819e+00
            +3.06759845e-04 * tc[1]
            -4.19613997e-08 * tc[2]
            +4.43820370e-12 * tc[3]
            -2.27287062e-16 * tc[4]
            -1.23393018e+03 * invT;
        /*species 1: N2 */
        species[1] =
            +1.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 2: E */
        species[2] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 3: On */
        species[3] =
            +1.54474869e+00
            -2.33347756e-05 * tc[1]
            +6.16374523e-09 * tc[2]
            -7.95398057e-13 * tc[3]
            +3.97925912e-17 * tc[4]
            +1.15042089e+04 * invT;
        /*species 4: O2n */
        species[4] =
            +2.95666294e+00
            +2.99070912e-04 * tc[1]
            -7.07113017e-08 * tc[2]
            +9.08168952e-12 * tc[3]
            -4.49978456e-16 * tc[4]
            -7.06287229e+03 * invT;
    }
    return;
}


/*compute the h/(RT) at the given temperature (Eq 20) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +3.21293640e+00
            +5.63743175e-04 * tc[1]
            -1.91871682e-07 * tc[2]
            +3.28469308e-10 * tc[3]
            -1.75371078e-13 * tc[4]
            -1.00524902e+03 * invT;
        /*species 1: N2 */
        species[1] =
            +3.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 2: E */
        species[2] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 3: On */
        species[3] =
            +2.90805921e+00
            -8.49024535e-04 * tc[1]
            +9.93566517e-07 * tc[2]
            -6.09587818e-10 * tc[3]
            +1.52245862e-13 * tc[4]
            +1.14357717e+04 * invT;
        /*species 4: O2n */
        species[4] =
            +3.66442522e+00
            -4.64370569e-04 * tc[1]
            +2.15159027e-06 * tc[2]
            -1.93675845e-09 * tc[3]
            +5.86665324e-13 * tc[4]
            -6.87076983e+03 * invT;
    } else {
        /*species 0: O2 */
        species[0] =
            +3.69757819e+00
            +3.06759845e-04 * tc[1]
            -4.19613997e-08 * tc[2]
            +4.43820370e-12 * tc[3]
            -2.27287062e-16 * tc[4]
            -1.23393018e+03 * invT;
        /*species 1: N2 */
        species[1] =
            +2.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 2: E */
        species[2] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 3: On */
        species[3] =
            +2.54474869e+00
            -2.33347756e-05 * tc[1]
            +6.16374523e-09 * tc[2]
            -7.95398057e-13 * tc[3]
            +3.97925912e-17 * tc[4]
            +1.15042089e+04 * invT;
        /*species 4: O2n */
        species[4] =
            +3.95666294e+00
            +2.99070912e-04 * tc[1]
            -7.07113017e-08 * tc[2]
            +9.08168952e-12 * tc[3]
            -4.49978456e-16 * tc[4]
            -7.06287229e+03 * invT;
    }
    return;
}


/*compute the S/R at the given temperature (Eq 21) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEntropy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: O2 */
        species[0] =
            +3.21293640e+00 * tc[0]
            +1.12748635e-03 * tc[1]
            -2.87807523e-07 * tc[2]
            +4.37959077e-10 * tc[3]
            -2.19213848e-13 * tc[4]
            +6.03473759e+00 ;
        /*species 1: N2 */
        species[1] =
            +3.29867700e+00 * tc[0]
            +1.40824000e-03 * tc[1]
            -1.98161100e-06 * tc[2]
            +1.88050500e-09 * tc[3]
            -6.11213750e-13 * tc[4]
            +3.95037200e+00 ;
        /*species 2: E */
        species[2] =
            +2.50000000e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -1.17208122e+01 ;
        /*species 3: On */
        species[3] =
            +2.90805921e+00 * tc[0]
            -1.69804907e-03 * tc[1]
            +1.49034977e-06 * tc[2]
            -8.12783757e-10 * tc[3]
            +1.90307328e-13 * tc[4]
            +2.80339097e+00 ;
        /*species 4: O2n */
        species[4] =
            +3.66442522e+00 * tc[0]
            -9.28741138e-04 * tc[1]
            +3.22738541e-06 * tc[2]
            -2.58234460e-09 * tc[3]
            +7.33331655e-13 * tc[4]
            +4.35140681e+00 ;
    } else {
        /*species 0: O2 */
        species[0] =
            +3.69757819e+00 * tc[0]
            +6.13519689e-04 * tc[1]
            -6.29420995e-08 * tc[2]
            +5.91760493e-12 * tc[3]
            -2.84108828e-16 * tc[4]
            +3.18916559e+00 ;
        /*species 1: N2 */
        species[1] =
            +2.92664000e+00 * tc[0]
            +1.48797700e-03 * tc[1]
            -2.84238050e-07 * tc[2]
            +3.36568000e-11 * tc[3]
            -1.68833775e-15 * tc[4]
            +5.98052800e+00 ;
        /*species 2: E */
        species[2] =
            +2.50000000e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -1.17208122e+01 ;
        /*species 3: On */
        species[3] =
            +2.54474869e+00 * tc[0]
            -4.66695513e-05 * tc[1]
            +9.24561785e-09 * tc[2]
            -1.06053074e-12 * tc[3]
            +4.97407390e-17 * tc[4]
            +4.52131015e+00 ;
        /*species 4: O2n */
        species[4] =
            +3.95666294e+00 * tc[0]
            +5.98141823e-04 * tc[1]
            -1.06066953e-07 * tc[2]
            +1.21089194e-11 * tc[3]
            -5.62473070e-16 * tc[4]
            +2.27871017e+00 ;
    }
    return;
}


/*save atomic weights into array */
void atomicWeight(double *  awt)
{
    awt[0] = 15.999400; /*O */
    awt[1] = 14.006700; /*N */
    awt[2] = 0.000549; /*E */

    return;
}


/* get temperature given internal energy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double ein  = *e;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double e1,emin,emax,cv,t1,dt;
    int i;/* loop counter */
    CKUBMS(&tmin, y, &emin);
    CKUBMS(&tmax, y, &emax);
    if (ein < emin) {
        /*Linear Extrapolation below tmin */
        CKCVBS(&tmin, y, &cv);
        *t = tmin - (emin-ein)/cv;
        *ierr = 1;
        return;
    }
    if (ein > emax) {
        /*Linear Extrapolation above tmax */
        CKCVBS(&tmax, y, &cv);
        *t = tmax - (emax-ein)/cv;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(emax-emin)*(ein-emin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKUBMS(&t1,y,&e1);
        CKCVBS(&t1,y,&cv);
        dt = (ein - e1) / cv;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}

/* get temperature given enthalpy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double hin  = *h;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double h1,hmin,hmax,cp,t1,dt;
    int i;/* loop counter */
    CKHBMS(&tmin, y, &hmin);
    CKHBMS(&tmax, y, &hmax);
    if (hin < hmin) {
        /*Linear Extrapolation below tmin */
        CKCPBS(&tmin, y, &cp);
        *t = tmin - (hmin-hin)/cp;
        *ierr = 1;
        return;
    }
    if (hin > hmax) {
        /*Linear Extrapolation above tmax */
        CKCPBS(&tmax, y, &cp);
        *t = tmax - (hmax-hin)/cp;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(hmax-hmin)*(hin-hmin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKHBMS(&t1,y,&h1);
        CKCPBS(&t1,y,&cp);
        dt = (hin - h1) / cp;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}


/*compute the critical parameters for each species */
void GET_CRITPARAMS(double *  Tci, double *  ai, double *  bi, double *  acentric_i)
{

    double   EPS[5];
    double   SIG[5];
    double    wt[5];
    double avogadro = 6.02214199e23;
    double boltzmann = 1.3806503e-16; //we work in CGS
    double Rcst = 83.144598; //in bar [CGS] !

    egtransetEPS(EPS);
    egtransetSIG(SIG);
    get_mw(wt);

    /*species 0: O2 */
    /*Imported from NIST */
    Tci[0] = 154.581000 ; 
    ai[0] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[0],2.0) / (pow(31.998800,2.0) * 50.430466); 
    bi[0] = 0.08664 * Rcst * Tci[0] / (31.998800 * 50.430466); 
    acentric_i[0] = 0.022200 ;

    /*species 1: N2 */
    /*Imported from NIST */
    Tci[1] = 126.192000 ; 
    ai[1] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[1],2.0) / (pow(28.013400,2.0) * 33.958000); 
    bi[1] = 0.08664 * Rcst * Tci[1] / (28.013400 * 33.958000); 
    acentric_i[1] = 0.037200 ;

    /*species 2: E */
    Tci[2] = 1.316 * EPS[2] ; 
    ai[2] = (5.55 * pow(avogadro,2.0) * EPS[2]*boltzmann * pow(1e-8*SIG[2],3.0) ) / (pow(wt[2],2.0)); 
    bi[2] = 0.855 * avogadro * pow(1e-8*SIG[2],3.0) / (wt[2]); 
    acentric_i[2] = 0.0 ;

    /*species 3: On */
    Tci[3] = 1.316 * EPS[3] ; 
    ai[3] = (5.55 * pow(avogadro,2.0) * EPS[3]*boltzmann * pow(1e-8*SIG[3],3.0) ) / (pow(wt[3],2.0)); 
    bi[3] = 0.855 * avogadro * pow(1e-8*SIG[3],3.0) / (wt[3]); 
    acentric_i[3] = 0.0 ;

    /*species 4: O2n */
    Tci[4] = 1.316 * EPS[4] ; 
    ai[4] = (5.55 * pow(avogadro,2.0) * EPS[4]*boltzmann * pow(1e-8*SIG[4],3.0) ) / (pow(wt[4],2.0)); 
    bi[4] = 0.855 * avogadro * pow(1e-8*SIG[4],3.0) / (wt[4]); 
    acentric_i[4] = 0.0 ;

    return;
}


void egtransetLENIMC(int* LENIMC ) {
    *LENIMC = 21;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 630;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 5;}


void egtransetNLITE(int* NLITE ) {
    *NLITE = 1;}


/*Patm in ergs/cm3 */
void egtransetPATM(double* PATM) {
    *PATM =   0.1013250000000000E+07;}


/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 3.19988000E+01;
    WT[1] = 2.80134000E+01;
    WT[2] = 5.48580100E-04;
    WT[3] = 1.59999486E+01;
    WT[4] = 3.19993486E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 1.07400000E+02;
    EPS[1] = 9.75300000E+01;
    EPS[2] = 8.50000000E+02;
    EPS[3] = 8.00000000E+01;
    EPS[4] = 1.07400000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 3.45800000E+00;
    SIG[1] = 3.62100000E+00;
    SIG[2] = 4.25000000E+02;
    SIG[3] = 2.75000000E+00;
    SIG[4] = 3.45800000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[0] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[4] = 0.00000000E+00;
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 1.60000000E+00;
    POL[1] = 1.76000000E+00;
    POL[2] = 0.00000000E+00;
    POL[3] = 0.00000000E+00;
    POL[4] = 1.60000000E+00;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 3.80000000E+00;
    ZROT[1] = 4.00000000E+00;
    ZROT[2] = 1.00000000E+00;
    ZROT[3] = 0.00000000E+00;
    ZROT[4] = 3.80000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 1;
    NLIN[1] = 1;
    NLIN[2] = 0;
    NLIN[3] = 0;
    NLIN[4] = 1;
}


/*Poly fits for the viscosities, dim NO*KK */
void egtransetCOFETA(double* COFETA) {
    COFETA[0] = -1.60066324E+01;
    COFETA[1] = 2.16753735E+00;
    COFETA[2] = -1.97226850E-01;
    COFETA[3] = 8.50065468E-03;
    COFETA[4] = -1.55270326E+01;
    COFETA[5] = 1.92766908E+00;
    COFETA[6] = -1.66518287E-01;
    COFETA[7] = 7.19100649E-03;
    COFETA[8] = -2.69531456E+01;
    COFETA[9] = -7.88142336E-01;
    COFETA[10] = 3.18038402E-01;
    COFETA[11] = -1.85151697E-02;
    COFETA[12] = -1.40007422E+01;
    COFETA[13] = 1.44311220E+00;
    COFETA[14] = -1.02767741E-01;
    COFETA[15] = 4.39662983E-03;
    COFETA[16] = -1.60066238E+01;
    COFETA[17] = 2.16753735E+00;
    COFETA[18] = -1.97226850E-01;
    COFETA[19] = 8.50065468E-03;
}


/*Poly fits for the conductivities, dim NO*KK */
void egtransetCOFLAM(double* COFLAM) {
    COFLAM[0] = -2.11869892E+00;
    COFLAM[1] = 2.98568651E+00;
    COFLAM[2] = -2.86879123E-01;
    COFLAM[3] = 1.23850873E-02;
    COFLAM[4] = 7.60997504E+00;
    COFLAM[5] = -1.18418698E+00;
    COFLAM[6] = 3.03558703E-01;
    COFLAM[7] = -1.54159597E-02;
    COFLAM[8] = 1.12880481E-01;
    COFLAM[9] = -7.88142336E-01;
    COFLAM[10] = 3.18038402E-01;
    COFLAM[11] = -1.85151697E-02;
    COFLAM[12] = 2.78452117E+00;
    COFLAM[13] = 1.44311220E+00;
    COFLAM[14] = -1.02767741E-01;
    COFLAM[15] = 4.39662983E-03;
    COFLAM[16] = -4.17001666E+00;
    COFLAM[17] = 3.85409566E+00;
    COFLAM[18] = -3.99415108E-01;
    COFLAM[19] = 1.68318766E-02;
}


/*Poly fits for the diffusion coefficients, dim NO*KK*KK */
void egtransetCOFD(double* COFD) {
    COFD[0] = -1.47079646E+01;
    COFD[1] = 3.10657376E+00;
    COFD[2] = -1.85922460E-01;
    COFD[3] = 7.92680827E-03;
    COFD[4] = -1.44285949E+01;
    COFD[5] = 2.99858376E+00;
    COFD[6] = -1.72232643E-01;
    COFD[7] = 7.34804765E-03;
    COFD[8] = -2.43418402E+01;
    COFD[9] = 5.33526425E+00;
    COFD[10] = -4.46630688E-01;
    COFD[11] = 1.81086658E-02;
    COFD[12] = -1.34772736E+01;
    COFD[13] = 2.80526779E+00;
    COFD[14] = -1.47524446E-01;
    COFD[15] = 6.29376912E-03;
    COFD[16] = -1.47079688E+01;
    COFD[17] = 3.10657376E+00;
    COFD[18] = -1.85922460E-01;
    COFD[19] = 7.92680827E-03;
    COFD[20] = -1.44285949E+01;
    COFD[21] = 2.99858376E+00;
    COFD[22] = -1.72232643E-01;
    COFD[23] = 7.34804765E-03;
    COFD[24] = -1.42056656E+01;
    COFD[25] = 2.91297621E+00;
    COFD[26] = -1.61544771E-01;
    COFD[27] = 6.90271324E-03;
    COFD[28] = -2.41515313E+01;
    COFD[29] = 5.28689054E+00;
    COFD[30] = -4.42932076E-01;
    COFD[31] = 1.80445515E-02;
    COFD[32] = -1.32254717E+01;
    COFD[33] = 2.69833584E+00;
    COFD[34] = -1.33541002E-01;
    COFD[35] = 5.68346517E-03;
    COFD[36] = -1.44285989E+01;
    COFD[37] = 2.99858376E+00;
    COFD[38] = -1.72232643E-01;
    COFD[39] = 7.34804765E-03;
    COFD[40] = -2.43418402E+01;
    COFD[41] = 5.33526425E+00;
    COFD[42] = -4.46630688E-01;
    COFD[43] = 1.81086658E-02;
    COFD[44] = -2.41515313E+01;
    COFD[45] = 5.28689054E+00;
    COFD[46] = -4.42932076E-01;
    COFD[47] = 1.80445515E-02;
    COFD[48] = -1.82216966E+01;
    COFD[49] = 1.68992430E+00;
    COFD[50] = 1.09701574E-01;
    COFD[51] = -8.81888811E-03;
    COFD[52] = -2.36348024E+01;
    COFD[53] = 5.13665150E+00;
    COFD[54] = -4.28328574E-01;
    COFD[55] = 1.75969700E-02;
    COFD[56] = -2.43418402E+01;
    COFD[57] = 5.33526425E+00;
    COFD[58] = -4.46630688E-01;
    COFD[59] = 1.81086658E-02;
    COFD[60] = -1.34772736E+01;
    COFD[61] = 2.80526779E+00;
    COFD[62] = -1.47524446E-01;
    COFD[63] = 6.29376912E-03;
    COFD[64] = -1.32254717E+01;
    COFD[65] = 2.69833584E+00;
    COFD[66] = -1.33541002E-01;
    COFD[67] = 5.68346517E-03;
    COFD[68] = -2.36348024E+01;
    COFD[69] = 5.13665150E+00;
    COFD[70] = -4.28328574E-01;
    COFD[71] = 1.75969700E-02;
    COFD[72] = -1.22676670E+01;
    COFD[73] = 2.49197560E+00;
    COFD[74] = -1.06603364E-01;
    COFD[75] = 4.50971238E-03;
    COFD[76] = -1.34772764E+01;
    COFD[77] = 2.80526779E+00;
    COFD[78] = -1.47524446E-01;
    COFD[79] = 6.29376912E-03;
    COFD[80] = -1.47079688E+01;
    COFD[81] = 3.10657376E+00;
    COFD[82] = -1.85922460E-01;
    COFD[83] = 7.92680827E-03;
    COFD[84] = -1.44285989E+01;
    COFD[85] = 2.99858376E+00;
    COFD[86] = -1.72232643E-01;
    COFD[87] = 7.34804765E-03;
    COFD[88] = -2.43418402E+01;
    COFD[89] = 5.33526425E+00;
    COFD[90] = -4.46630688E-01;
    COFD[91] = 1.81086658E-02;
    COFD[92] = -1.34772764E+01;
    COFD[93] = 2.80526779E+00;
    COFD[94] = -1.47524446E-01;
    COFD[95] = 6.29376912E-03;
    COFD[96] = -1.47079731E+01;
    COFD[97] = 3.10657376E+00;
    COFD[98] = -1.85922460E-01;
    COFD[99] = 7.92680827E-03;
}


/*List of specs with small weight, dim NLITE */
void egtransetKTDIF(int* KTDIF) {
    KTDIF[0] = 3;
}


/*Poly fits for thermal diff ratios, dim NO*NLITE*KK */
void egtransetCOFTD(double* COFTD) {
    COFTD[0] = -8.18816846E-02;
    COFTD[1] = 6.25102119E-04;
    COFTD[2] = -1.93882484E-07;
    COFTD[3] = 1.90693525E-11;
    COFTD[4] = -6.27142420E-02;
    COFTD[5] = 6.17357858E-04;
    COFTD[6] = -1.93698207E-07;
    COFTD[7] = 1.91874330E-11;
    COFTD[8] = 0.00000000E+00;
    COFTD[9] = 0.00000000E+00;
    COFTD[10] = 0.00000000E+00;
    COFTD[11] = 0.00000000E+00;
    COFTD[12] = -2.08077510E-02;
    COFTD[13] = 5.95806874E-04;
    COFTD[14] = -1.91160905E-07;
    COFTD[15] = 1.92016569E-11;
    COFTD[16] = -8.18816846E-02;
    COFTD[17] = 6.25102119E-04;
    COFTD[18] = -1.93882484E-07;
    COFTD[19] = 1.90693525E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

