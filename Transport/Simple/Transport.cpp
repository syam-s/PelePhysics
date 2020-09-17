#include "Transport.H"

void
transport_init()
{
  transport_params::init();
}

void
transport_close()
{
  transport_params::finalize();
}

AMREX_GPU_DEVICE
void
get_transport_coeffs(
  amrex::Box const& bx,
  amrex::Array4<const amrex::Real> const& Y_in,
  amrex::Array4<const amrex::Real> const& T_in,
  amrex::Array4<const amrex::Real> const& Rho_in,
  amrex::Array4<amrex::Real> const& D_out,
  amrex::Array4<amrex::Real> const& mu_out,
  amrex::Array4<amrex::Real> const& xi_out,
  amrex::Array4<amrex::Real> const& lam_out)
{

  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);

  bool wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag;

  wtr_get_xi = true;
  wtr_get_mu = true;
  wtr_get_lam = true;
  wtr_get_Ddiag = true;

  amrex::Real T;
  amrex::Real rho;
  amrex::Real massloc[NUM_SPECIES];

  amrex::Real muloc, xiloc, lamloc;
  amrex::Real Ddiag[NUM_SPECIES];

  for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
      for (int i = lo.x; i <= hi.x; ++i) {

        T = T_in(i, j, k);
        rho = Rho_in(i, j, k);
        for (int n = 0; n < NUM_SPECIES; ++n) {
          massloc[n] = Y_in(i, j, k, n);
        }

        transport(
          wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, T, rho, massloc,
          Ddiag, muloc, xiloc, lamloc);

        //   mu, xi and lambda are stored after D in the diffusion multifab
        for (int n = 0; n < NUM_SPECIES; ++n) {
          D_out(i, j, k, n) = Ddiag[n];
        }

        mu_out(i, j, k) = muloc;
        xi_out(i, j, k) = xiloc;
        lam_out(i, j, k) = lamloc;
      }
    }
  }
}

namespace transport_params {

AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_wt;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_iwt;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_eps;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_sig;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_dip;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_pol;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_zrot;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_fitmu;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_fitlam;
AMREX_GPU_DEVICE_MANAGED amrex::Real* trans_fitdbin;
AMREX_GPU_DEVICE_MANAGED int* trans_nlin;
AMREX_GPU_DEVICE_MANAGED int array_size = NUM_SPECIES;
AMREX_GPU_DEVICE_MANAGED int fit_length = NUM_FIT;

void
init()
{
  //    std::cout << " array_size " << array_size << std::endl;
  //    std::cout << " fit_length " << fit_length << std::endl;
  trans_wt = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_iwt = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_eps = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_sig = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_dip = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_pol = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));
  trans_zrot = static_cast<amrex::Real*>(
    amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real) * array_size));

  trans_fitmu = static_cast<amrex::Real*>(amrex::The_Managed_Arena()->alloc(
    sizeof(amrex::Real) * array_size * fit_length));
  trans_fitlam = static_cast<amrex::Real*>(amrex::The_Managed_Arena()->alloc(
    sizeof(amrex::Real) * array_size * fit_length));
  trans_fitdbin = static_cast<amrex::Real*>(amrex::The_Managed_Arena()->alloc(
    sizeof(amrex::Real) * array_size * array_size * fit_length));

  trans_nlin = static_cast<int*>(
    amrex::The_Managed_Arena()->alloc(sizeof(int) * array_size));

  egtransetWT(trans_wt);
  egtransetEPS(trans_eps);
  egtransetSIG(trans_sig);
  egtransetDIP(trans_dip);
  egtransetPOL(trans_pol);
  egtransetZROT(trans_zrot);
  egtransetNLIN(trans_nlin);
  egtransetCOFETA(trans_fitmu);
  egtransetCOFLAM(trans_fitlam);
  egtransetCOFD(trans_fitdbin);

  for (int i = 0; i < array_size; ++i) {
    trans_iwt[i] = 1. / trans_wt[i];
  }
}

void
finalize()
{
  amrex::The_Managed_Arena()->free(trans_wt);
  amrex::The_Managed_Arena()->free(trans_iwt);
  amrex::The_Managed_Arena()->free(trans_eps);
  amrex::The_Managed_Arena()->free(trans_sig);
  amrex::The_Managed_Arena()->free(trans_dip);
  amrex::The_Managed_Arena()->free(trans_pol);
  amrex::The_Managed_Arena()->free(trans_zrot);
  amrex::The_Managed_Arena()->free(trans_fitmu);
  amrex::The_Managed_Arena()->free(trans_fitlam);
  amrex::The_Managed_Arena()->free(trans_fitdbin);
  amrex::The_Managed_Arena()->free(trans_nlin);
}

} // namespace transport_params
