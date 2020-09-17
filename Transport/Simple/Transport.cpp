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

namespace transport_params {

amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_wt = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_iwt = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_eps = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_sig = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_dip = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_pol = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES> trans_zrot = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES * NUM_FIT> trans_fitmu = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES * NUM_FIT> trans_fitlam = {0.0};
amrex::GpuArray<amrex::Real,NUM_SPECIES * NUM_SPECIES * NUM_FIT> trans_fitdbin = {0.0};
amrex::GpuArray<int,NUM_SPECIES> trans_nlin = {0};

void
init()
{
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

  for (int i = 0; i < NUM_SPECIES; ++i) {
    trans_iwt[i] = 1. / trans_wt[i];
  }
}

void
finalize()
{}

} // namespace transport_params
