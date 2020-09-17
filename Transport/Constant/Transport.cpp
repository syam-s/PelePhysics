#include "Transport.H"

namespace transport {

void
transport_init(
  amrex::GpuArray<amrex::Real, max_prob_param>& transport_parm_real)
{
  transport_parm_real[const_viscosity] = 0.0;
  transport_parm_real[const_bulk_viscosity] = 0.0;
  transport_parm_real[const_diffusivity] = 0.0;
  transport_parm_real[const_conductivity] = 0.0;

  amrex::ParmParse pp("transport");
  pp.query("const_viscosity", transport_parm_real[const_viscosity]);
  pp.query("const_bulk_viscosity", transport_parm_real[const_bulk_viscosity]);
  pp.query("const_conductivity", transport_parm_real[const_conductivity]);
  pp.query("const_diffusivity", transport_parm_real[const_diffusivity]);
}

void
transport_close()
{
}

} // namespace transport
