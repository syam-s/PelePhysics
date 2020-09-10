module ckwc_wrapper_module 

use, intrinsic :: iso_c_binding

public :: ckwc_wrapper

contains

subroutine ckwc_wrapper (T, C, WDOT) bind(C, name="ckwc_wrapper")
  DOUBLE PRECISION, intent(in) :: T, C(*)
  DOUBLE PRECISION, intent(inout) :: WDOT(*)
  doubleprecision Tbefore, Tafter
  
  CALL CKWC(T, C, WDOT)
  
end subroutine ckwc_wrapper

end module ckwc_wrapper_module
