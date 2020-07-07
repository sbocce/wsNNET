!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! 
! This program computes the dimensionless eigenvalues from a dimensionless Jacobian for the 
! 5 moments system of maximum-entropy equations.
! 
! The user must choose the minimum and maximum values for adimensional heat fluxes q_star
! and the max and min values for sigma.
! The program scales them by defining functions:
!
! xi = log10 (sigma)
! th = asinh(10000 q_star)
!
! Then points are computed by sampling uniformly xi and th from the minimum to the maximum
! value requested.
! This allows to relief the singularity in the origin.
!
! The program then computes the base-10 logarithm of the dimensionless maximum and minimum 
! eigenvalues.
! The minimum eigenvalue is always negative, so before computing the log10 we change its sign.
! 
! OUTPUT:
! th, xi, log10(MAX_eig + 1), -log10(-MIN_eig + 1)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE VARIOUS

IMPLICIT NONE

CONTAINS

FUNCTION THETA_FROM_Q(Q, A)

  IMPLICIT NONE

  REAL(KIND=8), INTENT(IN) :: Q, A
  REAL(KIND=8) :: THETA_FROM_Q

  THETA_FROM_Q = ASINH(A*Q)

END FUNCTION THETA_FROM_Q

! -------------------------------------------

FUNCTION Q_FROM_THETA(THETA, A)

  IMPLICIT NONE

  REAL(KIND=8), INTENT(IN) :: THETA, A
  REAL(KIND=8) :: Q_FROM_THETA, SHTH
 
  Q_FROM_THETA = SINH(THETA)/A

END FUNCTION Q_FROM_THETA

! -------------------------------------------

SUBROUTINE LINSPACE(start_val, stop_val, Npoints, vect)

  REAL(KIND=8), INTENT(IN) :: start_val, stop_val
  INTEGER,      INTENT(IN) :: Npoints
  REAL(KIND=8), DIMENSION(Npoints), INTENT(OUT) :: vect

  INTEGER :: i
  REAL(KIND=8) :: delta
   
  IF (Npoints .EQ. 0) THEN

    RETURN

  ELSEIF (Npoints .EQ. 1) THEN 

    vect(1) = start_val

  ELSE

    delta = (stop_val - start_val)/(Npoints - 1.0d0) ! Division by integer is automatically upcast
  
    DO i = 1, Npoints

      vect(i) = start_val + (i-1)*delta

    END DO

  END IF

END SUBROUTINE LINSPACE

! ############################################

SUBROUTINE COMPUTE_JACOBIAN(rho, P, q, sigma, Jac)

REAL(KIND=8), INTENT(IN) :: rho, P, q, sigma
REAL(KIND=8), DIMENSION(:,:), INTENT(INOUT) :: Jac

REAL(KIND=8) :: r

r = q**2/sigma + 3 - 2*sigma

! ------ Jacobian --------

Jac(1,1) = 0
Jac(1,2) = 1
Jac(1,3) = 0
Jac(1,4) = 0
Jac(1,5) = 0

Jac(2,1) = 0
Jac(2,2) = 0
Jac(2,3) = 1
Jac(2,4) = 0
Jac(2,5) = 0

Jac(3,1) = 0
Jac(3,2) = 0
Jac(3,3) = 0
Jac(3,4) = 1
Jac(3,5) = 0

Jac(4,1) = 0
Jac(4,2) = 0
Jac(4,3) = 0
Jac(4,4) = 0
Jac(4,5) = 1

Jac(5,1) = ((-5.0)*P*(P/rho)**(3.0/2.0)*((q*(rho/P)**(3.0/2.0)*(1&
&0-4*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2&
&+3)))/rho+(16*q**3*(rho/P)**(9.0/2.0))/(rho**3*(sqrt((3-(r*rho)/P&
&**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)**2)))/(2.0*rho)+(P/rho&
&)**(5.0/2.0)*((q*(rho/P)**(3.0/2.0)*(10-4*sqrt(sqrt((3-(r*rho)/P*&
&*2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))/rho+(16*q**3*(rho/P)*&
&*(9.0/2.0))/(rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-&
&(r*rho)/P**2+3)**2))+(P/rho)**(5.0/2.0)*rho*((-(q*(rho/P)**(3.0/2&
&.0)*(10-4*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho&
&)/P**2+3)))/rho**2)+(3.0*q*sqrt(rho/P)*(10-4*sqrt(sqrt((3-(r*rho)&
&/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))/(2.0*P*rho)-(2*q*(&
&rho/P)**(3.0/2.0)*((((8*q**2)/P**3-(2*r*(3-(r*rho)/P**2))/P**2)/s&
&qrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3))/2.0-r/P**2))/(rho*sqr&
&t(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3))-(4&
&8*q**3*(rho/P)**(9.0/2.0))/(rho**4*(sqrt((3-(r*rho)/P**2)**2+(8*q&
&**2*rho)/P**3)-(r*rho)/P**2+3)**2)+(72*q**3*(rho/P)**(7.0/2.0))/(&
&P*rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**&
&2+3)**2)-(32*q**3*(rho/P)**(9.0/2.0)*((((8*q**2)/P**3-(2*r*(3-(r*&
&rho)/P**2))/P**2)/sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3))/2.&
&0-r/P**2))/(rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(&
&r*rho)/P**2+3)**3))
Jac(5,2) = (-3*P*(P/rho)**(5.0/2.0)*(((rho/P)**(3.0/2.0)*(10-4*sq&
&rt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))/&
&rho-(16*q**2*(rho/P)**(3.0/2.0))/(P**3*sqrt((3-(r*rho)/P**2)**2+(&
&8*q**2*rho)/P**3)*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3&
&)-(r*rho)/P**2+3))+(48*q**2*(rho/P)**(9.0/2.0))/(rho**3*(sqrt((3-&
&(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)**2)-(256*q**4&
&*(rho/P)**(9.0/2.0))/(P**3*rho**2*sqrt((3-(r*rho)/P**2)**2+(8*q**&
&2*rho)/P**3)*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)&
&/P**2+3)**3)))-4*q*(P/rho)**(5.0/2.0)*((-(2*q*(rho/P)**(3.0/2.0)*&
&((-(rho*(3-(r*rho)/P**2))/(P**2*sqrt((3-(r*rho)/P**2)**2+(8*q**2*&
&rho)/P**3)))-rho/P**2))/(rho*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**&
&2*rho)/P**3)-(r*rho)/P**2+3)))-(32*q**3*(rho/P)**(9.0/2.0)*((-(rh&
&o*(3-(r*rho)/P**2))/(P**2*sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P&
&**3)))-rho/P**2))/(rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/&
&P**3)-(r*rho)/P**2+3)**3))+(5*r)/rho
Jac(5,3) = (5.0*(P/rho)**(3.0/2.0)*((q*(rho/P)**(3.0/2.0)*(10-4*s&
&qrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))&
&/rho+(16*q**3*(rho/P)**(9.0/2.0))/(rho**3*(sqrt((3-(r*rho)/P**2)*&
&*2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)**2)))/2.0+(P/rho)**(5.0/2.0&
&)*rho*(((-3.0)*q*sqrt(rho/P)*(10-4*sqrt(sqrt((3-(r*rho)/P**2)**2+&
&(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))/(2.0*P**2)-(2*q*(rho/P)**(3.&
&0/2.0)*((((4*r*rho*(3-(r*rho)/P**2))/P**3-(24*q**2*rho)/P**4)/sqr&
&t((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3))/2.0+(2*r*rho)/P**3))/(r&
&ho*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+&
&3))-(72*q**3*(rho/P)**(7.0/2.0))/(P**2*rho**2*(sqrt((3-(r*rho)/P*&
&*2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)**2)-(32*q**3*(rho/P)**(&
&9.0/2.0)*((((4*r*rho*(3-(r*rho)/P**2))/P**3-(24*q**2*rho)/P**4)/s&
&qrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3))/2.0+(2*r*rho)/P**3))/&
&(rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2&
&+3)**3))
Jac(5,4) = (P/rho)**(5.0/2.0)*rho*(((rho/P)**(3.0/2.0)*(10-4*sqrt&
&(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)))/rh&
&o-(16*q**2*(rho/P)**(3.0/2.0))/(P**3*sqrt((3-(r*rho)/P**2)**2+(8*&
&q**2*rho)/P**3)*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-&
&(r*rho)/P**2+3))+(48*q**2*(rho/P)**(9.0/2.0))/(rho**3*(sqrt((3-(r&
&*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P**2+3)**2)-(256*q**4*(&
&rho/P)**(9.0/2.0))/(P**3*rho**2*sqrt((3-(r*rho)/P**2)**2+(8*q**2*&
&rho)/P**3)*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)-(r*rho)/P&
&**2+3)**3))
Jac(5,5) = (P/rho)**(5.0/2.0)*rho*((-(2*q*(rho/P)**(3.0/2.0)*((-(&
&rho*(3-(r*rho)/P**2))/(P**2*sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)&
&/P**3)))-rho/P**2))/(rho*sqrt(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rh&
&o)/P**3)-(r*rho)/P**2+3)))-(32*q**3*(rho/P)**(9.0/2.0)*((-(rho*(3&
&-(r*rho)/P**2))/(P**2*sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3)&
&))-rho/P**2))/(rho**3*(sqrt((3-(r*rho)/P**2)**2+(8*q**2*rho)/P**3&
&)-(r*rho)/P**2+3)**3))

END SUBROUTINE COMPUTE_JACOBIAN

! ******************************************************************

SUBROUTINE COMPUTE_EIGS_LAPACK_DGEEV(MAT, N, WR, WI, INFO)

! INPUT - MAT: ..
! INPUT - N: dimension of square matrix MAT
!
! OUTPUT - WR: real part of eigenvalues
! OUTPUT - WI: imaginary part of eigenvalues
! OUTPUT - INFO: if larger than zero, the algorithm has failed

IMPLICIT NONE

INTEGER, INTENT(IN) :: N ! Dimension of matrix "MAT"
REAL(KIND=8), DIMENSION(N, N), INTENT(IN)  :: MAT
REAL(KIND=8), DIMENSION(N),    INTENT(OUT) :: WR, WI
INTEGER, INTENT(OUT) :: INFO

! Local working variables
INTEGER      ::  LDA, LDVL, LDVR
INTEGER      ::  LWMAX
PARAMETER        ( LWMAX = 1000 )

INTEGER      ::  LWORK

REAL(KIND=8) :: VL(N,N), VR(N,N), WORK( LWMAX )

LDA  = N
LDVL = N
LDVR = N

! Query the optimal workspace
LWORK = -1
CALL DGEEV( 'N', 'N', N, MAT, LDA, WR, WI, VL, LDVL, &
              VR, LDVR, WORK, LWORK, INFO )
LWORK = MIN( LWMAX, INT( WORK( 1 ) ) )

! Solve eigenproblem
CALL DGEEV( 'N', 'N', N, MAT, LDA, WR, WI, VL, LDVL, &
           VR, LDVR, WORK, LWORK, INFO )

! Check for convergence
IF( INFO.GT.0 ) THEN
   WRITE(*,*)'The algorithm failed to compute eigenvalues.'
END IF

END SUBROUTINE COMPUTE_EIGS_LAPACK_DGEEV

END MODULE VARIOUS

! ####################################################################
! ####################################################################
! ####################################################################
! #                                   _                              #
! #                   _ __ ___   __ _(_)_ __                         #  
! #                  | '_ ` _ \ / _` | | '_ \                        #  
! #                  | | | | | | (_| | | | | |                       #  
! #                  |_| |_| |_|\__,_|_|_| |_|                       # 
! #                                                                  # 
! ####################################################################
! ####################################################################
! ####################################################################
! ####################################################################

PROGRAM compute_eigenvalues

USE VARIOUS 

IMPLICIT NONE

INTEGER, PARAMETER :: N_EQ = 5

REAL(KIND=8), DIMENSION(N_EQ)   :: W, eigReal, eigImag
REAL(KIND=8), DIMENSION(N_EQ,N_EQ) :: Jac
INTEGER :: INFO, jj, kk, numHeatflux, numSigma, eig_ID

REAL(KIND=8) :: rho, P, q, sigma
REAL(KIND=8) :: q_min, q_max, sig_min, sig_max

REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: th_vect, xi_vect

REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: maxAbsImagVEC
REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: maxEig, minEig

REAL(KIND=8), DIMENSION(N_EQ) :: eigProcessed

INTEGER :: nargs
CHARACTER(LEN=20) :: BUFFER

REAL(KIND=8), DIMENSION(7) :: testvect
REAL(KIND=8) :: A_SIGMOID 

! Extract command-line arguments
nargs = COMMAND_ARGUMENT_COUNT()

IF (nargs .lt. 1) THEN
  WRITE(*,*) "USAGE: "
  WRITE(*,*) "./a.out  q_MIN  q_MAX  Nq  sig_MIN  sig_MAX  Nsig > outp.csv"
  STOP
END IF

CALL GET_COMMAND_ARGUMENT(1,BUFFER)
READ(BUFFER,FMT='(D10.5)') q_min

CALL GET_COMMAND_ARGUMENT(2,BUFFER)
READ(BUFFER,FMT='(D10.5)') q_max

CALL GET_COMMAND_ARGUMENT(3,BUFFER)
READ(BUFFER,FMT='(I8)') numHeatflux

CALL GET_COMMAND_ARGUMENT(4,BUFFER)
READ(BUFFER,FMT='(D10.5)') sig_min

CALL GET_COMMAND_ARGUMENT(5,BUFFER)
READ(BUFFER,FMT='(D10.5)') sig_max

CALL GET_COMMAND_ARGUMENT(6,BUFFER)
READ(BUFFER,FMT='(I8)') numSigma

ALLOCATE(th_vect(numHeatFlux))
ALLOCATE(xi_vect(numSigma))

! Variables, nondimensionalized!
rho = 1.0d0
P   = 1.0d0

! Create array for xi = log10(sigma)
CALL LINSPACE(LOG10(sig_min), LOG10(sig_max), numSigma, xi_vect) 

! Create array for theta = q + q/(A + |q|)
A_SIGMOID = 10000.0d0
CALL LINSPACE(THETA_FROM_Q(q_min, A_SIGMOID), THETA_FROM_Q(q_max, A_SIGMOID), numHeatFlux, th_vect) 

! LOOP
DO jj = 1, numSigma

  sigma = 10.0d0**(xi_vect(jj)) ! Obtain sigma

  DO kk = 1, numHeatflux

    q = Q_FROM_THETA(th_vect(kk), A_SIGMOID) 

    ! Compute Jacobian
    CALL COMPUTE_JACOBIAN(rho, P, q, sigma, Jac)
      
    ! Compute eigenvalues
    CALL COMPUTE_EIGS_LAPACK_DGEEV(Jac, N_EQ, eigReal, eigImag, INFO)

    IF (INFO .lt. 0.0d0) THEN
      PRINT*, 'ATTENTION! Eigenvalues algorithm has failed!'
      STOP
    END IF
      
    ! Compute absolute value of eigenvalues
    ! FORMULA: W = SIGN(Re)*SQRT( Re**2 + Im**2 )
    DO eig_ID = 1,N_EQ
      eigProcessed(eig_ID) = DSIGN( SQRT(eigReal(eig_ID)**2 + eigImag(eig_ID)**2)  , eigReal(eig_ID))
    END DO
  
    PRINT*, "  ", th_vect(kk), ",  ", xi_vect(jj), ",  ", LOG10(MAXVAL(eigProcessed) + 1), ",  ", -LOG10(-MINVAL(eigProcessed) + 1)

  END DO

END DO

DEALLOCATE(th_vect)
DEALLOCATE(xi_vect)

END PROGRAM
